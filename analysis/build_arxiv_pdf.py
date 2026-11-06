"""Build the arXiv-format PDF: figures in paper mode, appendices from markdown, Tectonic.

Kept separate from build_pdf.py (the HTML-derived circulation artifact), which is
retained deliberately: the two serve different purposes and the markdown remains the
single source both derive from.

Requires `tectonic` on PATH or at --tectonic. Tectonic is a single ~20 MB binary that
fetches TeX packages on demand, chosen over a full TeXLive install.

Usage:
    python analysis/build_arxiv_pdf.py --tectonic <path-to-tectonic>
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXDIR = REPO_ROOT / "paper" / "tex"
FIGPAPER = REPO_ROOT / "paper" / "figures-paper"
OUT = REPO_ROOT / "paper" / "adapter-retention-arxiv.pdf"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


#: An Overfull box wider than this is content running off the page rather than a
#: typesetting nicety. Tables that overflowed the arXiv PDF for a whole draft cycle ran
#: 19-299pt over; ordinary prose boxes that no reader would notice run under 5pt.
OVERFULL_LIMIT_PT = 6.0

#: This gate matches Overfull \hbox ("too wide") only, and deliberately ignores Overfull
#: \vbox ("too high"). The two are not the same defect:
#:
#:   \hbox  content wider than the column. In a two-column layout it runs into the gutter
#:          or off the paper, and it is unreadable where it lands. Nothing recovers it.
#:   \vbox  a page's column ran taller than \textheight, so the last line sits lower than
#:          the grid intended. TeX does not clip; the text is fully rendered and simply
#:          eats into the bottom margin.
#:
#: Verified rather than assumed, on the build this note was written for: 42 overfull
#: vboxes, worst 8.41pt. Rasterizing every page and measuring the lowest body ink
#: (excluding the folio, which sits in the footer band by design) put the worst overhang
#: at 8.6pt into a 54pt margin, on page 13, with nothing leaving the physical page on any
#: page. At that scale it is invisible. If a vbox overfull ever approached the 54pt
#: margin it would be a real defect, so the number is stated here rather than left as
#: "vboxes are fine".
_OVERFULL = re.compile(r"([\w.]+):(\d+): Overfull .hbox \(([0-9.]+)pt too wide")


def check_overfull(log: str, limit_pt: float) -> bool:
    """Fail the build on any Overfull box beyond `limit_pt`.

    Same principle as the figure cross-checks and the non-ASCII gate: the failure is
    invisible in the source and silent in the output, so the build has to be the thing
    that notices. Tectonic reports an alignment's overfull box at the line that CLOSES
    it, not at the offending row, so line numbers point at \\end{center}.
    """
    worst: dict[tuple[str, int], float] = {}
    for m in _OVERFULL.finditer(log):
        key = (m.group(1), int(m.group(2)))
        worst[key] = max(worst.get(key, 0.0), float(m.group(3)))
    over = sorted(((v, f, l) for (f, l), v in worst.items() if v > limit_pt),
                  reverse=True)
    if not over:
        n = len(worst)
        print(f"     no overfull box beyond {limit_pt:.0f}pt "
              f"({n} under it)")
        return True
    print(f"     {len(over)} overfull boxes beyond {limit_pt:.0f}pt "
          f"-- content is running off the page:", file=sys.stderr)
    for amt, f, l in over[:12]:
        print(f"       {amt:>7.1f}pt  {f}:{l}", file=sys.stderr)
    if len(over) > 12:
        print(f"       ... and {len(over) - 12} more", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tectonic", default=shutil.which("tectonic"))
    ap.add_argument("--overfull-pt", type=float, default=OVERFULL_LIMIT_PT,
                    help="fail the build on an overfull box wider than this")
    args = ap.parse_args()
    if not args.tectonic:
        print("tectonic not found; pass --tectonic <path>", file=sys.stderr)
        return 1

    env = dict(os.environ, AR_FIG_PAPER="1", PYTHONPATH=str(REPO_ROOT / "src"))
    FIGPAPER.mkdir(parents=True, exist_ok=True)

    print("1/3  regenerating figures in paper mode (headers suppressed)")
    stub = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "from pathlib import Path\n"
        "import fig01_erasure_vs_survival as a, fig05_06_08 as b, fig_secondary as c\n"
        "for m in (a, b, c): m.FIGDIR = Path(r'%s')\n"
        "sys.argv = ['x']\n"
        "a.main(); b.main(); c.main()\n" % (REPO_ROOT / "analysis", FIGPAPER)
    )
    r = run([sys.executable, "-c", stub], env=env)
    if r.returncode:
        print(r.stdout[-3000:] + r.stderr[-3000:], file=sys.stderr)
        return 1
    if "MISMATCH" in r.stdout:
        print("figure cross-check failed:\n" + r.stdout, file=sys.stderr)
        return 1
    print(f"     {len(list(FIGPAPER.glob('*.pdf')))} vector figures, cross-checks passed")

    print("2/3  converting appendices from markdown")
    r = run([sys.executable, str(REPO_ROOT / "analysis" / "md_to_tex.py"), "--write"],
            env=env)
    print("     " + r.stdout.strip().replace("\n", "\n     "))
    # The exit code, not just the output. This check read stdout only, so when the
    # converter began raising on an unmapped glyph it left the previous appendices.tex
    # in place and the build carried on and reported success -- shipping an arXiv PDF
    # whose Appendix B was two rounds behind its own markdown. M.7: tooling reports
    # success on the operation, not on the outcome.
    if r.returncode:
        print(f"md_to_tex.py exited {r.returncode}; appendices.tex was NOT regenerated "
              f"and any PDF built now would carry the previous one:\n{r.stderr[-3000:]}",
              file=sys.stderr)
        return 1
    if "survived conversion" in r.stdout:
        print("non-ASCII survived conversion; fix before shipping", file=sys.stderr)
        return 1

    print("3/3  running tectonic")
    before = (TEXDIR / "main.pdf").stat().st_mtime if (TEXDIR / "main.pdf").exists() else 0
    r = run([args.tectonic, "-X", "compile", "main.tex"], cwd=TEXDIR)

    # Tectonic's exit code was never checked, and `main.pdf` exists from the previous
    # build, so a failed compile left every gate below inspecting a STALE artifact and
    # reporting it clean. A "Missing $ inserted" error passed the overfull check, the
    # cross-reference check and the cross-table check, all of which were reading
    # yesterday's PDF. Fail on the return code, and independently on the file not having
    # been rewritten, because the second catches a compiler that exits 0 without output.
    tex_errors = [l for l in (r.stdout + r.stderr).splitlines()
                  if l.startswith("error:")]
    if r.returncode or tex_errors:
        print(f"     tectonic failed (exit {r.returncode}):", file=sys.stderr)
        for line in tex_errors[:10]:
            print("       " + line, file=sys.stderr)
        if not tex_errors:
            print((r.stdout + r.stderr)[-2000:], file=sys.stderr)
        return 1
    if (TEXDIR / "main.pdf").exists() and (TEXDIR / "main.pdf").stat().st_mtime == before:
        print("     tectonic exited 0 but did not rewrite main.pdf; refusing to ship "
              "a stale artifact", file=sys.stderr)
        return 1

    bad = [l for l in (r.stdout + r.stderr).splitlines()
           if "Missing character" in l or "could not represent" in l]
    if bad:
        print(f"     {len(bad)} unicode errors:", file=sys.stderr)
        for l in bad[:5]:
            print("       " + l, file=sys.stderr)
        return 1

    if not check_overfull(r.stdout + r.stderr, args.overfull_pt):
        return 1

    # Standing gate, not a one-time fix: sections move between drafts and a dangling
    # reference is not a LaTeX error, just a sentence pointing nowhere.
    sys.path.insert(0, str(REPO_ROOT / "analysis"))
    import xref

    unresolved = xref.check((TEXDIR / "main.tex").read_text(encoding="utf-8"),
                            (TEXDIR / "appendices.tex").read_text(encoding="utf-8"))
    if unresolved:
        uniq = sorted({f"{k} {t}" for k, t, _ in unresolved})
        print(f"     {len(unresolved)} unresolved cross-references: {', '.join(uniq[:12])}",
              file=sys.stderr)
        return 1
    print("     all cross-references resolve")

    # And that they resolve to the RIGHT target. The appendix markdown writes its own
    # labels while this build lets LaTeX count, so inserting a subsection shifts every
    # later one and the check above still passes -- every shifted target exists.
    # Existence is not correctness: a reference translated into a section that exists but
    # is about something else passes every check above. Three did, for two rounds.
    misaligned = xref.section_alignment()
    if misaligned:
        print(f"     {len(misaligned)} references land on the wrong section:",
              file=sys.stderr)
        for label, target, best, got, want in misaligned:
            print(f"       report {label} -> paper {target} (content match {got:.2f}); "
                  f"paper {best} matches {want:.2f}", file=sys.stderr)
        return 1
    print("     every translated reference lands on the section it is about")

    drift = xref.numbering_drift()
    if drift:
        print(f"     {len(drift)} appendix headings disagree with the number LaTeX will "
              "give them:", file=sys.stderr)
        for name, label, why in drift:
            print(f"       [{name}] {label} {why}", file=sys.stderr)
        return 1
    print("     markdown appendix labels match the LaTeX numbering")

    # And across the PDF/repo boundary, in both directions. Round 8 moved content to
    # METHODOLOGY.md and PROMPTS.md; if those leave the reference gate they become the
    # next main.tex, which is the defect that survived three consecutive rounds.
    cross = xref.companion_refs((TEXDIR / "main.tex").read_text(encoding="utf-8"),
                                (TEXDIR / "appendices.tex").read_text(encoding="utf-8"))
    if cross:
        print(f"     {len(cross)} references dangle across the PDF/repo boundary:",
              file=sys.stderr)
        for kind, target, ctx in cross:
            print(f"       {kind} {target}: ...{ctx}", file=sys.stderr)
        return 1
    live = [c for c in xref.COMPANIONS if (REPO_ROOT / c).exists()]
    print(f"     references resolve across the boundary to {', '.join(live)}")

    # A citation block naming the wrong work is the boundary check one level up, and it
    # shipped for the whole project: the README's BibTeX title was not the paper's.
    titles = xref.title_disagreements()
    if titles:
        print(f"     {len(titles)} artifacts name the paper by a title it does "
              "not have:", file=sys.stderr)
        for where, got in titles:
            print(f"       {where}: {got!r}", file=sys.stderr)
        return 1
    authors = xref.author_disagreements()
    if authors:
        print(f"     {len(authors)} artifacts attribute the work to a different "
              "person:", file=sys.stderr)
        for where, got in authors:
            print(f"       {where}: {got!r}", file=sys.stderr)
        return 1
    print("     every citation block names the paper's actual title and author")

    # Table-to-table agreement. The prose-level check passed 18/18 while two tables
    # printed the same interval with different values, because both were tables.
    import tablecheck

    clash = tablecheck.disagreements()
    if clash:
        print(f"     {len(clash)} cells disagree between artifacts:", file=sys.stderr)
        for (a, b), entries in clash[:8]:
            print(f"       {a} x {b}: "
                  + "; ".join(f"{v} ({o})" for v, o in entries), file=sys.stderr)
        return 1
    print("     all cross-table cells agree")

    # Count words are claims the claim audit cannot see: not a printed measurement, and
    # what they count is a structure. Four were live when this was added.
    import countcheck

    have = countcheck.structures()
    miscounts = []
    for path in [TEXDIR / "main.tex"] + sorted((REPO_ROOT / "paper").glob("*.md")):
        for hit in countcheck.check(path.read_text(encoding="utf-8"), have):
            miscounts.append((path.name,) + hit)
    if miscounts:
        print(f"     {len(miscounts)} count words disagree with what they count:",
              file=sys.stderr)
        for name, phrase, claimed, actual, what, _ in miscounts[:10]:
            print(f"       [{name}] {phrase!r}: says {claimed}, there are "
                  f"{actual} {what}", file=sys.stderr)
        return 1
    print(f"     every count word agrees with what it counts "
          f"({len(countcheck.RULES)} rules)")

    # Retracted claims are invisible to every gate above: a sentence with no number in it
    # has nothing for the claim audit to recompute, nothing for countcheck to resolve and
    # no reference for xref to follow. Four corrections had landed in the appendix that
    # measured them and nowhere else when this was added (METHODOLOGY.md M.9).
    import retracted

    asserted = retracted.check()
    if asserted:
        print(f"     {len(asserted)} retracted wordings are asserted again:",
              file=sys.stderr)
        for rel, pat, replacement, where, ctx in asserted[:10]:
            print(f"       [{rel}] {pat}", file=sys.stderr)
            print(f"         retracted in {where}; now reads: {replacement}",
                  file=sys.stderr)
            print(f"         ...{ctx}...", file=sys.stderr)
        return 1
    print(f"     no retracted wording is asserted anywhere "
          f"({len(retracted.RETRACTED)} wordings, "
          f"{len(retracted.perimeter())} files)")

    # The other propagation mode. `retracted.py` catches a claim withdrawn in the
    # appendix and still asserted in the body; this catches one added to a summary and
    # never stated by a section. The round that built the first committed the second.
    import forward

    orphan = forward.unsourced(
        (TEXDIR / "main.tex").read_text(encoding="utf-8"),
        (TEXDIR / "appendices.tex").read_text(encoding="utf-8"))
    if orphan:
        print(f"     {len(orphan)} summary claims are stated by no section:",
              file=sys.stderr)
        for site, claim, ctx in orphan[:10]:
            print(f"       [{site}] {claim}\n         ...{ctx}...", file=sys.stderr)
        return 1
    print("     every summary claim is stated by a section that measures it "
          f"({len(forward.summary_regions((TEXDIR / 'main.tex').read_text(encoding='utf-8')))} sites)")

    built = TEXDIR / "main.pdf"
    if not built.exists():
        print((r.stdout + r.stderr)[-3000:], file=sys.stderr)
        return 1

    # Every gate above reads sources. This one reads what the reader sees, which is the
    # only place a destroyed control sequence is visible: it compiles to plain text and
    # TeX says nothing. Four rounds shipped one.
    import texcheck

    debris = texcheck.scan(texcheck.pdf_text(built))
    if debris:
        why = {n: reason for n, _, reason in texcheck.CHECKS}
        print(f"     {len(debris)} pieces of LaTeX/encoding debris in the rendered text:",
              file=sys.stderr)
        for name, hit, ctx in debris[:12]:
            print(f"       [{name}] {hit!r} -- {why[name]}", file=sys.stderr)
            print(f"         ...{ctx}...", file=sys.stderr)
        return 1
    print(f"     no LaTeX or encoding debris in the rendered text "
          f"({len(texcheck.CHECKS)} checks)")

    shutil.copy(built, OUT)
    try:
        from pypdf import PdfReader
        pages = len(PdfReader(str(OUT)).pages)
    except Exception:
        pages = -1
    print(f"     wrote {OUT.relative_to(REPO_ROOT)} "
          f"({OUT.stat().st_size / 1024 / 1024:.2f} MB, {pages} pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
