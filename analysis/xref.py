"""Resolve every cross-reference in the built paper against its actual structure.

The appendices were written against a section numbering that later changed, and about
forty references went stale: section 8 cites "Appendix E" for a table in Appendix C,
Appendix A cites subsections of sections that have none. Nothing noticed, because a
dangling reference in LaTeX is not an error -- it is a sentence that reads fine and
points nowhere.

This builds the document's real structure by numbering the LaTeX exactly as LaTeX does,
then resolves every reference found in the text against it.

The known trap, from this project's own record: a previous cross-reference checker
reported nine false positives because its pattern required whitespace after the section
number where the text uses a period. So the matcher here is tested in both directions --
it must flag a reference known to be broken and pass one known to be good -- and those
are regression tests, not a one-time check.

Usage:
    python analysis/xref.py            # report
    python analysis/xref.py --strict   # exit 1 on any unresolved reference
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXDIR = REPO_ROOT / "paper" / "tex"

#: Body sections are \section in main.tex; appendices restart lettering after \appendix.
_SEC = re.compile(r"^\\(section|subsection|subsubsection)\*?\{(.*)\}\s*$")
_APPENDIX = re.compile(r"^\\appendix\b")


def structure(main: str, appendices: str) -> tuple[set[str], set[str], dict[str, str]]:
    """(section labels, appendix labels, label -> title) as LaTeX will number them."""
    secs: set[str] = set()
    apps: set[str] = set()
    titles: dict[str, str] = {}
    in_appendix = False
    n = [0, 0, 0]           # section, subsection, subsubsection counters
    letter = 0

    for line in (main + "\n" + appendices).splitlines():
        if _APPENDIX.match(line):
            in_appendix = True
            n = [0, 0, 0]
            continue
        m = _SEC.match(line.strip())
        if not m:
            continue
        kind, title = m.group(1), m.group(2)
        if kind == "section":
            n[1] = n[2] = 0
            if in_appendix:
                letter += 1
                label = chr(ord("A") + letter - 1)
                apps.add(label)
            else:
                n[0] += 1
                label = str(n[0])
                secs.add(label)
            titles[label] = title
        elif kind == "subsection":
            n[1] += 1
            n[2] = 0
            head = chr(ord("A") + letter - 1) if in_appendix else str(n[0])
            label = f"{head}.{n[1]}"
            (apps if in_appendix else secs).add(label)
            titles[label] = title
        else:
            n[2] += 1
            head = chr(ord("A") + letter - 1) if in_appendix else str(n[0])
            label = f"{head}.{n[1]}.{n[2]}"
            (apps if in_appendix else secs).add(label)
            titles[label] = title
    return secs, apps, titles


#: A reference is the marker plus its target. The target ends at the first character that
#: cannot be part of one: crucially a period FOLLOWED BY whitespace ends a sentence, but a
#: period between digits does not. Requiring whitespace after the number -- the mistake
#: the earlier checker made -- misses every reference that ends a sentence.
REFS = [
    ("section", re.compile(r"(?:\\S\{\}|\\S|§)\s?(\d+(?:\.\d+)*)")),
    ("appendix", re.compile(r"Appendix~?\s*([A-Z])(?:\.(\d+(?:\.\d+)*))?\b")),
    # A BARE appendix reference -- "see D.1.2", "the breakdown in B.4". Same kind of
    # reference, same need to resolve, and it was outside this gate entirely because the
    # pattern above requires the literal word "Appendix". The reproduction appendix
    # shipped with four live references to D.1/D.1.1/D.1.2/D.6, every one of which
    # resolved to a different appendix, and nothing here objected.
    #
    # The lookbehind excludes a letter that is part of a word, a path, a version string
    # or a decimal, so `Qwen3-8B.1` and `v2.1` are not references. Requiring the letter
    # to be a capital in A-G excludes ordinary sentence-initial words.
    ("bare-appendix", re.compile(r"(?<![\w./~-])([A-G])(\.\d+(?:\.\d+)*)\b")),
    # "Fig 8" is the same reference as "Figure 8" and was outside this gate, which is how
    # two references to the pre-cut figure numbering survived in the practice appendix.
    ("figure", re.compile(r"\bFig(?:ure)?~?\.?\s*(\d+)\b")),
    ("table", re.compile(r"Table~?\s*(\d+)\b")),
    ("equation", re.compile(r"Equation~?\s*\((\d+)\)")),
    ("fw", re.compile(r"\bFW-(\d+)\b")),
]


def counts(main: str, appendices: str) -> dict[str, int]:
    both = main + "\n" + appendices
    return {
        "figure": len(re.findall(r"\\begin\{figure\*?\}", both)),
        "table": len(re.findall(r"\\begin\{table\*?\}", both)),
        "equation": len(re.findall(r"\\begin\{equation\}", both)),
        "fw": len(set(re.findall(r"\*\*FW-(\d+)", both) + re.findall(r"FW-(\d+)\*\*", both)
                      + re.findall(r"textbf\{FW-(\d+)", both))),
    }


#: Documents that left the PDF in the round-8 cut but did not leave the checks. Each is
#: (path, the heading prefix its own sections carry). A reference from one of these into
#: the paper must resolve against the paper's structure, and a reference from the paper
#: into one of these must resolve against that document's own headings -- otherwise
#: METHODOLOGY.md becomes the next main.tex, which is the defect that survived three
#: consecutive rounds.
COMPANIONS = ("METHODOLOGY.md", "PROMPTS.md", "README.md")


def companion_headings(name: str) -> set[str]:
    """The section labels a companion document defines, e.g. `M.3` in METHODOLOGY.md."""
    path = REPO_ROOT / name
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#{2,3}\s+([A-Z]+(?:\.\d+)+)\s+\S", line)
        if m:
            out.add(m.group(1))
    return out


def companion_refs(main: str, appendices: str) -> list[tuple[str, str, str]]:
    """References that cross the PDF/repo boundary in either direction and dangle.

    Two directions, and both were unchecked the moment content moved out:

    * a companion citing the paper -- `see 4.4` in METHODOLOGY.md -- resolves against the
      paper's structure, which this module already numbers exactly as LaTeX does;
    * the paper citing a companion -- `METHODOLOGY.md M.3` -- resolves against that
      document's own headings.
    """
    secs, apps, _ = structure(main, appendices)
    bad: list[tuple[str, str, str]] = []
    defined = {c: companion_headings(c) for c in COMPANIONS}

    for name in COMPANIONS:
        path = REPO_ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"§\s?(\d+(?:\.\d+)*)", text):
            if m.group(1) not in secs:
                lo = max(0, m.start() - 50)
                bad.append((f"{name}->paper", f"§{m.group(1)}",
                            " ".join(text[lo:m.end() + 20].split())))
        for m in re.finditer(r"(?:Appendix~?\s*)([A-G](?:\.\d+)*)\b", text):
            if m.group(1) not in apps:
                lo = max(0, m.start() - 50)
                bad.append((f"{name}->paper", f"Appendix {m.group(1)}",
                            " ".join(text[lo:m.end() + 20].split())))

    both = main + "\n" + appendices
    for name in COMPANIONS:
        stem = name.split(".")[0]
        for m in re.finditer(rf"{stem}\.md\s+((?:[A-Z]+\.\d+)(?:\.\d+)*)", both):
            if m.group(1) not in defined[name]:
                lo = max(0, m.start() - 50)
                bad.append((f"paper->{name}", m.group(1),
                            " ".join(both[lo:m.end() + 20].split())))
    return bad


def numbering_drift() -> list[tuple[str, str, str]]:
    """Where a markdown heading's own number disagrees with the number LaTeX will give it.

    The appendix markdown writes its own labels -- `## B.7 Paired contrasts` -- and the
    technical report renders them literally, while the arXiv build strips the number and
    lets LaTeX count. Those two agree only as long as the labels are sequential from 1,
    and inserting a section in the middle breaks every later one at once.

    That happened: a new B.6b was added between B.6 and B.7, so the arXiv build numbered
    it B.7 and shifted the paired contrasts to B.8, uniformity to B.11 and PG-2 to B.12,
    while five references still said B.6b and every reference to B.7 and beyond pointed
    one section short. `check` could not see it -- B.7 through B.13 all exist, so every
    reference resolved, to the wrong thing. This is the same failure §7.1 records for the
    renumbered section references: a gate that proves a target exists does not prove it is
    the right target.
    """
    bad: list[tuple[str, str, str]] = []
    for src in sorted((REPO_ROOT / "paper").glob("appendix-*.md")):
        bad += drift_in(src.name, src.read_text(encoding="utf-8"))
    return bad


def drift_in(name: str, text: str) -> list[tuple[str, str, str]]:
    """The pure half of `numbering_drift`, so it can be fed a known-bad document."""
    heads = re.findall(r"(?m)^##\s+([A-G])\.(\d+\w*)\s+(.*)$", text)
    return [(name, f"{letter}.{num}",
             f"is the {i}th subsection, so LaTeX will number it {letter}.{i}: "
             f"{title[:50]}")
            for i, (letter, num, title) in enumerate(heads, start=1) if num != str(i)]


def check(main: str, appendices: str) -> list[tuple[str, str, str]]:
    """Unresolved references as (kind, target, context)."""
    secs, apps, _ = structure(main, appendices)
    cnt = counts(main, appendices)
    text = main + "\n" + appendices
    bad: list[tuple[str, str, str]] = []
    for kind, pat in REFS:
        for m in pat.finditer(text):
            if kind == "section":
                target = m.group(1)
                ok = target in secs
            elif kind == "appendix":
                target = m.group(1) + (f".{m.group(2)}" if m.group(2) else "")
                ok = target in apps
            elif kind == "bare-appendix":
                target = m.group(1) + m.group(2)
                ok = target in apps
            elif kind in ("figure", "table", "equation"):
                target = m.group(1)
                ok = 1 <= int(target) <= max(cnt[kind], 0)
            else:
                target = m.group(1)
                ok = 1 <= int(target) <= max(cnt["fw"], 1)
            if not ok:
                lo = max(0, m.start() - 55)
                ctx = " ".join(text[lo:m.end() + 25].split())
                bad.append((kind, target, ctx))
    return bad


def title_disagreements() -> list[tuple[str, str]]:
    """(artifact, title) for any artifact naming the paper by a title it does not have.

    A reference that points at the wrong section is this file's subject; a citation
    block that names the wrong *work* is the same defect one level up, and it shipped:
    the README's BibTeX carried "Near-Total Weight-Space Erasure ... What Survives When
    a Merged LoRA Is Quantized" while the paper is titled "Weight-Space Erasure Without
    Behavioural Collapse in Quantized LoRA Adapters". Anyone citing from the repository
    attributed a work whose PDF says something else.

    `gen_readme.py` now derives both from `main.tex`, so this cannot drift by
    generation. It can still drift by hand-editing a generated file, which is exactly
    what M.4 says will happen eventually.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from gen_readme import paper_title

    want = paper_title()
    bad: list[tuple[str, str]] = []

    readme = REPO_ROOT / "README.md"
    if readme.exists():
        m = re.search(r"^\s*title\s*=\s*\{(.+?)\},\s*$", readme.read_text(
            encoding="utf-8"), re.M | re.S)
        if m is None:
            bad.append(("README.md", "no BibTeX title field found"))
        elif " ".join(m.group(1).split()) != want:
            bad.append(("README.md", " ".join(m.group(1).split())))

    cff = REPO_ROOT / "CITATION.cff"
    if cff.exists():
        m = re.search(r"^title:\s*>-\s*\n\s+(.+?)\s*$", cff.read_text(encoding="utf-8"),
                      re.M)
        if m is None:
            bad.append(("CITATION.cff", "no title field found"))
        elif " ".join(m.group(1).split()) != want:
            bad.append(("CITATION.cff", " ".join(m.group(1).split())))

    return bad


def author_disagreements() -> list[tuple[str, str]]:
    r"""(artifact, name) for any artifact attributing the work to a different person.

    Same shape as `title_disagreements`, one field over. The README's BibTeX and
    CITATION.cff are generated from the split declared in `gen_readme`, which is itself
    checked against `\author` in main.tex; LICENSE is not generated by anything, so it
    is the artifact most able to drift and the one a reader is most likely to trust.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from gen_readme import FAMILY, GIVEN, paper_author

    want = paper_author()
    bad: list[tuple[str, str]] = []

    lic = REPO_ROOT / "LICENSE"
    if lic.exists():
        m = re.search(r"^Copyright \(c\) \d{4} (.+?)(?:\s*<|$)",
                      lic.read_text(encoding="utf-8"), re.M)
        if m is None:
            bad.append(("LICENSE", "no copyright line found"))
        elif m.group(1).strip() != want:
            bad.append(("LICENSE", m.group(1).strip()))

    readme = REPO_ROOT / "README.md"
    if readme.exists():
        m = re.search(r"^\s*author\s*=\s*\{(.+?)\},\s*$", readme.read_text(
            encoding="utf-8"), re.M)
        if m is None:
            bad.append(("README.md", "no BibTeX author field found"))
        elif m.group(1).strip() != f"{FAMILY}, {GIVEN}":
            bad.append(("README.md", m.group(1).strip()))

    cff = REPO_ROOT / "CITATION.cff"
    if cff.exists():
        text = cff.read_text(encoding="utf-8")
        fam = re.search(r"^\s*-?\s*family-names:\s*(.+?)\s*$", text, re.M)
        giv = re.search(r"^\s*given-names:\s*(.+?)\s*$", text, re.M)
        if fam is None or giv is None:
            bad.append(("CITATION.cff", "no family-names/given-names pair found"))
        elif (fam.group(1), giv.group(1)) != (FAMILY, GIVEN):
            bad.append(("CITATION.cff", f"{giv.group(1)} {fam.group(1)}"))

    return bad


def main_() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    main = (TEXDIR / "main.tex").read_text(encoding="utf-8")
    apx = (TEXDIR / "appendices.tex").read_text(encoding="utf-8")
    secs, apps, _ = structure(main, apx)
    bad = check(main, apx)
    drift = numbering_drift()
    print(f"structure: {len(secs)} section labels, {len(apps)} appendix labels")
    print(f"           sections   {' '.join(sorted(secs, key=_key))}")
    print(f"           appendices {' '.join(sorted(apps, key=_key))}")
    if drift:
        print(f"\n{len(drift)} appendix headings disagree with the number LaTeX will "
              "give them:")
        for name, label, why in drift:
            print(f"  [{name}] {label} {why}")
        return 1
    print("           markdown appendix labels match the LaTeX numbering")
    cross = companion_refs(main, apx)
    if cross:
        print(f"\n{len(cross)} references dangle across the PDF/repo boundary:")
        for kind, target, ctx in cross:
            print(f"  {kind:<24} {target:<16} ...{ctx}")
        return 1
    live = [c for c in COMPANIONS if (REPO_ROOT / c).exists()]
    print(f"           references resolve across the boundary to {', '.join(live)}")
    titles = title_disagreements()
    if titles:
        print(f"\n{len(titles)} artifacts name the paper by a title it does not have:")
        for where, got in titles:
            print(f"  {where:<14} {got!r}")
        return 1
    authors = author_disagreements()
    if authors:
        print(f"\n{len(authors)} artifacts attribute the work to a different person:")
        for where, got in authors:
            print(f"  {where:<14} {got!r}")
        return 1
    print("           every citation block names the paper's actual title and author")
    if not bad:
        print("all cross-references resolve")
        return 0
    print(f"\n{len(bad)} unresolved references:")
    seen = set()
    for kind, target, ctx in bad:
        k = (kind, target)
        if k in seen:
            continue
        seen.add(k)
        n = sum(1 for a, b, _ in bad if (a, b) == k)
        print(f"  {kind:>9} {target:<8} x{n:<3} ...{ctx}")
    return 1 if args.strict else 0


def _key(s: str):
    return [int(p) if p.isdigit() else p for p in s.split(".")]



# ------------------------------------------------- semantic reference checking (M5)

_NUMTOK = re.compile(r"\d+\.\d{2,}|\d{3,}")


def _fingerprint(body: str) -> set[str]:
    """A section's numeric literals.

    Prose gets reworded between the two documents and numbers do not, so this identifies
    a section across a rewrite where title words and shared vocabulary both fail --
    "Secondary structure" and "An early-layer spike, and what it is not" are the same
    section under two names, and a title comparison calls them different.
    """
    return set(_NUMTOK.findall(body))


def _md_sections() -> dict[str, str]:
    """The report's own numbering: label -> body text, from the markdown sources."""
    out: dict[str, str] = {}
    for path in sorted((REPO_ROOT / "paper").glob("*.md")):
        label: str | None = None
        buf: list[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^#+\s+(\d+(?:\.\d+)*)\.?\s+(\S.*)$", line)
            if m:
                if label:
                    out[label] = "\n".join(buf)
                label, buf = m.group(1), []
            elif label:
                buf.append(line)
        if label:
            out[label] = "\n".join(buf)
    return out


def _tex_sections() -> dict[str, str]:
    """Paper sections keyed as LaTeX numbers them, with their bodies."""
    text = (REPO_ROOT / "paper" / "tex" / "main.tex").read_text(encoding="utf-8")
    out: dict[str, str] = {}
    n = [0, 0]
    label: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\\(section|subsection)\{", line.strip())
        if m:
            if label:
                out[label] = "\n".join(buf)
            if m.group(1) == "section":
                n[0] += 1
                n[1] = 0
                label = str(n[0])
            else:
                n[1] += 1
                label = f"{n[0]}.{n[1]}"
            buf = []
        elif label:
            buf.append(line)
    if label:
        out[label] = "\n".join(buf)
    return out


def section_alignment() -> list[tuple[str, str, str, float, float]]:
    """References translated to a section that is not the one they are about.

    `check` verifies that a translated reference resolves to a label that EXISTS. Any map
    satisfies that, including a map with a hole in it -- and a map with a hole is worse
    than no map, because the entries that are present make the omissions look deliberate.
    Section 5 loses a subsection in the paper, so everything after it shifts by one;
    `5.4 -> 5.3` was in the table and `5.3 -> 5.2` was not, and three references to the
    dissociation section resolved, silently, to the predictive gap: B.6's confound note,
    B.9's caption and Appendix C's P7 row.

    Content, not existence. Each report subsection is matched against every paper
    subsection of the same parent by numeric fingerprint, and if some other subsection
    matches better than the one the map sends the reference to, the map is wrong there.

    Returns (label, mapped target, best match, mapped score, best score).
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_m2t_for_xref", Path(__file__).resolve().parent / "md_to_tex.py")
    m2t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m2t)

    md, tex = _md_sections(), _tex_sections()
    bad: list[tuple[str, str, str, float, float]] = []
    for label, body in sorted(md.items()):
        if "." not in label:
            continue
        target = m2t.REFMAP.get(label, label)
        if "." not in target:
            continue                    # folded into a whole section; nothing to rank
        parent = label.split(".")[0]
        src = _fingerprint(body)
        if len(src) < 6:
            continue                    # too few numbers to identify anything
        cands = {k: len(src & _fingerprint(v)) / len(src)
                 for k, v in tex.items() if k.startswith(parent + ".")}
        if target not in cands:
            continue
        best = max(cands, key=lambda k: cands[k])
        if best != target and cands[best] > cands[target] + 0.15:
            bad.append((label, target, best, cands[target], cands[best]))
    return bad


if __name__ == "__main__":
    raise SystemExit(main_())
