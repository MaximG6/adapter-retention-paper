"""Convert the appendix markdown to LaTeX for the arXiv-format build.

Hand-converting 12,000 words would make the appendices a fork of the markdown that
drifts the moment either is edited — the failure mode of §7.8, applied to typesetting.
This converts the subset of markdown the manuscript actually uses, so the LaTeX
appendices remain derived from the same sources as the HTML report.

Handles: ATX headings, bold/italic/code spans, fenced code, pipe tables, bullet lists,
inline links, blockquotes, horizontal rules, and LaTeX-special characters.

Usage:
    python analysis/md_to_tex.py --write
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper"
OUT = PAPER / "tex" / "appendices.tex"

# (file, appendix title, figures to place at the end of that appendix)
APPENDICES = [
    ("appendix-A-tool.md", "The \\texttt{ar.predict} Tool",
     ["figA1_predict_validation"]),
    ("appendix-B-tables.md", "Full Tables",
     ["fig02_channel_model", "fig03_forest", "fig04_amplification",
      "fig11_layer_profile"]),
    ("07-registered-predictions.md", "Registered Predictions", []),
    ("06-results-advertised-vs-measured.md", "Advertised versus Measured: Full Detail",
     ["fig10_refusal"]),
    ("appendix-C-prompts.md", "Prompt Sets", []),
    ("appendix-D-reproduction.md", "Reproduction", []),
    (None, "Supplementary Figures",
     ["fig07_entropy_control", "fig09_bootstrap_intervals"]),
]

SPECIALS = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


#: Typewriter fonts carry no glyphs for these, and neither code spans nor verbatim
#: blocks are escaped, so both need an ASCII substitution pass of their own.
TT_ASCII = {"−": "-", "–": "-", "—": "--", "→": "->", "≥": ">=", "≤": "<=",
            "×": "x", "≈": "~", "Δ": "Delta", "δ": "delta", "α": "alpha",
            "√": "sqrt", "±": "+/-", "≠": "!=", "π": "pi", "σ": "sigma",
            "ρ": "rho", "∝": "prop", "∞": "inf", "·": "*", "…": "...",
           "τ": "tau", "θ": "theta", "μ": "mu", "λ": "lambda",
            "≡": "==", "ᵀ": "^T", "⊤": "^T", "∈": "in", "⟨": "<", "⟩": ">",
            "²": "^2", "³": "^3", "⁻": "^-", "⁰": "^0", "¹": "^1", "⁴": "^4",
            "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
            "é": "e", "É": "E", "ü": "u", "ö": "o", "ń": "n", "ś": "s",
            "‑": "-", "∀": "for all", "⚠": "", "✓": "OK",
            # Vulgar fractions. `u = frac(w/s + z + 1/2)` is written with the glyph in
            # the markdown, and its absence here crashed this converter -- which the
            # arXiv build did not notice, because it checked stdout and not the exit
            # code, so it shipped a stale appendices.tex for two rounds.
            "½": "1/2", "¼": "1/4", "¾": "3/4", "⅓": "1/3", "⅔": "2/3",
            "“": '"', "”": '"', "'": "'", "'": "'", "§": "S"}


def ascii_only(s: str) -> str:
    """Substitute the non-ASCII a typewriter font cannot render, and RAISE on the rest.

    This used to end `.encode("ascii", "replace")`, which turns anything unmapped into a
    literal `?`. Four characters took that path into the shipped PDF -- `Je suis
    d?sol?(e)`, `mean(Delta?)`, `A?A`, and the identity `cos x retention_ratio ?
    projection_coefficient` -- and the build's own non-ASCII gate reported clean, because
    by the time it looked there was no non-ASCII left to find. The gate was satisfied by
    the damage it existed to catch.

    Raising is the project's no-silent-fallback rule applied to typesetting: a character
    with no sensible ASCII form is a decision for a person, not for a codec.
    """
    for k, v in TT_ASCII.items():
        s = s.replace(k, v)
    bad = sorted({c for c in s if ord(c) > 127})
    if bad:
        raise ValueError(
            "No ASCII substitution for "
            + ", ".join(f"{c!r} (U+{ord(c):04X})" for c in bad)
            + f" in code span/verbatim: {s[:90]!r}. Add it to TT_ASCII, or take the "
              "character out of the code span -- do not let it become '?'."
        )
    return s


def esc(s: str) -> str:
    """Escape LaTeX specials outside code spans. Backslash first, always."""
    s = s.replace("\\", r"\textbackslash{}")
    for k, v in SPECIALS.items():
        s = s.replace(k, v)
    # Unicode the manuscript actually uses.
    for k, v in {"−": r"$-$", "—": "---", "–": "--", "×": r"$\times$", "≈": r"$\approx$",
                 "≥": r"$\geq$", "≤": r"$\leq$", "→": r"$\rightarrow$",
                 "Δ": r"$\Delta$", "δ": r"$\delta$", "α": r"$\alpha$",
                 "√": r"$\sqrt{\ }$", "σ": r"$\sigma$", "ρ": r"$\rho$",
                 "∝": r"$\propto$", "∞": r"$\infty$", "±": r"$\pm$",
                 "τ": r"$\tau$", "θ": r"$\theta$", "μ": r"$\mu$",
                 "λ": r"$\lambda$",
                 "'": "'", "'": "'", "\u201c": "``", "\u201d": "''",
                 "…": r"\ldots{}", "⚠": "", "§": r"\S{}", "✓": r"\checkmark{}",
                 "≠": r"$\neq$", "·": r"$\cdot$", "É": r"\'E", "é": r"\'e",
                 "ü": r'\"u', "ö": r'\"o', "ń": r"\'n", "ś": r"\'s",
                 "‑": "-", "⟨": r"$\langle$", "⟩": r"$\rangle$",
                 "⁻": r"$^{-}$", "⁰": r"$^{0}$", "¹": r"$^{1}$", "²": r"$^{2}$",
                 "³": r"$^{3}$", "⁴": r"$^{4}$", "⁵": r"$^{5}$", "⁶": r"$^{6}$",
                 "⁷": r"$^{7}$", "⁸": r"$^{8}$", "⁹": r"$^{9}$",
                 "ᵀ": r"$^{\top}$", "∈": r"$\in$", "∀": r"$\forall$",
                 "🔬": "", "⚡": "", "📊": ""}.items():
        s = s.replace(k, v)
    return s


def leftover_non_ascii(tex: str) -> dict[str, int]:
    """Any non-ASCII surviving conversion. TeX renders these as garbage or drops them
    silently, so the build reports them rather than shipping a PDF with holes in it."""
    from collections import Counter
    return dict(Counter(c for c in tex if ord(c) > 127))


#: The markdown is the technical report, where section 7.8 is a real heading. The arXiv
#: build cuts and re-orders: the methodological section becomes Appendix C, the
#: advertised-versus-measured section becomes Appendix D, and the body's subsections are
#: folded. A reference correct in one document is therefore wrong in the other, and that
#: is what left roughly forty dangling references in the built PDF. Rather than break the
#: markdown to suit the LaTeX, translate on the way through.
#:
#: Left side is the markdown's own numbering; right side is what LaTeX will number it.
REFMAP = {
    # Related work: seven subsections in the report, one section in the paper.
    **{f"2.{i}": "2" for i in range(1, 8) if i != 5},
    # Method: 3.8 and 3.10 previously folded into 3.7, which pointed the ground-truth
    # fixture and the refusal battery at the taboo battery instead -- eight references
    # resolving to the wrong section. Both subsections now exist in the body, so §3 is
    # numbered identically in both documents and needs no mapping at all.
    #
    # §2.5 is the reconciliation of the opposing result, which is §7 in the paper, not a
    # subsection of related work.
    "2.5": "7",
    # Results: 4.5.1 folds into 4.5. Section 5 loses one subsection in the paper --
    # "the degradation is not distribution flattening" is a sentence inside 5.1 there --
    # so everything after it shifts by one.
    #
    # Only the last shift was mapped. 5.4 -> 5.3 was here and 5.3 -> 5.2 was not, so
    # every reference to the dissociation section resolved, in the built paper, to the
    # predictive gap: B.6's confound note, B.9's caption and Appendix C's P7 row all
    # pointed one section past their subject. A partial map is worse than none, because
    # the entries that are present make the omissions look deliberate.
    #
    # 5.5 ("prior puzzles resolved") has no counterpart in the paper and nothing refers
    # to it, so it is deliberately absent rather than overlooked. If a reference to it is
    # ever written, `check_source_refs` sees the source side and the built-document gate
    # sees an untranslated 5.5, which does not exist in the paper.
    "4.5.1": "4.5", "5.2": "5.1", "5.3": "5.2", "5.4": "5.3",
    # Advertised versus measured becomes Appendix D, keeping its own numbering.
    **{f"6.{i}": f"D.{i}" for i in range(1, 6)},
    # Methodological practice becomes Appendix C, keeping its own order: heading 7.n is
    # subsection C.(n+1), because the appendix gains no heading the markdown lacks.
    #
    # This used to be a hand-written table from the PRE-CUT numbering to the post-cut
    # appendix, which meant the markdown's own references pointed at headings the markdown
    # did not have and only this table hid it. Every reference resolved, several to the
    # wrong entry, and one section cited itself. A one-line offset cannot do that, and
    # `check_source_refs` below fails the build if a reference in the markdown does not
    # match a heading in the markdown.
    #
    # Round 8 moved the practice entries to METHODOLOGY.md and left the registered
    # predictions as the whole of Appendix C, so there is no longer a 7.n to translate:
    # a reference to a practice entry now names the companion document instead, and
    # xref.companion_refs resolves it there.
    "7": "C",
    # Limitations has no subsections in the paper.
    **{f"8.{i}": "9" for i in range(1, 9)},
}

#: The report's own appendices are A-D; the paper's are A, B, then C and D taken by the
#: two body sections that became appendices, pushing prompts and reproduction to E and F.
#: Left unmapped, "Appendix D" in the reproduction appendix pointed at advertised-versus-
#: measured -- a target that exists, so the reference gate passed it.
APPENDIX_MAP = {"A": "A", "B": "B", "C": "E", "D": "F"}

_REF = re.compile(r"§\s?(\d+(?:\.\d+)*)")
#: Captures the letter and any subsection suffix. The suffix must be captured rather than
#: excluded: "Appendix D.4" needs the letter remapped and the number kept. An earlier
#: `(?!\.)` guard, meant to skip subsection references, also skipped every reference that
#: ended a sentence -- so "in Appendix D." stayed pointing at the wrong appendix.
_APX = re.compile(r"\bAppendix~?\s*([A-D])((?:\.\d+)*)\b")
#: A BARE appendix reference: "see D.1.2", "breakdown in D.6". These are the same kind of
#: reference as "Appendix D.6" and need the same remapping, but nothing rewrote them and
#: nothing checked them, because both the mapper and the cross-reference gate keyed on the
#: literal word "Appendix". The reproduction appendix -- which is Appendix F in the built
#: paper -- therefore shipped with live references to D.1, D.1.1, D.1.2 and D.6, all of
#: which resolve to the safety-adapter appendix.
_BARE_APX = re.compile(r"(?<![\w./-])([C-D])(\.\d+(?:\.\d+)*)\b")


def check_source_refs() -> list[tuple[str, str]]:
    """Every §7.x written in the markdown must match a `## 7.x` heading in the markdown.

    The gate that already exists checks the BUILT document, after REFMAP has translated
    everything, so a reference pointing at a cut entry still resolved -- to whatever the
    map sent it to. This checks the other end: the source has to be consistent with
    itself before translation, which is the only place a wrong-but-resolving reference is
    visible. Returns (reference, file) pairs that have no heading.
    """
    lessons = (PAPER / "07-registered-predictions.md").read_text(encoding="utf-8")
    have = set(re.findall(r"(?m)^##+\s+(7\.[0-9a]+)\s", lessons))
    have |= {"7"}
    bad: list[tuple[str, str]] = []
    for path in sorted(PAPER.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for ref in re.findall(r"§\s?(7\.[0-9a]+)", text):
            if ref not in have:
                bad.append((ref, path.name))
    return bad


def remap_refs(md: str) -> str:
    """Rewrite the report's section numbers to the paper's, for the LaTeX build."""

    def sub(m: re.Match[str]) -> str:
        target = REFMAP.get(m.group(1))
        if target is None:
            return m.group(0)
        return f"Appendix {target}" if target[0].isalpha() else f"§{target}"

    # Letters first, sections second. REFMAP turns §6.3 into "Appendix D.3"; if the
    # letter pass ran afterwards it would remap that D to F and corrupt a reference it
    # had just created. Only letters written in the source may be remapped.
    md = _APX.sub(lambda m: f"Appendix {APPENDIX_MAP[m.group(1)]}{m.group(2)}", md)
    md = _bare_apx(md)
    return _REF.sub(sub, md)


def _bare_apx(md: str) -> str:
    """Remap bare `C.n` / `D.n` references, skipping headings and fenced code.

    Headings are excluded because `## D.6 Expected runtimes` is the target, not a
    reference to it, and it is de-numbered later anyway. Code fences are excluded
    because a bare letter-dot-number inside a command or path is not a reference.
    """
    out: list[str] = []
    fenced = False
    for line in md.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            fenced = not fenced
            out.append(line)
            continue
        if fenced or stripped.startswith("#"):
            out.append(line)
            continue
        out.append(_BARE_APX.sub(
            lambda m: f"{APPENDIX_MAP[m.group(1)]}{m.group(2)}", line))
    return "".join(out)


def inline(s: str) -> str:
    """Inline spans. Code is extracted first so its contents are never escaped."""
    holds: list[str] = []

    def hold(m: re.Match[str]) -> str:
        body = m.group(1)
        # ASCII substitution runs BEFORE escaping, not after. Some substitutions emit
        # LaTeX specials of their own -- U+00B2 becomes "^2" -- and running them after
        # the escape pass puts a bare ^ inside \texttt{}, which is a "Missing $ inserted"
        # error, not a rendering nicety. Typewriter fonts lack these glyphs and code
        # spans are not escaped by esc(), so the substitution has to happen either way.
        body = ascii_only(body)
        for k, v in {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}",
                     "$": r"\$", "&": r"\&", "%": r"\%", "#": r"\#",
                     "_": r"\_", "^": r"\textasciicircum{}",
                     "~": r"\textasciitilde{}"}.items():
            body = body.replace(k, v)
        # A path in a code span is one unbreakable word. In a narrow table column it
        # overflows however the column is sized, so permit a break after the
        # separators. \allowbreak adds no hyphen, so the path stays copy-pasteable.
        body = re.sub(r"(?<=[/_.-])(?=[^\s/_.-])", r"\\allowbreak{}", body)
        holds.append(rf"\texttt{{{body}}}")
        return f"\x00{len(holds) - 1}\x00"

    s = re.sub(r"`([^`]+)`", hold, s)
    # Markdown backslash escapes, resolved before esc() turns the backslash itself into
    # \textbackslash{}. A footnote marker written `\*` reached the PDF as
    # "\textbackslash{}*", and `\|` inside a table cell was worse: it also split the cell.
    #
    # AFTER the code spans are held out, not before. Run first, this ate the backslashes
    # inside code spans, where they are literal content: a span containing `\\` was read
    # as an escaped backtick and the line became an invalid character in the built tex.
    # A backtick is not in the class for the same reason.
    s = re.sub(r"\\([*_|\[\]()#+\-.!])", "\x02\\1", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", s)          # links -> text
    # Markdown table cells cannot contain a newline, so the prompt tables use <br>. HTML
    # renders it; LaTeX printed it literally. \newline works inside a p/X column, which
    # is what every multi-line cell in this document sits in.
    s = re.sub(r"\s*<br\s*/?>\s*", "\x01", s, flags=re.I)
    s = esc(s)
    s = s.replace("\x01", r"\newline ")
    s = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", s)
    # Escaped characters restored after the emphasis pass, so an escaped asterisk cannot
    # be read as emphasis. `$|$` because a bare bar is a rule in some fonts.
    s = re.sub("\x02(.)", lambda m: "$|$" if m.group(1) == "|" else m.group(1), s)
    return re.sub(r"\x00(\d+)\x00", lambda m: holds[int(m.group(1))], s)


#: Page geometry, in points. article/10pt/twocolumn at margin=0.75in on letter.
#: Must match main.tex; TABLE_GEOMETRY_CHECK below fails the build if it drifts.
COLUMN_PT = 247.0
TEXT_PT = 505.0
#: Latin Modern mean advance, in points per character, at each size we emit.
CHAR_PT = {"footnotesize": 8.0 * 0.47, "scriptsize": 7.0 * 0.47}
#: booktabs inter-column gutter, both sides.
COL_PAD_PT = 12.0
#: Below this, a column is a label or a number and should size to its content.
NARROW_CH = 14
#: A wrapped column narrower than this reads one word per line.
MIN_MEASURE_CH = 26


def _demand(texts: list[str]) -> tuple[int, int]:
    """(longest cell, longest unbreakable word) in characters, markup removed."""
    plain = [re.sub(r"[*`\[\]]|\((?:https?|#)[^)]*\)", "", t) for t in texts]
    longest = max((len(t) for t in plain), default=0)
    word = max((len(w) for t in plain for w in t.split()), default=0)
    return longest, word


def table(rows: list[str]) -> str:
    """Lay a markdown table out to fit, measuring content instead of guessing.

    The previous version chose a fixed spec from the column count alone: bare `r`
    columns for numeric tables, which have no width bound and simply ran off the page,
    and a uniform `p{0.72/(n-1)}` for prose, which gave a 9-character measure at n=6.
    Both defects shipped in the arXiv PDF; neither is visible in the markdown, and only
    the first one warns.

    Each column's width demand is measured, then a container is chosen in increasing
    order of disruption -- one column, one column at \\scriptsize, then full width via
    table* -- and the width is distributed in proportion to demand. Columns that want
    less than NARROW_CH size to their content; the rest wrap, never below
    MIN_MEASURE_CH where the container allows it.
    """
    # `\|` inside a cell is an ESCAPED pipe -- markdown's only way to write a literal `|`
    # in a table -- and every table in this paper that names |delta| or |r| uses it. Split
    # naively it becomes both a spurious cell break and a `\textbackslash{}`, so
    # `\|delta\|/s` reached the built PDF as garbage across two extra columns, in the two
    # tables carrying the 2.3% and 10.4% headlines. Protected before the split and
    # restored after, as a literal bar rather than an escape LaTeX will mangle again.
    _BAR = "\x00BAR\x00"
    rows = [r.replace(r"\|", _BAR) for r in rows]
    cells = [[c.strip().replace(_BAR, "|") for c in r.strip().strip("|").split("|")]
             for r in rows]
    cells = [c for c in cells if not all(set(x) <= set("-: ") for x in c)]
    if not cells:
        return ""
    n = max(len(c) for c in cells)
    cells = [c + [""] * (n - len(c)) for c in cells]

    cols = [[row[c] for row in cells] for c in range(n)]
    longest, words = zip(*(_demand(col) for col in cols))
    numeric = [c > 0 and all(re.fullmatch(r"[-+~<>=$\\\d.,%/()\s]*", t or "0")
                             for t in cols[c][1:]) for c in range(n)]

    # Natural (unwrapped) width at each size, plus gutters.
    def natural(size: str) -> float:
        return sum(longest) * CHAR_PT[size] + n * COL_PAD_PT

    if natural("footnotesize") <= COLUMN_PT:
        size, avail, wide = "footnotesize", COLUMN_PT, False
    elif natural("scriptsize") <= COLUMN_PT:
        size, avail, wide = "scriptsize", COLUMN_PT, False
    elif natural("footnotesize") <= TEXT_PT:
        size, avail, wide = "footnotesize", TEXT_PT, True
    else:
        size, avail, wide = "scriptsize", TEXT_PT, True

    body = [r"\toprule",
            " & ".join(inline(x) for x in cells[0]) + r" \\",
            r"\midrule"]
    body += [" & ".join(inline(x) for x in row) + r" \\" for row in cells[1:]]
    body += [r"\bottomrule"]

    if avail == COLUMN_PT and not wide and natural(size) <= COLUMN_PT:
        # Fits as-is; let TeX size every column to its content.
        spec = "".join("r" if numeric[c] else "l" for c in range(n))
        inner = [rf"\begin{{tabular}}{{{spec}}}"] + body + [r"\end{tabular}"]
    else:
        # tabularx solves for the X widths so the total is exactly the container.
        # Estimated p{} widths cannot promise that: they are mixed with natural l/r
        # columns whose true width depends on glyph metrics, and the sum overran.
        # Weight the X columns by demand so the widest text column gets the widest
        # measure rather than an equal slice. The weights must sum to the number of X
        # columns for tabularx's own total to come out right.
        flex = [c for c in range(n) if longest[c] > NARROW_CH]
        if not flex:
            flex = [max(range(n), key=lambda c: longest[c])]
        tot = sum(longest[c] for c in flex) or 1
        k = len(flex)
        parts = []
        for c in range(n):
            if c not in flex:
                parts.append("r" if numeric[c] else "l")
                continue
            f = k * longest[c] / tot
            f = min(max(f, 0.45), k)               # keep any X column legible
            parts.append(f">{{\\hsize={f:.3f}\\hsize\\raggedright\\arraybackslash}}X")
        # Renormalise so the factors still sum to k after clamping.
        got = sum(float(re.search(r"hsize=([0-9.]+)", p).group(1))
                  for p in parts if "hsize=" in p)
        if got and abs(got - k) > 1e-6:
            parts = [re.sub(r"hsize=([0-9.]+)",
                            lambda m: f"hsize={float(m.group(1)) * k / got:.3f}", p)
                     for p in parts]
        spec = "".join(parts)
        width = r"\textwidth" if wide else r"\columnwidth"
        inner = ([rf"\begin{{tabularx}}{{{width}}}{{{spec}}}"] + body
                 + [r"\end{tabularx}"])

    if wide:
        # table* spans both columns. It floats, so it may move to the top of a page;
        # that is the cost of not truncating the data.
        return "\n".join([r"\begin{table*}[t]", rf"\centering\{size}"]
                         + inner + [r"\end{table*}"])
    return "\n".join([rf"\begin{{center}}\{size}"] + inner + [r"\end{center}"])


#: Characters that fit across \columnwidth in a verbatim block at each size, for the
#: fixed-pitch face (advance 0.6em). Verbatim cannot wrap, so a block that exceeds its
#: size simply runs off the page -- the same defect as the tables, in a different
#: environment, and equally silent in the markdown.
VERBATIM_FIT = ((51, "footnotesize"), (58, "scriptsize"), (82, "tiny"))


def verbatim_size(block: list[str]) -> str:
    longest = max((len(b) for b in block), default=0)
    for fits, size in VERBATIM_FIT:
        if longest <= fits:
            return size
    return VERBATIM_FIT[-1][1]


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append(r"\end{itemize}")
            in_list = False

    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("```"):
            close_list()
            i += 1
            buf: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            # verbatim passes bytes straight to the typewriter font, which has no
            # glyphs for the unicode the manuscript uses in code examples.
            buf = [ascii_only(b) for b in buf]
            out += [rf"\begin{{quote}}\{verbatim_size(buf)}\begin{{verbatim}}", *buf,
                    r"\end{verbatim}\end{quote}"]
            continue
        if re.match(r"^\s*\|", ln):
            close_list()
            buf = []
            while i < len(lines) and re.match(r"^\s*\|", lines[i]):
                buf.append(lines[i])
                i += 1
            out.append(table(buf))
            continue
        m = re.match(r"^(#{2,4})\s+(.*)$", ln)
        if m:
            close_list()
            lvl = len(m.group(1))
            # Strip the manual number the markdown carries for the technical report.
            # LaTeX numbers appendices itself, so leaving it produced "A.1 A.1 What it
            # does" on every appendix heading in the shipped PDF. The letter-prefixed
            # form was not caught because the old pattern only matched leading digits.
            # "D.7a" carries a letter suffix, so a pattern ending at a digit leaves it.
            title = re.sub(r"^(?:[A-G]\.)?\d[\d.]*[a-z]?\s+", "", m.group(2)).strip()
            title = re.sub(r"^[\d.]+\s*", "", title).strip()
            cmd = {2: "subsection", 3: "subsubsection", 4: "paragraph"}[min(lvl, 4)]
            out.append(rf"\{cmd}{{{inline(title)}}}")
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            if not in_list:
                out.append(r"\begin{itemize}\itemsep1pt")
                in_list = True
            item = [re.sub(r"^\s*[-*]\s+", "", ln)]
            # An indented continuation belongs to this bullet. Emitting it separately
            # closed the list and left an orphan paragraph between two bullets, which is
            # how the gate self-test list rendered in the built PDF.
            while (i + 1 < len(lines) and lines[i + 1].strip()
                   and lines[i + 1][:1].isspace()
                   and not re.match(r"^\s*[-*]\s+", lines[i + 1])):
                i += 1
                item.append(lines[i].strip())
            out.append(r"\item " + inline(" ".join(item)))
            i += 1
            continue
        if ln.strip().startswith(">"):
            close_list()
            # Consecutive quote lines are ONE block. Converting them line by line means
            # inline() sees half a **bold** span and leaves both delimiters literal --
            # the same defect already fixed for paragraphs, still live here, and it put
            # a pair of asterisks into A.3's correction box in the shipped PDF. An
            # empty ">" line separates paragraphs within the quote.
            para: list[str] = []
            paras: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                body = lines[i].strip().lstrip(">").strip()
                if body:
                    para.append(body)
                elif para:
                    paras.append(" ".join(para))
                    para = []
                i += 1
            if para:
                paras.append(" ".join(para))
            out.append(r"\begin{quote}\small")
            for n, block in enumerate(paras):
                if n:
                    out.append("")
                out.append(inline(block))
            out.append(r"\end{quote}")
            continue
        if re.match(r"^\s*(---+|\*\*\*+)\s*$", ln):
            close_list()
            i += 1
            continue
        if not ln.strip():
            close_list()
            out.append("")
            i += 1
            continue
        close_list()
        # Gather the whole paragraph before converting it. inline() is regex-based and
        # its spans cannot match across a line break, so a **bold** or *emphasis* the
        # author wrapped mid-phrase used to pass through as literal asterisks -- 70 of
        # them reached the built PDF. A markdown paragraph is its consecutive lines, so
        # joining first is also what the format means.
        para = [ln]
        while i + 1 < len(lines):
            nxt = lines[i + 1]
            if (not nxt.strip() or nxt.startswith("|") or nxt.strip().startswith((">", "```"))
                    or re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^#{2,4}\s", nxt)
                    or re.match(r"^\s*(---+|\*\*\*+)\s*$", nxt)):
                break
            i += 1
            para.append(lines[i])
        out.append(inline(" ".join(x.strip() for x in para)))
        i += 1
    close_list()
    return "\n".join(out)


#: Real captions for the appendix figures. Without these the caption was the generating
#: script's filename with the underscores taken out -- "Figure 04 amplification." -- on
#: eight of twelve figures, which tells a reader nothing and reads as unfinished.
#: A figure placed here with no entry fails the build rather than getting a filename.
FIG_CAPTIONS = {
    # The maximum errors are printed inside each panel by the figure itself, from the
    # raw records. They are deliberately NOT repeated here: a caption that restates a
    # number the figure computes is a second copy that can drift, and this one did --
    # the caption said 9.5% while the panel printed 10.4%, because the two aggregated
    # per-module predictions differently.
    "figA1_predict_validation":
        "\\texttt{ar.predict} against direct measurement, nine published adapters. "
        "Dashed line is exact agreement. Left: code-flip rate. Right: cosine, predicted "
        "per module as $\\sqrt{\\tau\\,|\\Delta|/s}$ with $\\tau$ the measured tail-shape "
        "constant, then averaged. Each panel prints its own maximum relative error; both "
        "are \\texttt{fixed\\_scale} (\\S3.3) and both are tabulated in A.3.",
    "fig02_channel_model":
        "The channel model against measurement across three decades of adapter magnitude, "
        "swept on a real \\texttt{q\\_proj} base at rank 32. The prediction has no fitted "
        "parameters.",
    "fig03_forest":
        "Weight-space cosine per adapter at INT4 g128, with intervals over layers. The "
        "ordering follows effective magnitude, not rank.",
    "fig04_amplification":
        "Subspace amplification against rank. Fitted exponents $-0.457$, $-0.455$, "
        "$-0.457$ against a predicted $-0.5$; generic-input amplification is flat at "
        "1.0, so the effect exists only on subspace-aligned inputs.",
    "fig11_layer_profile":
        "Bit-flip rate by layer, showing the layers 1--3 spike and the step-size "
        "distribution that drives it.",
    "fig10_refusal":
        "Refusal battery for the safety adapter, by prompt kind. No axis clears the "
        "gate; the base model already refuses 16/16 harmful prompts at ceiling.",
    "fig07_entropy_control":
        "Per-token decoding entropy across precisions. Flat at 1.35--1.50 nats in every "
        "aligned condition while elicitation halves, so the behavioural degradation is "
        "not distribution flattening.",
    "fig09_bootstrap_intervals":
        "Per-adapter behavioural retention with 95\\% intervals over intent clusters, "
        "paired. Pairs separate at every grid; only at INT3 do four of them.",
}


def figure(fig: str) -> str:
    if fig not in FIG_CAPTIONS:
        raise KeyError(
            f"No caption for {fig!r}. Add one to FIG_CAPTIONS rather than shipping the "
            "filename as the caption.")
    return "\n".join([
        r"\begin{figure}[htbp]", r"\centering",
        rf"\includegraphics[width=\columnwidth]{{{fig}.pdf}}",
        rf"\caption{{{FIG_CAPTIONS[fig]}}}",
        r"\end{figure}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    dangling = check_source_refs()
    if dangling:
        print(f"  {len(dangling)} section-7 references have no heading in the markdown:")
        for ref, name in dangling:
            print(f"    {ref}  in {name}")
        print("  These would still RESOLVE after REFMAP, at whatever it maps them to.")
        return 1

    parts = [r"\appendix", r"\onecolumn" if False else ""]
    for fname, title, figs in APPENDICES:
        parts.append(rf"\section{{{title}}}")
        if fname:
            md = (PAPER / fname).read_text(encoding="utf-8")
            md = re.sub(r"^#\s+.*$", "", md, count=1, flags=re.M)     # drop H1
            md = re.sub(r"^\*Draft\..*?\*\s*$", "", md, flags=re.M | re.S, count=1)
            md = re.sub(r"^\*Generated by .*?regenerate\.\*\s*$", "", md, flags=re.M)
            md = re.sub(r"^\*All values re-derived.*$", "", md, flags=re.M)
            parts.append(convert(remap_refs(md)))
        for f in figs:
            parts.append(figure(f))

    tex = "\n\n".join(p for p in parts if p)

    left = leftover_non_ascii(tex)
    if left:
        print(f"  {sum(left.values())} non-ASCII characters survived conversion:")
        for ch, n in sorted(left.items(), key=lambda kv: -kv[1]):
            print(f"    U+{ord(ch):04X} {ch!r} x{n}")
        print("  These render as garbage or vanish in TeX. Add them to esc()/TT_ASCII.")
    else:
        print("  no leftover non-ASCII")

    if args.write:
        OUT.write_text(tex, encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(tex):,} chars)")
    else:
        print(tex[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
