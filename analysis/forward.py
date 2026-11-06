r"""Every number a summary asserts must be stated somewhere that measures it.

`retracted.py` closes propagation failure on **retraction**: a claim is withdrawn in the
appendix that measured it and still asserted in the body. This closes the other mode,
propagation failure on **addition**, and the two are structurally different. A retraction
leaves a wording behind to search for. An addition leaves nothing: the abstract gains a
number, the section that should source it is never touched, and there is no string
anywhere whose presence is wrong.

The round that named partial propagation committed this variant of it. `[+4.2, +12.5]`
went into the abstract, the introduction, Figure 1's caption and the Conclusion, and
\S5.1 -- the section whose whole subject is that contrast -- never stated it. Every gate
passed: the claim audit recomputed the number and found it correct, `countcheck` had no
cardinal to resolve, `xref` had a reference that resolved, and `retracted.py` had nothing
retracted. The number was right, and it was asserted only in places whose job is to
summarise something said elsewhere.

**The rule.** A summary site -- the abstract, the introduction, any figure or table
caption, the conclusion -- may assert a number only if a body section or appendix also
states it. Summaries summarise; if nothing is being summarised, either the claim has no
source or the source has a hole in it. Both are defects and this cannot tell them apart,
which is fine: both need the same fix.

**What is deliberately not checked.** That the summary *cites* the source. A pointer is
good practice and this project follows it, but a missing pointer is a readability
complaint and a missing source is an unsupported claim. Only the second fails a build.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEXDIR = REPO_ROOT / "paper" / "tex"

#: Numbers that are structure, not measurement. Stripped before extraction, because a
#: reference to section 5.1 is not a claim that 5.1 was measured.
_STRUCTURE = (
    r"\\S\\ref\{[^}]*\}", r"\\S\{?\}?\s?\d+(?:\.\d+)*", r"\\ref\{[^}]*\}",
    r"\\cite\{[^}]*\}", r"\\label\{[^}]*\}",
    r"(?:Figure|Table|Appendix|Equation|Section)~?\s?[A-G0-9](?:\.\d+)*",
    r"\\includegraphics\[[^\]]*\]\{[^}]*\}",
    r"[A-Za-z]\d",                       # sm_120, INT4, BF16, Qwen3, 6^6 exponents
    r"\d+\s?(?:bits?|-bit)",
)

#: A claim token: a decimal, or an integer of three digits or more. Two-digit integers are
#: excluded -- "the six adapters", "8 intents", "20 candidates" are structure the body
#: states in words, and including them buries the real findings in noise.
_CLAIM = re.compile(r"(?<![\w.])(\d+\.\d+|\d{3,})(?![\w])")


def _strip_structure(s: str) -> str:
    for pat in _STRUCTURE:
        s = re.sub(pat, " ", s)
    return s


def summary_regions(main: str) -> dict[str, str]:
    """The four kinds of site whose job is to restate something measured elsewhere."""
    out: dict[str, str] = {}
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    if m:
        out["abstract"] = m.group(1)
    caps = _captions(main)
    m = re.search(r"\\section\{Introduction\}(.*?)(?=\\section\{)", main, re.S)
    if m:
        out["introduction"] = _without(m.group(1), caps)
    m = re.search(r"\\section\{Conclusion\}(.*?)(?=\\balance|\\bibliographystyle)",
                  main, re.S)
    if m:
        out["conclusion"] = _without(m.group(1), caps)
    for i, cap in enumerate(caps, 1):
        out[f"caption {i}"] = cap
    return out


def _without(text: str, caps: list[str]) -> str:
    """Captions are their own site. A figure floated into §1 would otherwise have every
    number in its caption reported twice, under two names, which trains a reader of this
    gate's output to skim it."""
    for c in caps:
        text = text.replace(c, " ")
    return text


def _captions(text: str) -> list[str]:
    """`\\caption{...}` bodies, brace-balanced. A regex to the first `}` truncates every
    caption in this paper at its first `\\texttt{}`."""
    out: list[str] = []
    for m in re.finditer(r"\\caption\{", text):
        depth, i = 1, m.end()
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        out.append(text[m.end():i - 1])
    return out


def body_text(main: str, appendices: str) -> str:
    """Everything that is not a summary site: the sections that do the measuring."""
    body = main
    for m in re.finditer(r"\\begin\{abstract\}.*?\\end\{abstract\}", body, re.S):
        body = body.replace(m.group(0), " ")
    m = re.search(r"\\section\{Introduction\}(.*?)(?=\\section\{)", body, re.S)
    if m:
        body = body.replace(m.group(1), " ")
    m = re.search(r"\\section\{Conclusion\}(.*?)(?=\\balance|\\bibliographystyle)",
                  body, re.S)
    if m:
        body = body.replace(m.group(1), " ")
    for cap in _captions(body):
        body = body.replace(cap, " ")
    return body + "\n" + appendices


#: An interval as a summary writes one: [+4.2, +12.5], [90.7\%, 107.6\%], [0.0, 20.8].
_INTERVAL = re.compile(r"\[\s*[+-]?(\d+\.?\d*)\\?%?\s*,\s*[+-]?(\d+\.?\d*)\\?%?\s*\]")

#: How far apart the two ends of an interval may sit in the body and still count as the
#: same interval. One sentence.
_WINDOW = 200


def _rounds_to(printed: str, have: list[float]) -> bool:
    """Does some measured value round to what the summary printed?

    A summary writing 0.363 for a measured 0.3634 is ordinary scientific prose, and a gate
    demanding digit-identical restatement would force the paper to quote every number at
    appendix precision. So resolution is by rounding at the precision the summary chose,
    which is also the precision at which the claim is being made.
    """
    dp = len(printed.split(".")[1]) if "." in printed else 0
    target = float(printed)
    return any(abs(round(v, dp) - target) < 10 ** -(dp + 3) for v in have)


def unsourced(main: str, appendices: str) -> list[tuple[str, str, str]]:
    """(site, claim, context) for every claim a summary asserts and no section states.

    Two rules, because a scalar and an interval fail differently. A scalar resolves if
    some measured value rounds to it. An interval resolves only if BOTH ends appear
    within one sentence of the body -- otherwise `[+4.2, +12.5]` is satisfied by a 4.2
    in one appendix and a 12.5 in another, which is how a gate on bare numbers would
    have passed the very defect it was written for.
    """
    body = _strip_structure(body_text(main, appendices))
    have = [float(x) for x in _CLAIM.findall(body)]
    bad: list[tuple[str, str, str]] = []
    for site, text in summary_regions(main).items():
        clean = _strip_structure(text)
        paired: set[int] = set()
        for m in _INTERVAL.finditer(clean):
            paired |= set(range(m.start(), m.end()))
            lo_s, hi_s = m.group(1), m.group(2)
            if not any(
                _rounds_to(lo_s, [float(x) for x in _CLAIM.findall(w)])
                and _rounds_to(hi_s, [float(x) for x in _CLAIM.findall(w)])
                for w in _windows(body, lo_s)
            ):
                ctx = " ".join(clean[max(0, m.start() - 40):m.end() + 40].split())
                bad.append((site, f"[{lo_s}, {hi_s}]", ctx))
        for m in _CLAIM.finditer(clean):
            if m.start() in paired or _rounds_to(m.group(1), have):
                continue
            lo = max(0, m.start() - 40)
            bad.append((site, m.group(1),
                        " ".join(clean[lo:m.end() + 40].split())))
    return bad


def _windows(body: str, anchor: str) -> list[str]:
    """Body neighbourhoods around every occurrence of `anchor`, for interval matching."""
    out = []
    for m in re.finditer(re.escape(anchor), body):
        out.append(body[max(0, m.start() - _WINDOW):m.end() + _WINDOW])
    return out


def check() -> list[tuple[str, str, str]]:
    return unsourced(
        (TEXDIR / "main.tex").read_text(encoding="utf-8"),
        (TEXDIR / "appendices.tex").read_text(encoding="utf-8"))


def main() -> int:
    bad = check()
    sites = len(summary_regions((TEXDIR / "main.tex").read_text(encoding="utf-8")))
    print(f"{sites} summary sites checked against the body and appendices")
    for site, num, ctx in bad:
        print(f"\nUNSOURCED  [{site}] {num}\n  ...{ctx}...")
    if bad:
        print(f"\n{len(bad)} numbers are asserted in a summary and stated nowhere else")
        return 1
    print("every number a summary asserts is stated by a section that measures it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
