r"""Universal quantifiers in the methods text, listed for a manual sweep.

This is **not** a pass/fail gate and is deliberately not wired into the build. It cannot
be one: whether "every per-adapter number is measured on layers 0, 12, 24 and 35" is true
depends on facts about two base models' layer counts that no regex reaches. What it does
is produce the list a human has to walk, and make the list *stable*, so a reviewer can
diff it between rounds and look only at what is new.

**Why it exists.** A universally quantified sentence creates an obligation across the
whole document, and fixing the one site that motivated it does not discharge that
obligation (`METHODOLOGY.md` M.11). Two shipped:

* "Every per-adapter number in this paper is measured on layers 0, 12, 24 and 35" ---
  written in the round that added the layer disclosure, true of Qwen3-8B, and layer 35
  does not exist on the 32-layer Llama that carries the safety adapter. Section 4.1 named
  that adapter's real layers, 0/10/21/31, four paragraphs later.
* "correlation below 0.0011 across all nine adapters" --- the control covers six. The
  released tool's own banner said six, correctly, and the paper drifted away from it.

Both were added by a round trying to be more precise, which is the pattern: **the sentence
advertising a standard is what invites the check that finds the violation.**

Usage:
    PYTHONPATH=src python analysis/quantifiers.py            # the list
    PYTHONPATH=src python analysis/quantifiers.py --since HEAD~1   # only what is new
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Files whose universal claims bind the whole document. Results sections are excluded:
#: "every adapter has signal exceeding noise" is a finding with a stated population, not a
#: standard the rest of the paper must meet.
SOURCES = ("paper/tex/main.tex", "paper/03-method.md", "METHODOLOGY.md")

QUANTIFIER = re.compile(
    r"\b(?:[Ee]very|[Aa]ll (?:nine|six|three|four|seven|eight|ten|eleven|\d+)?\s*"
    r"(?:adapters?|layers?|modules?|numbers?|runs?|conditions?|records?|sites?)|"
    r"[Ee]ach (?:adapter|layer|module|run|record)|"
    r"[Nn]o (?:adapter|number|claim|run)|[Aa]lways|[Nn]ever|throughout|"
    r"in this paper|anywhere)\b")

#: A quantified sentence that also pins a concrete value is the risky kind: the quantifier
#: promises coverage and the value is what a reader checks it against.
CONCRETE = re.compile(r"\d")


def sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.])\s+", text) if s.strip()]


def scan(text: str) -> list[str]:
    return [s for s in sentences(text)
            if QUANTIFIER.search(s) and CONCRETE.search(s) and len(s) < 500]


def at_revision(rev: str) -> set[str]:
    out: set[str] = set()
    for rel in SOURCES:
        r = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=REPO_ROOT,
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            out |= set(scan(r.stdout))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="list only quantified sentences added since this rev")
    args = ap.parse_args()

    now: set[str] = set()
    for rel in SOURCES:
        p = REPO_ROOT / rel
        if p.exists():
            now |= set(scan(p.read_text(encoding="utf-8")))

    if args.since:
        before = at_revision(args.since)
        new = sorted(now - before)
        print(f"{len(new)} universally quantified sentences added since {args.since}"
              f" (of {len(now)} total)\n")
        for s in new:
            print(f"  - {s[:300]}\n")
        print("Each one is an obligation across the whole document. Check it against "
              "every population it now covers, not only the one that motivated it.")
        return 0

    print(f"{len(now)} universally quantified sentences naming a value, "
          f"across {len(SOURCES)} sources\n")
    for s in sorted(now):
        print(f"  - {s[:220]}")
    print("\nThis is a worklist, not a verdict. Nothing here is checked automatically "
          "and nothing fails the build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
