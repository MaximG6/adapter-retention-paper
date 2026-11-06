"""Assemble the technical report PDF from the manuscript sections.

Markdown -> HTML (python-markdown) -> PDF (headless Edge/Chrome). Chosen over a LaTeX
toolchain because it needs no multi-gigabyte install and the browser is already present
on the target machine; the report is a technical report for direct circulation, not a
camera-ready submission.

Section order matches the manuscript. Figures are embedded as data URIs so the PDF is a
single self-contained file.

Usage:
    python analysis/build_pdf.py
"""

from __future__ import annotations

import argparse
import base64
import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER = REPO_ROOT / "paper"
FIGDIR = PAPER / "figures"
OUT_HTML = REPO_ROOT / "paper" / "adapter-retention-technical-report.html"
OUT_PDF = REPO_ROOT / "paper" / "adapter-retention-technical-report.pdf"

# Manuscript order. READTHROUGH.md is a working document and excluded.
SECTIONS = [
    "00-abstract.md",
    "01-introduction.md",
    "02-related-work.md",
    "03-method.md",
    "04-results-weight-space.md",
    "06-results-advertised-vs-measured.md",
    "07-registered-predictions.md",
    "08-09-limitations-conclusion.md",
    "appendix-A-tool.md",
    "appendix-B-tables.md",
    "appendix-C-prompts.md",
    "appendix-D-reproduction.md",
]

# Figures are referenced in prose as "Figure N"; place each after the section that
# first invokes it so a reader meets it where it is discussed.
FIGURE_FOR_SECTION = {
    "01-introduction.md": ["fig01_erasure_vs_survival"],
    "04-results-weight-space.md": [
        "fig02_channel_model", "fig03_forest", "fig04_amplification",
        "fig11_layer_profile", "fig05_dose_response", "fig06_benign_dissociation",
        "fig07_entropy_control", "fig08_predictive_gap", "fig09_bootstrap_intervals",
    ],
    "06-results-advertised-vs-measured.md": ["fig10_refusal"],
    "appendix-A-tool.md": ["figA1_predict_validation"],
}

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.5; color: #1a1a1a; max-width: 100%; }
h1 { font-size: 19pt; margin-top: 1.6em; border-bottom: 2px solid #1a1a1a;
     padding-bottom: 0.2em; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; margin-top: 0; }
h2 { font-size: 14pt; margin-top: 1.4em; color: #14213d; }
h3 { font-size: 11.5pt; margin-top: 1.2em; color: #14213d; }
p { margin: 0.6em 0; text-align: justify; }
code { font-family: 'Cascadia Mono', Consolas, monospace; font-size: 9pt;
       background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 2px; }
pre { background: #f7f7f7; border-left: 3px solid #c8c8c8; padding: 0.7em 0.9em;
      overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 8.8pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #d0d0d0; padding: 0.32em 0.5em; text-align: left; }
th { background: #eef1f5; font-weight: bold; }
tr:nth-child(even) td { background: #fbfbfb; }
blockquote { border-left: 3px solid #b0b0b0; margin-left: 0; padding-left: 1em;
             color: #444; font-style: italic; }
figure { margin: 1.4em 0; page-break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; border: 1px solid #e0e0e0; }
figcaption { font-size: 8.5pt; color: #666; margin-top: 0.4em; font-style: italic; }
hr { border: none; border-top: 1px solid #ddd; margin: 1.6em 0; }
.title-block { text-align: center; margin-bottom: 2.5em; page-break-after: always; }
.title-block h1 { border: none; font-size: 24pt; page-break-before: avoid; }
.title-block .sub { font-size: 12pt; color: #444; margin-top: 0.8em; }
.title-block .meta { font-size: 10pt; color: #666; margin-top: 2em; line-height: 1.8; }
"""

TITLE = """
<div class="title-block">
<h1>Near-Total Weight-Space Erasure Without Behavioural Collapse</h1>
<div class="sub">What Survives When a Merged LoRA Is Quantized</div>
<div class="meta">
Technical report &middot; Phase 0 and Phase 1<br>
All numbers reproducible from the accompanying repository<br>
<code>paper/appendix-D-reproduction.md</code><br>
November 2, 2026
</div>
</div>
"""


def embed(fig: str) -> str:
    png = FIGDIR / f"{fig}.png"
    if not png.exists():
        return ""
    b64 = base64.b64encode(png.read_bytes()).decode()
    num = re.match(r"fig(A?\d+)", fig).group(1).lstrip("0") or "1"
    return (f'<figure><img src="data:image/png;base64,{b64}">'
            f'<figcaption>Figure {num}</figcaption></figure>\n')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "md_in_html"])
    parts: list[str] = [TITLE]
    for name in SECTIONS:
        p = PAPER / name
        if not p.exists():
            print(f"  WARNING: {name} missing, skipped")
            continue
        text = p.read_text(encoding="utf-8")
        # Drop the drafting notes: they address us, not a reader.
        text = re.sub(r"^\*Draft\..*?\*\s*$", "", text, flags=re.M | re.S, count=1)
        text = re.sub(r"^\*Generated by .*?regenerate\.\*\s*$", "", text, flags=re.M)
        md.reset()
        parts.append(md.convert(text))
        for fig in FIGURE_FOR_SECTION.get(name, []):
            parts.append(embed(fig))

    html = (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>Adapter Retention Under Post-Training Quantization</title>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"wrote {OUT_HTML.relative_to(REPO_ROOT)} ({len(html):,} bytes)")
    if args.html_only:
        return 0

    browser = None
    for cand in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                 r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                 shutil.which("chromium"), shutil.which("google-chrome")):
        if cand and Path(cand).exists():
            browser = cand
            break
    if browser is None:
        print("No Edge/Chrome found; HTML written, PDF skipped.", file=sys.stderr)
        return 1

    cmd = [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if not OUT_PDF.exists():
        print(f"PDF not produced.\n{r.stdout}\n{r.stderr}", file=sys.stderr)
        return 1
    print(f"wrote {OUT_PDF.relative_to(REPO_ROOT)} "
          f"({OUT_PDF.stat().st_size / 1024 / 1024:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
