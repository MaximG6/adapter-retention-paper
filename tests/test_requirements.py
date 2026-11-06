"""requirements.txt must cover every third-party module the repository imports.

Appendix D tells a reader that `torch` from the CUDA-matched index plus this file is the
whole install. That claim has now been wrong twice. The first time, matplotlib, markdown
and pypdf were missing and every figure script and both PDF builds failed on import; a
comment was added to requirements.txt saying so. The second time, `pymupdf` was missing
and `analysis/texcheck.py` and `analysis/pagecost.py` failed -- found only when a fresh
clone was checked against the file rather than the file being read.

Both survived because the imports are lazy: `import fitz` sits inside the function that
needs it, so every module imports cleanly and the absence surfaces only when the gate
runs, in an environment that already had the package. A note in a file is not a check
(M.3, M.5), so this is the check.

It is deliberately static. Asking whether a package is *installed* would test this
machine; asking whether it is *declared* tests the claim Appendix D makes.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Import name -> distribution name, where PyPI disagrees with the module.
ALIAS = {"fitz": "pymupdf", "yaml": "pyyaml", "PIL": "pillow",
         "sklearn": "scikit-learn", "huggingface_hub": "huggingface-hub"}

#: Installed separately from the CUDA-matched index, because a mismatched build imports
#: cleanly and then produces wrong numbers on Blackwell. Appendix D says so explicitly,
#: and requirements.txt says why it is not listed there.
INSTALLED_SEPARATELY = {"torch"}

SKIP_DIRS = {".cache", ".venv", "__pycache__", ".git", ".pytest_cache"}


def _declared() -> set[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    return {re.split(r"[=<>!~\[]", line.strip())[0].lower()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def _local_modules() -> set[str]:
    """Modules importable by bare name because analysis/ and src/ go on sys.path."""
    names = {"ar"}
    for sub in ("analysis", "scripts", "tests", "src/ar"):
        names |= {p.stem for p in (ROOT / sub).glob("*.py")}
    return names


def _third_party_imports() -> dict[str, set[str]]:
    local, stdlib = _local_modules(), set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for py in sorted(ROOT.rglob("*.py")):
        if SKIP_DIRS & set(py.parts):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            else:
                continue
            for m in mods:
                top = m.split(".")[0]
                if top in stdlib or top in local:
                    continue
                dist = ALIAS.get(top, top).lower()
                found.setdefault(dist, set()).add(py.relative_to(ROOT).as_posix())
    return found


def test_every_third_party_import_is_declared() -> None:
    declared = _declared()
    missing = {m: sorted(f) for m, f in _third_party_imports().items()
               if m not in declared and m not in INSTALLED_SEPARATELY}
    assert not missing, (
        "imported but not in requirements.txt, so a reader following Appendix D "
        f"cannot run this code: {missing}")


def test_the_check_would_have_caught_the_omission_that_shipped() -> None:
    """Known-bad input: pymupdf removed from the declared set is exactly the state the
    file was in, and `fitz` must be reported under its distribution name rather than its
    module name (M.3)."""
    imports = _third_party_imports()
    assert "pymupdf" in imports, "fitz should be recorded as pymupdf"
    assert any("texcheck" in f for f in imports["pymupdf"])

    declared = _declared() - {"pymupdf"}
    missing = [m for m in imports if m not in declared and m not in INSTALLED_SEPARATELY]
    assert missing == ["pymupdf"], missing


def test_torch_is_not_declared_and_that_is_deliberate() -> None:
    """If torch were ever added to requirements.txt a reader could install a CPU or
    wrong-CUDA build over the one Appendix D directs them to, which fails silently on
    Blackwell rather than loudly."""
    assert "torch" not in _declared()
    assert "torch" in _third_party_imports()
