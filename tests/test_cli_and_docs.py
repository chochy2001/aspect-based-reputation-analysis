from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_help_does_not_import_heavy_dependencies() -> None:
    for script in ("train_svm.py", "train_beto.py", "eval_all.py"):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout


def test_paper_citations_exist_in_bibliography() -> None:
    paper = (REPO_ROOT / "docs" / "paper.md").read_text(encoding="utf-8")
    references = (REPO_ROOT / "docs" / "references.bib").read_text(encoding="utf-8")

    cited_keys = set(re.findall(r"@([A-Za-z0-9_:-]+)", paper))
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)", references))

    assert cited_keys, "El paper debe incluir citas bibliográficas."
    assert cited_keys <= bib_keys


def test_documentation_avoids_known_delivery_risks() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            REPO_ROOT / "README.md",
            REPO_ROOT / "data" / "README.md",
            REPO_ROOT / "docs" / "paper.md",
            REPO_ROOT / "docs" / "references.bib",
        ]
    )

    forbidden = [
        "paper.tex",
        "Cohen's $\\kappa$",
        "Bibliografía verificada",
        "chain-of-thought",
        "validando la hipótesis",
        "se valida",
        "desempeños sin precedentes",
    ]
    for phrase in forbidden:
        assert phrase not in combined
