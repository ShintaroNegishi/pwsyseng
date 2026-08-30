"""教材の番号・公開表現・既定の解析範囲の整合性を検査する。"""

from __future__ import annotations

import re
from pathlib import Path

from gridops import load_case
from gridops.plotting import COURSE_THEMES
from gridops.security import screen_n1

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_source_heading_matches_filename() -> None:
    for source in sorted((ROOT / "notebooks" / "src").glob("*.py")):
        expected = source.stem.split("_", 1)[0]
        text = source.read_text(encoding="utf-8")
        match = re.search(r"^# # (\d{2})(?:\s|$)", text, flags=re.MULTILINE)
        assert match is not None, source
        assert match.group(1) == expected, source


def test_public_docs_do_not_reference_private_workspace_paths() -> None:
    paths = [
        ROOT / "docs" / "references.md",
        ROOT / "docs" / "data_provenance.md",
        ROOT / "docs" / "solver_notes.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = (
        "研究室の `textbook/`",
        "`PWSYS/PWSYS/CPSDAM.py`",
        "`banditUC/`",
        "`inverseUC/`",
        "研究室の `OPF/`",
        "研究室の `energy-mix/`",
        "研究室の `制御系設計論/`",
    )
    for token in forbidden:
        assert token not in text


def test_default_n1_report_records_bridge_outages_as_unassessed() -> None:
    report = screen_n1(load_case("wscc9"), method="lodf", check_voltage=False)
    assert [key for key, _reason in report.skipped] == [(1, 4), (2, 7), (3, 9)]
    assert all("未評価" in reason or "適用でき" in reason for _key, reason in report.skipped)



def test_timescale_map_uses_integrated_course_numbers() -> None:
    labels = [theme[2] for theme in COURSE_THEMES]
    assert labels == [
        "pwsyseng 10-12, 17",
        "pwsyseng 13-16",
        "pwsyseng 01-04, 09",
        "pwsyseng 05-08",
        "pwsyseng 18",
    ]


def test_adequacy_notebook_describes_lole_as_an_expectation() -> None:
    text = (ROOT / "notebooks" / "src" / "18_adequacy.py").read_text(
        encoding="utf-8"
    )
    assert "供給不足状態になると期待されるか" in text
    assert "何時間足りないか」という**確率**" not in text
