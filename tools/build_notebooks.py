#!/usr/bin/env python
"""notebook のソース (.py) から .ipynb を生成する。

教材の原本は jupytext と同じ percent 形式の ``notebooks/src/*.py`` で持ち、
解答入り notebook と学生用の穴埋め版を機械的に生成する。生成時には、
見出し番号、解答ブロック、nbformat の妥当性も検査する。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebooks" / "src"
NOTEBOOK_DIR = ROOT / "notebooks"
EXERCISE_DIR = ROOT / "exercises"

SOLUTION_BEGIN = "# BEGIN SOLUTION"
SOLUTION_END = "# END SOLUTION"


def parse_percent_format(text: str) -> list[tuple[str, str]]:
    """percent 形式の文字列を ``(セル種別, 内容)`` のリストに分解する。"""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        if buffer and "".join(buffer).strip():
            cells.append((kind, "".join(buffer).rstrip("\n")))
        buffer.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
        elif kind == "markdown":
            if line.startswith("# "):
                buffer.append(line[2:])
            elif stripped == "#":
                buffer.append("\n")
            else:
                buffer.append(line)
        else:
            buffer.append(line)
    flush()
    return cells


def validate_solution_markers(source: str, path: Path) -> None:
    """解答ブロックの閉じ忘れ、余分な終了、入れ子を拒否する。"""
    inside = False
    begin_line = -1
    for line_number, line in enumerate(source.splitlines(), start=1):
        marker = line.strip()
        if marker == SOLUTION_BEGIN:
            if inside:
                raise ValueError(
                    f"{path}:{line_number}: {SOLUTION_BEGIN} が入れ子になっている"
                )
            inside = True
            begin_line = line_number
        elif marker == SOLUTION_END:
            if not inside:
                raise ValueError(
                    f"{path}:{line_number}: 対応する {SOLUTION_BEGIN} がない"
                )
            inside = False
    if inside:
        raise ValueError(
            f"{path}:{begin_line}: {SOLUTION_BEGIN} が {SOLUTION_END} で閉じていない"
        )


def validate_heading(source_path: Path, cells: list[tuple[str, str]]) -> None:
    """ファイル名の番号と最初の Markdown 見出し番号を一致させる。"""
    expected = source_path.stem.split("_", 1)[0]
    first_markdown = next((body for kind, body in cells if kind == "markdown"), "")
    match = re.search(r"^#\s+(\d{2})(?:\s|$)", first_markdown, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{source_path}: 最初の Markdown セルに '# NN' 見出しがない")
    actual = match.group(1)
    if actual != expected:
        raise ValueError(
            f"{source_path}: ファイル番号 {expected} と見出し番号 {actual} が一致しない"
        )


def strip_solutions(source: str, path: Path) -> str:
    """解答部分を削除し、同じインデントの ``...`` に置き換える。"""
    validate_solution_markers(source, path)
    output: list[str] = []
    inside = False
    indent = ""

    for line in source.splitlines():
        marker = line.strip()
        if marker == SOLUTION_BEGIN:
            inside = True
            indent = line[: len(line) - len(line.lstrip())]
            continue
        if marker == SOLUTION_END:
            inside = False
            output.append(f"{indent}...  # ここを埋めること")
            continue
        if not inside:
            output.append(line)
    return "\n".join(output)


def build(source_path: Path, *, with_solutions: bool) -> nbformat.NotebookNode:
    """1 つのソースから notebook を組み立て、構造を検証する。"""
    text = source_path.read_text(encoding="utf-8")
    validate_solution_markers(text, source_path)
    cells = parse_percent_format(text)
    validate_heading(source_path, cells)

    # 解答ブロックはコードセル専用。Markdown セルに置くと strip_solutions が
    # 通らず、穴埋め版に解答がそのまま残る（外部レビューの指摘 #9）。
    for kind, content in cells:
        if kind == "markdown" and SOLUTION_BEGIN in content:
            raise ValueError(
                f"{source_path}: Markdown セルに {SOLUTION_BEGIN} がある。"
                "解答ブロックはコードセルにだけ置けること。"
                "Markdown の模範解答はマーカーなしで書くこと。"
            )

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (pwsyseng)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}

    for kind, content in cells:
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(content))
        else:
            body = content if with_solutions else strip_solutions(content, source_path)
            if body.strip():
                notebook.cells.append(nbformat.v4.new_code_cell(body))

    # セル id を位置から決定的に振る。nbformat の既定は乱数の id で、
    # 再生成のたびに全ファイルが差分になり、CI の
    # 「再生成して git diff --exit-code」検査が内容と無関係に落ちる。
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"{source_path.stem}-{index:03d}"

    nbformat.validate(notebook)
    return notebook


def main(argv: list[str]) -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    EXERCISE_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(SOURCE_DIR.glob("*.py"))
    if argv:
        sources = [p for p in sources if any(p.name.startswith(a) for a in argv)]
    if not sources:
        print(f"変換対象が見つからない: {SOURCE_DIR}", file=sys.stderr)
        return 1

    for source in sources:
        name = source.stem + ".ipynb"
        solution = build(source, with_solutions=True)
        nbformat.write(solution, NOTEBOOK_DIR / name)

        exercise = build(source, with_solutions=False)
        nbformat.write(exercise, EXERCISE_DIR / name)

        has_blanks = SOLUTION_BEGIN in source.read_text(encoding="utf-8")
        note = "（穴埋めあり）" if has_blanks else "（穴埋めなし）"
        print(f"  {source.name} -> notebooks/{name}, exercises/{name} {note}")

    print(f"\n{len(sources)} 件の notebook を生成し、構造を検証した。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
