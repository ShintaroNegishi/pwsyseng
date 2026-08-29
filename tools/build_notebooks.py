#!/usr/bin/env python
"""notebook のソース (.py) から .ipynb を生成する。

なぜソースを .py で持つのか
---------------------------
.ipynb は JSON なので、差分が読めず、複数人での編集も衝突しやすい。
そこで教材の原本は jupytext と同じ「percent 形式」の .py で持ち、
配布用の .ipynb はここから機械的に生成する。教員が中身を直すときは
必ず ``notebooks/src/*.py`` を編集すること。

生成されるもの
--------------
``notebooks/``
    解答入りの notebook（教員用）。
``exercises/``
    ``# BEGIN SOLUTION`` から ``# END SOLUTION`` までを取り除き、
    代わりに ``# TODO:`` の指示だけを残した学生配布用 notebook。

使い方::

    python tools/build_notebooks.py            # すべて生成
    python tools/build_notebooks.py 01 04      # 指定した番号だけ
"""

from __future__ import annotations

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
    """percent 形式の文字列を (セル種別, 内容) のリストに分解する。"""
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
            # markdown セルは行頭の "# " を剥がす。
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


def strip_solutions(source: str) -> str:
    """解答部分を取り除く。

    ``# BEGIN SOLUTION`` から ``# END SOLUTION`` までを削除し、
    その位置に元のインデントを保った ``...`` を残す。直前に
    ``# TODO:`` で始まる行があれば、それは指示として残す。
    """
    lines = source.splitlines()
    output: list[str] = []
    inside = False
    indent = ""

    for line in lines:
        stripped = line.strip()
        if stripped == SOLUTION_BEGIN:
            inside = True
            indent = line[: len(line) - len(line.lstrip())]
            continue
        if stripped == SOLUTION_END:
            inside = False
            output.append(f"{indent}...  # ここを埋めること")
            continue
        if not inside:
            output.append(line)
    return "\n".join(output)


def build(source_path: Path, *, with_solutions: bool) -> nbformat.NotebookNode:
    """1 つのソースから notebook を組み立てる。"""
    text = source_path.read_text(encoding="utf-8")
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (gridops)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}

    for kind, content in parse_percent_format(text):
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(content))
        else:
            body = content if with_solutions else strip_solutions(content)
            if body.strip():
                notebook.cells.append(nbformat.v4.new_code_cell(body))
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

    print(f"\n{len(sources)} 件の notebook を生成した。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
