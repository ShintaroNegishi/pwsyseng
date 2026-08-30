"""パッケージ全体の整合性（統合の検証）。

個々のモジュールは自分のテストを持っているので、ここでは **モジュールを
またいだときにだけ壊れる性質**だけを見る。

1. ``gridops/__init__.py`` の docstring に書いた doctest が実際に通ること
   （教材の 1 行目が動かないのは、計算が間違っているのと同じくらい悪い）
2. ``__all__`` が実体と食い違っていないこと、アルファベット順であること
3. ``import gridops`` が ``genstab`` を要求しないこと
4. 統合のときに 3 モジュールから 1 箇所へ寄せた計算が、寄せる前と同じ
   答えを出すこと（:meth:`Case.scaled` の ``keep_generation`` と、
   :func:`gridops.solvers.problem` / :func:`gridops.solvers.lp_sum`）

いずれも実装の出力を実装で再計算するテストにはしない。1 は docstring と
実装の突き合わせ、4 は解析的に答えが分かる問題と手で組んだ注入との
突き合わせである。
"""

from __future__ import annotations

import doctest
import importlib
import subprocess
import sys

import numpy as np
import pytest

import gridops
from gridops import load_case, solvers

#: 再エクスポートの元になっているサブモジュール。
SUBMODULES = [
    "adequacy",
    "case",
    "commitment",
    "dc",
    "dispatch",
    "interop",
    "loader",
    "plotting",
    "powerflow",
    "security",
    "solvers",
    "voltage",
    "ybus",
]


@pytest.fixture(scope="module")
def case():
    """同梱の WSCC 9 母線ケース。"""
    return load_case("wscc9")


# ======================================================================
# 1. パッケージ docstring の doctest
# ======================================================================
def test_package_docstring_doctest_runs():
    """``gridops/__init__.py`` の使用例がそのまま動く。

    第 00 回の 1 セル目に貼る想定の例なので、ここが落ちたら教材が
    動かない。``pytest --doctest-modules`` を忘れても気づけるように、
    通常のテストからも回しておく。
    """
    results = doctest.testmod(gridops, verbose=False, raise_on_error=False)
    assert results.failed == 0, f"docstring の実行例が {results.failed} 件失敗した"
    # 例が 1 つも無いのに「通った」と報告される事故を防ぐ。
    assert results.attempted >= 8


# ======================================================================
# 2. 公開 API の整合性
# ======================================================================
def test_all_names_exist():
    """``__all__`` に書いた名前がすべて実体を持つ。"""
    missing = [name for name in gridops.__all__ if not hasattr(gridops, name)]
    assert missing == []


def test_all_is_sorted_and_unique():
    """``__all__`` はアルファベット順で重複がない（genstab に合わせた作法）。"""
    names = [name for name in gridops.__all__ if name != "__version__"]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert gridops.__all__[-1] == "__version__"


def test_submodule_all_names_exist():
    """各サブモジュールの ``__all__`` も実体と食い違っていない。"""
    for name in SUBMODULES:
        module = importlib.import_module(f"gridops.{name}")
        exported = getattr(module, "__all__", None)
        if exported is None:
            continue
        missing = [n for n in exported if not hasattr(module, n)]
        assert missing == [], f"gridops.{name}.__all__ に実体のない名前: {missing}"


def test_reexported_objects_are_the_same_objects():
    """再エクスポートは別名ではなく同一オブジェクトである。

    ``from gridops import screen_n1`` と
    ``from gridops.security import screen_n1`` が別物になっていると、
    ``isinstance`` や ``is`` の比較が notebook で謎に失敗する。
    """
    assert gridops.screen_n1 is gridops.security.screen_n1
    assert gridops.dc_opf is gridops.dispatch.dc_opf
    assert gridops.PowerFlowSolution is gridops.powerflow.PowerFlowSolution
    # solve は潮流と数理計画で衝突するので、直下には潮流のほうだけを
    # 別名で出している。
    assert gridops.solve_powerflow is gridops.powerflow.solve
    assert not hasattr(gridops, "solve")


def test_solvers_names_are_not_reexported_at_top_level():
    """ソルバ層の名前は直下に出さない（``pulp`` に触れる層を 1 つに保つため）。"""
    for name in ("solve", "Solution", "available_solver", "binary", "problem"):
        assert name not in gridops.__all__


# ======================================================================
# 3. 依存の分離
# ======================================================================
def test_importing_gridops_does_not_import_genstab():
    """``import gridops`` だけでは ``genstab`` を読み込まない。

    :mod:`gridops.interop` を ``__init__`` から eager に import しても、
    ``genstab`` の import が関数の中に閉じている限りこの性質は保たれる。
    別プロセスで確かめるのは、同じプロセスで先に他のテストが ``genstab``
    を読んでいると意味を失うためである。
    """
    code = "import gridops, sys; print('genstab' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


# ======================================================================
# 4. 統合で 1 箇所に寄せた計算
# ======================================================================
def test_scaled_keeps_generation_when_asked(case):
    """``keep_generation=True`` が参照解の**発電**を引き継ぐ。

    比べる相手は :meth:`Case.bus_injection` の出力ではなく、ケース
    ファイルの発電量と負荷から手で組んだ注入である。
    """
    factor = 1.5
    scaled = case.scaled(factor, keep_generation=True)
    p, q = scaled.bus_injection()

    for i, bus in enumerate(case.buses):
        pg, qg = case.reference.generation.get(bus.id, (0.0, 0.0))
        assert p[i] == pytest.approx(pg - bus.pd * factor, abs=1e-12)
        assert q[i] == pytest.approx(qg - bus.qd * factor, abs=1e-12)


def test_scaled_drops_generation_by_default(case):
    """既定 (``keep_generation=False``) では発電がゼロになる。

    これは仕様であって不具合ではない。ここを固定しておかないと、
    「倍率をかけたケースをそのまま潮流に渡すと別系統を解く」という罠が
    静かに変わってしまう。
    """
    scaled = case.scaled(1.5)
    assert scaled.reference is None
    p, _ = scaled.bus_injection()
    for i, bus in enumerate(case.buses):
        assert p[i] == pytest.approx(-bus.pd * 1.5, abs=1e-12)


def test_without_branch_keep_generation_matches_manual_replacement(case):
    """``without_branch(key, keep_generation=True)`` が枝だけの差し替えと一致する。

    N-1 の評価は「事故前の発電を据え置いて枝を 1 本外す」ものなので、
    :func:`dataclasses.replace` で枝だけを入れ替えたケースと注入が
    一致しなければならない。
    """
    from dataclasses import replace

    key = (5, 7)
    outaged = case.without_branch(key, keep_generation=True)
    manual = replace(case, branches=[b for b in case.branches if b.key() != key])

    assert [b.key() for b in outaged.branches] == [b.key() for b in manual.branches]
    p_out, q_out = outaged.bus_injection()
    p_man, q_man = manual.bus_injection()
    assert np.allclose(p_out, p_man, atol=1e-15)
    assert np.allclose(q_out, q_man, atol=1e-15)


def test_carried_reference_marks_the_voltages_as_stale(case):
    """引き継いだ参照解は「電圧は答えではない」と自己申告する。

    発電だけを引き継ぐので ``checks``（総損失・slack 出力）は空になり、
    ``source`` に注意書きが入る。ここを緩めると、変形後のケースの
    ``reference.v`` を答え合わせに使う誤りが静かに通ってしまう。
    """
    scaled = case.scaled(1.2, keep_generation=True)
    assert scaled.reference is not None
    assert dict(scaled.reference.checks) == {}
    assert "答え合わせには使えない" in scaled.reference.source
    assert scaled.reference.generation == case.reference.generation


def test_solvers_problem_and_lp_sum_solve_a_hand_checked_lp():
    """:func:`solvers.problem` と :func:`solvers.lp_sum` が素の PuLP と同じ答えを出す。

    比べる相手は解析解である。``min 2x + 5y`` を ``x + y == 10``,
    ``0 <= x <= 4`` で解くと ``x = 4, y = 6``、目的関数は
    :math:`2\\cdot4 + 5\\cdot6 = 38`、需給バランスの双対は高いほうの
    限界費用 5 になる（契約の符号規約：右辺に需要を正で置く）。
    """
    problem = solvers.problem("hand_checked")
    x = solvers.variable("x", 0.0, 4.0)
    y = solvers.variable("y", 0.0, 20.0)
    problem += solvers.lp_sum([2.0 * x, 5.0 * y]), "cost"
    problem += solvers.lp_sum([x, y]) == 10.0, "balance"

    result = solvers.solve(problem, context="単体テスト")
    assert result.is_optimal
    assert result.objective == pytest.approx(38.0, abs=1e-9)
    assert result.values["x"] == pytest.approx(4.0, abs=1e-9)
    assert result.values["y"] == pytest.approx(6.0, abs=1e-9)
    assert result.duals["balance"] == pytest.approx(5.0, abs=1e-9)


def test_solvers_problem_rejects_an_unknown_sense():
    """``sense`` の綴り間違いは日本語で止める。"""
    with pytest.raises(ValueError, match="sense は 'min' か 'max'"):
        solvers.problem("bad", sense="minimise")


def test_solvers_problem_accepts_names_with_forbidden_characters():
    """問題名に空白やハイフンがあっても警告を出さずに通る。"""
    problem = solvers.problem("unit commitment-24h")
    assert " " not in problem.name


def test_dispatch_and_dc_agree_on_the_susceptance_matrix(case):
    """直流の :math:`B'` が :mod:`gridops.dc` に一本化されている。

    統合前は :mod:`gridops.dispatch` にも同じ式が書かれていた。二重実装
    のままだと、片方だけタップの扱い（:math:`1/(\\tau x)` か
    :math:`1/(\\tau^2 x)` か）を直したときに気づけない。ここでは
    :math:`B' = A^{T}\\mathrm{diag}(b)A` を手で組み直して突き合わせる。
    """
    from gridops.dispatch import _susceptance

    B, b, A, _ = _susceptance(case)
    manual = A.T @ np.diag(b) @ A
    assert np.allclose(B, manual, atol=1e-15)
    assert np.allclose(B, gridops.susceptance_matrix(case), atol=1e-15)


# ======================================================================
# 5. 付属ツール
# ======================================================================
def _run_tool(name: str, *args: str) -> subprocess.CompletedProcess:
    """``tools/<name>`` を子プロセスで走らせる。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(root / "tools" / name), *args],
        capture_output=True,
        text=True,
        cwd=root,
    )


def test_check_case_reports_the_bundled_case_as_clean():
    """``tools/check_case.py`` が同梱ケースを健全と報告する。

    参照解は教科書の 4 桁の値なので、そこから組み直した注入には丸めに
    由来する残差が必ず残る。閾値を定数で直書きすると **正しいデータを
    「壊れている」と報告する**（実際に 1e-3 の直書きが誤警報を出していた）。
    いまは掲載桁数から上界を導いているので、同梱ケースは終了コード 0 で
    通らなければならない。
    """
    out = _run_tool("check_case.py")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "問題なし" in out.stdout


def test_check_case_rounding_bound_covers_the_reference_residual(case):
    """丸めの上界が実際の残差を上回り、かつ桁違いに緩くない。

    上界が残差より小さければ誤警報を出し、逆に何桁も大きければ本物の
    不整合を見逃す。1 次の摂動解析から出した上界なので、残差との比が
    1 を超え、かつ 1 桁以内に収まっていることを確かめる。
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "check_case.py"
    spec = importlib.util.spec_from_file_location("_check_case", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    Y = gridops.build_ybus(case)
    v = case.reference.voltage
    p, q = case.bus_injection()
    residual = np.max(np.abs(v * np.conj(Y @ v) - (p + 1j * q)))
    bound = module.rounding_bound(Y, v, case.reference.digits)

    assert residual < bound
    assert bound < 10.0 * residual


def test_design_ratings_reproduces_the_shipped_thermal_ratings():
    """``tools/design_ratings.py --check`` が契約の N-1 の数値を再現する。

    これは熱容量の設計根拠が失われていないことの確認である。同梱ケースの
    ねらいは「N-1 で拘束するのが 5-7 と 7-8 のちょうど 2 本」であり、
    その根拠となる負荷率は API 契約に載っている（拘束枝 112.4% / 112.5%、
    非拘束枝の最大 91.1%）。ツールは :mod:`gridops.security` とは独立に
    交流潮流を掃引するので、両者が同じ枝を指すことに意味がある。
    """
    out = _run_tool("design_ratings.py", "--check")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "N-1 で拘束する枝: [(5, 7), (7, 8)]（2 本）" in out.stdout
    assert "拘束しない枝の最大負荷率: 91.1%" in out.stdout


def test_build_notebooks_generates_a_notebook_with_the_pwsyseng_kernel(tmp_path):
    """``tools/build_notebooks.py`` が pwsyseng のカーネル名で notebook を作る。

    穴埋め（``# BEGIN SOLUTION`` 〜 ``# END SOLUTION``）が
    ``exercises/`` 側でだけ落ちることも合わせて確かめる。生成物を
    リポジトリに残さないよう、後片付けまでやる。
    """
    nbformat = pytest.importorskip("nbformat")
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "src" / "99_package_test.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(
        "# %% [markdown]\n"
        "# # 99 パッケージテスト\n"
        "\n"
        "# %%\n"
        "import gridops\n"
        "\n"
        "# TODO: 解を求めること\n"
        "# BEGIN SOLUTION\n"
        "answer = 42\n"
        "# END SOLUTION\n",
        encoding="utf-8",
    )
    produced = [
        root / "notebooks" / "99_package_test.ipynb",
        root / "exercises" / "99_package_test.ipynb",
    ]
    try:
        out = _run_tool("build_notebooks.py", "99")
        assert out.returncode == 0, out.stdout + out.stderr

        solution = nbformat.read(produced[0].open(encoding="utf-8"), as_version=4)
        exercise = nbformat.read(produced[1].open(encoding="utf-8"), as_version=4)

        assert solution.metadata["kernelspec"]["display_name"] == "Python 3 (pwsyseng)"
        assert exercise.metadata["kernelspec"]["display_name"] == "Python 3 (pwsyseng)"

        solution_code = "\n".join(c.source for c in solution.cells if c.cell_type == "code")
        exercise_code = "\n".join(c.source for c in exercise.cells if c.cell_type == "code")
        assert "answer = 42" in solution_code
        assert "answer = 42" not in exercise_code
        assert "# TODO: 解を求めること" in exercise_code
        assert "ここを埋めること" in exercise_code
    finally:
        source.unlink(missing_ok=True)
        for path in produced:
            path.unlink(missing_ok=True)


def test_readme_quick_start_actually_runs():
    """README のクイックスタートがそのまま動く。

    README は利用者が最初に読む文書なので、ここが動かないのは実装の
    バグと同じ重さの不具合である（実際、統合前は存在しない名前
    ``gridops.solve_power_flow`` を呼んでいた）。genstab を要求する最後の
    行を含むので、genstab が入っていない環境では飛ばす。
    """
    import re
    from pathlib import Path

    pytest.importorskip("genstab")

    readme = Path(__file__).resolve().parents[1] / "README.md"
    match = re.search(r"```python\n(import gridops\n.*?)```", readme.read_text("utf-8"), re.S)
    assert match is not None, "README に python のクイックスタートが見つからない"

    namespace: dict[str, object] = {}
    exec(compile(match.group(1), "README.md", "exec"), namespace)
    assert namespace["case"].n_bus == 9


def test_build_notebooks_rejects_solution_markers_in_markdown(tmp_path):
    """Markdown セルの解答ブロックを生成段階で拒否すること。

    外部レビューの指摘 #9。``strip_solutions`` はコードセルにしか
    掛からないので、Markdown に ``# BEGIN SOLUTION`` を置くと穴埋め版に
    解答がそのまま残る。検査で入口を塞ぐ。
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = root / "notebooks" / "src" / "98_markdown_leak.py"
    source.write_text(
        "# %% [markdown]\n"
        "# # 98 テスト\n"
        "#\n"
        "# # BEGIN SOLUTION\n"
        "# 模範解答\n"
        "# # END SOLUTION\n"
        "\n"
        "# %%\n"
        "import gridops\n",
        encoding="utf-8",
    )
    try:
        out = _run_tool("build_notebooks.py", "98")
        assert out.returncode != 0
        assert "Markdown セル" in (out.stdout + out.stderr)
        assert not (root / "exercises" / "98_markdown_leak.ipynb").exists()
    finally:
        source.unlink(missing_ok=True)
        for stale in (
            root / "notebooks" / "98_markdown_leak.ipynb",
            root / "exercises" / "98_markdown_leak.ipynb",
        ):
            stale.unlink(missing_ok=True)
