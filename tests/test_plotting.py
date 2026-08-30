"""作図ヘルパ（第 00 回〜第 11 回）の検証。

作図のテストで機械的に確かめられることは限られている。見た目の良し悪しは
判定できない。そこで **教材が止まる形の失敗**だけを漏れなく捕まえる。

1. 画面の無い環境（``matplotlib.use("Agg")``）で、すべての関数が例外なく
   動くこと。渡すのは wscc9 の **実データ**である（潮流解・経済負荷配分・
   直流最適潮流・起動停止計画・容量停止確率表・P-V 曲線・N-1 の結果）。
2. **文字列リテラルに非 ASCII 文字が現れないこと。** 軸ラベルや凡例に
   日本語が混ざると、日本語フォントの無い環境で豆腐（□）になり、その場で
   授業が止まる。``ast`` で文字列リテラルだけを走査して判定する
   （docstring とコメントは除外する。コメントはそもそも AST に残らない）。
3. ``ax=None`` なら新しい Axes を作り、``ax`` を渡せばそこに描くこと。
   notebook で図を並べたり、描いた図に描き足したりできるための条件である。
4. すべての Axes に xlabel と ylabel があること。単位の無い軸は教材では
   誤りである。

そのうえで、**渡した数値がそのまま図の中にあるか**を確かめる。棒の高さや
線の座標を取り出して、もとの解や API 契約の検証済みの数値と突き合わせる。
「例外が出ないこと」だけを見ていると、全部ゼロを描く実装でも通ってしまう。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # pyplot より前に置くこと（画面の無い環境で必要）

import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pytest                     # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from gridops import Case, load_case                                   # noqa: E402
from gridops import plotting                                          # noqa: E402
from gridops.adequacy import (                                        # noqa: E402
    annual_load,
    capacity_outage_table,
    lolp,
    monte_carlo_adequacy,
)
from gridops.commitment import demand_profile, unit_commitment        # noqa: E402
from gridops.dc import dc_powerflow                                   # noqa: E402
from gridops.dispatch import dc_opf                                   # noqa: E402
from gridops.powerflow import solve as solve_power_flow               # noqa: E402

#: API 契約の表にある基準潮流の負荷率（rate_a に対する比、枝の並び順）。
#: 表は小数 1 桁なので、突き合わせの許容差は 0.15 ポイントとする。
BASE_LOADING = (0.383, 0.816, 0.432, 0.561, 0.434, 0.791, 0.577, 0.697, 0.428)

#: 手計算した容量の合計 [MW]。60x3 + 90x2 + 50x2 = 460、24x3 + 36x2 + 15x2 = 174。
INSTALLED_MW = 460.0
MINIMUM_MW = 174.0

#: API 契約で検証済みの需要 315 MW における限界費用 [円/MWh]。
LAMBDA_AT_315 = 13090.0

#: 図を描く関数の一覧。:func:`drawings` の鍵と対応する。
PLOT_NAMES = (
    "timescale_map",
    "plot_voltage_profile",
    "plot_convergence",
    "plot_pv_curve",
    "plot_network_flows",
    "plot_network_flows (dc)",
    "plot_merit_order",
    "plot_lambda_search",
    "plot_lambda_search (infeasible demand)",
    "plot_lmp",
    "plot_commitment",
    "plot_commitment_schedule",
    "plot_duck_curve",
    "plot_capacity_outage_table",
    "plot_lolp_convergence",
    "plot_contingency_ranking",
)


# ----------------------------------------------------------------------
# 実データ（fixture は module スコープで 1 回だけ計算する）
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _close_figures():
    """テストごとに Figure を片づける（20 枚を超えると警告が出るため）。"""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def case() -> Case:
    return load_case("wscc9")


@pytest.fixture(scope="module")
def solutions(case: Case) -> dict:
    """3 つの解法の潮流解。収束の比較図にそのまま使う。"""
    return {
        method: solve_power_flow(case, method)
        for method in ("newton", "fast_decoupled", "gauss_seidel")
    }


@pytest.fixture(scope="module")
def solution(solutions: dict):
    return solutions["newton"]


@pytest.fixture(scope="module")
def dc_solution(case: Case):
    return dc_powerflow(case)


@pytest.fixture(scope="module")
def opf(case: Case):
    """直流最適潮流（CBC を使う）。混雑する枝 4-6 が拘束する。"""
    return dc_opf(case)


@pytest.fixture(scope="module")
def commitment(case: Case):
    """起動停止計画（CBC を使う）。light_load 形状は起動が 1 回で済む。"""
    return unit_commitment(case, demand_profile(case, "light_load"))


@pytest.fixture(scope="module")
def copt(case: Case):
    return capacity_outage_table(case.units)


@pytest.fixture(scope="module")
def monte_carlo(case: Case, copt):
    """標本数を 4 倍ずつ増やしたモンテカルロと、その解析解。"""
    load = annual_load(case, peak_mw=380.0)
    results = [
        monte_carlo_adequacy(case.units, load, n_samples=n, seed=0)
        for n in (1_000, 4_000, 16_000)
    ]
    analytic = float(np.mean([lolp(copt, value) for value in load]))
    return results, analytic


@pytest.fixture(scope="module")
def curve(case: Case):
    """P-V 曲線。ノーズ点を越えるまで掃引した実データ。"""
    voltage = pytest.importorskip("gridops.voltage")
    try:
        return voltage.pv_curve(case, step=0.05, max_factor=3.0)
    except Exception as error:   # pragma: no cover - 他モジュールの未完成時
        pytest.skip(f"gridops.voltage がまだ使えない: {error}")


@pytest.fixture(scope="module")
def report(case: Case):
    """N-1 の結果。交流で解き直して電圧まで見た実データ。"""
    security = pytest.importorskip("gridops.security")
    try:
        return security.screen_n1(case, method="ac", check_voltage=True)
    except Exception as error:   # pragma: no cover - 他モジュールの未完成時
        pytest.skip(f"gridops.security がまだ使えない: {error}")


@pytest.fixture(scope="module")
def drawings(
    case, solution, solutions, dc_solution, opf, commitment, copt, monte_carlo,
    curve, report,
):
    """``名前 -> ax を受け取って描く呼び出し`` の一覧。

    すべて wscc9 の実データを渡す。ここに載っていない公開関数があると
    :func:`test_every_public_plot_function_is_exercised` が落ちる。
    """
    results, analytic = monte_carlo
    return {
        "timescale_map": lambda ax: plotting.timescale_map(ax=ax),
        "plot_voltage_profile": lambda ax: plotting.plot_voltage_profile(solution, ax=ax),
        "plot_convergence": lambda ax: plotting.plot_convergence(solutions, ax=ax),
        "plot_pv_curve": lambda ax: plotting.plot_pv_curve(curve, 5, ax=ax),
        "plot_network_flows": lambda ax: plotting.plot_network_flows(solution, ax=ax),
        "plot_network_flows (dc)": lambda ax: plotting.plot_network_flows(
            dc_solution, ax=ax, limit="rate_b"
        ),
        "plot_merit_order": lambda ax: plotting.plot_merit_order(
            case, ax=ax, demand_mw=315.0
        ),
        "plot_lambda_search": lambda ax: plotting.plot_lambda_search(case, 315.0, ax=ax),
        # 需要が可動範囲の外（500 > 460 MW）。図は描けて注記が付く。
        "plot_lambda_search (infeasible demand)": lambda ax: plotting.plot_lambda_search(
            case, 500.0, ax=ax
        ),
        "plot_lmp": lambda ax: plotting.plot_lmp(opf, ax=ax),
        "plot_commitment": lambda ax: plotting.plot_commitment(commitment, ax=ax),
        "plot_commitment_schedule": lambda ax: plotting.plot_commitment_schedule(
            commitment, ax=ax
        ),
        "plot_duck_curve": lambda ax: plotting.plot_duck_curve(case, ax=ax),
        "plot_capacity_outage_table": lambda ax: plotting.plot_capacity_outage_table(
            copt, ax=ax
        ),
        "plot_lolp_convergence": lambda ax: plotting.plot_lolp_convergence(
            results, ax=ax, reference=analytic
        ),
        "plot_contingency_ranking": lambda ax: plotting.plot_contingency_ranking(
            report, ax=ax
        ),
    }


def _draw(drawings, name: str):
    """名前で引いた図を新しい Axes に描いて ``(fig, ax)`` を返す。"""
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    drawings[name](ax)
    return fig, ax


def _xdata(line) -> np.ndarray:
    """Line2D の x 座標。``axvline`` などは list を返すので配列に直す。"""
    return np.asarray(line.get_xdata(), dtype=float)


def _ydata(line) -> np.ndarray:
    """Line2D の y 座標（同上）。"""
    return np.asarray(line.get_ydata(), dtype=float)


# ======================================================================
# 1. 文字列リテラルに非 ASCII 文字が無いこと（豆腐の防止）
# ======================================================================
def _string_literals(path: Path) -> list[tuple[int, str]]:
    """docstring を除いた文字列リテラルを ``(行番号, 中身)`` で返す。

    コメントは AST に残らないので、``ast`` で走査すれば自動的に除外される。
    docstring は「モジュール・クラス・関数の本体の先頭にある文字列式」なので、
    その節点だけを覚えておいて弾く。f-string の中の定数部分は
    :class:`ast.Constant` として木に現れるので、これも検査の対象になる。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            docstrings.add(id(first.value))

    literals: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            literals.append((node.lineno, node.value))
    return literals


def test_no_non_ascii_string_literals_in_plotting():
    """plotting.py の文字列リテラルがすべて ASCII であること。

    軸ラベル・凡例・タイトル・注記・例外メッセージのどれか 1 つでも
    日本語が混ざると、日本語フォントの無い環境で豆腐（□）になる。
    目視の規律では必ず漏れるので機械的に固定する。docstring と
    コメントは対象外（そちらは日本語で書く規約である）。
    """
    path = Path(plotting.__file__)
    offenders = [
        (line, text) for line, text in _string_literals(path) if not text.isascii()
    ]
    assert offenders == [], (
        "plotting.py に非 ASCII の文字列リテラルがある（豆腐の原因）: "
        + "; ".join(f"{line}行目: {text!r}" for line, text in offenders[:5])
    )


def test_the_non_ascii_scan_is_not_vacuous():
    """上の検査が「何も見ていない」状態で通っていないことの確認。

    走査が壊れて空リストを返しても :func:`test_no_non_ascii_string_literals_in_plotting`
    は通ってしまう。文字列リテラルが十分な数見つかっていること、そして
    **docstring は日本語である**（＝除外がちゃんと効いている）ことを
    別々に確かめる。
    """
    path = Path(plotting.__file__)
    literals = _string_literals(path)
    assert len(literals) > 150, f"文字列リテラルが {len(literals)} 個しか見つからない"
    assert any("[MW]" in text for _, text in literals), "軸ラベルを拾えていない"
    assert not plotting.__doc__.isascii(), "モジュール docstring は日本語で書く規約"
    assert not plotting.timescale_map.__doc__.isascii(), "関数 docstring も日本語"


# ======================================================================
# 2. すべての関数が Agg で動き、ax の扱いが規約どおりであること
# ======================================================================
def test_every_public_plot_function_is_exercised(drawings):
    """公開している作図関数がすべてテストの一覧に載っていること。"""
    public = {
        name for name in plotting.__all__
        if name.startswith("plot_") or name == "timescale_map"
    }
    for name in sorted(public):
        assert any(
            key == name or key.startswith(f"{name} ") for key in drawings
        ), f"{name} を描くテストが無い"
    assert set(drawings) == set(PLOT_NAMES)


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_plot_runs_under_agg_and_labels_both_axes(drawings, name):
    """Agg で例外なく描け、すべての Axes に xlabel と ylabel があること。

    単位の書かれていない軸は教材では誤りである。カラーバーや twin 軸を
    作ると「ラベルの無い Axes」が figure に混ざるので、Figure に載って
    いる Axes すべてを検査する。
    """
    fig, ax = _draw(drawings, name)
    for index, axes in enumerate(fig.axes):
        assert axes.get_xlabel(), f"{name}: {index} 番目の Axes に xlabel が無い"
        assert axes.get_ylabel(), f"{name}: {index} 番目の Axes に ylabel が無い"
    assert len(fig.axes) == 1, f"{name}: Axes が {len(fig.axes)} 枚ある"


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_plot_draws_on_the_given_axes(drawings, name):
    """``ax`` を渡すとその Axes に描き、同じ Axes を返すこと。"""
    fig, ax = plt.subplots()
    other_fig, other_ax = plt.subplots()
    before = len(ax.get_children())

    returned = drawings[name](ax)

    assert returned is ax, f"{name}: 渡した ax が返らない"
    assert len(ax.get_children()) > before, f"{name}: ax に何も描かれていない"
    assert other_ax.get_xlabel() == "", f"{name}: 無関係の Axes に描いている"
    assert ax.figure is fig


@pytest.mark.parametrize("name", PLOT_NAMES)
def test_plot_creates_a_new_axes_when_ax_is_none(drawings, name):
    """``ax=None`` なら Figure ごと新しく作ること。"""
    plt.close("all")
    call = drawings[name]
    returned = call(None)
    assert isinstance(returned, Axes), f"{name}: Axes を返していない"
    assert len(plt.get_fignums()) == 1, f"{name}: Figure が 1 枚作られていない"
    assert returned.figure.number == plt.get_fignums()[0]


def test_use_gridops_style_updates_rcparams():
    """``use_gridops_style`` が rcParams を書き換えること。"""
    saved = {key: plt.rcParams[key] for key in plotting.GRIDOPS_RCPARAMS}
    try:
        plt.rcParams["axes.grid"] = False
        plotting.use_gridops_style()
        assert plt.rcParams["axes.grid"] is True
        assert plt.rcParams["figure.dpi"] == plotting.GRIDOPS_RCPARAMS["figure.dpi"]
        # フォント名は指定しない（環境に無いフォントを指定すると警告が出る）。
        assert not any("font.family" in key for key in plotting.GRIDOPS_RCPARAMS)
    finally:
        plt.rcParams.update(saved)


# ======================================================================
# 3. 図の中身が、渡した数値と一致すること
# ======================================================================
def test_timescale_map_spans_milliseconds_to_years(drawings):
    """時間スケールの地図が ms から年までを対数軸で覆うこと。

    5 テーマの帯が **隙間なく** 並ぶことも確かめる。時間スケールは連続で
    あって、テーマの境目に壁があるわけではない、というのがこの図の主張
    だからである。
    """
    fig, ax = _draw(drawings, "timescale_map")
    assert ax.get_xscale() == "log"
    left, right = ax.get_xlim()
    assert left <= 1.0e-3, "ミリ秒が入っていない"
    assert right >= 3.1536e7, "1 年が入っていない"

    bars = ax.containers
    assert len(bars) == len(plotting.COURSE_THEMES) == 5
    spans = sorted(
        (float(start), float(end)) for _, _, _, start, end in plotting.COURSE_THEMES
    )
    for (_, previous_end), (next_start, _) in zip(spans, spans[1:]):
        assert next_start <= previous_end, "テーマの帯に隙間がある"

    texts = " ".join(text.get_text() for text in ax.texts)
    assert "balanced" in texts, "科目全体の問いが図に書かれていない"


def test_voltage_profile_plots_the_solved_voltages(drawings, solution):
    """描かれた点の高さが解の |V| と一致すること。"""
    fig, ax = _draw(drawings, "plot_voltage_profile")
    drawn = sorted(
        float(_ydata(line)[0]) for line in ax.lines if _ydata(line).size == 1
    )
    assert drawn == pytest.approx(sorted(solution.v), abs=1e-12)
    # 教科書解（母線 5 が 0.9956）と突き合わせる。掲載桁は小数 4 桁。
    assert min(drawn) == pytest.approx(0.9956, abs=1e-4)


def test_convergence_shows_one_point_per_iteration_plus_the_initial_value(
    drawings, solutions
):
    """各手法の折れ線の長さが ``iterations + 1`` であること。

    ``mismatch_history[0]`` は初期値でのミスマッチなので 1 点多い。
    縦軸が対数であること（二次収束と一次収束を見分けるための条件）も
    ここで固定する。
    """
    fig, ax = _draw(drawings, "plot_convergence")
    assert ax.get_yscale() == "log"
    lengths = {int(_xdata(line).size) for line in ax.lines}
    expected = {solution.iterations + 1 for solution in solutions.values()}
    assert lengths == expected
    # Newton は 4 反復、Gauss-Seidel は 61 反復（powerflow の検証済みの値）。
    assert max(lengths) == 62 and min(lengths) == 5


def test_network_flows_bars_match_the_contract_table(drawings):
    """棒の長さが API 契約の基準負荷率の表と一致すること。"""
    fig, ax = _draw(drawings, "plot_network_flows")
    widths = [float(bar.get_width()) for bar in ax.containers[0]]
    assert widths == pytest.approx(BASE_LOADING, abs=1.5e-3)
    # 定格の位置に線が引いてあること（負荷率 1.0）。
    assert any(
        float(_xdata(line)[0]) == pytest.approx(1.0)
        for line in ax.lines
        if _xdata(line).size == 2
    )


def test_network_flows_of_the_dc_solution_is_lower_than_the_ac_one(
    drawings, solution, dc_solution
):
    """直流の枝 4-5 の負荷率が交流より小さいこと（無効電力が無いため）。

    「熱容量の判定を直流で行ってはいけない」という規約を図の側でも固定
    する。交流 0.561 に対し直流 0.380 で、47% の過小評価である。
    """
    fig, ax = _draw(drawings, "plot_network_flows (dc)")
    dc_widths = np.array([float(bar.get_width()) for bar in ax.containers[0]])
    index = [branch.key() for branch in dc_solution.case.branches].index((4, 5))
    ac_loading = solution.loading("rate_b")[(4, 5)]
    assert dc_widths[index] < ac_loading
    assert float(ac_loading / dc_widths[index]) == pytest.approx(1.477, abs=0.01)


def test_merit_order_covers_the_installed_capacity(drawings, case):
    """棒の幅の合計が設備容量に、高さが全負荷平均費用に一致すること。"""
    fig, ax = _draw(drawings, "plot_merit_order")
    bars = [bar for container in ax.containers for bar in container]
    assert len(bars) == len(case.units)
    heights = [float(bar.get_height()) for bar in bars]
    assert heights == sorted(heights), "安い順に並んでいない"
    assert heights == pytest.approx(
        sorted(unit.full_load_average_cost() for unit in case.units), rel=1e-12
    )
    total = sum(float(bar.get_width()) for bar in bars)
    # 棒の間に隙間を空けてあるぶんだけ幅の合計は 460 MW より少し小さい。
    assert total == pytest.approx(INSTALLED_MW * 0.98, rel=1e-9)


def test_lambda_search_curve_is_monotone_and_hits_the_known_lambda(drawings):
    """ΣP(λ) が単調非減少で、λ = 13,090 円/MWh に印が付くこと。

    単調性は二分法が必ず効く根拠そのものである。曲線の下端と上端は
    手計算した ΣPmin = 174 MW と ΣPmax = 460 MW に一致する。
    """
    fig, ax = _draw(drawings, "plot_lambda_search")
    total = _ydata(ax.lines[0])
    assert np.all(np.diff(total) >= -1e-9), "ΣP(λ) が単調非減少でない"
    assert float(total.min()) == pytest.approx(MINIMUM_MW, rel=1e-12)
    assert float(total.max()) == pytest.approx(INSTALLED_MW, rel=1e-12)
    marks = [
        float(_xdata(line)[0])
        for line in ax.lines
        if _xdata(line).size == 2
        and float(_xdata(line)[0]) == float(_xdata(line)[1])
    ]
    assert any(mark == pytest.approx(LAMBDA_AT_315, abs=1.0) for mark in marks)


def test_lambda_search_annotates_a_demand_outside_the_feasible_range(drawings):
    """需要が可動範囲の外でも図は描け、その旨が注記されること。

    例外で止めずに描くのは、「需要の線が階段の届かない高さにある」ことを
    目で見せるほうが、例外の文面より分かりやすいからである。
    """
    fig, ax = _draw(drawings, "plot_lambda_search (infeasible demand)")
    texts = " ".join(text.get_text() for text in ax.texts)
    assert "outside" in texts
    demand_lines = [
        float(_ydata(line)[0])
        for line in ax.lines
        if _ydata(line).size == 2 and _ydata(line)[0] == _ydata(line)[1]
    ]
    assert any(value == pytest.approx(500.0) for value in demand_lines)
    assert 500.0 > INSTALLED_MW   # 描いた需要線は階段の上端より上にある


def test_lmp_bars_match_the_result(drawings, opf):
    """棒の高さが母線ごとの LMP と一致し、混雑した枝が注記されること。"""
    fig, ax = _draw(drawings, "plot_lmp")
    heights = [float(bar.get_height()) for bar in ax.containers[0]]
    assert heights == pytest.approx([opf.lmp[bus] for bus in sorted(opf.lmp)], rel=1e-12)
    assert max(heights) > min(heights), "混雑しているのに価格が割れていない"
    texts = " ".join(text.get_text() for text in ax.texts)
    assert "4-6" in texts, "拘束した枝 4-6 が図に出ていない"


def test_commitment_stack_reproduces_the_dispatch(drawings, commitment):
    """積み上げの高さが号機ごとの出力と一致し、需要曲線が純需要であること。"""
    fig, ax = _draw(drawings, "plot_commitment")
    horizon = commitment.demand_mw.size
    stack = np.zeros(horizon)
    for container in ax.containers[: len(commitment.dispatch)]:
        stack += np.array([float(bar.get_height()) for bar in container])
    expected = sum(np.asarray(v, dtype=float) for v in commitment.dispatch.values())
    assert stack == pytest.approx(expected, rel=1e-12)

    demand = np.asarray(commitment.demand_mw, dtype=float)
    assert any(
        _ydata(line).size == horizon and np.allclose(_ydata(line), demand)
        for line in ax.lines
    ), "純需要の曲線が描かれていない"
    # 同期並列容量の破線は、どの時刻でも需要以上（その差が運転予備力）。
    committed = np.array([commitment.committed_mw(t) for t in range(horizon)])
    assert any(
        _ydata(line).size == horizon and np.allclose(_ydata(line), committed)
        for line in ax.lines
    ), "同期並列容量の曲線が描かれていない"
    assert np.all(committed >= demand - 1e-6)


def test_commitment_schedule_image_is_the_zero_one_matrix(drawings, commitment):
    """ヒートマップの中身が入切表そのものであること。"""
    fig, ax = _draw(drawings, "plot_commitment_schedule")
    image = np.asarray(ax.images[0].get_array())
    names = [unit.name for unit in commitment.case.units if unit.name in commitment.schedule]
    expected = np.array([commitment.schedule[name] for name in names], dtype=float)
    assert image.shape == expected.shape
    assert image == pytest.approx(expected)
    assert set(np.unique(image)) <= {0.0, 1.0}
    # 起動の印の数が n_startups() と一致すること。
    markers = [line for line in ax.lines if line.get_marker() == "^"]
    if commitment.n_startups():
        assert sum(_xdata(line).size for line in markers) == commitment.n_startups()


def test_duck_curve_keeps_the_net_demand_below_the_gross_demand(drawings, case):
    """純需要が需要以下で、差が VRE 出力であること。"""
    fig, ax = _draw(drawings, "plot_duck_curve")
    from gridops.commitment import demand_profile as profile
    from gridops.commitment import net_demand

    gross = np.asarray(profile(case, "summer_weekday"), dtype=float)
    net = np.asarray(net_demand(case, gross), dtype=float)
    drawn = [_ydata(line) for line in ax.lines if _ydata(line).size == gross.size]
    assert any(np.allclose(values, gross) for values in drawn), "需要が描かれていない"
    assert any(np.allclose(values, net) for values in drawn), "純需要が描かれていない"
    assert np.all(net <= gross + 1e-9)
    assert float((gross - net).max()) > 0.0, "VRE が効いていない"


def test_capacity_outage_table_uses_a_log_axis(drawings, copt):
    """縦軸が対数で、階段が累積確率と一致すること。

    線形軸では 1e-20 の桁が潰れて「先頭以外はゼロ」に見える。稀な多重停止
    こそが供給支障を決める、という第 18 回の要点が消えてしまう。
    """
    fig, ax = _draw(drawings, "plot_capacity_outage_table")
    assert ax.get_yscale() == "log"
    steps = [line for line in ax.lines if _ydata(line).size == copt.outage_mw.size]
    assert steps, "階段が描かれていない"
    cumulative = _ydata(steps[0])
    assert float(cumulative[0]) == pytest.approx(1.0, rel=1e-12)
    assert np.all(np.diff(cumulative) <= 1e-12), "累積確率が単調減少でない"
    texts = " ".join(text.get_text() for text in ax.get_legend().get_texts())
    assert "23.2" in texts, "期待停止容量 23.2 MW が凡例に出ていない"


def test_lolp_convergence_draws_the_intervals_and_the_reference(drawings, monte_carlo):
    """誤差棒が信頼区間と一致し、解析解の水平線が引かれること。"""
    results, analytic = monte_carlo
    fig, ax = _draw(drawings, "plot_lolp_convergence")
    assert ax.get_xscale() == "log"
    points = [line for line in ax.lines if _xdata(line).size == len(results)]
    assert points, "点推定が描かれていない"
    assert _ydata(points[0]) == pytest.approx([r.lolp for r in results], rel=1e-12)
    flat = [
        float(_ydata(line)[0])
        for line in ax.lines
        if _ydata(line).size == 2 and _ydata(line)[0] == _ydata(line)[1]
    ]
    assert any(value == pytest.approx(analytic, rel=1e-12) for value in flat)


def test_contingency_ranking_shows_that_pi_misranks(drawings, report):
    """PI の順位と実際の危険度がずれることが図に出ていること（masking）。

    棒は PI の降順に並ぶが、色は「実際に安全か」で塗る。wscc9 では
    枝 4-6 の開放が **熱容量では健全（75.7%）なのに母線 6 の電圧が
    0.9418 で下限 0.95 を割る**ため、PI が小さいのに赤い棒として現れる。
    PI の上位 3 件だけを見るスクリーニングはこの事故を取りこぼす。
    """
    fig, ax = _draw(drawings, "plot_contingency_ranking")
    bars = list(ax.containers[0])
    widths = [float(bar.get_width()) for bar in bars]
    assert widths == sorted(widths, reverse=True), "PI の降順に並んでいない"

    labels = [text.get_text() for text in ax.get_yticklabels()]
    assert labels[0] == "5-7", "PI が最大なのは枝 5-7 の開放"
    insecure = {
        f"{item.outage[0]}-{item.outage[1]}"
        for item in report.results
        if not item.is_secure
    }
    assert "4-6" in insecure, "4-6 は電圧で逸脱する（前提が崩れている）"
    assert "4-6" not in labels[:3], "masking の実例にならない並びになっている"

    red = matplotlib.colors.to_rgba(plotting.SEVERITY_COLORS["violation"])
    colors = {
        label: matplotlib.colors.to_rgba(bar.get_facecolor())
        for label, bar in zip(labels, bars)
    }
    assert colors["4-6"] == red, "逸脱する事故が赤く塗られていない"


# ======================================================================
# 4. 契約に書かれた最小限の属性だけで描けること（duck typing）
# ======================================================================
@dataclass
class _MinimalCurve:
    """:class:`gridops.voltage.PVCurve` の契約のうち、作図が使う分だけ。"""

    case: Case
    factors: np.ndarray
    voltages: np.ndarray
    converged: np.ndarray
    critical_index: int

    def nose(self, bus_id: int) -> tuple[float, float]:
        index = self.critical_index
        return (
            float(self.factors[index]),
            float(self.voltages[index, self.case.index_of(bus_id)]),
        )

    @property
    def loading_margin(self) -> float:
        return float(self.factors[self.critical_index]) - 1.0


@dataclass
class _MinimalContingency:
    """:class:`gridops.security.ContingencyResult` の契約のうち作図が使う分。"""

    outage: tuple[int, int]
    performance_index: float
    worst_branch: tuple[int, int]
    worst_loading: float
    is_secure: bool


def test_plot_pv_curve_needs_only_the_documented_attributes(case):
    """契約のフィールドだけを持つ代役でも P-V 曲線が描けること。

    voltage.py の実装が差し替わっても作図が壊れないことの確認である。
    値は手で置いた 3 点で、2 点目までが収束したことにしてある。
    """
    voltages = np.full((3, case.n_bus), np.nan)
    voltages[0, case.index_of(5)] = 0.99
    voltages[1, case.index_of(5)] = 0.90
    curve = _MinimalCurve(
        case=case,
        factors=np.array([1.0, 1.5, 2.0]),
        voltages=voltages,
        converged=np.array([True, True, False]),
        critical_index=1,
    )
    ax = plotting.plot_pv_curve(curve, 5)
    assert ax.get_xlabel() and ax.get_ylabel()
    drawn = [_ydata(line) for line in ax.lines if _ydata(line).size == 2]
    assert any(np.allclose(values, [0.99, 0.90]) for values in drawn)


def test_plot_contingency_ranking_needs_only_the_documented_attributes():
    """契約のフィールドだけを持つ代役でも N-1 の順位が描けること。"""
    results = [
        _MinimalContingency((4, 5), 1.35, (5, 7), 1.015, False),
        _MinimalContingency((8, 9), 1.00, (7, 8), 0.731, True),
    ]
    ax = plotting.plot_contingency_ranking(results)
    assert [text.get_text() for text in ax.get_yticklabels()] == ["4-5", "8-9"]
    assert ax.get_xlabel() and ax.get_ylabel()


def test_plot_pv_curve_refuses_to_guess_the_bus(case):
    """母線番号から列を引けないときは、黙って別の母線を描かないこと。"""
    curve = _MinimalCurve(
        case=None,
        factors=np.array([1.0, 1.1]),
        voltages=np.ones((2, 9)),
        converged=np.array([True, True]),
        critical_index=1,
    )
    with pytest.raises(ValueError, match="cannot map bus id"):
        plotting.plot_pv_curve(curve, 5)
