"""ソルバ層（Phase: 数理計画の土台）の検証。

このモジュールだけは :mod:`pulp` を直接 import する。検証の対象が
「PuLP と CBC の振る舞いをどう封じ込めたか」そのものだからである。
下流のテスト（dispatch / commitment）は :mod:`gridops.solvers` の
:class:`~gridops.solvers.Solution` だけを見ればよい。

突き合わせる独立な基準は次のとおり。

* 手計算できる 2 変数の線形計画（最適値・最適解・双対を紙で出せる）
* 目的関数の**数値微分**（双対の定義そのものと突き合わせる）
* 混合整数計画に対する**全列挙**
* :mod:`pulp` 自身が持つ名前の変換表（:func:`~gridops.solvers.safe_name`
  の答え合わせ）
"""

from __future__ import annotations

import itertools
import math

import pulp
import pytest

from gridops.solvers import (
    SOLVER_GAP,
    SOLVER_TIME_LIMIT,
    Solution,
    available_solver,
    binary,
    safe_name,
    solve,
    variable,
)

# ----------------------------------------------------------------------
# 手計算できる線形計画
#
#   min 10 x + 40 y
#   s.t. x + y == D          (D = 12)
#        0 <= x <= 10, 0 <= y <= 10
#
# 安い x を上限まで使い、残りを y が埋める。
#   x = 10, y = 2, 費用 = 100 + 80 = 180 円/h
# 需要が 1 増えれば高い方の y が 1 増えるので、限界費用は 40 円/MWh。
# D が [10, 20] の範囲にある限りこの関係は厳密に線形である。
# ----------------------------------------------------------------------
CHEAP_COST = 10.0     #: 安い機の限界費用 [円/MWh]
DEAR_COST = 40.0      #: 高い機の限界費用 [円/MWh]
CHEAP_MAX = 10.0      #: 安い機の上限 [MW]


def two_unit_lp(demand: float = 12.0, *, reversed_balance: bool = False):
    """2 機の経済負荷配分を線形計画で書く。

    Parameters
    ----------
    demand:
        需要 [MW]。
    reversed_balance:
        ``True`` なら ``demand - Σp == 0`` の向きで書く（規約に反する向き）。
    """
    problem = pulp.LpProblem("two_unit", pulp.LpMinimize)
    cheap = variable("G1-1", 0.0, CHEAP_MAX)
    dear = variable("G2-1", 0.0, 10.0)
    problem += CHEAP_COST * cheap + DEAR_COST * dear
    if reversed_balance:
        problem += demand - pulp.lpSum([cheap, dear]) == 0, "balance"
    else:
        problem += pulp.lpSum([cheap, dear]) == demand, "balance"
    return problem


# ======================================================================
# ソルバの発見
# ======================================================================
def test_available_solver_returns_a_usable_cbc():
    """例外なく何かが返り、それが実際に使えること。

    conda-forge の PuLP では ``COIN_CMD`` だけが、pip 版では
    ``PULP_CBC_CMD`` が使える。どちらの環境でも通ることを要求する。
    """
    solver = available_solver()
    assert solver.available()
    assert type(solver).__name__ in ("PULP_CBC_CMD", "COIN_CMD")


def test_available_solver_passes_the_options():
    """``time_limit`` と ``gap`` がソルバに渡ること。"""
    solver = available_solver(time_limit=5.0, gap=0.01, msg=True)
    assert solver.timeLimit == pytest.approx(5.0)
    assert solver.optionsDict["gapRel"] == pytest.approx(0.01)
    assert solver.msg is True

    default = available_solver()
    assert default.timeLimit == pytest.approx(SOLVER_TIME_LIMIT)
    assert default.optionsDict["gapRel"] == pytest.approx(SOLVER_GAP)
    assert default.msg is False


def test_time_limit_can_be_disabled():
    """非正または無限大の ``time_limit`` は「制限なし」として扱われること。"""
    assert available_solver(time_limit=0.0).timeLimit is None
    assert available_solver(time_limit=math.inf).timeLimit is None


# ======================================================================
# 線形計画: 最適値と双対
# ======================================================================
def test_lp_optimum_matches_hand_calculation():
    """最適値・最適解が手計算と一致すること。

    安い機を上限まで使う解 (10, 2) が唯一の最適解であり、費用は 180 円/h。
    線形計画の頂点解なので誤差は丸めだけである（許容差 1e-9）。
    """
    result = solve(two_unit_lp(), context="2 機の経済負荷配分")

    assert isinstance(result, Solution)
    assert result.status == "Optimal"
    assert result.is_optimal
    assert result.objective == pytest.approx(180.0, abs=1e-9)
    assert result.values["G1-1"] == pytest.approx(CHEAP_MAX, abs=1e-9)
    assert result.values["G2-1"] == pytest.approx(2.0, abs=1e-9)


def test_balance_dual_is_the_marginal_cost_with_the_documented_sign():
    """``lpSum(p) == 需要`` の向きなら双対が限界費用そのものになること。

    **符号込みで固定する。** 絶対値で比べてはならない。ここが反転すると
    LMP も混雑レントも符号ごと壊れるが、絶対値のテストでは気づけない。
    """
    result = solve(two_unit_lp())

    assert result.duals["balance"] == pytest.approx(DEAR_COST, abs=1e-9)
    assert result.duals["balance"] > 0.0


def test_balance_dual_equals_the_numerical_derivative_of_the_cost():
    """双対が ∂(総費用)/∂(需要) の数値微分と一致すること。

    双対の定義そのものと突き合わせる。目的関数は需要 [10, 20] MW の区間で
    厳密に線形（安い機が上限に張り付き、高い機が余裕を持つ）なので、
    前進差分の打切り誤差はゼロであり、残るのは丸めだけ（許容差 1e-9）。
    """
    step = 0.5
    low = solve(two_unit_lp(12.0 - step)).objective
    high = solve(two_unit_lp(12.0 + step)).objective
    derivative = (high - low) / (2.0 * step)

    assert derivative == pytest.approx(DEAR_COST, abs=1e-9)
    assert solve(two_unit_lp()).duals["balance"] == pytest.approx(derivative, abs=1e-9)


def test_reversed_balance_flips_the_dual_sign():
    """制約を逆向きに書くと双対の符号が反転すること。

    ``demand - Σp == 0`` は ``Σp == demand`` の両辺に -1 を掛けただけで、
    最適値も最適解も同じである。にもかかわらず双対だけが符号を変える。
    これが「呼び出し側がバランス制約の向きを守る」規約を必要とする理由で
    あり、規約が破れたことに気づくための番人でもある。
    """
    forward = solve(two_unit_lp())
    backward = solve(two_unit_lp(reversed_balance=True))

    assert backward.objective == pytest.approx(forward.objective, abs=1e-9)
    assert backward.values["G1-1"] == pytest.approx(forward.values["G1-1"], abs=1e-9)
    assert backward.duals["balance"] == pytest.approx(-DEAR_COST, abs=1e-9)
    assert backward.duals["balance"] < 0.0
    assert backward.duals["balance"] == pytest.approx(-forward.duals["balance"], abs=1e-9)


def test_inequality_dual_signs_are_the_measured_ones():
    """不等式制約の双対の符号を実測値で固定すること。

    最小化問題で

        min 10 a + 40 b,  a + b >= 12,  a <= K,  0 <= a, b <= 10

    K = 6 では a = 6, b = 6 で費用 300 円/h、K = 7 では a = 7, b = 5 で
    270 円/h。よって ∂(費用)/∂K = -30 であり、容量制約の双対は **負**に
    なる。需要制約の双対は +40 で正。**絶対値で比べず符号ごと assert する。**
    """
    def capacity_lp(cap: float):
        problem = pulp.LpProblem("capacity", pulp.LpMinimize)
        cheap = variable("a", 0.0, 10.0)
        dear = variable("b", 0.0, 10.0)
        problem += CHEAP_COST * cheap + DEAR_COST * dear
        problem += pulp.lpSum([cheap, dear]) >= 12.0, "demand"
        problem += cheap <= cap, "cap_a"
        return problem

    result = solve(capacity_lp(6.0))
    assert result.objective == pytest.approx(300.0, abs=1e-9)

    # >= 制約の双対は正（需要が増えると費用が増える）
    assert result.duals["demand"] > 0.0
    assert result.duals["demand"] == pytest.approx(DEAR_COST, abs=1e-9)

    # <= 制約の双対は負（容量が増えると費用が減る）
    assert result.duals["cap_a"] < 0.0
    assert result.duals["cap_a"] == pytest.approx(-30.0, abs=1e-9)

    # 数値微分でも同じ符号・同じ大きさになること
    derivative = (solve(capacity_lp(7.0)).objective - result.objective) / 1.0
    assert derivative == pytest.approx(result.duals["cap_a"], abs=1e-9)


def test_feasibility_problem_has_zero_objective():
    """目的関数を置かない問題でも例外にならず、最適値が 0 になること。"""
    problem = pulp.LpProblem("feasibility", pulp.LpMinimize)
    x = variable("x", 0.0, 10.0)
    problem += x >= 3.0, "lower"

    result = solve(problem)
    assert result.is_optimal
    assert result.objective == pytest.approx(0.0, abs=1e-12)
    assert result.values["x"] >= 3.0 - 1e-9


def test_seconds_is_measured_and_small():
    """``seconds`` が実時間として妥当な範囲にあること。"""
    result = solve(two_unit_lp())
    assert 0.0 < result.seconds < SOLVER_TIME_LIMIT


# ======================================================================
# 例外
# ======================================================================
def test_infeasible_problem_raises_a_japanese_value_error():
    """実行不可能な問題が日本語の ValueError になること。

    ``pulp.value(problem.objective)`` は実行不可能な問題に対しても数値を
    返す（CBC が途中で持っていた値が残る）。状態を見ずに読むと
    「解けているが間違っている」結果が下流に流れるので、ここで堰き止める。
    """
    problem = pulp.LpProblem("infeasible", pulp.LpMinimize)
    x = variable("x", 0.0, 1.0)
    problem += x
    problem += x >= 5.0, "impossible"

    with pytest.raises(ValueError, match="実行不可能"):
        solve(problem, context="単体テスト")


def test_infeasible_error_carries_the_context_and_the_checklist():
    """例外メッセージに文脈と切り分けの手掛かりが載ること。"""
    problem = pulp.LpProblem("infeasible", pulp.LpMinimize)
    x = variable("x", 0.0, 1.0)
    problem += x
    problem += x >= 5.0, "impossible"

    with pytest.raises(ValueError) as excinfo:
        solve(problem, context="起動停止計画 (T=24)")

    message = str(excinfo.value)
    assert "起動停止計画 (T=24)" in message
    assert "Case.check()" in message      # データを先に疑わせる
    assert "単位の取り違え" in message


def test_unbounded_problem_raises_a_japanese_value_error():
    """非有界な問題が日本語の ValueError になること。"""
    problem = pulp.LpProblem("unbounded", pulp.LpMinimize)
    x = pulp.LpVariable("x", None, None)   # 下限なし
    problem += x
    problem += x <= 100.0, "upper"

    with pytest.raises(ValueError, match="非有界"):
        solve(problem)


def test_solve_rejects_a_non_problem():
    """``LpProblem`` 以外を渡したら型エラーになること。"""
    with pytest.raises(TypeError, match="pulp.LpProblem"):
        solve("これは問題ではない")


# ======================================================================
# 混合整数計画
# ======================================================================
#: ナップサック問題（価値 [円]、重量 [MW]、容量 [MW]）。
KNAPSACK_VALUES = (3.0, 4.0, 5.0, 8.0, 10.0, 2.0, 7.0, 6.0)
KNAPSACK_WEIGHTS = (2.0, 3.0, 4.0, 5.0, 9.0, 1.0, 6.0, 5.0)
KNAPSACK_CAPACITY = 12.0


def brute_force_knapsack() -> float:
    """全列挙による最適値。ソルバとは無関係な独立の基準。"""
    best = 0.0
    for pattern in itertools.product((0, 1), repeat=len(KNAPSACK_VALUES)):
        weight = sum(p * w for p, w in zip(pattern, KNAPSACK_WEIGHTS))
        if weight <= KNAPSACK_CAPACITY:
            best = max(best, sum(p * v for p, v in zip(pattern, KNAPSACK_VALUES)))
    return best


def knapsack_problem():
    """0-1 ナップサックを PuLP で書く。号機名を模してハイフンを含める。"""
    problem = pulp.LpProblem("knapsack", pulp.LpMaximize)
    picks = [variable(f"z-{i}", cat="Binary") for i in range(len(KNAPSACK_VALUES))]
    problem += pulp.lpSum(v * z for v, z in zip(KNAPSACK_VALUES, picks))
    problem += (
        pulp.lpSum(w * z for w, z in zip(KNAPSACK_WEIGHTS, picks)) <= KNAPSACK_CAPACITY,
        "capacity",
    )
    return problem, picks


def test_milp_optimum_matches_brute_force():
    """混合整数計画の最適値が全列挙と一致すること。

    2^8 = 256 通りを数え上げた値と突き合わせる。ソルバの出力をソルバで
    確かめるのではなく、定義に戻って確かめている。
    """
    problem, picks = knapsack_problem()
    result = solve(problem, context="ナップサック")

    assert result.is_optimal
    assert result.objective == pytest.approx(brute_force_knapsack(), abs=1e-6)

    # 解が実行可能で、目的関数値と整合していること。
    chosen = [binary(result.values[f"z-{i}"]) for i in range(len(picks))]
    weight = sum(z * w for z, w in zip(chosen, KNAPSACK_WEIGHTS))
    value = sum(z * v for z, v in zip(chosen, KNAPSACK_VALUES))
    assert weight <= KNAPSACK_CAPACITY + 1e-9
    assert value == pytest.approx(result.objective, abs=1e-6)


def test_milp_returns_no_duals():
    """混合整数計画では ``duals`` が空になること。

    整数変数を含む問題の最適値は右辺について階段関数であり、双対（微分）が
    そもそも存在しない。CBC は最後の緩和問題の双対を返してくるが、その値は
    探索経路に依存し限界費用としての意味を持たないので捨てる。
    """
    problem, _ = knapsack_problem()
    result = solve(problem)

    assert result.duals == {}
    assert problem.isMIP()
    # 同じ問題を線形緩和にすれば双対は取れる（捨てているのは意味の問題であって
    # 取れないからではない、ということの確認）。
    relaxed = pulp.LpProblem("relaxed", pulp.LpMaximize)
    picks = [variable(f"z-{i}", 0.0, 1.0) for i in range(len(KNAPSACK_VALUES))]
    relaxed += pulp.lpSum(v * z for v, z in zip(KNAPSACK_VALUES, picks))
    relaxed += (
        pulp.lpSum(w * z for w, z in zip(KNAPSACK_WEIGHTS, picks)) <= KNAPSACK_CAPACITY,
        "capacity",
    )
    assert solve(relaxed).duals["capacity"] != 0.0


# ======================================================================
# 0-1 変数の丸め
# ======================================================================
def test_binary_rounds_solver_noise():
    """分枝限定が返す端数を 0 と 1 に丸めること。

    ``int(0.9999999998)`` は切り捨てで 0 になる。丸めであって切り捨てでは
    ないことを固定する。
    """
    assert binary(0.9999999998) == 1
    assert binary(1e-9) == 0
    assert binary(1.0) == 1
    assert binary(0.0) == 0
    assert binary(-3e-11) == 0        # 負側にはみ出す値も来る
    assert binary(1.0000000004) == 1
    assert isinstance(binary(0.9999999998), int)


def test_binary_rejects_a_fractional_value():
    """0 からも 1 からも離れた値は黙って丸めず例外にすること。

    0.5 を勝手に 0 か 1 にすると、起動していない号機が出力を持つ矛盾した
    計画が出来上がる。整数宣言の忘れをここで捕まえる。
    """
    with pytest.raises(ValueError, match="0-1 変数"):
        binary(0.5)
    with pytest.raises(ValueError, match="cat='Binary'"):
        binary(0.4)


def test_binary_rejects_none_and_non_finite():
    """未求解（None）や NaN を渡したら例外になること。"""
    with pytest.raises(ValueError, match="None"):
        binary(None)
    with pytest.raises(ValueError, match="有限でない"):
        binary(float("nan"))


def test_binary_tolerance_is_adjustable():
    """許容差を広げれば通ること（既定は CBC の整数許容誤差）。"""
    with pytest.raises(ValueError):
        binary(0.999)
    assert binary(0.999, tolerance=1e-2) == 1


# ======================================================================
# 名前の落とし穴
# ======================================================================
@pytest.mark.parametrize(
    "name",
    ["p_G1-1_3", "u[G1-1,0]", "plain_name", "with space", "a+b", "a>b", "a/b"],
)
def test_safe_name_matches_pulp_translation(name):
    """:func:`safe_name` が PuLP 自身の置換と一致すること。

    独立な基準は PuLP そのものである。変数と制約で禁止文字の集合が違う
    （変数は ``>`` と ``/`` も置換されるが、制約は置換されない）ので、
    両方を確かめる。
    """
    var = pulp.LpVariable(name)
    assert safe_name(name, kind="variable") == var.name

    constraint = pulp.LpConstraint(var, name=name)
    assert safe_name(name, kind="constraint") == constraint.name


def test_safe_name_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        safe_name("x", kind="objective")


def test_values_are_keyed_by_the_caller_name():
    """``variable()`` で作った変数は元の名前で引けること。

    PuLP は変数名のハイフンを ``_`` に置き換える。号機名 ``G1-1`` で
    組み立てた変数を ``G1-1`` のまま引けなければ、下流の dispatch /
    commitment が必ず踏む。
    """
    result = solve(two_unit_lp())

    assert set(result.values) == {"G1-1", "G2-1"}
    assert result.values["G1-1"] == pytest.approx(CHEAP_MAX, abs=1e-9)
    assert "G1-1" in result.values


def test_values_can_be_read_with_a_raw_pulp_variable_name():
    """素の ``pulp.LpVariable`` を使っても置換前の名前で引けること。

    ``variable()`` を使わずに書いた下流コードでも壊れないようにするための
    二重の手当て。置換後の名前でも同じ値が引ける。
    """
    problem = pulp.LpProblem("raw", pulp.LpMinimize)
    p = pulp.LpVariable("p_G1-1_3", 0.0, 10.0)
    problem += p
    problem += p >= 3.0, "balance-3"

    result = solve(problem)

    assert set(result.values) == {"p_G1_1_3"}       # PuLP が置換した名前で格納
    assert result.values["p_G1-1_3"] == pytest.approx(3.0, abs=1e-9)   # 元の名前でも引ける
    assert result.values["p_G1_1_3"] == pytest.approx(3.0, abs=1e-9)
    assert result.values.get("p_G1-1_3") == pytest.approx(3.0, abs=1e-9)


def test_duals_can_be_read_with_the_original_constraint_name():
    """制約名も置換前・置換後の両方で引けること。"""
    problem = pulp.LpProblem("raw", pulp.LpMinimize)
    p = variable("p", 0.0, 10.0)
    problem += CHEAP_COST * p
    problem += pulp.lpSum([p]) == 4.0, "balance-bus-5"

    result = solve(problem)

    assert result.duals["balance-bus-5"] == pytest.approx(CHEAP_COST, abs=1e-9)
    assert result.duals["balance_bus_5"] == pytest.approx(CHEAP_COST, abs=1e-9)
    assert "balance-bus-5" in result.duals


def test_missing_name_raises_a_helpful_key_error():
    """存在しない名前を引いたら、置換の話を含む日本語で落ちること。"""
    result = solve(two_unit_lp())

    with pytest.raises(KeyError, match="解に含まれていない"):
        result.values["G9-9"]
    assert result.values.get("G9-9", -1.0) == -1.0
    assert "G9-9" not in result.values


def test_lookup_dict_behaves_like_a_plain_dict():
    """別名で引けても、辞書としての長さや反復は素直であること。

    別名をキーとして二重に持ってしまうと、下流で「号機の台数」を
    ``len(values)`` で数えたときに壊れる。
    """
    result = solve(two_unit_lp())

    assert len(result.values) == 2
    assert sorted(result.values.items()) == [("G1-1", 10.0), ("G2-1", 2.0)]
    assert dict(result.values) == {"G1-1": 10.0, "G2-1": 2.0}


# ======================================================================
# ソルバが無いとき・打ち切られたとき
# ======================================================================
class _Unavailable:
    """常に「使えない」と答えるソルバ。CBC の無い環境を模す。"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def available(self):
        return False


def test_missing_cbc_raises_a_japanese_runtime_error(monkeypatch):
    """CBC が 1 つも無い環境で、直し方を案内する RuntimeError になること。

    学生が最初に踏むのはここである。「ソルバが無い」だけでは何をすれば
    よいか分からないので、conda 環境の有効化と導入コマンドを本文に出す。
    """
    monkeypatch.setattr(pulp, "PULP_CBC_CMD", _Unavailable)
    monkeypatch.setattr(pulp, "COIN_CMD", _Unavailable)

    with pytest.raises(RuntimeError, match="CBC ソルバが見つからない"):
        available_solver()

    with pytest.raises(RuntimeError) as excinfo:
        available_solver()
    message = str(excinfo.value)
    assert "conda activate pwsyseng" in message
    assert "conda install -c conda-forge coin-or-cbc" in message
    # 商用ソルバには触れない（ライセンスの有無で再現性が崩れるため）。
    assert "gurobipy" not in message


def test_solver_search_order_prefers_the_bundled_cbc(monkeypatch):
    """``PULP_CBC_CMD`` が使えるならそちらを先に採ること。

    pip 版の PuLP（CBC 同梱）と conda-forge 版（同梱なし）の両方で
    同じコードが動くようにするための探索順を固定する。
    """
    class _Bundled(_Unavailable):
        def available(self):
            return True

    monkeypatch.setattr(pulp, "PULP_CBC_CMD", _Bundled)
    assert isinstance(available_solver(), _Bundled)


def _stop_at(status, sol_status):
    """CBC が打ち切ったときの状態を模す ``LpProblem.solve`` の差し替え。"""

    def fake_solve(self, solver=None, **kwargs):
        self.status = status
        self.sol_status = sol_status
        for var in self.variables():
            var.varValue = 0.0
        return self.status

    return fake_solve


def test_time_limited_but_feasible_solution_is_returned_with_a_warning(monkeypatch):
    """時間切れでも実行可能解があれば、警告つきで返すこと。

    起動停止計画で ``time_limit`` を短く切ったときに相当する。粗くても
    使える答えを捨てる理由はないが、最適性は保証されないので
    ``is_optimal`` は ``False`` になり、警告が出る。
    """
    problem, _ = knapsack_problem()
    monkeypatch.setattr(
        pulp.LpProblem, "solve",
        _stop_at(pulp.LpStatusNotSolved, pulp.LpSolutionIntegerFeasible),
    )

    with pytest.warns(UserWarning, match="time_limit"):
        result = solve(problem, context="打ち切りの確認")

    assert result.status == "Not Solved"
    assert result.is_optimal is False


def test_stopped_without_any_solution_raises(monkeypatch):
    """実行可能解が 1 つも無いまま打ち切られたら例外になること。"""
    problem, _ = knapsack_problem()
    monkeypatch.setattr(
        pulp.LpProblem, "solve",
        _stop_at(pulp.LpStatusNotSolved, pulp.LpSolutionNoSolutionFound),
    )

    with pytest.raises(ValueError, match="求解できなかった"):
        solve(problem, context="打ち切りの確認")
