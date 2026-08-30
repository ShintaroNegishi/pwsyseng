"""経済負荷配分（第 05 回）と直流最適潮流（第 06 回）の検証。

方針は 3 つある。

1. **閉形式と突き合わせる。** λ 法の解は KKT 条件の場合分けで解析的に
   書ける。ここでは「どの号機が上限／下限／内点か」の割り当てを全通り
   列挙して連立式を解く独立実装を置き、二分法の答えと比べる。さらに
   :func:`scipy.optimize.minimize` の制約付き最小化とも比べ、**三者が
   一致する**ことを確かめる。実装の出力を実装で再計算するだけの
   自己参照テストにしないための足場である。
2. **双対は「値」ではなく「意味」で検証する。** LMP が母線ごとの注入
   等式の双対として何番目に出てきたか、ではなく、**その母線の負荷を
   1 MW 増やして解き直したときの総費用の増分**と一致するかを見る。
   これが LMP の定義そのものであり、符号の取り違えを唯一確実に捕まえる。
3. **恒等式は 2 通り計算して比べる。** 混雑レントを定数で直書きせず、
   価格差形式と影値形式の両方を計算して一致を見る。
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy.optimize import fsolve, minimize

from gridops import load_case
from gridops.case import Branch, Bus, BusType, Case, Unit
from gridops.dispatch import (
    DCOPFResult,
    DispatchResult,
    dc_opf,
    dispatch_with_losses,
    economic_dispatch,
    merit_order,
    penalty_factors,
)
from gridops.ybus import build_ybus


# ======================================================================
# 題材
# ======================================================================
#: 閉形式で追える 3 機の小系統。2 次係数をすべて正にしてあるので、
#: どの需要でも最適解が一意に決まる。
THREE_UNITS = (
    Unit(name="U1", bus=1, p_max_mw=100.0, p_min_mw=10.0, var_cost=8000.0, quadratic=10.0),
    Unit(name="U2", bus=1, p_max_mw=80.0, p_min_mw=20.0, var_cost=9000.0, quadratic=20.0),
    Unit(name="U3", bus=1, p_max_mw=150.0, p_min_mw=0.0, var_cost=9500.0, quadratic=5.0),
)

#: 3 機系の供給可能範囲 [MW]。
THREE_UNITS_PMIN = 30.0
THREE_UNITS_PMAX = 330.0


@pytest.fixture(scope="module")
def wscc9() -> Case:
    return load_case("wscc9")


@pytest.fixture(scope="module")
def toy() -> Case:
    """3 機を 1 母線に並べただけのケース（λ 法の検算専用）。"""
    buses = [Bus(id=1, type=BusType.SLACK, v_set=1.0), Bus(id=2, type=BusType.PQ, pd=1.0)]
    branches = [Branch(1, 2, x=0.1)]
    return Case(name="toy-3-unit", buses=buses, branches=branches, units=list(THREE_UNITS))


def congested_three_bus(*, rate_a: float = 0.50) -> Case:
    """混雑を起こす 3 母線ループ系統。

    リアクタンスの等しい 3 本の線路で 1-2-3 を結び、母線 3 に 100 MW の
    負荷を置く。安い号機 A は母線 1、高い号機 B は母線 2 にある。
    枝 1-3 の熱容量を ``rate_a`` [p.u.] で絞ると混雑が起きる。

    等リアクタンスのループでは、母線 3 を基準にした注入分配（PTDF）が
    1 に対して 2/3 と 1/3 に分かれる。この分数のおかげで潮流と価格を
    手計算で追えるので、教材の題材として使いやすい。
    """
    buses = [
        Bus(id=1, type=BusType.SLACK, v_set=1.0, name="cheap"),
        Bus(id=2, type=BusType.PV, v_set=1.0, name="expensive"),
        Bus(id=3, type=BusType.PQ, pd=1.00, name="load"),
    ]
    branches = [
        Branch(1, 2, x=0.1, rate_a=math.inf, rate_b=math.inf),
        Branch(1, 3, x=0.1, rate_a=rate_a, rate_b=rate_a),
        Branch(2, 3, x=0.1, rate_a=math.inf, rate_b=math.inf),
    ]
    units = [
        Unit(name="A", bus=1, p_max_mw=200.0, p_min_mw=0.0, var_cost=10000.0),
        Unit(name="B", bus=2, p_max_mw=200.0, p_min_mw=0.0, var_cost=20000.0),
    ]
    return Case(name="loop3", buses=buses, branches=branches, units=units)


def radial_three_bus() -> Case:
    """安い号機だけが遠い放射状の系統（ペナルティファクタの題材）。

    母線 1（slack）に高い号機、母線 3 に安い号機を置き、母線 2 の負荷まで
    の線路の抵抗を **母線 3 側だけ 3 倍**にしてある。増分損失は slack を
    基準に測るので、slack から遠い母線 3 の号機だけがペナルティを受ける。
    """
    buses = [
        Bus(id=1, type=BusType.SLACK, v_set=1.02, name="near"),
        Bus(id=2, type=BusType.PQ, pd=2.00, qd=0.20, name="load"),
        Bus(id=3, type=BusType.PV, v_set=1.02, name="far"),
    ]
    branches = [
        Branch(1, 2, r=0.010, x=0.050, b=0.02),
        Branch(3, 2, r=0.030, x=0.150, b=0.05),
    ]
    units = [
        Unit(name="NEAR", bus=1, p_max_mw=200.0, p_min_mw=0.0,
             var_cost=12000.0, quadratic=10.0),
        Unit(name="FAR", bus=3, p_max_mw=200.0, p_min_mw=0.0,
             var_cost=9000.0, quadratic=10.0),
    ]
    return Case(name="radial3", buses=buses, branches=branches, units=units)


# ======================================================================
# 独立実装（テストの基準。gridops.dispatch を一切呼ばない）
# ======================================================================
def kkt_closed_form(units, demand_mw):
    """KKT 条件を場合分けして解く閉形式（独立実装）。

    各号機が「下限に張り付く / 内点 / 上限に張り付く」のどれかを全通り
    列挙し、内点の号機について

    .. math:: \\lambda = \\frac{D - \\sum_{hi} P^{max} - \\sum_{lo} P^{min}
              + \\sum_{int} b_i/(2 c_i)}{\\sum_{int} 1/(2 c_i)}

    を解いてから、割り当てが KKT 条件と整合するかを検査する。二分法とは
    まったく別の道筋なので、両者の一致は意味のある検証になる。

    Returns
    -------
    tuple[float, dict[str, float]]
        ``(λ, 号機名 -> 出力)``。整合する割り当てが 1 つも無ければ
        :class:`AssertionError`。
    """
    labels = ("lo", "int", "hi")
    for assignment in itertools.product(labels, repeat=len(units)):
        interior = [u for u, a in zip(units, assignment) if a == "int"]
        if not interior:
            continue
        fixed = 0.0
        for unit, tag in zip(units, assignment):
            if tag == "hi":
                fixed += unit.p_max_mw
            elif tag == "lo":
                fixed += unit.p_min_mw
        weight = sum(1.0 / (2.0 * u.quadratic) for u in interior)
        offset = sum(u.var_cost / (2.0 * u.quadratic) for u in interior)
        lam = (demand_mw - fixed + offset) / weight

        dispatch, ok = {}, True
        for unit, tag in zip(units, assignment):
            if tag == "int":
                p = (lam - unit.var_cost) / (2.0 * unit.quadratic)
                if not (unit.p_min_mw < p < unit.p_max_mw):
                    ok = False
                    break
                dispatch[unit.name] = p
            elif tag == "hi":
                if unit.incremental_cost(unit.p_max_mw) > lam:
                    ok = False
                    break
                dispatch[unit.name] = unit.p_max_mw
            else:
                if unit.incremental_cost(unit.p_min_mw) < lam:
                    ok = False
                    break
                dispatch[unit.name] = unit.p_min_mw
        if ok:
            return lam, dispatch
    raise AssertionError(f"需要 {demand_mw} MW に整合する KKT の割り当てが見つからない")


def scipy_dispatch(units, demand_mw):
    """``scipy.optimize.minimize``（SLSQP）で解く（独立実装）。

    2 次計画をそのまま数値最適化に投げる。λ も KKT も使わないので、
    等 λ 法とは共有するものが目的関数と制約しかない。
    """
    quadratic = np.array([u.quadratic for u in units])
    linear = np.array([u.var_cost for u in units])
    lower = np.array([u.p_min_mw for u in units])
    upper = np.array([u.p_max_mw for u in units])
    scale = 1.0e6  # 目的関数の桁を落として SLSQP の ftol を効かせる

    def cost(p):
        return float(np.sum(quadratic * p**2 + linear * p) / scale)

    def gradient(p):
        return (2.0 * quadratic * p + linear) / scale

    start = lower + (upper - lower) * 0.5
    start = start * (demand_mw / start.sum())
    start = np.clip(start, lower, upper)
    result = minimize(
        cost,
        start,
        jac=gradient,
        bounds=list(zip(lower, upper)),
        constraints=[{
            "type": "eq",
            "fun": lambda p: p.sum() - demand_mw,
            "jac": lambda p: np.ones_like(p),
        }],
        method="SLSQP",
        options={"ftol": 1e-14, "maxiter": 500},
    )
    assert result.success, result.message
    return {u.name: float(v) for u, v in zip(units, result.x)}


def independent_losses(case, p_spec, q_spec, v0, theta0):
    """``scipy.optimize.fsolve`` で潮流を解き直し、総損失 [p.u.] を返す。

    :mod:`gridops.dispatch` の内部 Newton（差分ヤコビアン）とは別の
    アルゴリズム（Powell の hybrid 法）なので、ペナルティファクタの
    独立な基準になる。
    """
    Y = build_ybus(case)
    slack, _, pq = case.type_indices()
    non_slack = [i for i in range(case.n_bus) if i != int(slack[0])]
    n_theta = len(non_slack)

    def residual(x):
        v = np.asarray(v0, float).copy()
        theta = np.asarray(theta0, float).copy()
        theta[non_slack] = x[:n_theta]
        v[pq] = x[n_theta:]
        voltage = v * np.exp(1j * theta)
        s = voltage * np.conj(Y @ voltage)
        return np.concatenate(
            [s.real[non_slack] - p_spec[non_slack], s.imag[pq] - q_spec[pq]]
        )

    start = np.concatenate([np.asarray(theta0)[non_slack], np.asarray(v0)[pq]])
    x, _, flag, message = fsolve(residual, start, full_output=True, xtol=1e-13)
    assert flag == 1, message
    v = np.asarray(v0, float).copy()
    theta = np.asarray(theta0, float).copy()
    theta[non_slack] = x[:n_theta]
    v[pq] = x[n_theta:]
    voltage = v * np.exp(1j * theta)
    return float(np.sum((voltage * np.conj(Y @ voltage)).real))


# ======================================================================
# 1. 優先順位
# ======================================================================
def test_merit_order_is_ascending_in_full_load_average_cost(wscc9):
    """優先順位が全負荷平均費用の昇順であること。

    「安い順」の定義が増分費用ではなく :math:`C(P^{max})/P^{max}` である
    ことを、値そのもので固定する。
    """
    order = merit_order(wscc9)
    costs = [unit.full_load_average_cost() for unit in order]
    assert costs == sorted(costs)
    assert [unit.name for unit in order] == [
        "G1-1", "G1-2", "G1-3", "G2-1", "G2-2", "G3-1", "G3-2"
    ]
    # 無負荷費を含むので、増分費用の順とは一致しない点も確認しておく。
    expected = [10220.0, 10320.0, 10420.0, 1204800 / 90, 1249800 / 90, 21500.0, 22000.0]
    assert costs == pytest.approx(expected, rel=1e-12)


def test_merit_order_accepts_an_explicit_unit_list(wscc9):
    """units 引数で並べ替える対象を絞れること。"""
    subset = [u for u in wscc9.units if u.plant == "G3"]
    assert [u.name for u in merit_order(wscc9, units=subset)] == ["G3-1", "G3-2"]


# ======================================================================
# 2. 等 λ 法 — 閉形式・二分法・scipy の三者一致
# ======================================================================
@pytest.mark.parametrize(
    "demand_mw, expected_lambda",
    [
        (157.5, 9900.0),    # 3 機とも内点
        (237.5, 10500.0),   # U1 が上限に張り付く
        (70.0, 9000.0),     # U2 と U3 が下限に張り付く
    ],
)
def test_bisection_matches_closed_form_and_scipy(toy, demand_mw, expected_lambda):
    """二分法・KKT の閉形式・SLSQP の三者が一致すること。

    許容差は 2 つに分かれる。閉形式との比較は倍精度の丸めだけが誤差源
    なので相対 1e-9 で見る。SLSQP は反復法なので、その収束条件
    （ftol=1e-14、目的関数を 1e6 で割ってある）で決まる誤差を見込み、
    相対 1e-6 で見る。
    """
    result = economic_dispatch(toy, demand_mw)
    lam_closed, dispatch_closed = kkt_closed_form(THREE_UNITS, demand_mw)
    dispatch_scipy = scipy_dispatch(THREE_UNITS, demand_mw)

    assert result.converged
    assert lam_closed == pytest.approx(expected_lambda, rel=1e-12)
    assert result.lam == pytest.approx(lam_closed, rel=1e-9)
    assert result.total_mw() == pytest.approx(demand_mw, rel=1e-12)

    for unit in THREE_UNITS:
        assert result.dispatch[unit.name] == pytest.approx(
            dispatch_closed[unit.name], rel=1e-9, abs=1e-9
        ), f"{unit.name}: 二分法と閉形式が食い違う"
        assert result.dispatch[unit.name] == pytest.approx(
            dispatch_scipy[unit.name], rel=1e-6, abs=1e-6
        ), f"{unit.name}: 二分法と SLSQP が食い違う"


def test_total_cost_matches_the_sum_of_fuel_costs(toy):
    """総費用が各号機の燃料費（無負荷費込み）の合計であること。"""
    result = economic_dispatch(toy, 157.5)
    expected = sum(u.fuel_cost(result.dispatch[u.name]) for u in THREE_UNITS)
    assert result.total_cost == pytest.approx(expected, rel=1e-12)


def test_lambda_dispatch_is_monotone_in_demand(toy):
    """需要を増やすと λ が単調に増えること（二分法が成り立つ根拠の裏返し）。"""
    lambdas = [economic_dispatch(toy, d).lam for d in np.linspace(35.0, 325.0, 40)]
    assert all(a <= b + 1e-9 for a, b in zip(lambdas, lambdas[1:]))


# ======================================================================
# 3. KKT の不等式（等号ではない）
# ======================================================================
def test_kkt_inequalities_hold_strictly_at_the_bounds(toy):
    """上限では増分費用 < λ、下限では > λ が **不等式として** 成り立つこと。

    「等 λ 法だから全号機の増分費用が等しい」という思い込みを潰すための
    テストなので、等号で assert してはならない。ここでは差が丸めでは
    説明できない大きさ（1 円/MWh 以上）であることまで確かめる。
    """
    result = economic_dispatch(toy, 237.5)      # U1 が上限
    lam = result.lam
    assert result.marginal_units == ("U2", "U3")

    u1 = THREE_UNITS[0]
    assert result.dispatch["U1"] == pytest.approx(u1.p_max_mw, rel=1e-12)
    assert u1.incremental_cost(result.dispatch["U1"]) < lam - 1.0

    low = economic_dispatch(toy, 70.0)          # U2, U3 が下限
    assert low.marginal_units == ("U1",)
    for name in ("U2", "U3"):
        unit = next(u for u in THREE_UNITS if u.name == name)
        assert low.dispatch[name] == pytest.approx(unit.p_min_mw, abs=1e-12)
        assert unit.incremental_cost(low.dispatch[name]) > low.lam + 1.0

    # 内点の号機だけが λ に一致する。
    marginal = next(u for u in THREE_UNITS if u.name == "U1")
    assert marginal.incremental_cost(low.dispatch["U1"]) == pytest.approx(
        low.lam, rel=1e-9
    )


def test_kkt_inequalities_hold_on_the_course_case(wscc9):
    """WSCC 9 母線（315 MW）でも KKT の不等式が成り立つこと。

    λ = 13090 円/MWh、限界号機は G2-1 と G2-2 になる。この値は
    :math:`\\sum P(\\lambda) = 210 + (2\\lambda - 24500)/16 = 315` を
    手で解いた結果であり、実装とは独立に決まる。
    """
    result = economic_dispatch(wscc9, 315.0)
    assert result.lam == pytest.approx(13090.0, rel=1e-9)
    assert result.dispatch["G2-1"] == pytest.approx(68.125, rel=1e-9)
    assert result.dispatch["G2-2"] == pytest.approx(36.875, rel=1e-9)
    assert set(result.marginal_units) == {"G2-1", "G2-2"}

    for unit in wscc9.units:
        p = result.dispatch[unit.name]
        ic = unit.incremental_cost(p)
        if unit.name in result.marginal_units:
            assert ic == pytest.approx(result.lam, rel=1e-9)
        elif p == pytest.approx(unit.p_max_mw, rel=1e-12):
            assert ic < result.lam, f"{unit.name}: 上限で dC/dP <= λ が破れている"
        else:
            assert ic > result.lam, f"{unit.name}: 下限で dC/dP >= λ が破れている"


def test_marginal_units_are_exactly_the_unbound_ones(wscc9):
    """marginal_units に張り付いた号機が混ざっていないこと。"""
    result = economic_dispatch(wscc9, 315.0)
    for unit in wscc9.units:
        p = result.dispatch[unit.name]
        interior = unit.p_min_mw + 1e-9 < p < unit.p_max_mw - 1e-9
        assert (unit.name in result.marginal_units) is interior


# ======================================================================
# 4. 需要が供給可能範囲の外にあるとき
# ======================================================================
def test_demand_above_capacity_raises_japanese_error(toy):
    """需要が ΣPmax を超えると、不足量つきの日本語 ValueError になること。

    この検査が無いと二分法は λ_hi に張り付いたまま「収束したふり」を
    する（区間幅だけが縮むので converged が真になってしまう）。
    """
    with pytest.raises(ValueError, match="供給力が 70 MW 不足している"):
        economic_dispatch(toy, THREE_UNITS_PMAX + 70.0)


def test_demand_below_minimum_output_raises_japanese_error(toy):
    """需要が ΣPmin を下回ると、超過量つきの日本語 ValueError になること。"""
    with pytest.raises(ValueError, match="最低出力が 20 MW 超過している"):
        economic_dispatch(toy, THREE_UNITS_PMIN - 20.0)


def test_error_message_names_the_running_units(toy):
    """例外に運転中の号機名が並ぶこと（どれを起動／解列するかの手掛かり）。"""
    with pytest.raises(ValueError) as excinfo:
        economic_dispatch(toy, 1000.0)
    message = str(excinfo.value)
    assert "収束したふり" in message
    for name in ("U1", "U2", "U3"):
        assert name in message


def test_the_boundaries_themselves_are_feasible(toy):
    """ΣPmin と ΣPmax ちょうどは実行可能で、例外にならないこと。"""
    low = economic_dispatch(toy, THREE_UNITS_PMIN)
    high = economic_dispatch(toy, THREE_UNITS_PMAX)
    assert low.total_mw() == pytest.approx(THREE_UNITS_PMIN, abs=1e-9)
    assert high.total_mw() == pytest.approx(THREE_UNITS_PMAX, abs=1e-9)
    assert low.marginal_units == ()
    assert high.marginal_units == ()


def test_committed_selects_units_and_rejects_unknown_names(wscc9):
    """committed で号機を絞れること、未知の名前は日本語で止まること。"""
    result = economic_dispatch(wscc9, 150.0, committed=["G1-1", "G1-2", "G2-1"])
    assert set(result.dispatch) == {"G1-1", "G1-2", "G2-1"}
    assert result.total_mw() == pytest.approx(150.0, rel=1e-12)
    with pytest.raises(ValueError, match="committed に未知の号機名がある"):
        economic_dispatch(wscc9, 150.0, committed=["G9-9"])


# ======================================================================
# 5. 2 次係数がゼロ（階段状の P(λ)）
# ======================================================================
def test_linear_costs_stack_in_merit_order():
    """線形費用（quadratic=0）でも優先順位どおりに積み上がること。

    :math:`P(\\lambda)` が階段関数になり λ が一意に決まらない場合でも、
    出力は「安い順に上限まで、最後の 1 台が端数」という自明な答えに
    なるはずである。閉形式は手で書ける。
    """
    units = [
        Unit(name="L1", bus=1, p_max_mw=100.0, p_min_mw=0.0, var_cost=10000.0),
        Unit(name="L2", bus=1, p_max_mw=100.0, p_min_mw=0.0, var_cost=20000.0),
        Unit(name="L3", bus=1, p_max_mw=100.0, p_min_mw=0.0, var_cost=30000.0),
    ]
    case = Case(
        name="linear",
        buses=[Bus(id=1, type=BusType.SLACK)],
        branches=[],
        units=units,
    )
    result = economic_dispatch(case, 150.0)
    assert result.dispatch == pytest.approx({"L1": 100.0, "L2": 50.0, "L3": 0.0}, abs=1e-9)
    assert result.lam == pytest.approx(20000.0, rel=1e-9)
    assert result.marginal_units == ("L2",)
    assert result.total_cost == pytest.approx(100 * 10000 + 50 * 20000, rel=1e-12)

    # 段差のちょうど上（1 台がまるごと入りきる需要）でも壊れないこと。
    exact = economic_dispatch(case, 100.0)
    assert exact.dispatch == pytest.approx({"L1": 100.0, "L2": 0.0, "L3": 0.0}, abs=1e-9)
    assert exact.total_mw() == pytest.approx(100.0, abs=1e-9)


def test_linear_and_quadratic_units_can_be_mixed():
    """線形費用と 2 次費用が混ざっても需給が合うこと。"""
    units = [
        Unit(name="Q", bus=1, p_max_mw=100.0, p_min_mw=0.0, var_cost=8000.0, quadratic=20.0),
        Unit(name="L", bus=1, p_max_mw=100.0, p_min_mw=0.0, var_cost=10000.0),
    ]
    case = Case(name="mixed", buses=[Bus(id=1, type=BusType.SLACK)], branches=[], units=units)
    for demand in (10.0, 50.0, 120.0, 190.0):
        result = economic_dispatch(case, demand)
        assert result.converged
        assert result.total_mw() == pytest.approx(demand, abs=1e-8)


# ======================================================================
# 6. 直流最適潮流 — LMP を符号込みで
# ======================================================================
def test_lmp_signs_and_values_in_a_congested_loop():
    """混雑した 3 母線ループで LMP が符号込みで手計算と一致すること。

    等リアクタンスのループでは、母線 3 を基準にした注入 1 MW のうち
    枝 1-3 に 2/3、枝 2-3 経由に 1/3 が流れる。よって

    .. math:: f_{13} = \\tfrac{2}{3}p_A + \\tfrac{1}{3}p_B
              = \\tfrac{1}{3}p_A + \\tfrac{D}{3}

    となり、:math:`f_{13} \\le 50` は :math:`p_A \\le 150 - D` を意味する。
    :math:`D = 100` では :math:`p_A = 50, p_B = 50`、費用は
    :math:`10000 p_A + 20000 p_B = 30000 D - 30000 \\bar f`。したがって
    :math:`\\pi_3 = 30000` 円/MWh である。
    """
    case = congested_three_bus()
    result = dc_opf(case)

    assert result.is_congested()
    assert result.dispatch["A"] == pytest.approx(50.0, abs=1e-6)
    assert result.dispatch["B"] == pytest.approx(50.0, abs=1e-6)
    assert result.flows[1] == pytest.approx(0.50, abs=1e-9)      # 枝 1-3 が限界
    assert result.flows[0] == pytest.approx(0.00, abs=1e-9)      # 枝 1-2 は無潮流

    assert result.lmp[1] == pytest.approx(10000.0, rel=1e-9)
    assert result.lmp[2] == pytest.approx(20000.0, rel=1e-9)
    assert result.lmp[3] == pytest.approx(30000.0, rel=1e-9)

    # 混雑料金は正の量として保持する（'<=' の双対をそのまま入れると負）。
    assert result.congestion_price[(1, 3)] == pytest.approx(30000.0, rel=1e-9)
    assert result.congestion_price[(1, 2)] == 0.0
    assert result.congestion_price[(2, 3)] == 0.0


def test_congested_lmp_can_exceed_the_most_expensive_incremental_cost():
    """混雑時に LMP が最高の限界費用を超えうること（バグではない）。

    母線 3 の負荷を 1 MW 増やすには、混雑した枝 1-3 を守るために安い
    号機 A を 1 MW 下げ、高い号機 B を 2 MW 上げなければならない。
    費用増は :math:`-10000 + 2\\times 20000 = 30000` 円で、これは最も高い
    号機の限界費用 20000 円/MWh を上回る。**送電制約のある系統では
    価格が号機の費用の範囲を出る**ことを示す。
    """
    case = congested_three_bus()
    result = dc_opf(case)
    highest = max(unit.var_cost for unit in case.units)
    assert result.lmp[3] > highest
    assert result.lmp[3] == pytest.approx(2.0 * 20000.0 - 10000.0, rel=1e-9)

    # 再給電の中身（A を下げ B を上げる）まで確かめる。
    stressed = dc_opf(case, loads={1: 0.0, 2: 0.0, 3: 101.0})
    assert stressed.dispatch["A"] - result.dispatch["A"] == pytest.approx(-1.0, abs=1e-6)
    assert stressed.dispatch["B"] - result.dispatch["B"] == pytest.approx(+2.0, abs=1e-6)


def test_uncongested_lmp_is_uniform_and_equals_lambda():
    """混雑が無ければ全母線の LMP が等しく、λ に一致すること。

    熱容量を外した同じ系統で解き、線形費用の等 λ 法（安い号機だけが
    運転する）と比べる。
    """
    case = congested_three_bus(rate_a=math.inf)
    result = dc_opf(case)
    reference = economic_dispatch(case, 100.0)

    assert not result.is_congested()
    assert result.congestion_rent(method="price") == pytest.approx(0.0, abs=1e-6)
    assert result.congestion_rent(method="shadow") == 0.0
    values = list(result.lmp.values())
    assert max(values) - min(values) < 1e-9
    assert values[0] == pytest.approx(reference.lam, rel=1e-9)
    assert result.dispatch["A"] == pytest.approx(100.0, abs=1e-6)
    assert result.dispatch["B"] == pytest.approx(0.0, abs=1e-6)


def test_uncongested_lmp_on_the_course_case(wscc9):
    """WSCC 9 母線でも、熱容量を外せば LMP が 1 つの値に揃うこと。

    もとのケースは枝 4-6 が拘束するので LMP が母線ごとに分かれる。
    熱容量を無限大にした複製で解き直し、分かれる原因が費用ではなく
    **送電制約**であることを示す。
    """
    from dataclasses import replace

    relaxed = replace(
        wscc9,
        branches=[replace(b, rate_a=math.inf, rate_b=math.inf) for b in wscc9.branches],
        reference=None,
    )
    result = dc_opf(relaxed)
    values = list(result.lmp.values())
    assert max(values) - min(values) < 1e-6
    assert not result.is_congested()

    congested = dc_opf(wscc9)
    assert congested.is_congested()
    assert max(congested.lmp.values()) - min(congested.lmp.values()) > 1000.0


# ======================================================================
# 7. LMP の「意味」— これが最も重要
# ======================================================================
@pytest.mark.parametrize("bus_id", [1, 2, 3])
def test_lmp_equals_the_cost_of_one_extra_mw_in_the_loop(bus_id):
    """負荷を 1 MW 増やしたときの総費用の増分が LMP に一致すること。

    双対がどの制約から来たか（値）ではなく、それが本当に
    :math:`\\partial(\\text{総費用})/\\partial d_b` になっているか（意味）を
    確かめる。符号を取り違えていれば必ずここで落ちる。前進差分と
    後退差分が一致することも見て、1 MW の摂動が線形領域に収まって
    いる（基底が変わっていない）ことを確認する。
    """
    case = congested_three_bus()
    base_loads = {1: 0.0, 2: 0.0, 3: 100.0}
    base = dc_opf(case, loads=base_loads)

    up = dict(base_loads)
    up[bus_id] += 1.0
    down = dict(base_loads)
    down[bus_id] -= 1.0
    forward = dc_opf(case, loads=up).total_cost - base.total_cost
    backward = base.total_cost - dc_opf(case, loads=down).total_cost

    assert forward == pytest.approx(backward, rel=1e-9)
    assert forward == pytest.approx(base.lmp[bus_id], rel=1e-6)


def test_lmp_meaning_holds_on_the_course_case(wscc9):
    """WSCC 9 母線の全母線で LMP = 1 MW あたりの費用増になっていること。

    許容差 1e-6（相対）は CBC が解ファイルに書き出す桁数（有効 9 桁
    程度）で決まる。実測の最大相対誤差は 1.8e-7 である。
    """
    base = dc_opf(wscc9)
    loads = dict(base.loads_mw)
    for bus_id in wscc9.bus_ids:
        raised = dict(loads)
        raised[bus_id] += 1.0
        increment = dc_opf(wscc9, loads=raised).total_cost - base.total_cost
        assert increment == pytest.approx(base.lmp[bus_id], rel=1e-6), (
            f"母線 {bus_id}: LMP が 1 MW の費用増と一致しない"
        )


# ======================================================================
# 8. 混雑レント（2 通りの計算）
# ======================================================================
def test_congestion_rent_agrees_between_the_two_formulas():
    """価格差形式と影値形式の混雑レントが一致すること。

    **定数を直書きせず、2 通り計算して比べる。** 3 母線ループでは
    どちらも厳密に同じ数になる（線形計画の解が有理数で、CBC が丸めを
    起こさないため）ので、相対 1e-9 で見る。
    """
    case = congested_three_bus()
    result = dc_opf(case)
    by_price = result.congestion_rent(method="price")
    by_shadow = result.congestion_rent(method="shadow")

    assert by_price == pytest.approx(by_shadow, rel=1e-9)
    assert by_price > 0.0

    # 3 番目の道筋: 負荷の支払い - 発電の受取り。恒等式であることを示す。
    payments = sum(result.lmp[b] * result.loads_mw[b] for b in case.bus_ids)
    revenue = sum(
        result.lmp[u.bus] * result.dispatch[u.name] for u in case.units
    )
    assert payments - revenue == pytest.approx(by_price, rel=1e-9)


def test_congestion_rent_agrees_on_the_course_case(wscc9):
    """WSCC 9 母線でも 2 通りの混雑レントが一致すること。

    許容差を 1e-6（相対）にしてあるのは、CBC が返す位相の桁数が
    有効 9 桁ほどで、影値形式が使う「熱容量ちょうど」と価格差形式が
    使う「解から計算した潮流」が最終桁でずれるためである。実測の
    相対差は 8.5e-8（枝 4-6 の潮流が 0.800000054 p.u. と報告される）。
    """
    result = dc_opf(wscc9)
    assert result.is_congested()
    by_price = result.congestion_rent(method="price")
    by_shadow = result.congestion_rent(method="shadow")
    assert by_price == pytest.approx(by_shadow, rel=1e-6)

    payments = sum(result.lmp[b] * result.loads_mw[b] for b in wscc9.bus_ids)
    revenue = sum(result.lmp[u.bus] * result.dispatch[u.name] for u in wscc9.units)
    assert payments - revenue == pytest.approx(by_price, rel=1e-6)


def test_congestion_rent_rejects_an_unknown_method():
    case = congested_three_bus()
    result = dc_opf(case)
    with pytest.raises(ValueError, match="method は"):
        result.congestion_rent(method="magic")


# ======================================================================
# 9. 直流最適潮流の物理
# ======================================================================
def test_dc_opf_respects_the_nodal_balance_and_the_thermal_limits(wscc9):
    """母線ごとの注入等式と熱容量が実際に満たされていること。

    直流近似には損失が無いので、発電の合計は負荷の合計にちょうど等しい。
    位相から組み直した注入との突き合わせは 1e-6 p.u.（0.1 kW）で見る。
    CBC が解ファイルに書き出す位相の桁数（有効 9 桁程度）に、枝の
    サセプタンス 1/x ~ 17 と接続行列が掛かるので、この程度の残差は
    定式化の誤りではなく報告桁数の丸めである（実測 1.2e-7 p.u.）。
    """
    result = dc_opf(wscc9)
    assert sum(result.dispatch.values()) == pytest.approx(
        sum(result.loads_mw.values()), rel=1e-9
    )

    A = np.zeros((wscc9.n_branch, wscc9.n_bus))
    for k, branch in enumerate(wscc9.branches):
        A[k, wscc9.index_of(branch.from_bus)] = 1.0
        A[k, wscc9.index_of(branch.to_bus)] = -1.0
    injection = A.T @ result.flows      # p.u.
    for i, bus in enumerate(wscc9.buses):
        generated = sum(
            result.dispatch[u.name] for u in wscc9.units if u.bus == bus.id
        )
        expected = wscc9.to_pu(generated - result.loads_mw[bus.id])
        assert injection[i] == pytest.approx(expected, abs=1e-6)

    for k, branch in enumerate(wscc9.branches):
        assert abs(result.flows[k]) <= branch.rate_a + 1e-6


def test_dc_opf_scales_the_load_profile_with_demand_mw(wscc9):
    """demand_mw が母線の負荷分布を保ったまま合計を合わせること。"""
    result = dc_opf(wscc9, demand_mw=250.0)
    assert sum(result.loads_mw.values()) == pytest.approx(250.0, rel=1e-12)
    ratio = result.loads_mw[5] / result.loads_mw[6]
    assert ratio == pytest.approx(1.25 / 0.90, rel=1e-12)


def test_dc_opf_uses_only_the_linear_cost(wscc9):
    """total_cost が :math:`\\sum b_i P_i` そのもの（2 次項も無負荷費も無し）であること。

    PuLP と CBC は線形計画しか扱えないので 2 次項は落ちる。この住み分けを
    値で固定しておかないと、DispatchResult.total_cost と取り違えられる。
    """
    result = dc_opf(wscc9)
    linear = sum(u.var_cost * result.dispatch[u.name] for u in wscc9.units)
    assert result.total_cost == pytest.approx(linear, rel=1e-12)
    quadratic_total = sum(u.fuel_cost(result.dispatch[u.name]) for u in wscc9.units)
    assert quadratic_total > result.total_cost


def test_dc_opf_summary_runs(wscc9):
    """summary() が例外なく文字列を返すこと。"""
    text = dc_opf(wscc9).summary()
    assert "DC optimal power flow" in text
    assert "LMP" in text


def test_dc_opf_rejects_an_unknown_limit_name(wscc9):
    with pytest.raises(ValueError, match="limit="):
        dc_opf(wscc9, limit="rate_z")


def test_dc_opf_rejects_loads_on_unknown_buses(wscc9):
    with pytest.raises(ValueError, match="loads に存在しない母線番号がある"):
        dc_opf(wscc9, loads={99: 10.0})


# ======================================================================
# 10. 送電損失とペナルティファクタ
# ======================================================================
def test_penalty_factor_of_the_slack_bus_is_one(wscc9):
    """slack 母線のペナルティファクタが 1 であること。

    増分損失は slack を基準に測るので、基準点自身の増分損失は定義により
    ゼロである。ここが 1 でなければ基準の取り方が壊れている。
    """
    factors = penalty_factors(wscc9)
    slack_idx, _, _ = wscc9.type_indices()
    slack_bus = wscc9.buses[int(slack_idx[0])].id
    assert factors[slack_bus] == pytest.approx(1.0, abs=1e-12)
    assert set(factors) == set(wscc9.bus_ids)


def test_penalty_factors_match_an_independent_power_flow(wscc9):
    """ペナルティファクタが独立な潮流計算の数値微分と一致すること。

    :func:`scipy.optimize.fsolve`（Powell の hybrid 法）で潮流を解き直し、
    刻みを変えて（1e-3 ではなく 5e-4）中心差分を取る。ソルバも刻みも
    違うので、実装の出力を実装で再計算する自己参照にはならない。
    許容差 1e-6（相対）は、刻みの違いによる中心差分の打切り誤差
    :math:`O(\\delta^2)` の差と fsolve の収束条件 xtol=1e-13 から見込んだ
    値である（実測の最大相対差は 5.2e-10）。
    """
    factors = penalty_factors(wscc9)
    Y = build_ybus(wscc9)
    v = wscc9.reference.v
    theta = np.radians(wscc9.reference.angle_deg)
    voltage = v * np.exp(1j * theta)
    s0 = voltage * np.conj(Y @ voltage)
    p_spec, q_spec = s0.real.copy(), s0.imag.copy()
    slack_idx, _, _ = wscc9.type_indices()
    slack = int(slack_idx[0])

    step = 5e-4
    for i, bus in enumerate(wscc9.buses):
        if i == slack:
            continue
        plus, minus = p_spec.copy(), p_spec.copy()
        plus[i] += step
        minus[i] -= step
        loss_plus = independent_losses(wscc9, plus, q_spec, v, theta)
        loss_minus = independent_losses(wscc9, minus, q_spec, v, theta)
        itl = (loss_plus - loss_minus) / (2.0 * step)
        expected = 1.0 / (1.0 - itl)
        assert factors[bus.id] == pytest.approx(expected, rel=1e-6), (
            f"母線 {bus.id} のペナルティファクタが独立計算と食い違う"
        )


def test_penalty_factors_are_close_to_one_on_a_lightly_loaded_system(wscc9):
    """基準潮流の損失が小さいので、ペナルティファクタが 1 の近傍にあること。

    総損失は 4.64 MW（3.15 MW の負荷に対し 1.5%）なので、増分損失は
    数 % のオーダーに収まるはずである。桁が違えば符号か基準の取り違えを
    疑う。
    """
    factors = penalty_factors(wscc9)
    assert all(0.9 < value < 1.1 for value in factors.values()), factors


def test_losses_push_down_the_cheap_unit_far_from_the_load():
    """損失を入れると、負荷から遠い安い号機の出力が減ること。

    放射状の 3 母線系統で、安い号機だけが抵抗の大きい線路の先にある。
    損失を無視すれば安い号機が 175 MW まで出すが、増分損失を織り込むと
    ペナルティファクタが 1 を超え、実効的な増分費用
    :math:`L\\,dC/dP` が押し上げられて出力が下がる。**総発電量は
    損失の分だけ増えるのに、遠い安い号機の出力は減る**ところが要点で、
    「安い機を目一杯焚けばよい」という直観が崩れる場面である。
    """
    case = radial_three_bus()
    lossless = economic_dispatch(case, 200.0)
    with_losses = dispatch_with_losses(case, 200.0)

    assert with_losses.converged
    assert lossless.dispatch["FAR"] == pytest.approx(175.0, rel=1e-9)

    # 遠い安い号機は減り、近い高い号機は増える。
    assert with_losses.dispatch["FAR"] < lossless.dispatch["FAR"] - 10.0
    assert with_losses.dispatch["NEAR"] > lossless.dispatch["NEAR"] + 10.0

    # 総発電量は損失の分だけ増える。
    assert with_losses.total_mw() > lossless.total_mw()
    assert with_losses.total_mw() == pytest.approx(
        200.0 + with_losses.losses_mw, rel=1e-6
    )

    # ペナルティを受けるのは遠い号機だけ（slack は基準なので 1）。
    assert with_losses.penalty["NEAR"] == pytest.approx(1.0, abs=1e-12)
    assert with_losses.penalty["FAR"] > 1.02

    # 最適性条件 L_i dC_i/dP_i = λ が両機で成り立っていること。
    for unit in case.units:
        p = with_losses.dispatch[unit.name]
        effective = with_losses.penalty[unit.name] * unit.incremental_cost(p)
        assert effective == pytest.approx(with_losses.lam, rel=1e-6)


def test_losses_are_consistent_with_an_independent_power_flow():
    """報告された損失が独立な潮流計算の損失と一致すること。"""
    case = radial_three_bus()
    result = dispatch_with_losses(case, 200.0)
    p_spec, q_spec = case.bus_injection(result.dispatch)
    v0 = np.array([bus.v_set if bus.type is not BusType.PQ else 1.0 for bus in case.buses])
    losses_pu = independent_losses(case, p_spec, q_spec, v0, np.zeros(case.n_bus))
    assert case.to_mw(losses_pu) == pytest.approx(result.losses_mw, rel=1e-6)


def test_dispatch_with_losses_on_the_course_case(wscc9):
    """WSCC 9 母線でも損失込みの反復が収束すること。

    交流の総損失（基準潮流で 4.64 MW）と同じ桁になることを確かめる。
    需要 315 MW は基準潮流と同じ総負荷なので、損失も近い値になるはず
    だが、出力配分が違うので一致はしない。
    """
    result = dispatch_with_losses(wscc9, 315.0)
    assert result.converged
    assert result.iterations <= 20
    assert 0.5 < result.losses_mw < 10.0
    assert result.total_mw() == pytest.approx(315.0 + result.losses_mw, rel=1e-6)
    assert result.total_cost > economic_dispatch(wscc9, 315.0).total_cost


def test_penalty_factors_reject_a_mismatched_solution(wscc9):
    """別のケースの潮流解を渡すと日本語 ValueError になること。"""
    bad = np.ones(3)
    with pytest.raises(ValueError, match="母線数"):
        penalty_factors(wscc9, type("S", (), {"v": bad, "theta": np.zeros(3)})())


def test_penalty_factors_need_a_reference_solution():
    """参照解も引数も無ければ、何が足りないかを日本語で言うこと。"""
    case = congested_three_bus()
    with pytest.raises(ValueError, match="参照解がない"):
        penalty_factors(case)


# ======================================================================
# 11. 結果オブジェクトの体裁
# ======================================================================
def test_dispatch_result_summary_runs(wscc9):
    result = economic_dispatch(wscc9, 315.0)
    text = result.summary()
    assert "lambda" in text
    assert "marginal" in text
    assert isinstance(result, DispatchResult)


def test_dc_opf_returns_the_documented_types(wscc9):
    result = dc_opf(wscc9)
    assert isinstance(result, DCOPFResult)
    assert result.theta.shape == (wscc9.n_bus,)
    assert result.flows.shape == (wscc9.n_branch,)
    assert set(result.lmp) == set(wscc9.bus_ids)
    assert set(result.congestion_price) == {b.key() for b in wscc9.branches}
    slack_idx, _, _ = wscc9.type_indices()
    assert result.theta[int(slack_idx[0])] == pytest.approx(0.0, abs=1e-12)


def test_a_case_without_units_is_reported_in_japanese():
    """units 層を持たないケースを渡すと、どのケースを使えばよいか言うこと。"""
    empty = Case(name="network-only", buses=[Bus(id=1, type=BusType.SLACK)], branches=[])
    with pytest.raises(ValueError, match="号機がない"):
        economic_dispatch(empty, 10.0)
    with pytest.raises(ValueError, match="号機がない"):
        merit_order(empty)


# ======================================================================
# 外部レビュー（2026-08-30）の回帰
# ======================================================================
def test_dc_opf_rejects_non_rating_limit_attributes(wscc9):
    """``limit`` に熱容量でない Branch 属性を渡すと即座に止まること。

    外部レビューの指摘 #2。``hasattr`` だけの検査だと ``limit="x"`` が
    リアクタンスを送電容量と解釈し、実行可能なら「もっともらしい誤答」を
    静かに返す。名前を入口で検査する。
    """
    for bogus in ("x", "tap", "shift_deg", "r"):
        with pytest.raises(ValueError, match="許容容量ではない"):
            dc_opf(wscc9, limit=bogus)


def test_congestion_rent_matches_between_methods_with_parallel_lines():
    """並列 2 回線の片方だけが拘束しても、price 法と shadow 法が一致すること。

    外部レビューの指摘 #7。混雑価格を ``Branch.key()`` で合算すると、
    拘束していない側の回線の容量にまで価格が掛かり、shadow 法のレントが
    二重計上される（この構成では 6.4M 円/h と誤答していた。正しくは
    価格差法と同じ 2.4M 円/h）。
    """
    from gridops.case import Bus, BusType, Branch, Case, Unit

    parallel = Case(
        name="parallel-circuit",
        base_mva=100.0,
        buses=[Bus(id=1, type=BusType.SLACK), Bus(id=2, type=BusType.PQ, pd=0.8)],
        branches=[
            Branch(from_bus=1, to_bus=2, x=0.1, rate_a=0.30),
            Branch(from_bus=1, to_bus=2, x=0.1, rate_a=0.50),
        ],
        units=[
            Unit(name="cheap", bus=1, p_max_mw=200.0, var_cost=10_000.0),
            Unit(name="dear", bus=2, p_max_mw=200.0, var_cost=50_000.0),
        ],
    )
    result = dc_opf(parallel)
    price = result.congestion_rent(method="price")
    shadow = result.congestion_rent(method="shadow")
    assert result.is_congested()
    assert price == pytest.approx(2_400_000.0, rel=1e-9)
    assert shadow == pytest.approx(price, rel=1e-9)
