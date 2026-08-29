"""電圧安定性（第 11 回）の検証。

このモジュールの主張は「潮流が収束しなくなる点が電圧安定の限界である」
という **手続きの言い換え**になりやすい。手続きが自分の答えを定義して
しまわないよう、実装の外にある基準とだけ突き合わせる。使う基準は 5 つ。

1. **2 母線の閉形式解**（:func:`gridops.voltage.two_bus_nose`）。
   4 次方程式 :math:`|V|^4 + (2QX - E^2)|V|^2 + X^2(P^2+Q^2) = 0` の
   判別式をゼロにして得た :math:`P_{max}`, :math:`|V|_{crit}` は、
   継続法とはまったく別の道筋で出てくる。**これが最重要の独立基準**である。
2. **数値微分**。無効電力を少し変えて潮流を解き直した電圧の差分と
   :func:`gridops.voltage.voltage_sensitivity` を比べる。
3. **ケースファイルの教科書解**（Anderson & Fouad）。倍率 1.0 の点が
   これと一致することで、掃引が発電を落としていないことを確かめる。
4. **数値微分で組み直したヤコビアン**の最小特異値。解析式で組んだ
   ヤコビアンの :func:`min_singular_value` と突き合わせる。
5. **折り目（fold）の平方根則** :math:`\\sigma_{min} \\propto
   \\sqrt{\\lambda_{max} - \\lambda}`。分岐理論から出る性質で、
   実装の中には書かれていない。

「収束しなかったから限界だ」で終わらせないために、**解析解が
「解が無い」と言う倍率でだけ潮流が失敗すること**を両方向で固定してある。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gridops import Branch, Bus, BusType, Case, load_case
from gridops.powerflow import jacobian_blocks, mismatch, solve
from gridops.voltage import (
    PVCurve,
    min_singular_value,
    pv_curve,
    two_bus_nose,
    two_bus_voltages,
    voltage_sensitivity,
)
from gridops.ybus import build_ybus

#: WSCC 9 母線の教科書解との差の上限 [p.u.]。API 契約の実測値 4.705e-05
#: （掲載 4 桁の丸めで説明できる大きさ）に安全率を見た値。
REFERENCE_TOL = 1e-4

#: 2 母線の答え合わせで要求する相対差。担当指示の値そのもの。
#: 倍率については実測 1e-9 の水準、電圧については 1e-5 の水準で一致するので
#: 2 桁以上の余裕がある。電圧のほうが緩いのはノーズ点で
#: :math:`dV/d\\lambda \\to \\infty` になり、倍率の誤差 δ に対して
#: 電圧の誤差が :math:`\\sqrt{\\delta}` でしか縮まないためである。
NOSE_RTOL = 2e-3


# ----------------------------------------------------------------------
# 足場（ケースの組み立て）
# ----------------------------------------------------------------------
def two_bus_case(
    e: float = 1.0, x: float = 0.1, p0: float = 0.5, power_factor: float = 1.0
) -> Case:
    """無限大母線 + リアクタンス + 力率一定負荷の 2 母線ケース。

    ``casedata/`` にファイルを足さずに済ませるため、その場で組み立てる。
    抵抗と充電容量をゼロにしてあるので :func:`two_bus_nose` の前提
    （純リアクタンス）を厳密に満たす。母線 1 は slack なので発電の
    指定は要らず、負荷と損失をすべて引き受ける。
    """
    magnitude = abs(power_factor)
    sin_phi = math.copysign(math.sqrt(1.0 - magnitude**2), power_factor)
    q0 = p0 * sin_phi / magnitude
    return Case(
        name=f"two-bus (pf={power_factor})",
        buses=[
            Bus(id=1, type=BusType.SLACK, v_set=e, name="infinite bus"),
            Bus(id=2, type=BusType.PQ, pd=p0, qd=q0, v_min=0.0, v_max=2.0, name="load"),
        ],
        branches=[Branch(from_bus=1, to_bus=2, r=0.0, x=x, b=0.0)],
    )


def scaled_with_generation(case: Case, factor: float) -> Case:
    """負荷を ``factor`` 倍し、**発電を据え置いた**ケース。

    :meth:`Case.scaled` は参照解を落とすので、そのまま解くと発電ゼロの
    別系統になる。テスト側でも同じ手当てを **自前で**書いておくことで、
    実装の私有ヘルパに依存せずに済ませる。
    """
    return replace(case.scaled(factor), reference=case.reference)


@pytest.fixture(scope="module")
def case():
    return load_case("wscc9")


@pytest.fixture(scope="module")
def wscc9_curve(case):
    return pv_curve(case)


# ======================================================================
# 1. 2 母線 — 解析解と継続法の一致（最重要の独立基準）
# ======================================================================
@pytest.mark.parametrize(
    "e, x, p0, pf",
    [
        (1.0, 0.10, 0.50, 1.00),    # 力率 1。P_max = E^2/(2X) の教科書値
        (1.0, 0.10, 0.50, 0.95),    # 遅れ
        (1.0, 0.10, 0.50, -0.95),   # 進み
        (1.0, 0.10, 0.50, 0.90),
        (1.04, 0.08, 1.20, 0.98),   # E と X と基準負荷を全部動かす
    ],
)
def test_pv_curve_matches_the_two_bus_analytic_nose(e, x, p0, pf):
    """継続法のノーズ点が 4 次方程式の閉形式解と一致すること。

    実装の外にある基準はこの 1 本だけである。判別式をゼロにして得た
    :math:`P_{max} = E^2 (1-\\sin\\phi) / (2X\\cos\\phi)` と
    :math:`|V|_{crit} = E/\\sqrt{2(1+\\sin\\phi)}` に対し、負荷倍率を
    上げながら潮流を解いて挟み撃ちにした結果を相対 2e-3 で比べる。
    """
    expected_factor, expected_voltage = two_bus_nose(e, x, p0, power_factor=pf)
    curve = pv_curve(two_bus_case(e, x, p0, pf), step=0.05, max_factor=40.0)
    factor, voltage = curve.nose(2)

    assert factor == pytest.approx(expected_factor, rel=NOSE_RTOL)
    assert voltage == pytest.approx(expected_voltage, rel=NOSE_RTOL)
    # 数値解は必ず「解が存在すると確かめられた」側、すなわち解析解の手前にある。
    assert factor <= expected_factor * (1.0 + 1e-9)


def test_two_bus_curve_matches_the_analytic_upper_branch(case):
    """ノーズ点だけでなく **曲線そのもの**が解析解と一致すること。

    ノーズ点の一致は「1 点が合った」に過ぎない。倍率 1.0 から
    ノーズ点の 98% までの各点で、数値解の電圧と 4 次方程式の上枝の解を
    比べる。上枝と下枝の取り違え（Newton が下枝へ飛ぶ事故）もここで捕まる。
    """
    e, x, p0 = 1.0, 0.1, 0.5
    curve = pv_curve(two_bus_case(e, x, p0), step=0.05, max_factor=40.0)
    nose_factor, _ = curve.nose(2)

    checked = 0
    for factor, voltage, ok in zip(curve.factors, curve.voltage_of(2), curve.converged):
        if not ok or factor > 0.98 * nose_factor:
            continue
        upper, lower = two_bus_voltages(e, x, p0 * factor)
        assert voltage == pytest.approx(upper, rel=1e-9), f"倍率 {factor} で上枝から外れた"
        assert voltage > lower, "下枝へ飛び移っている"
        checked += 1
    assert checked >= 10


def test_analytic_nose_is_the_double_root_of_the_quartic():
    """ノーズ点で 4 次方程式が重根を持つこと（判別式ゼロの言い換え）。

    :func:`two_bus_nose` と :func:`two_bus_voltages` は同じ方程式を
    別の解き方で扱っているので、ここが一致しなければ導出のどこかが
    間違っている。ノーズ点をわずかに超えると解が消えることも見る。
    """
    e, x, pf = 1.04, 0.07, 0.92
    p_max, v_crit = two_bus_nose(e, x, power_factor=pf)

    upper, lower = two_bus_voltages(e, x, p_max, power_factor=pf)
    assert upper == pytest.approx(v_crit, rel=1e-7)
    assert lower == pytest.approx(v_crit, rel=1e-7)

    # 1e-6 だけ手前ではまだ 2 つの解が分かれている。
    upper, lower = two_bus_voltages(e, x, p_max * (1 - 1e-6), power_factor=pf)
    assert upper > lower
    # わずかに超えると実数解が消える（例外ではなく nan で返る規約）。
    beyond = two_bus_voltages(e, x, p_max * (1 + 1e-6), power_factor=pf)
    assert all(math.isnan(value) for value in beyond)


def test_p_factor_only_rescales_the_returned_multiplier():
    """``p_factor`` は倍率への読み替えだけを行い、物理を変えないこと。"""
    p_max, v_crit = two_bus_nose(1.0, 0.1, power_factor=0.95)
    factor, voltage = two_bus_nose(1.0, 0.1, 0.4, power_factor=0.95)
    assert factor == pytest.approx(p_max / 0.4, rel=1e-14)
    assert voltage == pytest.approx(v_crit, rel=1e-14)


def test_unity_power_factor_reproduces_the_textbook_values():
    """力率 1 で :math:`P_{max}=E^2/(2X)`, :math:`|V|_{crit}=E/\\sqrt2` になること。"""
    for e, x in ((1.0, 0.1), (1.05, 0.25), (0.98, 0.05)):
        p_max, v_crit = two_bus_nose(e, x)
        assert p_max == pytest.approx(e**2 / (2 * x), rel=1e-14)
        assert v_crit == pytest.approx(e / math.sqrt(2.0), rel=1e-14)


# ======================================================================
# 2. 力率とノーズ点
# ======================================================================
def test_leading_power_factor_extends_the_nose_analytically():
    """進み力率のほうがノーズが伸びること（解析式の側から）。

    :math:`P_{max} \\propto (1 - \\sin\\phi)/\\cos\\phi` なので、
    :math:`\\sin\\phi < 0`（進み）で大きく、:math:`\\sin\\phi > 0`
    （遅れ）で小さくなる。臨界電圧は逆に進みのほうが高い。
    """
    lagging, v_lag = two_bus_nose(1.0, 0.1, power_factor=0.95)
    unity, v_one = two_bus_nose(1.0, 0.1, power_factor=1.0)
    leading, v_lead = two_bus_nose(1.0, 0.1, power_factor=-0.95)

    assert lagging < unity < leading
    assert v_lag < v_one < v_lead
    # 遅れ 0.95 で限界は 1 割以上落ち、進み 0.95 で 3 割以上伸びる。
    assert lagging / unity < 0.9
    assert leading / unity > 1.3


@pytest.mark.parametrize("pf", [0.90, 0.95, 1.00, -0.95, -0.90])
def test_numeric_nose_moves_with_the_power_factor(pf):
    """力率を変えるとノーズ点が動き、その動きが解析解と一致すること。

    「進みのほうが伸びる」を数値の側でも確かめる。基準は各力率での
    閉形式解であり、数値どうしの比較ではない。
    """
    expected, _ = two_bus_nose(1.0, 0.1, 0.5, power_factor=pf)
    curve = pv_curve(two_bus_case(power_factor=pf), step=0.05, max_factor=40.0)
    factor, _ = curve.nose(2)
    assert factor == pytest.approx(expected, rel=NOSE_RTOL)


def test_numeric_noses_are_ordered_by_power_factor():
    """遅れ < 力率 1 < 進み の順序が数値解でも保たれること。"""
    noses = []
    for pf in (0.95, 1.0, -0.95):
        curve = pv_curve(two_bus_case(power_factor=pf), step=0.05, max_factor=40.0)
        noses.append(curve.nose(2)[0])
    assert noses[0] < noses[1] < noses[2]


# ======================================================================
# 3. WSCC 9 母線 — 負荷余裕と最小特異値
# ======================================================================
def test_base_point_reproduces_the_textbook_solution(case, wscc9_curve):
    """倍率 1.0 の点が教科書の潮流解に一致すること。

    これは掃引が **発電を落としていない**ことの検査でもある。
    :meth:`Case.scaled` は参照解を落とすので、素直に渡すと発電ゼロの
    別系統を解いてしまい、この比較が真っ先に壊れる。
    """
    assert wscc9_curve.factors[0] == pytest.approx(1.0)
    assert bool(wscc9_curve.converged[0])
    assert wscc9_curve.voltages[0] == pytest.approx(case.reference.v, abs=REFERENCE_TOL)


def test_loading_margin_is_positive_and_consistent(case, wscc9_curve):
    """負荷余裕が正であり、ノーズ点の倍率と整合すること。"""
    factor, voltage = wscc9_curve.nose(5)
    assert factor > 1.0
    assert wscc9_curve.loading_margin == pytest.approx(factor - 1.0, rel=1e-12)
    # 電圧崩壊は負荷の最も重い母線 5 から始まり、臨界電圧は 0.5〜0.8 の間。
    assert wscc9_curve.critical_bus()[0] == 5
    assert 0.5 < voltage < 0.8


def test_nose_factor_is_bracketed_by_solvability(case, wscc9_curve):
    """ノーズ点の直前は解け、直後は解けないこと（挟み撃ちの検算）。

    :class:`PVCurve` が返す倍率が「解ける最大の倍率」であることを、
    曲線の記録ではなく **その場で解き直して**確かめる。
    """
    nose_factor, _ = wscc9_curve.nose(5)
    inside = solve(scaled_with_generation(case, nose_factor * (1 - 1e-6)))
    assert inside.converged
    with pytest.raises(RuntimeError):
        solve(scaled_with_generation(case, nose_factor * (1 + 1e-6)))


def test_min_singular_value_falls_monotonically_to_zero(wscc9_curve):
    """最小特異値がノーズ点に向かって単調に減り、ゼロに近づくこと。

    「解けた／解けない」の 0/1 判定ではなく連続量として限界への近さが
    読めることが、この指標の存在理由である。
    """
    sigma = wscc9_curve.min_singular_values[wscc9_curve.converged]
    assert sigma[0] > 0.9                       # 基準点では十分に非特異
    assert np.all(np.diff(sigma) < 0.0)         # 単調減少（等号も許さない）
    assert sigma[-1] < 1e-3                     # ノーズ点でほぼゼロ
    assert sigma[-1] < sigma[0] * 1e-4


def test_min_singular_value_follows_the_square_root_law(case, wscc9_curve):
    """:math:`\\sigma_{min} \\propto \\sqrt{\\lambda_{max}-\\lambda}` になること。

    折り目分岐（saddle-node）では固有値がゼロを二次の形で通るので、
    最小特異値は距離の平方根で縮む。分岐理論から出る性質であって
    実装のどこにも書かれていないので、独立な基準として使える。
    距離を 4 倍にすれば最小特異値はおよそ 2 倍になるはずである。
    """
    nose_factor, _ = wscc9_curve.nose(5)
    values = []
    for distance in (0.04, 0.01):
        scaled = scaled_with_generation(case, nose_factor - distance)
        values.append(min_singular_value(scaled, solve(scaled)))
    ratio = values[0] / values[1]
    # 高次項があるので厳密に 2 にはならない（実測 1.93）。10% の幅を取る。
    assert 1.8 < ratio < 2.2


def test_min_singular_value_matches_a_numerically_built_jacobian(case):
    """最小特異値が **数値微分で組んだ**ヤコビアンのものと一致すること。

    :func:`min_singular_value` は解析式のヤコビアンを使う。ここでは
    :func:`gridops.powerflow.mismatch` を中心差分して行列を組み直し、
    さらに :math:`\\sigma_{min} = 1/\\lVert J^{-1}\\rVert_2` という
    特異値分解を経ない定義で比べる。行列の取り違えと「最大特異値を
    返している」種類の誤りが同時に捕まる。
    """
    solution = solve(case, tol=1e-12)
    Y = build_ybus(case)
    _, pv, pq = case.type_indices()
    non_slack = np.sort(np.concatenate([pv, pq]))
    n = case.n_unknowns()

    def residual(theta, v):
        # mismatch は ΔS = S_sp - S(x) なので、ヤコビアンの符号は反転する。
        return -mismatch(case, Y, v, theta)

    numeric = np.zeros((n, n))
    step = 1e-6
    for column, index in enumerate(non_slack):
        for sign in (+1, -1):
            theta = solution.theta.copy()
            theta[index] += sign * step
            numeric[:, column] += sign * residual(theta, solution.v) / (2 * step)
    for offset, index in enumerate(pq):
        column = len(non_slack) + offset
        for sign in (+1, -1):
            v = solution.v.copy()
            # 未知数は Δ|V|/|V| なので、刻みも相対で入れる。
            v[index] *= 1.0 + sign * step
            numeric[:, column] += sign * residual(solution.theta, v) / (2 * step)

    expected = 1.0 / np.linalg.norm(np.linalg.inv(numeric), 2)
    got = min_singular_value(case, solution)
    # 中心差分の打切り誤差 O(step^2)=1e-12 と丸め 1e-16/step=1e-10 の和。
    assert got == pytest.approx(expected, rel=1e-7)
    assert 0.5 < got < 1.5


# ======================================================================
# 4. ノーズ点を超えたら潮流は解けない
# ======================================================================
@pytest.mark.parametrize("pf", [1.0, 0.95, -0.95])
def test_beyond_the_analytic_nose_the_power_flow_raises(pf):
    """解析解が「解が無い」と言う倍率で潮流が RuntimeError になること。

    **初期値のせいではないこと**を示すため、ノーズ点の直前の解を
    そのまま初期値に与えてから解かせる。良い初期値を与えても解けない
    ことが「解が無い」の証拠になる。
    """
    e, x, p0 = 1.0, 0.1, 0.5
    limit, _ = two_bus_nose(e, x, p0, power_factor=pf)
    base = two_bus_case(e, x, p0, pf)

    warm = solve(base.scaled(limit * 0.999))
    assert warm.converged
    with pytest.raises(RuntimeError):
        solve(base.scaled(limit * 1.001), v0=warm.v, theta0=warm.theta)


def test_beyond_the_nose_the_wscc9_power_flow_raises(case, wscc9_curve):
    """9 母線でもノーズ点を超えた倍率では収束しないこと。"""
    nose_factor, _ = wscc9_curve.nose(5)
    warm_case = scaled_with_generation(case, nose_factor)
    warm = solve(warm_case)
    with pytest.raises(RuntimeError):
        solve(
            scaled_with_generation(case, nose_factor * 1.02),
            v0=warm.v,
            theta0=warm.theta,
        )


def test_failed_points_are_recorded_rather_than_hidden(wscc9_curve):
    """収束しなかった点が捨てられず、``nan`` の行として残ること。

    「解けなかった」は測定値であって失敗ではない、というのがこの
    モジュールの立場である。作図でノーズ点の外側が空くのはそのため。
    """
    assert not bool(wscc9_curve.converged.all())
    failed = ~wscc9_curve.converged
    assert np.all(np.isnan(wscc9_curve.voltages[failed]))
    assert np.all(np.isnan(wscc9_curve.min_singular_values[failed]))
    assert np.all(wscc9_curve.iterations[failed] == -1)
    # 収束した点はすべて有限で、倍率は昇順に並んでいる。
    assert np.all(np.isfinite(wscc9_curve.voltages[wscc9_curve.converged]))
    assert np.all(np.diff(wscc9_curve.factors) > 0)
    # 失敗した点はすべてノーズ点より上にある（手前で失敗していない）。
    nose_factor = wscc9_curve.factors[wscc9_curve.critical_index]
    assert np.all(wscc9_curve.factors[failed] > nose_factor)


# ======================================================================
# 5. 調相設備はノーズを伸ばす
# ======================================================================
def test_shunt_compensation_extends_the_nose(case, wscc9_curve):
    """母線 5 に調相設備（``Bus.bs``）を足すとノーズ点が伸びること。

    負荷母線に容量性のサセプタンスを置くと、実質的に力率が進み側に
    寄る。:func:`two_bus_nose` が示す
    :math:`P_{max} \\propto (1-\\sin\\phi)/\\cos\\phi` の効き方が、
    9 母線でも同じ向きに出ることを確かめる。
    """
    base_factor, _ = wscc9_curve.nose(5)

    buses = list(case.buses)
    index = case.index_of(5)
    buses[index] = replace(buses[index], bs=buses[index].bs + 0.5)
    compensated = replace(case, buses=buses)

    curve = pv_curve(compensated)
    factor, _ = curve.nose(5)

    assert factor > base_factor
    # 0.5 p.u. (= 50 Mvar 相当) の投入で 2% 以上は伸びる。
    assert factor / base_factor > 1.02
    # 基準点の電圧も上がる（同じ負荷を高い電圧で運べる）。
    assert curve.voltages[0][index] > wscc9_curve.voltages[0][index]


def test_larger_shunt_extends_the_nose_further(case):
    """調相設備を増やすほどノーズが伸びること（単調性）。"""
    factors = []
    for shunt in (0.0, 0.3, 0.6):
        buses = list(case.buses)
        index = case.index_of(5)
        buses[index] = replace(buses[index], bs=shunt)
        curve = pv_curve(replace(case, buses=buses), step=0.05)
        factors.append(curve.nose(5)[0])
    assert factors[0] < factors[1] < factors[2]


# ======================================================================
# 6. 電圧-無効電力感度
# ======================================================================
def numeric_voltage_sensitivity(case: Case, solution, delta: float = 1e-4) -> np.ndarray:
    """数値微分による :math:`\\partial |V|/\\partial Q`（独立な基準）。

    PQ 母線の ``qd`` を :math:`\\mp\\delta` 動かして潮流を解き直す。
    ``qd`` を減らすことが無効電力を **注入する**ことに当たる
    （:meth:`Case.bus_injection` の符号）。中心差分なので打切り誤差は
    :math:`O(\\delta^2)`。
    """
    _, _, pq = case.type_indices()
    result = np.zeros((pq.size, pq.size))
    for column, index in enumerate(pq):
        columns = []
        for sign in (+1, -1):
            buses = list(case.buses)
            bus = buses[index]
            buses[index] = replace(bus, qd=bus.qd - sign * delta)
            perturbed = replace(case, buses=buses)
            columns.append(
                solve(perturbed, tol=1e-13, v0=solution.v, theta0=solution.theta).v[pq]
            )
        result[:, column] = (columns[0] - columns[1]) / (2 * delta)
    return result


def test_voltage_sensitivity_matches_numerical_differentiation(case):
    """感度行列が「Q を動かして解き直した差分」と一致すること。

    これが :func:`voltage_sensitivity` の定義そのものである。刻み
    1e-4 の中心差分なので打切り誤差は :math:`O(10^{-8})`、ソルバの
    許容差 1e-13 に由来する丸めは :math:`10^{-13}/10^{-4} = 10^{-9}` で、
    要求する相対 1e-4 には 4 桁以上の余裕がある。
    """
    solution = solve(case, tol=1e-13)
    expected = numeric_voltage_sensitivity(case, solution)
    got = voltage_sensitivity(case, solution)

    assert got.shape == expected.shape == (6, 6)
    assert np.max(np.abs(got - expected) / np.abs(expected)) < 1e-4


def test_naive_l_inverse_does_not_match_numerical_differentiation(case):
    """素の :math:`L^{-1}` では数値微分と合わないこと。

    「L ブロックの逆行列」を字義どおりに取ると減結合近似になり、
    :math:`\\Delta P = 0` ではなく :math:`\\Delta\\theta = 0` を課した
    量になってしまう。WSCC 9 母線では 10% 以上ずれる。**この差が
    Fast Decoupled 法の前提が近似であることの大きさそのもの**であり、
    縮約が必要な理由でもある。
    """
    solution = solve(case, tol=1e-13)
    expected = numeric_voltage_sensitivity(case, solution)
    _, _, pq = case.type_indices()
    _, _, _, L = jacobian_blocks(case, build_ybus(case), solution.v, solution.theta)
    naive = np.diag(solution.v[pq]) @ np.linalg.inv(L)

    error = np.max(np.abs(naive - expected) / np.abs(expected))
    assert error > 0.05, "減結合近似が偶然一致してしまっている（縮約の検査にならない）"
    assert error < 0.5


def test_voltage_sensitivity_is_symmetric_in_sign_and_grows_near_the_nose(case):
    """感度の符号と、ノーズ点に近づくと発散すること。

    対角成分は正（無効電力を入れれば自分の電圧は上がる）。ノーズ点に
    近づくと縮約ヤコビアンが特異に近づくので感度は大きくなる。
    """
    base = voltage_sensitivity(case, solve(case, tol=1e-12))
    assert np.all(np.diag(base) > 0.0)

    # ノーズ点 2.3739 のすぐ手前まで負荷を上げる。
    near = scaled_with_generation(case, 2.37)
    stressed = voltage_sensitivity(near, solve(near, tol=1e-12))
    growth = np.diag(stressed) / np.diag(base)
    assert np.all(growth > 1.5), "どの母線でも感度は上がるはず"
    # 崩壊が始まる母線 5 では 5 倍以上に跳ね上がる（実測 9.1 倍）。
    assert growth.max() > 5.0
    assert int(np.argmax(growth)) == 1


def test_voltage_sensitivity_row_and_column_order(case):
    """行と列の並びが :meth:`Case.type_indices` の PQ の並びであること。

    母線番号ではなく **母線の並び順**であることを固定する。ここを
    取り違えると値は妥当に見えるまま結論だけが入れ替わる。
    """
    _, _, pq = case.type_indices()
    assert [case.buses[i].id for i in pq] == [4, 5, 6, 7, 8, 9]

    solution = solve(case, tol=1e-13)
    got = voltage_sensitivity(case, solution)
    expected = numeric_voltage_sensitivity(case, solution)
    # 母線 5 に Q を入れたときに最も動くのは母線 5 自身である。
    column = 1
    assert int(np.argmax(got[:, column])) == column
    assert got[:, column] == pytest.approx(expected[:, column], rel=1e-4)


# ======================================================================
# 引数の検査と縁の場合
# ======================================================================
def test_two_bus_nose_rejects_bad_arguments():
    """物理的に意味のない引数を日本語の ValueError で止めること。"""
    with pytest.raises(ValueError, match="無限大母線の電圧は正"):
        two_bus_nose(0.0, 0.1)
    with pytest.raises(ValueError, match="誘導性リアクタンス"):
        two_bus_nose(1.0, -0.1)
    with pytest.raises(ValueError, match="基準負荷"):
        two_bus_nose(1.0, 0.1, 0.0)
    with pytest.raises(ValueError, match="power_factor がゼロ"):
        two_bus_nose(1.0, 0.1, power_factor=0.0)
    with pytest.raises(ValueError, match="絶対値が 1 を超えている"):
        two_bus_nose(1.0, 0.1, power_factor=1.05)


def test_pv_curve_rejects_bad_arguments(case):
    """掃引の設定が矛盾していれば解く前に止めること。"""
    with pytest.raises(ValueError, match="非正"):
        pv_curve(case, step=0.0)
    with pytest.raises(ValueError, match="1 以下"):
        pv_curve(case, max_factor=1.0)


def test_pv_curve_refuses_a_base_case_that_cannot_be_solved():
    """基準ケースが解けないときは、ノーズ点の話にせず区別して止めること。

    基準負荷 10 p.u. は :math:`P_{max}=E^2/(2X)=5` を超えているので、
    倍率 1.0 の時点で解が存在しない。これを「余裕ゼロ」と報告すると
    データの誤りが隠れるので、``ValueError`` で別扱いにしている。
    """
    with pytest.raises(ValueError, match="既に潮流が解けない"):
        pv_curve(two_bus_case(p0=10.0))


def test_pv_curve_warns_when_the_nose_is_out_of_range():
    """``max_factor`` の内側にノーズ点が無ければ警告すること。

    黙って上限の点を返すと「余裕は 0.5 だった」という誤読を招く。
    """
    with pytest.warns(UserWarning, match="ノーズ点を"):
        curve = pv_curve(two_bus_case(), step=0.1, max_factor=1.5)
    assert bool(curve.converged.all())
    assert curve.loading_margin == pytest.approx(0.5, abs=0.11)


def test_pv_curve_without_refinement_is_only_as_good_as_the_step():
    """``refine=False`` では刻みの精度でしかノーズ点が決まらないこと。

    二分法が効いていることを、効かせなかった場合との差で示す。
    """
    expected, _ = two_bus_nose(1.0, 0.1, 0.5)   # = 10.0
    # 刻み 0.35 の格子は 9.75 の次が 10.10 なので、ノーズ点 10.0 に乗らない。
    step = 0.35
    coarse = pv_curve(two_bus_case(), step=step, max_factor=40.0, refine=False)
    fine = pv_curve(two_bus_case(), step=step, max_factor=40.0, refine=True)

    coarse_error = expected - coarse.nose(2)[0]
    fine_error = expected - fine.nose(2)[0]
    assert 0.0 <= fine_error < 1e-6
    assert coarse_error > 0.1            # 刻みの粗さがそのまま残る
    assert coarse_error < step + 1e-9    # ただし刻みより悪くはならない


def test_scaling_only_selected_buses(case):
    """``buses`` で指定した母線だけが重くなること。

    母線 5 だけを重くすると、全母線を一律に重くするより倍率の余裕は
    大きくなる（同じ倍率でも系統全体に加わる負荷が小さいため）。
    """
    all_buses = pv_curve(case, step=0.05)
    only_bus5 = pv_curve(case, buses=[5], step=0.05, max_factor=8.0)
    assert only_bus5.scaled_buses == (5,)
    assert only_bus5.nose(5)[0] > all_buses.nose(5)[0]
    # 母線 5 以外の負荷は動いていないので、基準点の電圧は一致する。
    assert only_bus5.voltages[0] == pytest.approx(all_buses.voltages[0], abs=1e-10)


def test_pv_curve_without_a_case_cannot_resolve_bus_numbers():
    """``case`` を持たない :class:`PVCurve` は母線番号を引けないこと。"""
    curve = PVCurve(
        factors=np.array([1.0, 1.1]),
        voltages=np.array([[1.0, 0.99], [1.0, 0.95]]),
        converged=np.array([True, True]),
        critical_index=1,
    )
    assert curve.loading_margin == pytest.approx(0.1)
    with pytest.raises(ValueError, match="case を持っていない"):
        curve.nose(2)


def test_summary_is_japanese_and_mentions_the_margin(wscc9_curve):
    """要約が日本語で、負荷余裕と最小特異値に触れること。"""
    text = wscc9_curve.summary()
    assert "負荷余裕" in text
    assert "最小特異値" in text
    assert f"{wscc9_curve.loading_margin:.6f}" in text
