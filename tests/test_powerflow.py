"""交流潮流計算（第 02 回）の検証。

このモジュールの正しさは、**実装の外にある基準**とだけ突き合わせて
確かめる。実装の出力を実装で計算し直すだけのテストは書かない。
使った基準は次の 5 つである。

1. 教科書の潮流解（Anderson & Fouad, 2003, Ch.2）— ケースファイルの
   ``solution`` 層に入っている。掲載は 4 桁なので、許容差はその丸めで決まる
2. API 契約に載っている **独立に検証済みの数値**（slack 出力・総損失・
   枝潮流・負荷率・N-1 の結果）
3. **中心差分**による数値微分 — 解析式のヤコビアンとは導出の道筋が違う
4. **物理的な不変性** — 位相を一律にずらしても枝潮流は変わらない、
   注入の総和は枝損失の合計に等しい
5. **3 つの解法の相互一致** — Newton / Gauss-Seidel / Fast Decoupled は
   アルゴリズムが独立で、同じ方程式を解いている

許容差にはすべて根拠を書く。「だいたい合っている」で緩めた許容差は、
後でドリフトが入り込む隙間になる。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gridops import BusType, Case, load_case
from gridops.powerflow import (
    DEFAULT_MAX_ITER,
    PowerFlowSolution,
    jacobian,
    jacobian_blocks,
    mismatch,
    solve,
)
from gridops.ybus import build_ybus

#: API 契約の「検証済みの数値」。独立な実装で確認された基準潮流の枝潮流。
#: ``枝: (両端の |S| の大きい方, from 側の有効電力, rate_a に対する負荷率 [%])``
VERIFIED_FLOWS = {
    (1, 4): (0.7658, 0.7164, 38.3),
    (2, 7): (1.6326, 1.6300, 81.6),
    (3, 9): (0.8631, 0.8500, 43.2),
    (4, 5): (0.5614, 0.4094, 56.1),
    (4, 6): (0.3473, 0.3070, 43.4),
    (5, 7): (0.8702, -0.8432, 79.1),
    (6, 9): (0.6345, -0.5946, 57.7),
    (7, 8): (0.7666, 0.7638, 69.7),
    (8, 9): (0.3422, -0.2410, 42.8),
}

#: API 契約の N-1 の結果（交流潮流、rate_b に対する負荷率）。
#: ``開放する枝: (最悪の枝, 負荷率 [%], 最低電圧の母線, その電圧)``
VERIFIED_N1 = {
    (4, 5): ((5, 7), 101.5, 5, 0.8388),
    (4, 6): ((7, 8), 75.7, 6, 0.9418),
    (5, 7): ((7, 8), 112.5, 5, 0.9380),
    (6, 9): ((5, 7), 101.5, 6, 0.9639),
    (7, 8): ((5, 7), 112.4, 8, 0.9690),
    (8, 9): ((7, 8), 73.1, 8, 0.9783),
}


@pytest.fixture(scope="module")
def case() -> Case:
    return load_case("wscc9")


@pytest.fixture(scope="module")
def solution(case: Case) -> PowerFlowSolution:
    return solve(case)


# ----------------------------------------------------------------------
# テスト用のヘルパ
# ----------------------------------------------------------------------
def _complex_power(Y: np.ndarray, v_mag: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """:math:`\\bar S = \\bar V (Y\\bar V)^{*}` を複素数のまま計算する。

    数値微分の土台。実装のヤコビアンは三角関数表現から解析的に
    導かれているので、こちらとは式の形が違う。
    """
    voltage = v_mag * np.exp(1j * theta)
    return voltage * np.conj(Y @ voltage)


def _outage(case: Case, key: tuple[int, int]) -> Case:
    """枝を 1 本開放したケースを、**参照解を残したまま**作る。

    :meth:`Case.without_branch` は ``reference=None`` にしてしまうので、
    そのまま潮流を解くと :meth:`Case.bus_injection` が発電ゼロを返し、
    「slack 母線 1 台で全負荷を賄う」という別の系統を解くことになる。
    N-1 の評価では発電機の出力は事故前のまま据え置くのが前提なので、
    ここでは枝だけを差し替える。
    """
    target = (min(key), max(key))
    remaining = [branch for branch in case.branches if branch.key() != target]
    assert len(remaining) == len(case.branches) - 1
    return replace(case, branches=remaining)


# ======================================================================
# 1. 教科書解との一致
# ======================================================================
def test_newton_matches_textbook_solution(case, solution):
    """Newton 解が Anderson & Fouad の掲載値と一致すること。

    許容差 1e-3 は掲載桁数（|V| は小数 4 桁、位相は小数 4 桁 [deg]）で
    決まる丸めの上限である。実測の差は |V| で 4.705e-05、位相で
    4.867e-05 deg なので、桁数由来の丸めよりさらに 1 桁小さい。
    """
    reference = case.reference
    assert solution.converged
    assert np.max(np.abs(solution.v - reference.v)) < 1e-3
    assert np.max(np.abs(solution.angle_deg - reference.angle_deg)) < 1e-3

    # ドリフト検出。ここが緩むと「合っているつもり」が始まる。
    assert np.max(np.abs(solution.v - reference.v)) == pytest.approx(4.705e-05, abs=1e-7)
    assert np.max(
        np.abs(solution.angle_deg - reference.angle_deg)
    ) == pytest.approx(4.867e-05, abs=1e-7)


def test_slack_power_and_losses_match_case_checks(case, solution):
    """slack 出力と総損失が、ケースファイルの独立な検算値と一致すること。

    ``checks`` はケースファイル側に置かれた期待値であり、実装が計算した
    ものではない。許容差 1e-5 は ``checks`` の記載桁数（小数 6 桁）による。
    """
    checks = case.reference.checks
    assert solution.slack_power.real == pytest.approx(checks["slack_p"], abs=1e-5)
    assert solution.slack_power.imag == pytest.approx(checks["slack_q"], abs=1e-5)
    assert solution.losses == pytest.approx(checks["losses_pu"], abs=1e-5)


def test_textbook_solution_is_the_same_root(case, solution):
    """教科書解を初期値にしても、フラットスタートと同じ解に落ち着くこと。

    潮流方程式は非線形なので原理的には複数の解を持つ。掲載値から出発した
    Newton が別の点に行かないことを確かめておかないと、「教科書解に近い
    別の解」を掴んでいる可能性を否定できない。掲載値のミスマッチは丸めの
    ぶんだけ残るので 2 反復ほどで収束する。
    """
    from_textbook = solve(
        case, v0=case.reference.v, theta0=np.radians(case.reference.angle_deg)
    )
    assert from_textbook.iterations <= 3
    assert np.max(np.abs(from_textbook.v - solution.v)) < 1e-9
    assert np.max(np.abs(from_textbook.theta - solution.theta)) < 1e-9


def test_injections_match_the_trigonometric_form(case, solution):
    """注入電力が、三角関数で書いた式と一致すること。

    実装は :math:`\\bar S = \\bar V (Y \\bar V)^{*}` を複素数のまま計算する。
    こちらは教科書の式

    .. math:: P_i = \\sum_j |V_i||V_j|(G\\cos\\theta_{ij} + B\\sin\\theta_{ij})

    をそのまま書き下したもので、両者は独立な道筋である。許容差 1e-12 は
    倍精度の丸めの水準。
    """
    Y = build_ybus(case)
    G, B = Y.real, Y.imag
    difference = solution.theta[:, None] - solution.theta[None, :]
    products = np.outer(solution.v, solution.v)
    p_expected = np.sum(products * (G * np.cos(difference) + B * np.sin(difference)), axis=1)
    q_expected = np.sum(products * (G * np.sin(difference) - B * np.cos(difference)), axis=1)

    p, q = solution.injections()
    assert p == pytest.approx(p_expected, abs=1e-12)
    assert q == pytest.approx(q_expected, abs=1e-12)


def test_mismatch_vanishes_at_the_solution(case, solution):
    """収束した解でミスマッチが tol 以下であること、長さが未知数と合うこと。"""
    residual = mismatch(case, build_ybus(case), solution.v, solution.theta)
    assert residual.shape == (case.n_unknowns(),)
    assert np.max(np.abs(residual)) < 1e-10
    # 方程式の数 = 2 n_PQ + n_PV。数え違いはここで露見する。
    _, pv, pq = case.type_indices()
    assert residual.size == 2 * len(pq) + len(pv)


# ======================================================================
# 2. ヤコビアン — 解析式と中心差分
# ======================================================================
def _central_difference_blocks(case, Y, v_mag, theta, step=1e-6):
    """4 ブロックを中心差分で作る（解析式とは独立な基準）。

    刻み幅 1e-6 は打切り誤差 :math:`O(h^2) \\sim 10^{-12}` と丸め誤差
    :math:`O(\\varepsilon |f| / h) \\sim 10^{-9}` の釣り合う点である
    （genstab.linearize の DEFAULT_STEP と同じ考え方）。
    """
    _, pv, pq = case.type_indices()
    non_slack = np.sort(np.concatenate([pv, pq]))

    H = np.zeros((non_slack.size, non_slack.size))
    N = np.zeros((non_slack.size, pq.size))
    M = np.zeros((pq.size, non_slack.size))
    L = np.zeros((pq.size, pq.size))

    for column, j in enumerate(non_slack):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += step
        minus[j] -= step
        derivative = (
            _complex_power(Y, v_mag, plus) - _complex_power(Y, v_mag, minus)
        ) / (2 * step)
        H[:, column] = derivative.real[non_slack]
        M[:, column] = derivative.imag[pq]

    for column, j in enumerate(pq):
        plus, minus = v_mag.copy(), v_mag.copy()
        plus[j] += step
        minus[j] -= step
        derivative = (
            _complex_power(Y, plus, theta) - _complex_power(Y, minus, theta)
        ) / (2 * step)
        # N, L は ∂/∂|V| に |V| を掛けた形で定義されている。
        N[:, column] = derivative.real[non_slack] * v_mag[j]
        L[:, column] = derivative.imag[pq] * v_mag[j]

    return H, N, M, L


def test_jacobian_matches_central_difference(case):
    """解析ヤコビアンの 4 ブロックが中心差分と一致すること。

    解析式は写し間違いを実行時に教えてくれない。**この一致がヤコビアンの
    唯一の外的な検証**である。収束点ではなく乱数で散らした点で比べるのは、
    収束点だと :math:`P, Q` が指定値に等しくなって対角項の誤りが
    打ち消されうるためである。許容差 1e-6 は中心差分の丸め誤差
    （実測で 4e-9 程度）に対して 2 桁以上の余裕がある。
    """
    Y = build_ybus(case)
    rng = np.random.default_rng(3)
    v_mag = np.array([bus.v_set for bus in case.buses]) + rng.normal(0.0, 0.02, case.n_bus)
    theta = rng.normal(0.0, 0.1, case.n_bus)

    analytic = jacobian_blocks(case, Y, v_mag, theta)
    numeric = _central_difference_blocks(case, Y, v_mag, theta)

    for name, exact, approximate in zip("HNML", analytic, numeric):
        assert exact.shape == approximate.shape, f"{name} ブロックの形が違う"
        assert np.max(np.abs(exact - approximate)) < 1e-6, f"{name} ブロックが不一致"


def test_jacobian_matches_central_difference_with_phase_shifter(case):
    """位相調整器（非対称 Ybus）でも解析式が中心差分と一致すること。

    ``shift_deg != 0`` があると :math:`Y_{ft} \\ne Y_{tf}` となり Ybus は
    非対称になる。解析式の導出は対称性を使っていないので成り立つはず
    だが、対称性を暗に仮定した式を書いてしまう事故が起こりやすい箇所
    なので、ここで固定しておく。
    """
    shifted = replace(
        case,
        branches=[
            replace(branch, shift_deg=5.0, tap=1.05) if branch.key() == (1, 4) else branch
            for branch in case.branches
        ],
    )
    Y = build_ybus(shifted)
    assert not np.allclose(Y, Y.T)   # 前提の確認: 非対称であること

    rng = np.random.default_rng(11)
    v_mag = np.array([bus.v_set for bus in shifted.buses]) + rng.normal(0.0, 0.02, 9)
    theta = rng.normal(0.0, 0.1, 9)

    for name, exact, approximate in zip(
        "HNML",
        jacobian_blocks(shifted, Y, v_mag, theta),
        _central_difference_blocks(shifted, Y, v_mag, theta),
    ):
        assert np.max(np.abs(exact - approximate)) < 1e-6, f"{name} ブロックが不一致"


def test_jacobian_lays_out_the_four_blocks(case, solution):
    """``jacobian`` が ``[[H, N], [M, L]]`` の並びであること。

    並びが崩れると Newton は「収束しないか、遅く収束する」という形でしか
    異常を教えてくれない。未知数の数との一致もここで確かめる。
    """
    Y = build_ybus(case)
    H, N, M, L = jacobian_blocks(case, Y, solution.v, solution.theta)
    matrix = jacobian(case, Y, solution.v, solution.theta)

    assert matrix.shape == (case.n_unknowns(), case.n_unknowns())
    rows, columns = H.shape
    assert matrix[:rows, :columns] == pytest.approx(H, abs=0.0)
    assert matrix[:rows, columns:] == pytest.approx(N, abs=0.0)
    assert matrix[rows:, :columns] == pytest.approx(M, abs=0.0)
    assert matrix[rows:, columns:] == pytest.approx(L, abs=0.0)


def test_offdiagonal_blocks_are_small_in_high_voltage_network(case, solution):
    """高電圧系統で :math:`|N|, |M| \\ll |H|, |L|` であること（減結合の根拠）。

    送電線は :math:`x \\gg r` なので、有効電力は主に位相差で、無効電力は
    主に電圧差で決まる。WSCC 9 母線の枝の :math:`r/x` は最大 0.229 で、
    実測の比は :math:`\\max|N|/\\max|H| = 0.084`,
    :math:`\\max|M|/\\max|L| = 0.102` と同じオーダーに収まる。閾値 0.15 は
    この実測と :math:`r/x` の最大値の間に取った。

    抵抗を 10 倍にすると比は 0.41 / 0.49 まで上がる。Fast Decoupled 法の
    前提が **系統の性質であって手法の性質ではない**ことがここで見える。
    """
    Y = build_ybus(case)
    H, N, M, L = jacobian_blocks(case, Y, solution.v, solution.theta)

    assert np.abs(N).max() / np.abs(H).max() < 0.15
    assert np.abs(M).max() / np.abs(L).max() < 0.15
    assert np.linalg.norm(N) / np.linalg.norm(H) < 0.15
    assert np.linalg.norm(M) / np.linalg.norm(L) < 0.15

    resistive = replace(
        case, branches=[replace(branch, r=branch.r * 10.0) for branch in case.branches]
    )
    resistive_solution = solve(resistive)
    Hr, Nr, Mr, Lr = jacobian_blocks(
        resistive, build_ybus(resistive), resistive_solution.v, resistive_solution.theta
    )
    assert np.abs(Nr).max() / np.abs(Hr).max() > 3.0 * np.abs(N).max() / np.abs(H).max()


# ======================================================================
# 3. 3 つの解法
# ======================================================================
@pytest.fixture(scope="module")
def by_method(case) -> dict[str, PowerFlowSolution]:
    """3 つの交流解法の解をまとめて作る。"""
    return {name: solve(case, name) for name in ("newton", "gauss_seidel", "fast_decoupled")}


def test_methods_agree_on_the_same_solution(by_method):
    """Newton / Gauss-Seidel / Fast Decoupled が同じ解に収束すること。

    3 つはアルゴリズムが独立だが **同じ方程式**を解いている。違うのは
    「どう近づくか」だけなので、収束すれば同じ点に行き着く。ここが
    崩れるとしたら、どれかが別の方程式（別の母線種別、別の注入）を
    解いている。許容差 1e-6 は tol=1e-10 のミスマッチが電圧に与える
    影響（実測で 3e-12）に対して十分に緩い。
    """
    reference = by_method["newton"]
    for name, other in by_method.items():
        assert np.max(np.abs(other.v - reference.v)) < 1e-6, name
        assert np.max(np.abs(other.theta - reference.theta)) < 1e-6, name
        assert other.losses == pytest.approx(reference.losses, abs=1e-9), name
        assert other.converged and other.method == name


def test_iteration_counts_show_the_order_of_convergence(by_method):
    """反復回数が収束次数を反映すること。

    Newton は二次収束なので 6 回以内（実測 4 回）。Gauss-Seidel は
    一次収束なので桁違いに多い（実測 61 回）。Fast Decoupled は
    Newton より多く Gauss-Seidel よりずっと少ない（実測 8 回）。
    **反復回数の多さは実装の良し悪しではなく収束次数の帰結**である。
    """
    newton = by_method["newton"].iterations
    gauss = by_method["gauss_seidel"].iterations
    fast = by_method["fast_decoupled"].iterations

    assert newton <= 6
    assert gauss > 5 * newton and gauss >= 30
    assert newton <= fast < gauss
    # 既定の上限は解法ごとに違う。揃えると Gauss-Seidel が打ち切られる。
    assert DEFAULT_MAX_ITER["gauss_seidel"] > DEFAULT_MAX_ITER["newton"]


def test_newton_converges_quadratically(by_method):
    """Newton のミスマッチが前回の 2 乗のオーダーで減ること。

    二次収束とは :math:`e_{k+1} \\le C e_k^2` が成り立つことである。
    実測の :math:`C = e_{k+1}/e_k^2` は 0.061〜0.157 なので、閾値 0.5 は
    3 倍以上の余裕がある。履歴は 1.63 → 1.88e-1 → 2.15e-3 → 3.42e-7 →
    1.84e-14 と、有効桁が反復ごとに倍になっていく。
    """
    history = by_method["newton"].mismatch_history
    assert len(history) == by_method["newton"].iterations + 1
    assert history[-1] <= 1e-10

    for previous, current in zip(history, history[1:]):
        assert current <= 0.5 * previous**2, f"{previous:.3e} -> {current:.3e}"


def test_gauss_seidel_converges_linearly(by_method):
    """Gauss-Seidel のミスマッチが一定の比で減ること（一次収束）。

    一次収束は :math:`e_{k+1} \\approx r e_k`（:math:`r` は定数）である。
    終盤の比は実測で 0.695〜0.701 とほぼ一定で、二次収束の指標
    :math:`e_{k+1}/e_k^2` は 1e9 の桁まで膨らむ（二次では **ない**）。
    対数軸で描くと Newton が下に折れ曲がるのに対し、こちらは直線になる。
    """
    history = by_method["gauss_seidel"].mismatch_history
    ratios = [
        current / previous for previous, current in zip(history[-8:], history[-7:])
    ]
    assert max(ratios) - min(ratios) < 0.05      # 比が一定であること
    assert 0.5 < min(ratios) < max(ratios) < 0.9
    # 同じ区間で二次収束の定数は発散する。
    assert min(
        current / previous**2 for previous, current in zip(history[-8:], history[-7:])
    ) > 1e6


def test_pv_buses_hold_their_setpoint(case, by_method):
    """どの解法でも PV・slack 母線の |V| が設定値に一致すること。

    Gauss-Seidel で「Q を計算したあと |V| を設定値に戻す」段を忘れると、
    PV 母線が実質 PQ 母線になって設定値からずれる。収束はするので
    気づきにくい。許容差 1e-12 は倍精度の丸めの水準。
    """
    for name, result in by_method.items():
        for i, bus in enumerate(case.buses):
            if bus.type is BusType.PQ:
                continue
            assert result.v[i] == pytest.approx(bus.v_set, abs=1e-12), f"{name}: 母線 {bus.id}"


def test_flat_start_and_warm_start_reach_the_same_point(case, solution):
    """収束した解を初期値に与えると 0 反復で返ること。

    収束判定が「更新してから」ではなく「更新する前」に行われている
    ことの確認でもある。ここが逆だと必ず 1 回余計に反復する。
    """
    warm = solve(case, v0=solution.v, theta0=solution.theta)
    assert warm.iterations == 0
    assert len(warm.mismatch_history) == 1


# ======================================================================
# 4. 物理的な不変性
# ======================================================================
def test_branch_flows_are_invariant_under_uniform_phase_shift(solution):
    """全母線の位相を一律にずらしても枝潮流が変わらないこと。

    潮流は位相 **差**だけで決まる。この対称性が崩れていれば、枝の
    アドミタンス行列か電力の式のどこかで絶対位相を使っている。
    許容差 1e-12 は倍精度の丸めの水準（実測 3.1e-15）。
    """
    shifted = replace(solution, theta=solution.theta + math.radians(37.0))
    base_flows = solution.branch_flows()
    shifted_flows = shifted.branch_flows()

    assert set(base_flows) == set(shifted_flows)
    for key, (s_ft, s_tf) in base_flows.items():
        assert abs(s_ft - shifted_flows[key][0]) < 1e-12, key
        assert abs(s_tf - shifted_flows[key][1]) < 1e-12, key
    assert shifted.losses == pytest.approx(solution.losses, abs=1e-12)


def test_losses_equal_the_sum_of_injections(solution):
    """注入の実部の総和が枝損失の合計に一致すること（損失の保存）。

    「発電の合計 - 負荷の合計 = 送電損失」という当たり前の関係だが、
    枝潮流を枝ごとに組み立てる過程と、母線注入を :math:`Y` から計算する
    過程は別物なので、一致は組み立ての検算になる。WSCC 9 母線は母線
    シャント ``gs`` がゼロなので厳密に一致する。許容差 1e-10 は
    倍精度の丸め（実測 1.3e-15）。
    """
    p, _ = solution.injections()
    assert float(p.sum()) == pytest.approx(solution.losses, abs=1e-10)
    # 損失は正であり、最大の枝潮流よりずっと小さい。
    assert 0.0 < solution.losses < 0.1


def test_branch_losses_are_positive_and_reciprocal(solution):
    """各枝の両端の有効電力の和が正（枝は電力を消費する）であること。

    符号の規約「その端子から枝に流れ込む向きが正」が守られていれば、
    抵抗のある枝の :math:`\\mathrm{Re}(S_{ft} + S_{tf})` は必ず正になる。
    変圧器は ``r = 0`` なので損失ゼロ（許容差 1e-12）である。
    """
    for branch, (s_ft, s_tf) in zip(
        solution.case.branches, solution._branch_powers()
    ):
        loss = (s_ft + s_tf).real
        if branch.r == 0.0:
            assert abs(loss) < 1e-12, branch.label
        else:
            assert loss > 0.0, branch.label


def test_branch_flows_match_the_verified_table(solution):
    """枝潮流が API 契約の検証済みの表と一致すること。

    表は独立な実装で確認された値であり、掲載は小数 4 桁。許容差 1e-4 は
    その丸め（実測の最大差は |S| で 4.6e-05、P で 4.6e-05）。
    """
    magnitudes = solution.apparent_flows()
    flows = solution.branch_flows()
    for key, (expected_s, expected_p, _) in VERIFIED_FLOWS.items():
        assert magnitudes[key] == pytest.approx(expected_s, abs=1e-4), key
        assert flows[key][0].real == pytest.approx(expected_p, abs=1e-4), key


def test_loading_matches_the_verified_table(case, solution):
    """負荷率が検証済みの表と一致すること（rate_a に対する百分率）。

    許容差 0.15 ポイントは、表が小数 1 桁で丸められていること（枝 4-5 の
    56.1% は実測 56.14%）による。負荷率は **皮相電力**を熱容量で割った
    値であって、直流潮流の有効電力ではない。
    """
    loading = solution.loading()
    for key, (expected_s, _, expected_percent) in VERIFIED_FLOWS.items():
        assert loading[key] * 100.0 == pytest.approx(expected_percent, abs=0.15), key
        # 定義の確認: |S| / rate_a であること。
        rating = getattr(case.branches[[b.key() for b in case.branches].index(key)], "rate_a")
        assert loading[key] == pytest.approx(
            solution.apparent_flows()[key] / rating, abs=1e-12
        )
    assert max(loading.values()) < 1.0        # 基準状態は熱容量に対して健全


def test_apparent_flow_takes_the_larger_end(solution):
    """``apparent_flows`` が両端の大きい方を採っていること。

    線路の損失と充電容量のぶんだけ両端の |S| は異なる。小さい方を
    採ると過負荷を見落とす。
    """
    for branch, (s_ft, s_tf) in zip(solution.case.branches, solution._branch_powers()):
        magnitude = solution.apparent_flows()[branch.key()]
        assert magnitude == pytest.approx(max(abs(s_ft), abs(s_tf)), abs=1e-15)
        assert magnitude >= min(abs(s_ft), abs(s_tf))
    # 充電容量のある線路では両端が実際に食い違う（同じなら比較の意味がない）。
    line = [branch for branch in solution.case.branches if branch.b > 0][0]
    index = list(solution.case.branches).index(line)
    s_ft, s_tf = solution._branch_powers()[index]
    assert abs(abs(s_ft) - abs(s_tf)) > 1e-3


# ======================================================================
# 5. N-1 と逸脱の判定
# ======================================================================
def test_n1_results_match_the_verified_table(case):
    """6 つの想定事故の結果が API 契約の検証済みの表と一致すること。

    最悪の枝・その負荷率（rate_b に対する）・最低電圧の母線とその値を
    まとめて突き合わせる。表は独立な実装で確認された値である。許容差は
    負荷率が 0.15 ポイント（表は小数 1 桁）、電圧が 1e-4（同 4 桁）。
    """
    for key, (worst, percent, low_bus, low_voltage) in VERIFIED_N1.items():
        result = solve(_outage(case, key))
        loading = result.loading("rate_b")
        worst_key = max(loading, key=loading.get)
        bus_id, magnitude = result.min_voltage()

        assert worst_key == worst, f"開放 {key}: 最悪の枝が違う"
        assert loading[worst_key] * 100.0 == pytest.approx(percent, abs=0.15), key
        assert bus_id == low_bus, f"開放 {key}: 最低電圧の母線が違う"
        assert magnitude == pytest.approx(low_voltage, abs=1e-4), key


def test_thermal_screening_misses_the_voltage_violation(case):
    """枝 4-6 の開放は熱容量では健全なのに電圧の下限を割ること。

    第 09 回の主教材。最悪の枝でも rate_b の 75.7% で熱容量は健全だが、
    母線 6 の電圧が 0.9418 p.u. まで落ちて下限 0.95 を割る。**熱容量
    だけを見る N-1 スクリーニングはこの事故を「健全」と誤判定する。**
    ``violations`` が両方を返すのはこのためである。
    """
    result = solve(_outage(case, (4, 6)))
    loading = result.loading("rate_b")
    assert max(loading.values()) < 1.0            # 熱容量は健全

    messages = result.violations("rate_b")
    assert messages, "電圧の逸脱が報告されていない"
    assert all("枝" not in message for message in messages)
    assert any("母線 6" in message and "下限" in message for message in messages)
    assert result.min_voltage() == (6, pytest.approx(0.9418, abs=1e-4))


def test_base_case_has_no_violations(solution):
    """基準状態では熱容量にも電圧にも逸脱がないこと。"""
    assert solution.violations() == []
    assert solution.violations("rate_b") == []


def test_without_branch_drops_the_reference_generation(case):
    """``Case.without_branch`` が参照解を落とすことを固定する（落とし穴）。

    :meth:`Case.without_branch` は ``reference=None`` にするので、
    :meth:`Case.bus_injection` が **発電ゼロ**を返す。そのまま潮流を
    解くと「slack 1 台で全負荷を賄う」別の系統を解くことになり、
    WSCC 9 母線では収束しない。N-1 の評価では発電機の出力を事故前の
    まま据え置くのが前提なので、枝だけを差し替えるか ``dispatch`` を
    与えること（本テストの ``_outage`` ヘルパがそれである）。
    """
    stripped = case.without_branch((4, 5))
    assert stripped.reference is None
    p_stripped, _ = stripped.bus_injection()
    assert p_stripped[stripped.index_of(2)] == pytest.approx(0.0)

    p_kept, _ = _outage(case, (4, 5)).bus_injection()
    assert p_kept[case.index_of(2)] == pytest.approx(1.630)


# ======================================================================
# 6. 無効電力の上下限
# ======================================================================
def test_q_limit_switches_pv_bus_to_pq(case):
    """Q が上限を超える PV 母線が PQ に切り替わり、電圧が下がること。

    母線 2 の 2 号機の ``q_max`` を 0.02 p.u. ずつ（合計 0.04）に絞ると、
    基準解で必要な 0.067 p.u. を出せなくなる。このとき発電機の Q は
    上限に張り付き、母線 2 の電圧は設定値 1.025 を **保てなくなって
    下がる**。これは近似ではなく物理そのものである。
    """
    tightened = replace(
        case,
        units=[
            replace(unit, q_max=0.02) if unit.bus == 2 else unit for unit in case.units
        ],
    )
    limited = solve(tightened, enforce_q_limits=True)
    index = tightened.index_of(2)
    _, q = limited.injections()

    assert limited.q_limited == (2,)
    assert q[index] == pytest.approx(0.04, abs=1e-9)     # 上限に張り付いている
    assert limited.v[index] < 1.025 - 1e-3               # 設定値を保てない
    # 有効電力の指定は変わらない（PV 母線が失うのは電圧の支持だけ）。
    p, _ = limited.injections()
    assert p[index] == pytest.approx(1.630, abs=1e-9)
    assert "Q 制限" in limited.summary()


def test_q_limits_are_inactive_when_generators_have_headroom(case, solution):
    """余裕があるときは ``enforce_q_limits`` が解を変えないこと。

    WSCC 9 母線の基準状態では、母線 2 の Q = 0.067 も母線 3 の
    Q = -0.109 も可能範囲の内側にある。切り替えが起きないことと、
    解が 1 ビットも変わらないことを確かめる。
    """
    enforced = solve(case, enforce_q_limits=True)
    assert enforced.q_limited == ()
    assert enforced.v == pytest.approx(solution.v, abs=0.0)
    assert enforced.theta == pytest.approx(solution.theta, abs=0.0)


# ======================================================================
# 7. 収束しないとき
# ======================================================================
def _heavy(case: Case, factor: float) -> Case:
    """発電を据え置いたまま負荷だけを ``factor`` 倍する。"""
    return replace(
        case,
        buses=[replace(bus, pd=bus.pd * factor, qd=bus.qd * factor) for bus in case.buses],
    )


def test_diverging_case_raises_japanese_runtime_error(case):
    """負荷 5 倍で収束せず、切り分けの手掛かりを含む例外が出ること。

    メッセージには (a) 最大ミスマッチとその母線、(b) 発電機の無効電力と
    上下限、(c) 島の有無 が並ぶ。この 3 つがあれば「解が無い」のか
    「初期値が悪い」のかを学生が自分で切り分けられる。
    """
    with pytest.raises(RuntimeError, match="収束しなかった") as error:
        solve(_heavy(case, 5.0))

    message = str(error.value)
    assert "最大ミスマッチ" in message
    assert "発電機の無効電力" in message
    assert "上限" in message                      # 制限を超えた発電機が挙がる
    assert "トポロジー" in message
    assert "系統は連結" in message                # 島ではないことも言う
    assert "ミスマッチの推移" in message
    assert "Case.check()" in message


def test_islanded_case_reports_the_topology(case):
    """橋を開放したケースで、島に分かれている事実が報告されること。

    枝 1-4 は橋なので、開放すると slack 母線 1 が単独の島になる。
    これは「ソルバの失敗」ではなく **位相の事実**であり、メッセージも
    そう読めるようになっていなければならない。
    """
    with pytest.raises(RuntimeError, match="島に分かれている"):
        solve(_outage(case, (1, 4)))


def test_max_iter_is_respected(case):
    """``max_iter`` で打ち切られたら例外になること（黙って返さない）。"""
    with pytest.raises(RuntimeError, match="1 回の反復で収束しなかった"):
        solve(case, max_iter=1)


def test_gauss_seidel_diverges_with_too_large_acceleration(case):
    """加速係数を大きくしすぎると発散すること。

    :math:`\\alpha \\ge 2` はほぼ必ず発散する。既定の 1.6 が「安全側に
    寄せた実用値」であることを、失敗の側から示しておく。
    """
    with pytest.raises(RuntimeError):
        solve(case, "gauss_seidel", acceleration=2.4, max_iter=200)


# ======================================================================
# 8. 引数の検査と入口の振る舞い
# ======================================================================
def test_rejects_unknown_method(case):
    """解法の名前が違えば日本語の ValueError で止まること。"""
    with pytest.raises(ValueError, match="method='nr' は使えない"):
        solve(case, "nr")


def test_rejects_non_positive_acceleration(case):
    with pytest.raises(ValueError, match="加速係数"):
        solve(case, "gauss_seidel", acceleration=0.0)


def test_rejects_initial_value_of_wrong_length(case):
    with pytest.raises(ValueError, match="母線数"):
        solve(case, v0=[1.0, 1.0, 1.0])


def test_rejects_unknown_limit_name(solution):
    with pytest.raises(ValueError, match="rate_a"):
        solution.loading("rate_c")


def test_missing_slack_is_rejected(case):
    """slack 母線がないケースは潮流計算が定義できないこと。"""
    no_slack = replace(
        case,
        buses=[
            replace(bus, type=BusType.PV) if bus.type is BusType.SLACK else bus
            for bus in case.buses
        ],
    )
    with pytest.raises(ValueError, match="slack 母線"):
        solve(no_slack)


def test_dc_method_delegates_to_gridops_dc(case):
    """``method="dc"`` が :mod:`gridops.dc` に委譲されること。

    直流解は :math:`|V| = 1`、反復なしの直接解法である。位相は交流解と
    1 deg 以内で一致する（実測の最大差 0.52 deg, 母線 2）。これが直流
    近似の精度そのものであり、**枝潮流は交流の式を通しても直流の値には
    ならない**（充電容量と損失のぶんずれる）。
    """
    dc = pytest.importorskip("gridops.dc")
    result = solve(case, "dc")
    linear = dc.dc_powerflow(case)

    assert result.method == "dc"
    assert result.converged and result.iterations == 0
    assert result.v == pytest.approx(np.ones(case.n_bus), abs=0.0)
    assert result.theta == pytest.approx(np.asarray(linear.theta), abs=1e-12)

    reference = solve(case)
    assert np.max(np.abs(result.angle_deg - reference.angle_deg)) < 1.0


def test_summary_reports_the_essentials(solution):
    """``summary`` に収束・反復回数・slack 出力・損失・最低電圧が出ること。"""
    text = solution.summary()
    assert "newton" in text and "収束" in text
    assert "slack 出力" in text and "0.716410" in text
    assert "総損失" in text and "0.046410" in text
    assert "最低電圧" in text and "母線 5" in text
