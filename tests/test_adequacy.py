"""アデカシー評価（第 18 回）の検証。

このモジュールの正しさは、**gridops の外にある基準**と突き合わせて確かめる。

1. 同一容量の号機だけなら :func:`scipy.stats.binom.pmf` と厳密に一致する
   （畳み込みの実装とは式の出所がまったく違う。これが一番強い基準）
2. 7 号機の :math:`2^7 = 128` 通りの **全列挙**と一致する
   （同じ確率モデルを、まとめずに素朴に数え上げた別実装）
3. 不変量（確率の総和 1、期待停止容量 :math:`\\sum P_{max,i} FOR_i`）
4. 2 状態しかない系統に対する **手計算**の LOLE と EUE
5. モンテカルロは点推定の一致ではなく、**95% 信頼区間が解析解を含むこと**と
   変動係数が :math:`\\beta = \\sqrt{(1-p)/(pN)}` に合うことで確かめる

実装の出力を実装で再計算するだけのテストは書かない。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import binom

from gridops import Case, load_case
from gridops.case import Unit
from gridops.adequacy import (
    CapacityOutageTable,
    annual_load,
    capacity_outage_table,
    elcc,
    eue,
    load_duration_curve,
    lole,
    lolp,
    monte_carlo_adequacy,
)

#: モンテカルロを回すときの需要のピーク [MW]。設備容量 460 MW に対して
#: 380 MW まで持ち上げてあるのは、LOLP を 3e-3 程度にして 10 万標本でも
#: 正規近似が効く（不足の標本が 300 件以上出る）ようにするためである。
STRESSED_PEAK_MW = 380.0


def _unit(name: str, capacity_mw: float, outage_rate: float) -> Unit:
    """アデカシーに必要な 2 つの諸元だけを持つ号機を作る。"""
    return Unit(
        name=name, bus=1, p_max_mw=capacity_mw, forced_outage_rate=outage_rate
    )


@pytest.fixture(scope="module")
def case() -> Case:
    return load_case("wscc9")


@pytest.fixture(scope="module")
def copt(case) -> CapacityOutageTable:
    return capacity_outage_table(case.units)


@pytest.fixture(scope="module")
def stressed_load(case) -> np.ndarray:
    """LOLP がおよそ 3.5e-3 になる年間需要。"""
    return annual_load(case, peak_mw=STRESSED_PEAK_MW)


# ----------------------------------------------------------------------
# 1. 二項分布との厳密一致（最重要の独立基準）
# ----------------------------------------------------------------------
def test_identical_units_match_binomial_pmf():
    """同一容量 5 台の COPT が二項分布に厳密に一致すること。

    容量 :math:`C`、強制停止率 :math:`p` の号機が :math:`n` 台なら、
    停止容量 :math:`kC` の確率は :math:`\\binom{n}{k} p^k (1-p)^{n-k}` に
    なる。畳み込みの実装と ``scipy.stats.binom`` は式の出所がまったく
    違うので、これが独立基準として最も強い。許容差 1e-14 は倍精度の
    丸め（確率は 1e-1 の桁なので相対 1e-16 が数十回積み上がる程度）で
    決めた。
    """
    units = [_unit(f"U{i}", 40.0, 0.01) for i in range(5)]
    table = capacity_outage_table(units)

    assert table.installed_mw == 200.0
    assert table.outage_mw == pytest.approx([0.0, 40.0, 80.0, 120.0, 160.0, 200.0])

    expected = binom.pmf(np.arange(6), 5, 0.01)
    assert np.max(np.abs(table.probability - expected)) < 1e-14


def test_cumulative_matches_binomial_survival():
    """累積確率 P(停止容量 >= x) が二項分布の生存関数に一致すること。

    停止台数が :math:`k` 台以上である確率は ``binom.sf(k-1, n, p)``。
    ``cumulative`` と ``probability_of_at_least`` の 2 つの入口を同じ
    基準で確かめる。
    """
    units = [_unit(f"U{i}", 40.0, 0.01) for i in range(5)]
    table = capacity_outage_table(units)

    expected = binom.sf(np.arange(6) - 1, 5, 0.01)
    assert np.max(np.abs(table.cumulative - expected)) < 1e-14

    for k in range(6):
        assert table.probability_of_at_least(40.0 * k) == pytest.approx(
            expected[k], abs=1e-14
        )
    # 表にない値でも引ける（39.5 MW 以上 = 1 台以上停止）。
    assert table.probability_of_at_least(39.5) == pytest.approx(expected[1], abs=1e-14)


# ----------------------------------------------------------------------
# 2. 全列挙との一致
# ----------------------------------------------------------------------
def test_seven_units_match_full_enumeration(case, copt):
    """7 号機の 128 通りの全列挙と畳み込みが一致すること。

    畳み込みは同じ停止容量の状態をその場でまとめるが、全列挙は
    :math:`2^7` 通りの入切の組をそのまま数え上げる。アルゴリズムが
    違うので独立な基準になる。許容差 1e-12 は 128 個の積を足し上げる
    際の丸めの上限として取った（実測の差は 1e-17 の桁）。
    """
    capacities = [u.p_max_mw for u in case.units]
    rates = [u.forced_outage_rate for u in case.units]

    enumerated: dict[float, float] = {}
    for mask in range(2 ** len(capacities)):
        probability = 1.0
        outage = 0.0
        for i, (capacity, rate) in enumerate(zip(capacities, rates)):
            if mask >> i & 1:
                probability *= rate
                outage += capacity
            else:
                probability *= 1.0 - rate
        enumerated[outage] = enumerated.get(outage, 0.0) + probability

    assert copt.outage_mw.size == len(enumerated)
    for outage, probability in zip(copt.outage_mw, copt.probability):
        assert probability == pytest.approx(enumerated[outage], abs=1e-12)


# ----------------------------------------------------------------------
# 3. 不変量
# ----------------------------------------------------------------------
def test_probabilities_sum_to_one(copt):
    """確率の総和が 1 であること。"""
    assert copt.probability.sum() == pytest.approx(1.0, abs=1e-14)
    assert np.all(copt.probability >= 0.0)
    assert np.all(np.diff(copt.outage_mw) > 0.0)      # 昇順で重複がない


def test_expected_outage_matches_sum_of_capacity_times_for(case, copt):
    """期待停止容量が Sum(Pmax_i * FOR_i) に一致すること。

    独立性から導かれる不変量で、表の中身がどう並んでいても成り立つ。
    教材ケースでは 3*60*0.04 + 2*90*0.05 + 2*50*0.07 = 23.2 MW。
    """
    expected = sum(u.p_max_mw * u.forced_outage_rate for u in case.units)
    assert expected == pytest.approx(23.2, abs=1e-12)
    assert copt.expected_outage_mw() == pytest.approx(expected, abs=1e-12)


def test_available_capacity_is_installed_minus_outage(copt):
    """利用可能容量が設備容量 - 停止容量であること。"""
    assert copt.available_mw() == pytest.approx(copt.installed_mw - copt.outage_mw)
    assert copt.available_mw()[0] == pytest.approx(460.0)


# ----------------------------------------------------------------------
# 4. 丸め
# ----------------------------------------------------------------------
def test_rounding_preserves_probability_and_expectation():
    """丸めても確率の総和と期待停止容量が保存されること。

    格子の間の状態を「近い方に丸める」のではなく両隣に **内分**して
    配っているので、重み付き平均が元の容量に戻り期待値が保存される。
    容量が 10 MW の倍数でない 4 台で確かめる（倍数だと丸めが何もせず
    テストが素通りしてしまう）。
    """
    units = [
        _unit("A", 23.0, 0.05),
        _unit("B", 37.0, 0.08),
        _unit("C", 41.0, 0.10),
        _unit("D", 55.0, 0.04),
    ]
    exact = capacity_outage_table(units)
    rounded = capacity_outage_table(units, rounding_mw=10.0)

    assert rounded.probability.sum() == pytest.approx(1.0, abs=1e-14)
    assert rounded.expected_outage_mw() == pytest.approx(
        exact.expected_outage_mw(), abs=1e-12
    )
    assert rounded.expected_outage_mw() == pytest.approx(10.41, abs=1e-12)
    # 状態はすべて 10 MW の格子の上に乗る。
    grid = rounded.outage_mw / 10.0
    assert np.max(np.abs(grid - np.round(grid))) < 1e-9
    # LOLP は保存されない（丸めは近似であって整理ではない）。
    load = exact.installed_mw - 40.0
    assert lolp(rounded, load) != pytest.approx(lolp(exact, load), rel=1e-6)


def test_rounding_reduces_the_number_of_states():
    """丸めの目的（状態数を抑えること）が実際に果たされること。

    容量が互いに素に近い 12 台では、部分和がほとんど重ならないので
    状態数が 399 まで膨らむ。10 MW の格子に載せると 57 状態で収まる。

    格子の点数（設備容量 496 MW / 10 + 1 = 51）より少し多いのは、
    丸めを 1 台ごとに行うため最大の停止容量が設備容量を超えて
    はみ出すからである（台数 x 刻みが上限）。はみ出した状態の確率は
    合計しても 1e-15 の桁にしかならないが、利用可能容量が負になる
    状態が表に載ることは知っておく必要がある。
    """
    capacities = [23, 37, 41, 55, 29, 61, 17, 43, 59, 31, 47, 53]
    units = [_unit(f"U{i}", float(c), 0.05) for i, c in enumerate(capacities)]

    exact = capacity_outage_table(units)
    rounded = capacity_outage_table(units, rounding_mw=10.0)

    assert exact.outage_mw.size > 300
    assert rounded.outage_mw.size < exact.outage_mw.size / 5
    # 格子の点数 + はみ出し（台数 x 刻み）が上限。
    assert rounded.outage_mw.size <= int(
        (exact.installed_mw + len(units) * 10.0) / 10.0
    ) + 1
    assert rounded.outage_mw.max() > exact.installed_mw
    assert rounded.probability[rounded.outage_mw > exact.installed_mw].sum() < 1e-12
    assert rounded.expected_outage_mw() == pytest.approx(
        exact.expected_outage_mw(), abs=1e-10
    )


def test_rounding_underestimates_lolp_at_a_grid_threshold():
    """丸めが LOLP を過小評価側にずらすこと（向きまで固定する）。

    しきい値 :math:`x = 設備容量 - 需要` が格子点の上にあるとき、
    :math:`x` と次の格子点の間にあった状態は確率の一部を :math:`x` 自身へ
    落とす。:math:`x` ちょうどは「不足」に数えない規約なので、その分だけ
    :math:`P(停止容量 > x)` が減る。**丸めは LOLP を必ず減らす側にずらす。**

    ここでは 60/60/90 MW（すべて格子の倍数）に 41 MW を 1 台だけ足し、
    格子から外れる丸めが 1 回だけ起きるようにしてある。この構成なら
    上の議論がそのまま成り立つ。容量が格子から外れたまま何段も丸めると
    偏りの向きは保証できない（丸めた表の数値をそのまま設備計画に
    使ってはいけない理由がこれである）。
    """
    units = [
        _unit("A", 60.0, 0.04),
        _unit("B", 60.0, 0.04),
        _unit("C", 90.0, 0.05),
        _unit("D", 41.0, 0.10),
    ]
    exact = capacity_outage_table(units)
    rounded = capacity_outage_table(units, rounding_mw=10.0)

    # x = 40 MW: 41 MW の状態がちょうど 40 と 50 の間にあるので確率が落ちる。
    load = exact.installed_mw - 40.0
    assert lolp(rounded, load) < lolp(exact, load)
    assert lolp(exact, load) == pytest.approx(0.212032, abs=1e-12)

    # x = 50, 60 MW: (50, 60) と (60, 70) に状態が無いのでずれない。
    for threshold in (50.0, 60.0):
        level = exact.installed_mw - threshold
        assert lolp(rounded, level) == pytest.approx(lolp(exact, level), abs=1e-12)

    # 全格子点で「増える方向にはずれない」ことを確かめる。
    for threshold in np.arange(0.0, exact.installed_mw, 10.0):
        level = exact.installed_mw - threshold
        assert lolp(rounded, level) <= lolp(exact, level) + 1e-12


def test_rounding_step_must_be_positive():
    """丸めの刻みが非正なら日本語で止まること。"""
    units = [_unit("A", 60.0, 0.04)]
    with pytest.raises(ValueError, match="rounding_mw"):
        capacity_outage_table(units, rounding_mw=0.0)


# ----------------------------------------------------------------------
# 5. LOLP の規約
# ----------------------------------------------------------------------
def test_lolp_excludes_equality():
    """利用可能容量が需要にちょうど等しい状態を不足に数えないこと。

    離散容量では等号が実際に起こる。100 MW（必ず健全）+ 50 MW（FOR 0.1）
    の系統に 100 MW の需要を与えると、50 MW 機が止まった状態の
    利用可能容量がちょうど 100 MW になる。この規約は文献ごとに違うので
    テストで固定しておく。
    """
    units = [_unit("firm", 100.0, 0.0), _unit("var", 50.0, 0.1)]
    table = capacity_outage_table(units)

    assert lolp(table, 100.0) == 0.0            # 等号は不足でない
    assert lolp(table, 100.000001) == pytest.approx(0.1, abs=1e-12)
    assert lolp(table, 150.0) == pytest.approx(0.1, abs=1e-12)   # 150 も等号
    assert lolp(table, 150.000001) == pytest.approx(1.0, abs=1e-12)


def test_lolp_matches_binomial_cdf():
    """LOLP が二項分布の累積分布に一致すること。

    40 MW x 5 台（FOR 0.01）で需要 100 MW なら、40k < 100 すなわち
    健全 2 台以下が不足である。等号を含めない規約なので、需要 120 MW
    でも「健全 3 台 = 120 MW」は不足に数えず、しきい値は変わらない。
    """
    units = [_unit(f"U{i}", 40.0, 0.01) for i in range(5)]
    table = capacity_outage_table(units)

    expected = binom.cdf(2, 5, 0.99)
    assert lolp(table, 100.0) == pytest.approx(expected, abs=1e-14)
    assert lolp(table, 120.0) == pytest.approx(expected, abs=1e-14)
    assert lolp(table, 120.5) == pytest.approx(binom.cdf(3, 5, 0.99), abs=1e-14)


# ----------------------------------------------------------------------
# 6. LOLE の単位
# ----------------------------------------------------------------------
def test_lole_days_is_not_hours_divided_by_24(copt, stressed_load):
    """LOLE の "days" が "hours" の 1/24 でないこと。

    "days" は日ごとの **最大**需要に対する LOLP の和である。LOLP は
    需要について単調非減少だから日ピークの LOLP は日平均以上であり、
    lole(days) >= lole(hours)/24 が常に成り立つ。教材ケースでは
    3 倍以上開く。「0.1 day/year」を「2.4 h/year」と読み替えては
    ならないことの数値的な裏づけ。
    """
    hours = lole(copt, stressed_load, unit="hours")
    days = lole(copt, stressed_load, unit="days")

    assert hours > 0.0 and days > 0.0
    assert days > hours / 24.0
    assert days / (hours / 24.0) > 2.0
    assert days != pytest.approx(hours / 24.0, rel=0.5)


def test_lole_days_equals_sum_of_daily_peak_lolp(copt, stressed_load):
    """"days" が日ピークの LOLP の和であることを、別に組んだ和で確かめる。

    実装は (n_day, 24) に整形してから最大を取るが、こちらは日ごとに
    スライスして最大を取り :func:`lolp` を素朴に足す。同じ定義の別実装。
    """
    daily = sum(
        lolp(copt, stressed_load[24 * d : 24 * (d + 1)].max()) for d in range(365)
    )
    assert lole(copt, stressed_load, unit="days") == pytest.approx(daily, abs=1e-12)


def test_lole_hours_equals_sum_of_hourly_lolp(copt):
    """"hours" が毎時 LOLP の素朴な和に一致すること。"""
    profile = np.linspace(300.0, 460.0, 48)
    assert lole(copt, profile) == pytest.approx(
        sum(lolp(copt, level) for level in profile), abs=1e-12
    )


def test_lole_days_requires_whole_days(copt):
    """"days" に 24 の倍数でない系列を渡すと日本語で止まること。"""
    with pytest.raises(ValueError, match="24 の倍数"):
        lole(copt, np.full(30, 300.0), unit="days")


def test_lole_rejects_unknown_unit(copt):
    """unit が 'hours' / 'days' 以外なら止まること。"""
    with pytest.raises(ValueError, match="'hours' か 'days'"):
        lole(copt, np.full(24, 300.0), unit="years")


# ----------------------------------------------------------------------
# 7. LOLE と EUE は別のものを測る
# ----------------------------------------------------------------------
def test_same_lole_but_different_eue():
    """LOLE が同じで EUE が 9 倍違う 2 つの系統を作れること。

    どちらも設備容量 200 MW、需要は 110 MW 一定、10 時間の期間である。

    ==========  ==================  ======  ==========  =====
    系統        構成                LOLE    不足の深さ  EUE
    ==========  ==================  ======  ==========  =====
    A           100(確実)+100(5%)   0.5 h   10 MW        5 MWh
    B            20(確実)+180(5%)   0.5 h   90 MW       45 MWh
    ==========  ==================  ======  ==========  =====

    **LOLE は不足の「深さ」を見ない。** 供給支障が起きる確率は同じでも、
    起きたときに 10 MW 足りないのと 90 MW 足りないのでは意味がまるで
    違う。設備計画を LOLE 単独で決めてはならないことの実演である。
    期待値はすべて手計算で、gridops の出力ではない。
    """
    load = np.full(10, 110.0)
    system_a = [_unit("firm", 100.0, 0.0), _unit("var", 100.0, 0.05)]
    system_b = [_unit("firm", 20.0, 0.0), _unit("var", 180.0, 0.05)]

    copt_a = capacity_outage_table(system_a)
    copt_b = capacity_outage_table(system_b)

    assert lole(copt_a, load) == pytest.approx(0.5, abs=1e-14)
    assert lole(copt_b, load) == pytest.approx(0.5, abs=1e-14)

    assert eue(copt_a, load) == pytest.approx(5.0, abs=1e-13)     # 0.05*10MW*10h
    assert eue(copt_b, load) == pytest.approx(45.0, abs=1e-13)    # 0.05*90MW*10h
    assert eue(copt_b, load) / eue(copt_a, load) == pytest.approx(9.0, abs=1e-12)


def test_eue_matches_hand_computed_value(copt):
    """EUE が状態ごとの手計算の和に一致すること。

    需要 1 点だけの系列に対して、表の全状態について
    ``P_i * max(D - A_i, 0)`` を素朴に足した値と比べる。
    """
    load = 420.0
    expected = float(
        np.sum(copt.probability * np.maximum(load - copt.available_mw(), 0.0))
    )
    assert eue(copt, [load]) == pytest.approx(expected, abs=1e-12)
    assert expected > 0.0


def test_load_duration_curve_is_a_sorted_permutation(stressed_load):
    """需要持続曲線が降順の並べ替えであること（元の系列は壊さない）。"""
    original = stressed_load.copy()
    curve = load_duration_curve(stressed_load)

    assert curve.shape == stressed_load.shape
    assert np.all(np.diff(curve) <= 0.0)
    assert np.array_equal(np.sort(curve), np.sort(stressed_load))
    assert np.array_equal(stressed_load, original)


# ----------------------------------------------------------------------
# 8. モンテカルロ
# ----------------------------------------------------------------------
def test_monte_carlo_interval_contains_the_analytic_lolp(case, copt, stressed_load):
    """95% 信頼区間が解析解（畳み込み）を含むこと。

    **点推定の一致では書かない。** 標本ごとに値が動くのが正しい振る舞い
    だからである。回帰値（seed=0 で固定）は実装が黙って変わったことを
    検出するためだけに置く。
    """
    result = monte_carlo_adequacy(
        case.units, stressed_load, n_samples=100_000, seed=0
    )
    exact_lolp = lole(copt, stressed_load) / stressed_load.size

    low, high = result.lolp_interval(0.95)
    assert low < exact_lolp < high

    exact_eue = eue(copt, stressed_load)
    assert abs(result.eue - exact_eue) < 1.96 * result.eue_stderr

    # seed 固定の回帰値（ドリフト検出）。
    assert result.n_samples == 100_000
    assert result.lolp == pytest.approx(0.00339, rel=1e-12)
    assert result.eue == pytest.approx(787.2039700855149, rel=1e-9)


def test_monte_carlo_interval_covers_the_analytic_value_across_seeds(
    case, copt, stressed_load
):
    """複数の seed で 95% 区間の被覆率が妥当であること。

    区間が「たまたま当たった」のではないことを確かめる。8 個の seed の
    うち少なくとも 6 個で解析解を含めば、95% 区間としておかしくない
    （3 個以上外れる確率は二項分布で 0.6% 未満）。
    """
    exact_lolp = lole(copt, stressed_load) / stressed_load.size
    covered = 0
    for seed in range(8):
        result = monte_carlo_adequacy(
            case.units, stressed_load, n_samples=50_000, seed=seed
        )
        low, high = result.lolp_interval(0.95)
        covered += int(low <= exact_lolp <= high)
    assert covered >= 6


def test_monte_carlo_coefficient_of_variation_matches_theory(
    case, copt, stressed_load
):
    """変動係数が beta = sqrt((1-p)/(pN)) の予測と相対 0.2 以内で合うこと。

    :math:`p` には解析解を入れる。標本から出した :math:`\\hat p` の
    ゆらぎ（相対で数%）が :math:`\\beta` に伝わるので、一致を要求できる
    のは相対 0.2 程度までである。標本数を 4 倍にすると変動係数が
    半分になること（:math:`1/\\sqrt{N}` の法則）も併せて固定する。
    """
    exact_lolp = lole(copt, stressed_load) / stressed_load.size

    for n_samples in (50_000, 200_000):
        result = monte_carlo_adequacy(
            case.units, stressed_load, n_samples=n_samples, seed=1
        )
        beta = math.sqrt((1.0 - exact_lolp) / (exact_lolp * n_samples))
        assert abs(result.coefficient_of_variation() - beta) / beta < 0.2

    coarse = monte_carlo_adequacy(case.units, stressed_load, n_samples=50_000, seed=1)
    fine = monte_carlo_adequacy(case.units, stressed_load, n_samples=200_000, seed=1)
    ratio = coarse.coefficient_of_variation() / fine.coefficient_of_variation()
    assert ratio == pytest.approx(2.0, rel=0.15)


def test_monte_carlo_is_reproducible(case, stressed_load):
    """同じ seed なら同じ結果、違う seed なら違う結果になること。"""
    first = monte_carlo_adequacy(case.units, stressed_load, n_samples=20_000, seed=3)
    second = monte_carlo_adequacy(case.units, stressed_load, n_samples=20_000, seed=3)
    other = monte_carlo_adequacy(case.units, stressed_load, n_samples=20_000, seed=4)

    assert first.lolp == second.lolp
    assert first.eue == second.eue
    assert first.lolp != other.lolp


def test_monte_carlo_block_splitting_does_not_change_the_stream(case, stressed_load):
    """ブロック分割の境界（10 万標本）をまたいでも結果が決まること。

    標本数を大きくしても使用メモリが増えないようにブロックに切って
    回しているので、境界の前後で結果が再現することを固定しておく。
    """
    a = monte_carlo_adequacy(case.units, stressed_load, n_samples=150_000, seed=7)
    b = monte_carlo_adequacy(case.units, stressed_load, n_samples=150_000, seed=7)
    assert a.lolp == b.lolp and a.eue == b.eue
    assert a.n_samples == 150_000


def test_monte_carlo_interval_and_cv_edge_cases():
    """信頼水準の検査と、不足がゼロのときの変動係数。"""
    units = [_unit("firm", 100.0, 0.0)]
    result = monte_carlo_adequacy(units, np.full(10, 50.0), n_samples=1000, seed=0)
    assert result.lolp == 0.0
    assert result.coefficient_of_variation() == math.inf
    with pytest.raises(ValueError, match="level"):
        result.lolp_interval(1.5)


# ----------------------------------------------------------------------
# 9. 年間需要の合成
# ----------------------------------------------------------------------
def test_annual_load_is_reproducible_and_scaled_to_the_peak(case):
    """同じ seed なら同じ系列、長さ 8760、最大値がちょうど peak_mw。

    最大値で割ってから peak を掛けているので、規格化は厳密である。
    「年間最大需要を固定して設備構成を比べる」議論を単純にするための
    性質なので、丸め誤差ではなく厳密一致で固定する。
    """
    first = annual_load(case)
    second = annual_load(case)

    assert first.shape == (8760,)
    assert np.array_equal(first, second)
    assert first.max() == 315.0                      # commitment.peak_mw
    assert first.min() > 0.0

    custom = annual_load(case, peak_mw=400.0, hours=48)
    assert custom.shape == (48,)
    assert custom.max() == 400.0


def test_annual_load_has_the_seasonal_and_weekend_structure(case):
    """季節変動と週末係数がケースの諸元どおりに効いていること。

    ``reliability.annual`` は weekend_factor = 0.88、
    seasonal_amplitude = 0.18 である。週末（先頭を月曜とみなした 6, 7
    日目）の平均が平日の 0.88 倍前後になり、冬（1 月）の平均が春（4 月）
    を上回ることを確かめる。雑音の分だけずれるので許容は緩い。
    """
    load = annual_load(case)
    day = np.arange(load.size) // 24
    weekend = day % 7 >= 5

    ratio = load[weekend].mean() / load[~weekend].mean()
    factor = case.reliability["annual"]["weekend_factor"]
    assert ratio == pytest.approx(factor, rel=0.02)

    january = load[day < 31].mean()
    april = load[(day >= 90) & (day < 121)].mean()
    july = load[(day >= 181) & (day < 212)].mean()
    assert january > april          # 冬が最大
    assert july > april             # 夏が第 2 の山、春が谷


def test_annual_load_requires_the_reliability_block():
    """reliability 層の無いケースでは Case.require が案内すること。"""
    bare = Case(name="bare")
    with pytest.raises(ValueError, match="reliability"):
        annual_load(bare)


# ----------------------------------------------------------------------
# 10. 等価容量価値
# ----------------------------------------------------------------------
def test_elcc_is_close_to_capacity_times_availability(case, stressed_load):
    """ELCC が 追加容量 x (1 - FOR) の前後に来ること。

    在来型の電源（出力が時刻に依存しない）なら、等価容量価値は
    :math:`P_{max}(1-FOR)` の近くに来るのが理論的な目安である。厳密に
    一致しないのは、需要の分布と COPT の階段の形に依存するためで、
    実際には目安をやや下回る（既設の信頼度が高いほど追加分の価値は
    逓減する）。範囲を緩く取っているのはそのためで、ここで確かめたいのは
    「設備容量そのものではなく、稼働率で割り引いた量になる」ことである。
    """
    for capacity, rate in ((50.0, 0.10), (60.0, 0.04)):
        new_unit = _unit("NEW", capacity, rate)
        value = elcc(case.units, stressed_load, new_unit)
        target = capacity * (1.0 - rate)

        assert 0.70 * target < value < 1.05 * target
        assert value < capacity          # 設備容量そのものにはならない


def test_elcc_restores_the_original_lole(case, stressed_load):
    """ELCC の定義（元の LOLE に戻る需要増分）を定義式で確かめる。

    求めた :math:`\\Delta` だけ需要を上げたとき、新電源を含む系統の
    LOLE が元の系統の LOLE を超えないこと。さらに :math:`\\Delta` を
    1 MW 上乗せすると超えること（二分法が上限まで詰めていること）。
    """
    new_unit = _unit("NEW", 50.0, 0.10)
    delta = elcc(case.units, stressed_load, new_unit, tol=0.05)

    base = capacity_outage_table(case.units)
    expanded = capacity_outage_table(list(case.units) + [new_unit])
    target = lole(base, stressed_load)

    assert lole(expanded, stressed_load + delta) <= target + 1e-12
    assert lole(expanded, stressed_load + delta + 1.0) > target


def test_elcc_warns_when_the_base_system_never_fails():
    """既設の LOLE がゼロなら警告すること（答えが意味を持たないため）。"""
    units = [_unit("firm", 500.0, 0.0), _unit("var", 100.0, 0.02)]
    load = np.full(240, 100.0)
    with pytest.warns(UserWarning, match="LOLE がゼロ"):
        value = elcc(units, load, _unit("NEW", 50.0, 0.1))
    assert value == pytest.approx(50.0)


def test_elcc_rejects_a_zero_capacity_unit(case, stressed_load):
    """容量ゼロの号機は等価容量価値を定義できないので止まること。"""
    with pytest.raises(ValueError, match="p_max_mw が非正"):
        elcc(case.units, stressed_load, _unit("ZERO", 0.0, 0.1))


# ----------------------------------------------------------------------
# 11. 入力の検査と表示
# ----------------------------------------------------------------------
def test_case_instead_of_units_raises_type_error(case):
    """Case をそのまま渡したら case.units を渡すよう案内すること。"""
    with pytest.raises(TypeError, match="case.units"):
        capacity_outage_table(case)


def test_empty_units_raise_value_error():
    """号機が空なら日本語で止まること。"""
    with pytest.raises(ValueError, match="号機が 1 台もない"):
        capacity_outage_table([])


def test_out_of_range_outage_rate_raises():
    """FOR が [0, 1] の外なら百分率との取り違えを疑わせて止まること。"""
    with pytest.raises(ValueError, match="百分率ではない"):
        capacity_outage_table([_unit("bad", 60.0, 4.0)])


def test_empty_load_profile_raises(copt):
    """需要系列が空なら止まること。"""
    with pytest.raises(ValueError, match="需要系列が空"):
        lole(copt, [])


def test_summaries_are_readable(case, copt, stressed_load):
    """summary() が例外なく主要な数値を含む文字列を返すこと。"""
    text = copt.summary()
    assert "460.0 MW" in text
    assert "23.2000 MW" in text

    result = monte_carlo_adequacy(case.units, stressed_load, n_samples=5_000, seed=0)
    report = result.summary()
    assert "LOLP" in report and "95% CI" in report


# ======================================================================
# 外部レビュー（2026-08-30）の回帰
# ======================================================================
def test_zero_event_interval_does_not_claim_certainty():
    """供給支障 0 件の信頼区間が ``(0, 0)`` にならないこと。

    外部レビューの指摘 #5。Wald 区間（正規近似）は 0 件のとき幅ゼロと
    なり「真の LOLP は確実にゼロ」という誤った断定になる。Wilson 区間は
    0 件でも上限 :math:`\\approx z^2/(N+z^2)` を残す。
    """
    from gridops.adequacy import MonteCarloResult

    result = MonteCarloResult(
        lolp=0.0, eue=0.0, n_samples=10_000, lolp_stderr=0.0, eue_stderr=0.0
    )
    low, high = result.lolp_interval()
    assert low == 0.0
    assert high > 0.0
    assert high == pytest.approx(1.96 ** 2 / (10_000 + 1.96 ** 2), rel=1e-3)


def test_wilson_interval_still_contains_the_analytic_answer():
    """Wilson 区間でも「解析解を含む」検証がそのまま成り立つこと。"""
    units = [_unit(f"U{i}", 50.0, 0.05) for i in range(4)]
    copt = capacity_outage_table(units)
    load = np.full(200, 130.0)
    analytic = lolp(copt, copt.installed_mw - (copt.installed_mw - 130.0))
    result = monte_carlo_adequacy(units, load, n_samples=20_000, seed=3)
    low, high = result.lolp_interval()
    assert low <= analytic <= high
