"""静的セキュリティ解析（第 09 回）の検証。

このモジュールの正しさは、**実装の外にある基準**と突き合わせて確かめる。
使う基準は次の 6 つである。

1. API 契約に載っている N-1 の表（交流潮流を独立に確認した値）。
   6 候補それぞれの最悪枝・負荷率・最低電圧母線・電圧を固定する。
2. 枝を消して :math:`B'` を組み直した直流潮流（:func:`dc_powerflow`）。
   LODF によるスクリーニングと 1e-12 で一致しなければならない。
3. 手計算できる性能指標（容量を 1.0 に揃えた小さな例。PI は暗算で出る）。
4. 優先順位法の手計算による経済配分（SCED の「制約なし」の基準）。
5. 線形計画の実行可能領域の包含関係（是正的 ⊇ 予防的）。
   費用の大小はこの包含から **論理的に**決まるので、数値が合う／合わない
   ではなく不等式そのものを固定する。
6. :func:`gridops.ybus.bridges` が独立に検出する橋。

**この回は誤判定そのものをテストに書いてある。** ``check_voltage=False``
にすると枝 4-6 の開放が「健全」と報告される。これはバグではなく、
熱容量だけを見る N-1 スクリーニングが実際に犯す誤りであり、教材として
固定する価値がある。同じ理由で、直流スクリーニングの閾値を 100% に
上げると枝 4-5 の開放（交流では 101.5% で逸脱）を取りこぼすことも
テストにしてある。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gridops import Case, load_case
from gridops.dc import dc_powerflow
from gridops.powerflow import solve as solve_powerflow
from gridops.security import (
    ContingencyResult,
    SCEDResult,
    SecurityReport,
    performance_index,
    sced,
    screen_n1,
)
from gridops.ybus import bridges

# ----------------------------------------------------------------------
# API 契約の N-1 の表（交流潮流で独立に確認済みの値）
#   開放する枝: (最悪の枝, rate_b 負荷率 [%], 最低電圧の母線, その電圧)
# ----------------------------------------------------------------------
CONTRACT_N1 = {
    (4, 5): ((5, 7), 101.5, 5, 0.8388),
    (4, 6): ((7, 8), 75.7, 6, 0.9418),
    (5, 7): ((7, 8), 112.5, 5, 0.9380),
    (6, 9): ((5, 7), 101.5, 6, 0.9639),
    (7, 8): ((5, 7), 112.4, 8, 0.9690),
    (8, 9): ((7, 8), 73.1, 8, 0.9783),
}

#: 枝ごとの N-1 最大負荷率 [%]（rate_b 基準、参照解の発電）。契約の値。
CONTRACT_WORST_BY_BRANCH = {
    (4, 5): 91.1,
    (4, 6): 90.5,
    (5, 7): 112.4,
    (6, 9): 88.2,
    (7, 8): 112.5,
    (8, 9): 88.3,
}

#: 橋（変圧器 3 本）。開放すると発電機母線が島になる。
BRIDGES = [(1, 4), (2, 7), (3, 9)]

#: 電圧の下限。ケースの voltage_limits と同じ値を独立に書いてある。
V_MIN_LIMIT = 0.95

#: 契約の表は小数 1 桁（負荷率）・小数 4 桁（電圧）で丸められている。
#: 丸めの幅はそれぞれ 0.05 ポイント・5e-5 なので、その 10 倍を許容差にする。
LOADING_TOL = 1e-3          # 負荷率（1.0 = 100%）の許容差
VOLTAGE_TOL = 1e-3          # 電圧 [p.u.] の許容差


@pytest.fixture(scope="module")
def case() -> Case:
    return load_case("wscc9")


@pytest.fixture(scope="module")
def report(case: Case) -> SecurityReport:
    """既定の設定（直流で絞り込み、交流で判定）の N-1 スクリーニング。"""
    return screen_n1(case)


# ======================================================================
# 1. スクリーニングと直接再計算の一致
# ======================================================================
def test_lodf_screening_matches_direct_dc_recomputation(case: Case) -> None:
    """LODF のスクリーニングを、枝を消して解き直した直流潮流と比べる。

    :func:`screen_n1` は事故前の直流潮流 1 回と LODF の掛け算だけで
    事故後潮流を出す。比較相手の :func:`dc_powerflow` は事故ごとに
    :math:`B'` を組み直して連立 1 次方程式を解き直す **別の定式化**で
    ある。両者が倍精度の丸め（1e-12）まで一致することが、
    「潮流を解き直さずに N-1 を掃ける」という直流近似の主張そのもの。
    """
    screened = screen_n1(case, method="lodf", check_voltage=False)
    assert screened.results, "候補が 1 件も無い"
    for result in screened.results:
        assert result.method == "lodf"
        direct = dc_powerflow(case, removed_branches=[result.outage])
        np.testing.assert_allclose(
            result.flows, np.abs(direct.flows), rtol=0.0, atol=1e-12
        )
        # 開放した枝の潮流は厳密にゼロでなければならない。
        index = [b.key() for b in case.branches].index(result.outage)
        assert result.flows[index] == 0.0


def test_screening_error_is_not_one_sided(case: Case, report: SecurityReport) -> None:
    """直流スクリーニングの誤差は **安全側に偏っていない**。

    定格は皮相電力 :math:`|S|` の制限なのに直流が持っているのは有効電力
    :math:`P` だけなので、直流はふつう負荷率を過小評価する（6 事故中
    5 件でそうなる。枝 4-5 の開放では 86.2% 対 101.5% で 15 ポイントの
    差）。しかし直流は :math:`P` そのものも近似しているので、**上に
    外れることもある**（枝 6-9 の開放では 102.1% 対 101.5%）。

    つまり「直流の値に一律の余裕を足せば安全側になる」とは言えない。
    余裕の取り方ではなく、**交流で解き直すこと**でしか判定はできない。
    """
    for result in report.results:
        assert math.isfinite(result.screening_loading)
    errors = {
        result.outage: result.worst_loading - result.screening_loading
        for result in report.results
    }
    assert sum(1 for value in errors.values() if value > 0) == 5   # 過小評価
    assert sum(1 for value in errors.values() if value < 0) == 1   # 過大評価
    assert errors[(4, 5)] == pytest.approx(0.1534, abs=1e-3)
    assert errors[(6, 9)] == pytest.approx(-0.0059, abs=1e-3)

    # 枝 4-5 の開放では、直流 86.2% に対し交流 101.5%。閾値を 100% に
    # 置くと「健全」に見えて交流に回らない、という差である。
    result = next(r for r in report.results if r.outage == (4, 5))
    assert result.screening_loading < 1.0 < result.worst_loading


# ======================================================================
# 2. 契約の N-1 の表を交流で再現する
# ======================================================================
@pytest.mark.parametrize("outage", sorted(CONTRACT_N1))
def test_ac_n1_reproduces_the_contract_table(case: Case, outage) -> None:
    """6 候補それぞれの最悪枝・負荷率・最低電圧母線・電圧を契約と比べる。

    契約の値は本実装とは独立に（教科書の潮流解を出発点にした
    Newton-Raphson で）確認されたものである。
    """
    worst_branch, worst_percent, v_bus, v_value = CONTRACT_N1[outage]
    report = screen_n1(case, method="ac", contingencies=[outage])
    assert len(report.results) == 1
    result = report.results[0]

    assert result.outage == outage
    assert result.converged is True
    assert result.worst_branch == worst_branch
    assert result.worst_loading == pytest.approx(worst_percent / 100.0, abs=LOADING_TOL)
    assert result.v_min_bus == v_bus
    assert result.v_min == pytest.approx(v_value, abs=VOLTAGE_TOL)


def test_worst_loading_by_branch_matches_the_contract(report: SecurityReport) -> None:
    """枝ごとの N-1 最大負荷率が契約の 6 つの値と一致する。

    「どの事故が重いか」ではなく「どの設備が運転点を縛るか」の表である。
    """
    worst = report.worst_loading_by_branch()
    for key, percent in CONTRACT_WORST_BY_BRANCH.items():
        assert worst[key] == pytest.approx(percent / 100.0, abs=LOADING_TOL)


def test_two_stage_screening_agrees_with_full_ac(case: Case, report: SecurityReport) -> None:
    """既定の 2 段構え（lodf + 交流判定）は全件交流と同じ結論に至る。

    絞り込みの閾値が既定の 0.0（= 全候補を交流で解き直す）である限り、
    ``method`` は結論を変えない。**変えないことを確かめておかないと、
    閾値を上げたときの取りこぼしが「method の違い」に紛れてしまう。**
    """
    full = screen_n1(case, method="ac")
    by_outage = {r.outage: r for r in full.results}
    for result in report.results:
        other = by_outage[result.outage]
        assert result.worst_branch == other.worst_branch
        assert result.worst_loading == pytest.approx(other.worst_loading, rel=1e-12)
        assert result.v_min == pytest.approx(other.v_min, rel=1e-12)
        assert result.is_secure == other.is_secure


# ======================================================================
# 3. 主教材: 熱容量は健全なのに電圧が下限を割る
# ======================================================================
def test_outage_4_6_is_thermally_secure_but_voltage_insecure(report: SecurityReport) -> None:
    """枝 4-6 の開放は熱容量では健全、電圧では逸脱である。

    第 09 回の主教材そのもの。最悪の枝でも rate_b の 75.7% で熱容量には
    24 ポイントの余裕があるのに、母線 6 の電圧が 0.9418 p.u. まで落ちて
    下限 0.95 を割る。**熱容量と電圧は独立に見なければならない。**
    """
    result = next(r for r in report.results if r.outage == (4, 6))

    assert result.thermal_secure is True
    assert result.worst_loading == pytest.approx(0.757, abs=LOADING_TOL)

    assert result.voltage_secure is False
    assert result.v_min_bus == 6
    assert result.v_min == pytest.approx(0.9418, abs=VOLTAGE_TOL)
    assert result.v_min < V_MIN_LIMIT

    assert result.is_secure is False
    # 逸脱の一覧には電圧の話だけが載り、枝の過負荷は 1 件も無い。
    problems = result.violations()
    assert len(problems) == 1
    assert "母線 6" in problems[0]
    assert "下限" in problems[0]


def test_outage_4_5_overloads_so_it_is_not_the_thermal_secure_example(
    report: SecurityReport,
) -> None:
    """枝 4-5 の開放は電圧が最も低いが、熱容量でも逸脱している。

    「電圧が低い事故」と「熱容量では健全なのに電圧が低い事故」を
    取り違えないための対照実験である。4-5 は母線 5 が 0.8388 p.u. と
    6 事故で最も低いが、枝 5-7 が 101.5% で過負荷も起こしているので、
    熱容量だけを見るスクリーニングでも捕まる。
    """
    result = next(r for r in report.results if r.outage == (4, 5))
    assert result.thermal_secure is False
    assert result.voltage_secure is False
    assert result.v_min == pytest.approx(0.8388, abs=VOLTAGE_TOL)


# ======================================================================
# 4. 誤判定そのものを固定する
# ======================================================================
@pytest.mark.parametrize("method", ["lodf", "ac"])
def test_check_voltage_false_declares_outage_4_6_secure(case: Case, method: str) -> None:
    """``check_voltage=False`` は枝 4-6 の開放を「健全」と誤判定する。

    **これはバグではない。** 熱容量だけを見る N-1 スクリーニングが実際に
    犯す誤りであり、第 09 回はこの誤りを見せるための回である。
    ``method="ac"`` でも同じ誤判定が起きる（電圧の値そのものは結果に
    残っているのに、判定には使われない）ことまで固定してある。
    """
    report = screen_n1(case, method=method, check_voltage=False)
    result = next(r for r in report.results if r.outage == (4, 6))

    assert result.voltage_checked is False
    assert result.thermal_secure is True
    assert result.voltage_secure is True      # ← 見ていないので「健全」
    assert result.is_secure is True           # ← 誤判定

    if method == "ac":
        # 交流は解いたので電圧の値は残っている。それでも判定は健全。
        assert result.v_min == pytest.approx(0.9418, abs=VOLTAGE_TOL)
        assert result.v_min < V_MIN_LIMIT
    # 逸脱の一覧には「電圧を見ていない」という但し書きが必ず入る。
    assert any("電圧は見ていない" in message for message in result.violations())


def test_raising_the_screening_threshold_misses_outage_4_5(case: Case) -> None:
    """絞り込みの閾値を 100% に上げると、交流で逸脱する事故を取りこぼす。

    枝 4-5 の開放は直流の最悪負荷率が 86.2% なので「直流が定格を超えた
    ものだけ交流で見る」という素朴な 2 段構えでは交流に回らない。だが
    交流では枝 5-7 が 101.5% で逸脱している。**スクリーニングは
    絞り込みであって判断ではない。**
    """
    lenient = screen_n1(case, screen_threshold=1.0)
    missed = next(r for r in lenient.results if r.outage == (4, 5))
    assert missed.method == "lodf"          # 交流に回らなかった
    assert missed.is_secure is True         # ← 取りこぼし

    strict = screen_n1(case, screen_threshold=0.0)
    caught = next(r for r in strict.results if r.outage == (4, 5))
    assert caught.method == "ac"
    assert caught.is_secure is False

    # 閾値をいくら下げても直流では電圧を捕まえられない、という対比。
    assert next(r for r in lenient.results if r.outage == (4, 6)).is_secure is True
    assert next(r for r in strict.results if r.outage == (4, 6)).is_secure is False


# ======================================================================
# 5. 拘束する枝はちょうど 2 本
# ======================================================================
def test_exactly_two_branches_bind_under_n1(report: SecurityReport) -> None:
    """N-1 で拘束する枝の集合が ``{(5,7), (7,8)}`` にちょうど一致する。

    9 本の枝のうち 2 本だけが運転点を縛る。非拘束枝の最大は 91.1% で
    100% まで 9 ポイントの余裕があるので、この 2 本は「たまたま境界に
    近い」のではなく系統の隘路である。
    """
    assert report.binding_branches() == [(5, 7), (7, 8)]

    worst = report.worst_loading_by_branch()
    binding = {(5, 7), (7, 8)}
    for key, ratio in worst.items():
        if key in binding:
            assert ratio > 1.0
        else:
            assert ratio <= 1.0
    # 非拘束枝の最大は 91.1%（枝 4-5）。境界からの距離を数値で固定する。
    slack = max(ratio for key, ratio in worst.items() if key not in binding)
    assert slack == pytest.approx(0.911, abs=LOADING_TOL)


def test_binding_branch_5_7_is_the_genstab_fault_branch(
    case: Case, report: SecurityReport
) -> None:
    """拘束する枝 5-7 は、過渡安定度の標準事故で開放される枝と同じである。

    静的セキュリティ（過負荷）と過渡安定度（脱調）という別々の物理が、
    同じ 1 本の枝で出会う。ケースの ``stability`` 層から独立に読み出して
    突き合わせる（テスト側に (5, 7) を直書きしない）。
    """
    tripped = case.stability["fault"]["tripped_branches"]
    keys = [
        (min(b["from"], b["to"]), max(b["from"], b["to"])) for b in tripped
    ]
    assert keys == [(5, 7)]
    assert keys[0] in report.binding_branches()


# ======================================================================
# 6. 橋は候補から外し、外した事実を残す
# ======================================================================
def test_bridges_are_skipped_and_never_appear_in_results(case: Case) -> None:
    """全枝を候補に与えても、橋 3 本は ``skipped`` に回って ``results`` に入らない。

    候補の一覧はケースの ``contingencies`` 層ではなく全枝から作るので、
    「もともと候補に入っていなかっただけ」ではないことが分かる。
    橋の判定は :func:`gridops.ybus.bridges` が独立に行っている。
    """
    all_keys = [branch.key() for branch in case.branches]
    assert bridges(case) == BRIDGES

    report = screen_n1(case, contingencies=all_keys)

    skipped_keys = [key for key, _ in report.skipped]
    assert sorted(skipped_keys) == BRIDGES
    assert sorted(r.outage for r in report.results) == sorted(
        set(all_keys) - set(BRIDGES)
    )
    for key, reason in report.skipped:
        assert "橋" in reason
        assert "島" in reason
        # 「除外は健全という意味ではない」ことが理由に書かれている。
        assert "健全" in reason
    assert "除外は「健全」ではない" in report.summary()


def test_default_candidates_come_from_the_case(case: Case, report: SecurityReport) -> None:
    """既定の評価対象はケースの候補で、対象外の橋も未評価として記録する。"""
    assert sorted(r.outage for r in report.results) == sorted(case.contingencies)
    assert [key for key, _reason in report.skipped] == BRIDGES
    assert all("未評価" in reason or "適用でき" in reason
               for _key, reason in report.skipped)


# ======================================================================
# 7. 性能指標の masking
# ======================================================================
def test_performance_index_matches_hand_calculation() -> None:
    """PI の値を暗算と突き合わせる（容量を 1.0 に揃えた例）。

    :math:`\\mathrm{PI} = \\sum w (f/f^{max})^{2n} / (2n)` なので、
    0.95 が 10 本なら :math:`10 \\times 0.9025 / 2 = 4.5125` である。
    """
    limits = np.ones(10)
    mild = np.full(10, 0.95)
    assert performance_index(mild, limits) == pytest.approx(4.5125, rel=1e-12)

    severe = np.array([2.0] + [0.1] * 9)
    assert performance_index(severe, limits) == pytest.approx(2.045, rel=1e-12)

    # n を上げると 8 乗になる。2^8 / 8 = 32 が支配的。
    assert performance_index(severe, limits, n=4) == pytest.approx(
        (2.0 ** 8 + 9 * 0.1 ** 8) / 8, rel=1e-12
    )
    # 重みは線形に効く。
    assert performance_index(mild, limits, weights=np.full(10, 2.0)) == pytest.approx(
        2 * 4.5125, rel=1e-12
    )
    # 容量が inf の枝は寄与しない。
    limits_inf = np.array([np.inf] + [1.0] * 9)
    assert performance_index(severe, limits_inf) == pytest.approx(
        9 * 0.01 / 2, rel=1e-12
    )


def _fake_result(outage, flows, limits, *, n=1) -> ContingencyResult:
    """手で作った枝潮流から :class:`ContingencyResult` を組み立てる補助。"""
    flows = np.asarray(flows, dtype=float)
    limits = np.asarray(limits, dtype=float)
    loading = flows / limits
    worst = int(np.argmax(loading))
    return ContingencyResult(
        outage=outage,
        flows=flows,
        loading=loading,
        v_min=1.0,
        v_min_bus=1,
        performance_index=performance_index(flows, limits, n=n),
        worst_branch=(worst, worst),
        worst_loading=float(loading[worst]),
        islanding=False,
        converged=True,
    )


def test_performance_index_masks_a_severe_single_overload() -> None:
    """PI (n=1) の順位が、実際の最悪度の順位と食い違う（masking）。

    事故 A は 10 本すべてが 95% で **1 本も定格を超えていない**。
    事故 B は 1 本が 200% で深刻に過負荷している。それでも PI は
    A (4.5125) を B (2.045) より上に置く。軽い負荷が多数あると、
    2 乗の和が重大な 1 本を隠してしまう。

    **PI ランキングは順位を誤るものである。** これは実装の粗さでは
    なく指標の性質であり、だから PI は候補を並べる道具であって
    判定ではない。
    """
    limits = np.ones(10)
    mild = _fake_result((1, 2), np.full(10, 0.95), limits)
    severe = _fake_result((3, 4), np.array([2.0] + [0.1] * 9), limits)

    # 実際の最悪度: B のほうが圧倒的に危険（A は逸脱ゼロ）。
    assert mild.worst_loading == pytest.approx(0.95)
    assert severe.worst_loading == pytest.approx(2.0)
    assert mild.thermal_secure is True
    assert severe.thermal_secure is False

    # ところが PI の順位は逆である。
    assert mild.performance_index > severe.performance_index

    report = SecurityReport(base=None, results=[mild, severe])
    assert [r.outage for r in report.ranked(by="performance_index")] == [(1, 2), (3, 4)]
    assert [r.outage for r in report.ranked(by="worst_loading")] == [(3, 4), (1, 2)]
    # 2 つの順位が食い違うことそのものを固定する。
    assert [r.outage for r in report.ranked(by="performance_index")] != [
        r.outage for r in report.ranked(by="worst_loading")
    ]


def test_higher_exponent_unmasks_the_severe_contingency() -> None:
    """指数を上げる (n=4) と masking が解ける。

    ただし 8 乗は 1.05 と 1.10 の差をほとんど潰すので、上げれば良いと
    いう話ではない。順位づけの道具に判定を任せないこと、が結論である。
    """
    limits = np.ones(10)
    mild = _fake_result((1, 2), np.full(10, 0.95), limits, n=4)
    severe = _fake_result((3, 4), np.array([2.0] + [0.1] * 9), limits, n=4)
    assert severe.performance_index > mild.performance_index
    assert severe.performance_index == pytest.approx(32.0, abs=1e-6)
    assert mild.performance_index == pytest.approx(0.8292755, rel=1e-6)


def test_rankings_disagree_on_the_real_case(report: SecurityReport) -> None:
    """WSCC 9 母線でも PI の 1 位と電圧の 1 位は違う事故である。

    PI（熱容量だけの指標）の 1 位は枝 5-7 の開放（7-8 が 112.5%）だが、
    電圧の 1 位は枝 4-5 の開放（母線 5 が 0.8388 p.u.）である。
    「最悪の事故」と言うときは、どの順位で並べたかを必ず添えること。
    """
    assert report.ranked(by="performance_index")[0].outage == (5, 7)
    assert report.ranked(by="worst_loading")[0].outage == (5, 7)
    assert report.ranked(by="v_min")[0].outage == (4, 5)


# ======================================================================
# 8-9. セキュリティ制約付き経済配分
# ======================================================================
def _merit_order_cost(case: Case) -> tuple[float, dict[str, float]]:
    """優先順位法（手計算と同じ手順）で経済配分の費用を出す独立実装。

    安い号機から上限まで積む。全号機が運転している前提なので、まず
    全機を最低出力に置いてから残余を配る。線形費用なのでこれが最適解。
    """
    demand = float(case.to_mw(sum(bus.pd for bus in case.buses)))
    dispatch = {unit.name: unit.p_min_mw for unit in case.units}
    remaining = demand - sum(dispatch.values())
    for unit in sorted(case.units, key=lambda u: u.var_cost):
        step = min(remaining, unit.p_max_mw - dispatch[unit.name])
        dispatch[unit.name] += step
        remaining -= step
    assert remaining == pytest.approx(0.0, abs=1e-9)
    cost = sum(unit.var_cost * dispatch[unit.name] for unit in case.units)
    return float(cost), dispatch


@pytest.fixture(scope="module")
def preventive(case: Case) -> SCEDResult:
    return sced(case, mode="preventive")


@pytest.fixture(scope="module")
def corrective(case: Case) -> SCEDResult:
    return sced(case, mode="corrective")


def test_unconstrained_cost_matches_the_merit_order(
    case: Case, preventive: SCEDResult
) -> None:
    """SCED が基準にする「制約なし」の費用が、優先順位法の手計算と一致する。

    テスト側の実装は線形計画をまったく使わず、安い順に積むだけである。
    """
    expected, dispatch = _merit_order_cost(case)
    assert expected == pytest.approx(3_343_500.0, rel=1e-12)  # 手で検算できる値
    assert preventive.unconstrained_cost == pytest.approx(expected, rel=1e-9)


def test_cost_ordering_preventive_ge_corrective_ge_unconstrained(
    preventive: SCEDResult, corrective: SCEDResult
) -> None:
    """予防的の費用 >= 是正的の費用 >= 制約なしの費用。

    これは数値の偶然ではなく実行可能領域の包含から出る。是正的は
    :math:`p^c = p` と置けば予防的の解をそのまま含み、制約なしは
    さらに送電制約を落としたものだからである。**3 つとも真に異なる**
    ことまで確かめる（等号で通ってしまうと包含が効いていない実装でも
    テストが緑になる）。
    """
    assert preventive.total_cost >= corrective.total_cost - 1e-6
    assert corrective.total_cost >= preventive.unconstrained_cost - 1e-6
    assert preventive.total_cost > corrective.total_cost + 1.0
    assert corrective.total_cost > preventive.unconstrained_cost + 1.0

    # セキュリティの値段は非負で、予防的のほうが高い。
    base = preventive.unconstrained_cost
    assert preventive.cost_of_security(base) > corrective.cost_of_security(base) > 0.0
    assert preventive.cost_of_security(base) == pytest.approx(
        preventive.total_cost - base, rel=1e-12
    )


def test_corrective_without_ramp_room_equals_preventive(
    case: Case, preventive: SCEDResult
) -> None:
    """事故後に 1 MW も動けない是正的は、予防的とまったく同じ問題になる。

    包含関係の端の場合を押さえるテストである。``corrective_ramp_fraction``
    をゼロにすると :math:`p^c = p` が強制されるので、費用は一致しなければ
    ならない。
    """
    frozen = sced(case, mode="corrective", corrective_ramp_fraction=0.0)
    assert frozen.total_cost == pytest.approx(preventive.total_cost, rel=1e-9)


@pytest.mark.parametrize("mode", ["preventive", "corrective"])
def test_sced_solution_is_feasible_for_every_contingency(case: Case, mode: str) -> None:
    """SCED の解が全事故で実行可能であることを、独立に再計算して確かめる。

    SCED は PTDF と LODF から潮流を線形式で組み立てている。ここでの
    検算は :func:`gridops.dc.dc_powerflow` に出力を渡し、事故ごとに
    :math:`B'` を組み直して解き直す **別の定式化**である。予備力や
    需給の帳尻も同時に見る。
    """
    result = sced(case, mode=mode)
    rate_a = np.array([b.rate_a for b in case.branches])
    rate_b = np.array([b.rate_b for b in case.branches])
    demand = float(case.to_mw(sum(bus.pd for bus in case.buses)))

    # 需給と上下限。
    assert sum(result.dispatch.values()) == pytest.approx(demand, abs=1e-6)
    for unit in case.units:
        assert unit.p_min_mw - 1e-9 <= result.dispatch[unit.name] <= unit.p_max_mw + 1e-9

    # 事故前は rate_a 以内。
    base_flow = dc_powerflow(case, dispatch=result.dispatch).flows
    assert np.all(np.abs(base_flow) <= rate_a + 1e-6)

    # 事故後は rate_b 以内。是正的なら事故ごとの出力を使う。
    for key in case.contingencies:
        used = result.corrective_dispatch.get(key, result.dispatch)
        assert sum(used.values()) == pytest.approx(demand, abs=1e-6)
        post = dc_powerflow(case, dispatch=used, removed_branches=[key]).flows
        assert np.all(np.abs(post) <= rate_b + 1e-6), (
            f"事故 {key} で rate_b を超えた: "
            f"{np.round(np.abs(post) / rate_b * 100, 2)}"
        )
        # 是正的の再給電はランプ率の範囲に収まっていなければならない。
        for unit in case.units:
            delta = used[unit.name] - result.dispatch[unit.name]
            assert -unit.ramp_down - 1e-6 <= delta <= unit.ramp_up + 1e-6


@pytest.mark.parametrize("mode", ["preventive", "corrective"])
def test_sced_binding_constraints_are_tight(case: Case, mode: str) -> None:
    """報告された拘束制約が、実際に等号で成立していることを確かめる。

    「制約を足した」ことと「その制約が効いた」ことは別である。
    :attr:`SCEDResult.binding` は後者だけを載せる。検算は
    :func:`gridops.dc.dc_powerflow`（:math:`B'` を組み直す別の定式化）で
    行う。
    """
    result = sced(case, mode=mode)
    assert result.binding, "1 本以上は拘束するはず"
    rate_b = {b.key(): b.rate_b for b in case.branches}
    keys = [b.key() for b in case.branches]
    for outage, branch in result.binding:
        assert outage in [tuple(k) for k in case.contingencies]
        assert branch in keys
        used = result.corrective_dispatch.get(outage, result.dispatch)
        post = dc_powerflow(case, dispatch=used, removed_branches=[outage])
        flow = abs(post.flow_of(branch))
        assert flow == pytest.approx(rate_b[branch], abs=1e-5)


def test_constraint_generation_needs_only_a_few_rounds(
    preventive: SCEDResult, corrective: SCEDResult
) -> None:
    """制約生成は数周で止まる（最後の 1 周は違反ゼロの確認）。

    6 事故 x 9 枝 = 54 本の潮流制約を最初から並べる必要はない、という
    のが制約生成の値打ちである。
    """
    assert 1 <= preventive.iterations <= 5
    assert 1 <= corrective.iterations <= 5
    assert preventive.status == "Optimal"
    assert corrective.status == "Optimal"


def test_sced_result_is_secure_under_dc_screening(case: Case, preventive: SCEDResult) -> None:
    """予防的 SCED の運転点は、直流の N-1 スクリーニングを通る。

    SCED は直流で解いているので、直流の意味では必ず健全になる。
    交流で見ると負荷率が上がるので、**この確認だけで安心してはいけない**
    ことも同時に固定する（交流の最悪負荷率が直流より大きいこと）。
    """
    base = solve_powerflow(case, dispatch=preventive.dispatch)
    dc_report = screen_n1(case, base, method="lodf", check_voltage=False)
    assert dc_report.is_secure is True

    ac_report = screen_n1(case, base, method="ac")
    for result in ac_report.results:
        assert result.worst_loading >= (
            next(
                r.worst_loading
                for r in dc_report.results
                if r.outage == result.outage
            )
            - 1e-9
        )


def test_scaling_demand_changes_the_cost_of_security(case: Case) -> None:
    """需要を下げるとセキュリティの値段が下がる（送電制約が緩む）。

    ``demand_mw`` が母線の比を保って一律にスケールすることの確認でも
    ある。軽負荷では送電制約が効かなくなり、SCED の費用が制約なしの
    経済配分に一致する。
    """
    light = sced(case, demand_mw=200.0, mode="preventive")
    assert light.cost_of_security(light.unconstrained_cost) == pytest.approx(0.0, abs=1.0)
    assert light.binding == []
    assert sum(light.dispatch.values()) == pytest.approx(200.0, abs=1e-6)


# ======================================================================
# 表示と例外
# ======================================================================
def test_report_text_mentions_the_voltage_violation(report: SecurityReport) -> None:
    """``summary`` と ``to_table`` が読める形で出る（表の見出しは英語）。"""
    text = report.summary()
    assert "母線 6" in text and "0.9418" in text
    assert "拘束する枝" in text and "5-7" in text and "7-8" in text

    table = report.to_table()
    assert "outage" in table and "worst branch" in table and "PI" in table
    assert "INSECURE" in table
    assert len(table.splitlines()) == 3 + len(report.results) + len(report.skipped)

    one = next(r for r in report.results if r.outage == (4, 6)).summary()
    assert "熱容量" not in one          # 過負荷は 1 件も無い
    assert "0.9418" in one


def test_summary_warns_when_voltage_is_not_checked(case: Case) -> None:
    """電圧を見ていない報告には、その但し書きが必ず入る。"""
    text = screen_n1(case, check_voltage=False).summary()
    assert "電圧を見ていない" in text
    assert "誤判定" in text
    assert "voltage = OFF" in screen_n1(case, check_voltage=False).to_table()


def test_sced_summary_is_readable(preventive: SCEDResult) -> None:
    text = preventive.summary()
    assert "preventive" in text
    assert "セキュリティの値段" in text
    assert "N-1 で拘束" in text


@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        ({"method": "dc"}, "method="),
        ({"limit": "rate_c"}, "limit="),
        ({"contingencies": [(4, 99)]}, "はケース"),
        ({"contingencies": [(4, 5, 6)]}, "組のリスト"),
    ],
)
def test_screen_n1_rejects_bad_arguments(case: Case, kwargs, pattern) -> None:
    """誤った引数は日本語の :class:`ValueError` で止める（文言まで固定）。"""
    with pytest.raises(ValueError, match=pattern):
        screen_n1(case, **kwargs)


def test_ranked_rejects_an_unknown_key(report: SecurityReport) -> None:
    with pytest.raises(ValueError, match="performance_index"):
        report.ranked(by="cost")


@pytest.mark.parametrize(
    "kwargs, pattern",
    [
        ({"mode": "postventive"}, "mode="),
        ({"limit": "rate_c"}, "limit="),
        ({"max_iter": 0}, "max_iter="),
        ({"corrective_ramp_fraction": -1.0}, "corrective_ramp_fraction="),
    ],
)
def test_sced_rejects_bad_arguments(case: Case, kwargs, pattern) -> None:
    with pytest.raises(ValueError, match=pattern):
        sced(case, **kwargs)


@pytest.mark.parametrize(
    "args, pattern",
    [
        (([1.0, 2.0], [1.0]), "長さ"),
        (([1.0], [1.0]), None),
    ],
)
def test_performance_index_checks_shapes(args, pattern) -> None:
    if pattern is None:
        assert performance_index(*args) == pytest.approx(0.5)
    else:
        with pytest.raises(ValueError, match=pattern):
            performance_index(*args)


def test_performance_index_rejects_bad_exponent_and_limits() -> None:
    with pytest.raises(ValueError, match="n=0"):
        performance_index([1.0], [1.0], n=0)
    with pytest.raises(ValueError, match="非正"):
        performance_index([1.0], [0.0])
    with pytest.raises(ValueError, match="weights"):
        performance_index([1.0, 1.0], [1.0, 1.0], weights=[1.0])


def test_non_convergent_contingency_is_not_reported_as_secure(case: Case) -> None:
    """事故後の潮流が収束しない候補を「健全」に丸めないこと。

    負荷を 1.5 倍にすると、事故前は解けるのに枝 4-5 を開放した系統は
    解が無くなる（電圧崩壊）。掃引は例外で
    止まらず、``converged=False`` の結果として残り、``is_secure`` は
    ``False`` になる（「分からない」を「健全」に丸めない）。
    """
    heavy = replace(
        case,
        buses=[replace(b, pd=b.pd * 1.5, qd=b.qd * 1.5) for b in case.buses],
    )
    report = screen_n1(heavy, method="ac", contingencies=[(4, 5)])
    result = report.results[0]
    assert result.converged is False
    assert result.thermal_secure is False
    assert result.is_secure is False
    assert report.is_secure is False


def test_outaged_case_keeps_the_reference_generation(case: Case, report: SecurityReport) -> None:
    """事故後のケースは枝だけを差し替え、発電を事故前に据え置く。

    :meth:`Case.without_branch` を使うと参照解が落ち、発電ゼロの別系統を
    解いてしまう（WSCC 9 母線では収束しない）。それを踏んでいないことを、
    契約の N-1 の表が再現できることで確かめている。ここでは
    ``without_branch`` を使った場合に本当に収束しないことを対比として
    固定する。
    """
    stripped = case.without_branch((4, 5))
    assert stripped.reference is None
    with pytest.raises(RuntimeError):
        solve_powerflow(stripped)
    # 一方、screen_n1 の結果は契約の表と一致している。
    result = next(r for r in report.results if r.outage == (4, 5))
    assert result.converged is True
    assert result.worst_loading == pytest.approx(1.015, abs=LOADING_TOL)


# ======================================================================
# 追加の性質（引数の伝播・部分集合の候補・結果の読み方）
# ======================================================================
def test_pi_exponent_reaches_the_results(case: Case) -> None:
    """``pi_n`` が各事故の PI に伝わり、指数を上げると順位が変わりうる。

    PI は事故ごとに ``flows`` と熱容量から計算されるので、指数を変えた
    結果は :func:`performance_index` をテスト側で呼び直して再現できる。
    """
    report = screen_n1(case, pi_n=4)
    rates = np.array([b.rate_b for b in case.branches])
    for result in report.results:
        assert result.performance_index == pytest.approx(
            performance_index(result.flows, rates, n=4), rel=1e-12
        )
    # n=1 と n=4 で値そのものは必ず変わる。
    default = {r.outage: r.performance_index for r in screen_n1(case).results}
    assert all(
        result.performance_index != pytest.approx(default[result.outage], rel=1e-6)
        for result in report.results
    )


def test_contingency_subset_is_honoured(case: Case) -> None:
    """候補を絞ると、その事故だけが評価される。"""
    report = screen_n1(case, contingencies=[(5, 7), (7, 8)])
    assert sorted(r.outage for r in report.results) == [(5, 7), (7, 8)]
    # 部分集合でも、その範囲での拘束枝は同じ 2 本になる。
    assert report.binding_branches() == [(5, 7), (7, 8)]


def test_insecure_is_sorted_by_loading(report: SecurityReport) -> None:
    """``insecure()`` は逸脱した事故だけを負荷率の高い順に返す。"""
    bad = report.insecure()
    assert [r.outage for r in bad] == [(5, 7), (7, 8), (4, 5), (6, 9), (4, 6)]
    assert all(not r.is_secure for r in bad)
    assert (8, 9) not in [r.outage for r in bad]
    # 電圧だけで落ちた 4-6 が、熱容量では最下位に来る（順位の意味の対比）。
    assert bad[-1].outage == (4, 6)
    assert bad[-1].thermal_secure is True


def test_violations_lists_every_overloaded_branch(report: SecurityReport) -> None:
    """過負荷の一覧は枝の番号と負荷率を伴って出る。"""
    result = next(r for r in report.results if r.outage == (5, 7))
    problems = result.violations()
    assert any("枝 7-8" in message and "112.5%" in message for message in problems)
    assert any("母線 5" in message for message in problems)


def test_sced_accepts_a_contingency_subset(case: Case, preventive: SCEDResult) -> None:
    """想定事故を減らすと制約が緩み、費用は下がる（か同じ）。

    「セキュリティの値段」が **どの事故を想定したか**に依存する量で
    あることの確認である。想定を減らせば安くなるが、想定から外した
    事故が起きたときに耐えられる保証は無くなる。
    """
    fewer = sced(case, contingencies=[(4, 6), (8, 9)])
    assert fewer.total_cost <= preventive.total_cost + 1e-6
    assert fewer.unconstrained_cost == pytest.approx(
        preventive.unconstrained_cost, rel=1e-12
    )
    for outage, _branch in fewer.binding:
        assert outage in [(4, 6), (8, 9)]


def test_corrective_actually_redispatches(case: Case, corrective: SCEDResult) -> None:
    """是正的の解は、事故ごとに違う出力を持っている。

    「是正的が安い」のは事故後に動く余地を織り込めるからであって、
    制約を無視したからではない。実際に動いていることを確かめる。
    """
    assert corrective.corrective_dispatch, "是正的なら事故ごとの出力があるはず"
    moved = False
    for key, post in corrective.corrective_dispatch.items():
        assert key in [tuple(k) for k in case.contingencies]
        for unit in case.units:
            delta = post[unit.name] - corrective.dispatch[unit.name]
            if abs(delta) > 1e-6:
                moved = True
    assert moved, "どの事故でも 1 MW も動いていない"


def test_preventive_has_no_corrective_dispatch(preventive: SCEDResult) -> None:
    """予防的では事故後の再給電が存在しない（同じ 1 組の出力で耐える）。"""
    assert preventive.corrective_dispatch == {}
    assert preventive.mode == "preventive"


def test_candidates_fall_back_to_all_branches(case: Case) -> None:
    """``contingencies`` 層を持たないケースでは全枝が候補になる。

    そのとき橋 3 本は ``skipped`` に回るので、評価されるのは残り 6 本と
    なり、ケースの ``contingencies`` 層とちょうど同じ集合になる。
    **ケースの但し書きではなくアルゴリズムが同じ答えを出す**ことの確認。
    """
    bare = replace(case, contingencies=())
    report = screen_n1(bare)
    assert sorted(r.outage for r in report.results) == sorted(case.contingencies)
    assert sorted(key for key, _ in report.skipped) == BRIDGES


def test_bridges_only_candidates_refuse_to_pretend_security(case: Case) -> None:
    """橋だけを候補にすると、黙って ``is_secure=True`` を返さず止まること。

    以前は評価対象ゼロ件のまま ``True`` を返しており、「何も検査して
    いないのに N-1 健全」と誤読される API だった（外部レビューの指摘 #6）。
    いまは評価できる事故が 1 件もなければ日本語の ValueError になる。
    """
    with pytest.raises(ValueError, match="評価できる想定事故"):
        screen_n1(case, contingencies=BRIDGES)


# ======================================================================
# 外部レビュー（2026-08-30）の回帰
# ======================================================================
def test_report_exposes_unassessed_contingencies(report: SecurityReport) -> None:
    """未評価の事故が残っていることを ``has_unassessed`` が明示すること。

    外部レビューの指摘 #6。``is_secure`` は評価した事故だけの判定なので、
    橋 3 本が未評価のまま「N-1 健全」と読まれないよう、まずこのフラグを
    見る運用にする。
    """
    assert report.has_unassessed is True
    assert len(report.skipped) == len(BRIDGES)


def test_screening_with_no_assessable_contingency_raises(case: Case) -> None:
    """評価できる想定事故が 1 件も無ければ日本語で止まること。

    空リストや「すべて橋」の指定は、黙って ``is_secure=True`` を返すと
    「検査して健全だった」と誤読される（外部レビューの指摘 #6）。
    """
    with pytest.raises(ValueError, match="評価できる想定事故"):
        screen_n1(case, contingencies=[])
    with pytest.raises(ValueError, match="評価できる想定事故"):
        screen_n1(case, contingencies=list(BRIDGES))
