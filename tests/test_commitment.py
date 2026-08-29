"""起動停止計画（第 07 回・第 08 回）の検証。

突き合わせる **独立な基準**は次のとおり。実装の出力を実装で数え直す
自己参照のテストは書かない。

* **全列挙**（:func:`~gridops.commitment.enumerate_commitment`）。分枝限定と
  まったく別の道筋で最適値に到達する。1 時刻の配分も線形計画を使わず、
  凸な区分線形費用を安いセグメントから埋める貪欲法で解いている。
* **このファイルに書いた検算コード**。需給・上下限・最低運転停止時間・
  ランプ・予備力を、モデルとは別に組んだ状態機械と不等式で確かめる。
* **厳密な経済負荷配分**（等 λ 法の二分法）。区分線形近似の誤差が
  分割数とともに単調に減り、必ず過大評価側であることを見る。
* **数値微分**。限界費用（双対）を、需要を動かしたときの費用の差分商と
  突き合わせる。
* **理論上界** :math:`\\sum_i c_{2,i} L_i^2 / 4`。区分線形の 1 セグメント
  あたりの最大誤差（2 次曲線と割線の差の最大値）。

MILP を解くテストがあるので、実行前に必ず ``conda activate pwsyseng`` して
おくこと。CBC は conda 環境の ``bin/`` に入るので、環境を有効にしないと
PATH に載らず、:mod:`gridops.solvers` が RuntimeError を投げる。
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np
import pytest

from gridops import load_case
from gridops.case import Case, Unit
from gridops.commitment import (
    DEFAULT_SEGMENTS,
    DEFAULT_VOLL,
    CommitmentResult,
    demand_profile,
    enumerate_commitment,
    marginal_prices,
    net_demand,
    priority_list,
    unit_commitment,
)

#: 総費用を厳密に比べるときのソルバの相対ギャップ。既定の 1e-4 のままだと
#: 「最適値から 0.01% 以内」で打ち切られてしまい、1e-6 の比較ができない。
EXACT_GAP = 1e-9


# ======================================================================
# 独立に書いた検算コード（実装を参照しない）
# ======================================================================
def cost_blocks(unit: Unit, n_segments: int) -> list[tuple[float, float]]:
    """区分線形近似の ``(傾き, 幅)`` を、breakpoint から素直に作る。

    実装側は差分商で傾きを求めているが、こちらは
    :math:`s_k = c_1 + c_2 (x_{k-1} + x_k)` の閉じた式で作る。同じ値に
    なるはずの量を別の式で出すのが目的である。
    """
    edges = np.linspace(unit.p_min_mw, unit.p_max_mw, n_segments + 1)
    return [
        (unit.var_cost + unit.quadratic * (a + b), b - a)
        for a, b in zip(edges[:-1], edges[1:])
        if b > a
    ]


def piecewise_fuel(unit: Unit, p_mw: float, n_segments: int) -> float:
    """区分線形近似で評価した燃料費（無負荷費を除く）[円/h]。"""
    total = unit.quadratic * unit.p_min_mw**2 + unit.var_cost * unit.p_min_mw
    left = p_mw - unit.p_min_mw
    for slope, width in cost_blocks(unit, n_segments):
        take = min(max(left, 0.0), width)
        total += slope * take
        left -= take
    return total


def hour_oracle(units, on, previous, following, demand, reserve, n_segments,
                voll=DEFAULT_VOLL):
    """入切を固定した 1 時刻の最小費用 [円]。**ランプ率が非拘束の場合のみ**。

    起動停止が決まっていて時間方向の結合が無ければ、各時刻は独立に解ける。
    可変費用は凸なので「安いブロックから順に埋める」が最適である。
    限界費用（最後に埋めたブロックの傾き）も同時に返す。

    Returns
    -------
    tuple
        ``(費用, 総出力, 供給不足, 出力抑制, 限界費用)``。
    """
    base = 0.0
    low = high = capacity = 0.0
    blocks: list[tuple[float, float]] = []
    for unit, is_on, was_on, next_on in zip(units, on, previous, following):
        if not is_on:
            continue
        capacity += unit.p_max_mw
        base += unit.noload_cost + unit.quadratic * unit.p_min_mw**2 \
            + unit.var_cost * unit.p_min_mw
        if not was_on:
            base += unit.startup_cost
        top = unit.p_max_mw
        if not was_on:
            top = min(top, unit.su_ramp)
        if next_on is not None and not next_on:
            top = min(top, unit.sd_ramp)
        low += unit.p_min_mw
        high += top
        room = top - unit.p_min_mw
        for slope, width in cost_blocks(unit, n_segments):
            if room <= 0.0:
                break
            blocks.append((slope, min(width, room)))
            room -= width
    ceiling = min(high, capacity - reserve)
    assert ceiling >= low - 1e-9, "この入切では予備力を確保できない"
    target = min(max(demand, low), ceiling)
    shed = max(0.0, demand - ceiling)
    spill = max(0.0, low - demand)

    blocks.sort()
    cost = base
    left = target - low
    price = 0.0
    for slope, width in blocks:
        if left <= 1e-12:
            break
        take = min(width, left)
        cost += slope * take
        left -= take
        price = slope
    if shed > 0.0:
        price = voll
    return cost + voll * shed, target, shed, spill, price


def exact_dispatch(units, demand: float) -> tuple[float, float]:
    """厳密な経済負荷配分（等 λ 法の二分法）。``(λ, 総費用)`` を返す。

    2 次費用のまま解く。区分線形近似はこの値を **上から**近似する。
    """
    low = min(unit.var_cost for unit in units)
    high = max(unit.incremental_cost(unit.p_max_mw) for unit in units)

    def output(lam: float) -> list[float]:
        return [
            min(max((lam - unit.var_cost) / (2.0 * unit.quadratic), unit.p_min_mw),
                unit.p_max_mw)
            for unit in units
        ]

    for _ in range(200):
        lam = 0.5 * (low + high)
        if sum(output(lam)) < demand:
            low = lam
        else:
            high = lam
    lam = 0.5 * (low + high)
    return lam, float(sum(u.fuel_cost(p) for u, p in zip(units, output(lam))))


def verify_result(
    result: CommitmentResult, *, reserve_rate: float, tol: float = 1e-5
) -> None:
    """得られた計画が制約をすべて満たすことを、モデルとは別に確かめる。

    ここが本ファイルの要である。ソルバが返した数字を信じずに、需給・
    上下限・最低運転停止時間・ランプ・予備力を組み直して当てる。
    """
    case = result.case
    units = [unit for unit in case.units if unit.name in result.schedule]
    horizon = len(result.demand_mw)

    # --- 需給バランス --------------------------------------------------
    supplied = np.zeros(horizon)
    for unit in units:
        supplied = supplied + result.dispatch[unit.name]
    residual = supplied + result.shortfall_mw - result.spill_mw - result.demand_mw
    assert np.max(np.abs(residual)) < tol, f"需給が閉じていない: {residual}"
    assert np.all(result.shortfall_mw >= -tol)
    assert np.all(result.spill_mw >= -tol)

    for unit in units:
        commitment = result.schedule[unit.name]
        output = result.dispatch[unit.name]
        assert set(np.unique(commitment)) <= {0.0, 1.0}, f"{unit.name} の u が 0/1 でない"

        # --- 出力の上下限 ----------------------------------------------
        assert np.all(output <= unit.p_max_mw * commitment + tol), unit.name
        assert np.all(output >= unit.p_min_mw * commitment - tol), unit.name

        # --- 最低運転停止時間（初期条件つき）---------------------------
        state, timer = int(unit.u0), int(unit.hours_in_state)
        for t in range(horizon):
            nxt = int(commitment[t])
            if nxt != state:
                need = unit.min_up if state == 1 else unit.min_down
                assert timer >= need, (
                    f"{unit.name}: 時刻 {t} で状態 {state} -> {nxt} に変わったが、"
                    f"その状態は {timer} 時間しか続いておらず、必要な {need} 時間に"
                    "満たない（初期条件の持ち越しを含む）"
                )
                state, timer = nxt, 1
            else:
                timer += 1

        # --- ランプ率 --------------------------------------------------
        for t in range(1, horizon):
            started = commitment[t] > commitment[t - 1]
            stopped = commitment[t] < commitment[t - 1]
            up_limit = (unit.ramp_up if commitment[t - 1] > 0.5 else 0.0) \
                + (unit.su_ramp if started else 0.0)
            down_limit = (unit.ramp_down if commitment[t] > 0.5 else 0.0) \
                + (unit.sd_ramp if stopped else 0.0)
            assert output[t] - output[t - 1] <= up_limit + tol, f"{unit.name} t={t} 増出力"
            assert output[t - 1] - output[t] <= down_limit + tol, f"{unit.name} t={t} 減出力"
        if unit.u0 == 0:
            # 停止していた号機の 1 時刻目だけは初期出力が 0 と分かる。
            assert output[0] <= unit.su_ramp * commitment[0] + tol, f"{unit.name} 起動時"

    # --- 運転予備力 ----------------------------------------------------
    required = reserve_rate * np.maximum(result.demand_mw, 0.0)
    unloaded = np.zeros(horizon)
    for unit in units:
        unloaded = unloaded + unit.p_max_mw * result.schedule[unit.name] \
            - result.dispatch[unit.name]
    assert np.all(unloaded >= required - tol), (
        f"予備力が不足: {unloaded - required}"
    )
    assert np.allclose(unloaded, result.reserve_mw(), atol=tol)


# ======================================================================
# ケースと共通の結果（module スコープで使い回す）
# ======================================================================
@pytest.fixture(scope="module")
def case() -> Case:
    return load_case("wscc9")


@pytest.fixture(scope="module")
def summer(case) -> np.ndarray:
    return demand_profile(case, "summer_weekday")


@pytest.fixture(scope="module")
def summer_result(case, summer) -> CommitmentResult:
    return unit_commitment(case, summer)


def relaxed(unit: Unit, **changes) -> Unit:
    """ランプ率を非拘束にした号機を作る（全列挙の基準と比べるため）。"""
    return replace(unit, ramp_up=math.inf, ramp_down=math.inf, **changes)


@pytest.fixture(scope="module")
def tiny(case) -> Case:
    """3 号機 x 6 時刻の小さなケース。全列挙が現実的な時間で終わる規模。

    ランプ率は非拘束にしてある（全列挙は時間方向に結合する制約を扱わない）。
    最低運転停止時間は 2 時間に縮め、入切表に自由度を残す。
    """
    by_name = {unit.name: unit for unit in case.units}
    units = [
        relaxed(by_name["G1-1"], min_up=2, min_down=2, u0=1, hours_in_state=4),
        relaxed(by_name["G2-1"], min_up=2, min_down=2, u0=0, hours_in_state=4),
        relaxed(by_name["G3-1"], min_up=2, min_down=2, u0=1, hours_in_state=1),
    ]
    return replace(case, units=units, commitment={"reserve_rate": 0.10})


# ======================================================================
# 1. 全列挙との突き合わせ
# ======================================================================
@pytest.mark.parametrize(
    "demand",
    [
        [110.0, 85.0, 70.0, 95.0, 150.0, 175.0],
        [70.0, 60.0, 55.0, 120.0, 175.0, 140.0],
        [130.0, 110.0, 80.0, 60.0, 120.0, 170.0],
    ],
)
def test_milp_matches_exhaustive_enumeration(tiny, demand):
    """3 号機 x 6 時刻で、混合整数計画と全列挙の総費用が一致すること。

    **入切表そのものは比べない。** 同じ費用の別解（費用が縮退した解）が
    あり得るので、比べてよいのは最適値だけである。ソルバの相対ギャップは
    1e-9 まで絞る（既定の 1e-4 のままだと 0.01% 以内で打ち切られる）。
    """
    demand = np.array(demand)
    reference = enumerate_commitment(tiny, demand)
    result = unit_commitment(tiny, demand, gap=EXACT_GAP)

    assert reference.status == "Optimal"
    assert result.status == "Optimal"
    assert result.total_cost == pytest.approx(reference.total_cost, rel=1e-6)
    # 全列挙が最適である以上、混合整数計画がそれを下回ることはあり得ない。
    assert result.total_cost >= reference.total_cost * (1.0 - 1e-9)

    verify_result(result, reserve_rate=0.10)
    verify_result(reference, reserve_rate=0.10)


def test_enumeration_rejects_binding_ramp_rates(case):
    """ランプ率が拘束しうる号機を渡したら、日本語で止まること。

    全列挙は各時刻を独立に解くので、時間方向に結合するランプ制約を
    扱えない。黙って無視すると「別の問題の最適値」を基準にしてしまう。
    """
    small = replace(case, units=list(case.units[:2]), commitment={"reserve_rate": 0.0})
    with pytest.raises(ValueError, match="ランプ率が拘束しうる号機がある"):
        enumerate_commitment(small, np.array([80.0, 90.0]))


def test_enumeration_refuses_large_problems(case):
    """規模が大きすぎるときに日本語 ValueError を投げること。"""
    with pytest.raises(ValueError, match="全列挙の規模が大きすぎる"):
        enumerate_commitment(case, demand_profile(case))


# ======================================================================
# 2. 最低運転停止時間の初期条件
# ======================================================================
def two_unit_case(case: Case, *, u0: int, hours_in_state: int, min_up: int,
                  min_down: int, var_cost: float) -> Case:
    """安いベース機 1 台と、初期条件を振る試験機 1 台からなるケース。

    試験機の経済性を極端にしておき、「経済的には逆にしたいのに、最低運転
    停止時間のせいでそうできない」状況を作る。
    """
    base = Unit(
        name="BASE", bus=1, p_max_mw=100.0, p_min_mw=10.0, var_cost=5000.0,
        quadratic=1.0, noload_cost=10000.0, startup_cost=0.0,
        min_up=1, min_down=1, ramp_up=math.inf, ramp_down=math.inf,
        u0=1, hours_in_state=24,
    )
    test = Unit(
        name="TEST", bus=2, p_max_mw=50.0, p_min_mw=20.0, var_cost=var_cost,
        quadratic=1.0, noload_cost=50000.0, startup_cost=100000.0,
        min_up=min_up, min_down=min_down, ramp_up=math.inf, ramp_down=math.inf,
        u0=u0, hours_in_state=hours_in_state,
    )
    return replace(case, units=[base, test], commitment={"reserve_rate": 0.0})


@pytest.mark.parametrize(
    "u0, hours_in_state, min_up, min_down, var_cost, demand, state, length",
    [
        # 運転中で、最低運転時間 8 時間のうち 1 時間しか経っていない。
        # 経済的には即座に止めたいが、7 時間は止められない。
        (1, 1, 8, 2, 30000.0, 60.0, 1, 7),
        (1, 4, 8, 2, 30000.0, 60.0, 1, 4),
        (1, 7, 8, 2, 30000.0, 60.0, 1, 1),
        # すでに 8 時間運転しているので、1 時刻目から止められる。
        (1, 8, 8, 2, 30000.0, 60.0, 1, 0),
        (1, 24, 8, 2, 30000.0, 60.0, 1, 0),
        # 停止中で、最低停止時間 8 時間のうち 1 時間しか経っていない。
        # 需要が足りず起動したいが、7 時間は起動できない（供給不足になる）。
        (0, 1, 2, 8, 4000.0, 130.0, 0, 7),
        (0, 6, 2, 8, 4000.0, 130.0, 0, 2),
        (0, 8, 2, 8, 4000.0, 130.0, 0, 0),
    ],
)
def test_initial_state_forces_commitment(
    case, u0, hours_in_state, min_up, min_down, var_cost, demand, state, length
):
    """初期状態から持ち越した最低運転／停止時間が、実際に課されること。

    ``Unit.remaining_min_up()`` / ``remaining_min_down()`` が返す時間だけは
    入切が固定され、その直後の時刻では **経済的に望ましいほうへ切り替わる**
    ことまで確かめる。固定されているだけなら「たまたま動かなかった」のか
    「拘束が効いた」のか区別がつかないためである。
    """
    small = two_unit_case(
        case, u0=u0, hours_in_state=hours_in_state,
        min_up=min_up, min_down=min_down, var_cost=var_cost,
    )
    horizon = 12
    result = unit_commitment(small, np.full(horizon, demand), gap=EXACT_GAP)
    schedule = result.schedule["TEST"]
    unit = [u for u in small.units if u.name == "TEST"][0]

    expected = unit.remaining_min_up() if u0 == 1 else unit.remaining_min_down()
    assert expected == length, "remaining_min_up/down の値そのものが想定と違う"
    assert np.all(schedule[:length] == float(state)), (
        f"初期条件で固定されるはずの {length} 時刻が {schedule[:length]} になっている"
    )
    # 拘束が解けた瞬間に、経済的に望ましい状態へ切り替わること。
    assert schedule[length] == float(1 - state)
    verify_result(result, reserve_rate=0.0)


def test_long_min_up_unit_cannot_be_stopped_in_the_first_hour(case):
    """min_up=8 の機を 1 時刻目に止められないこと（窓和の切り詰めの罠）。

    最低運転時間の窓和を「期間の内側だけ」で書くと、この号機の 1 時刻目の
    拘束が消え、経済的に不利な号機が即座に止まる解が出る。教材として
    もっとも踏みやすい落とし穴なので単独のテストにしてある。
    """
    small = two_unit_case(
        case, u0=1, hours_in_state=1, min_up=8, min_down=2, var_cost=30000.0
    )
    result = unit_commitment(small, np.full(12, 60.0), gap=EXACT_GAP)
    assert result.schedule["TEST"][0] == 1.0
    assert result.dispatch["TEST"][0] >= 20.0 - 1e-6      # 運転中なら Pmin 以上


# ======================================================================
# 3. 制約充足の独立な再計算
# ======================================================================
def test_full_case_solution_satisfies_every_constraint(case, summer_result):
    """wscc9 の 24 時間解が、モデルとは別に組んだ検算をすべて通ること。"""
    assert summer_result.status == "Optimal"
    verify_result(summer_result, reserve_rate=case.commitment["reserve_rate"])


def test_priority_list_solution_satisfies_every_constraint(case, summer):
    """優先順位法の解も同じ検算を通ること（入切表がヒューリスティックでも、
    出力は入切を固定した線形計画で決めているので厳密に実行可能である）。"""
    result = priority_list(case, summer)
    verify_result(result, reserve_rate=case.commitment["reserve_rate"])


def test_reserve_is_unloaded_synchronised_capacity(case):
    """予備力が Σ(Pmax u - p) であって Σ Pmax u >= (1+r)D ではないこと。

    需要を極端に大きくすると供給不足が立つ。このとき同期並列容量は
    ``(1+r) D`` にまったく届かないが、**出していない容量**としての予備力は
    要求を満たしている。2 つの書き方が別物であることの実例である。
    """
    demand = demand_profile(case) * 2.5
    result = unit_commitment(case, demand)
    rate = case.commitment["reserve_rate"]
    committed = np.array([result.committed_mw(t) for t in range(len(demand))])

    assert np.all(result.shortfall_mw > 0.0)
    assert np.all(committed < (1.0 + rate) * result.demand_mw)      # 素朴な式なら実行不可能
    assert np.all(result.reserve_mw() >= rate * result.demand_mw - 1e-6)
    verify_result(result, reserve_rate=rate)


# ======================================================================
# 4. 区分線形近似の収束
# ======================================================================
@pytest.fixture(scope="module")
def single_hour(case) -> tuple[Case, list[Unit], float]:
    """1 時刻・全機運転が強制されたケース。入切の自由度を消してある。

    ``min_up=24`` で ``hours_in_state=1`` にすると
    ``remaining_min_up() = 23`` となり、期間中は全機が運転を強制される。
    こうすると混合整数計画の自由度は出力配分だけになり、厳密な経済負荷
    配分と直接比べられる。
    """
    by_name = {unit.name: unit for unit in case.units}
    units = [
        relaxed(by_name[name], min_up=24, min_down=24, u0=1, hours_in_state=1,
                startup_cost=0.0)
        for name in ("G1-1", "G2-1", "G3-1")
    ]
    return replace(case, units=units, commitment={"reserve_rate": 0.10}), units, 150.0


def test_piecewise_error_shrinks_monotonically_and_overestimates(single_hour):
    """分割数を増やすと誤差が単調に減り、常に過大評価側であること。

    割線は凸な 2 次曲線の上側に来るので、区分線形の最適値は厳密解を
    **下回れない**。分割を細かくすると誤差は単調に減り、1 セグメントあたりの
    理論上界 :math:`c_2 L^2 / 4` を超えない。
    """
    small, units, demand = single_hour
    _lam, exact_cost = exact_dispatch(units, demand)

    errors = []
    for n_segments in (1, 2, 4, 8):
        result = unit_commitment(
            small, np.array([demand]), n_segments=n_segments, gap=EXACT_GAP
        )
        error = result.total_cost - exact_cost
        bound = sum(
            unit.quadratic * ((unit.p_max_mw - unit.p_min_mw) / n_segments) ** 2 / 4.0
            for unit in units
        )
        assert error > 0.0, f"K={n_segments} で過小評価になっている"
        assert error <= bound, f"K={n_segments} で理論上界 {bound} を超えた"
        errors.append(error)

    assert errors == sorted(errors, reverse=True), f"単調でない: {errors}"
    assert errors[-1] < errors[0] / 10.0, "分割を 8 倍にしても誤差がほとんど減っていない"


def test_piecewise_cost_matches_independent_evaluation(case, summer_result):
    """出力から数え直した区分線形費用が、内訳の燃料費と一致すること。"""
    n_segments = summer_result.options["n_segments"]
    fuel = 0.0
    for unit in case.units:
        commitment = summer_result.schedule[unit.name]
        output = summer_result.dispatch[unit.name]
        for t in range(len(commitment)):
            if commitment[t] > 0.5:
                fuel += piecewise_fuel(unit, float(output[t]), n_segments)
    assert fuel == pytest.approx(summer_result.cost_breakdown["fuel"], rel=1e-9)
    # 2 次曲線そのものの費用は、区分線形の評価を必ず下回る。
    exact = sum(
        float(np.sum(
            unit.quadratic * summer_result.dispatch[unit.name] ** 2
            + unit.var_cost * summer_result.dispatch[unit.name]
        ))
        for unit in case.units
    )
    assert exact < summer_result.cost_breakdown["fuel"]


# ======================================================================
# 5. 優先順位法との比較
# ======================================================================
@pytest.fixture(scope="module")
def light_pair(case) -> tuple[CommitmentResult, CommitmentResult]:
    """軽負荷日の需要に対する、混合整数計画と優先順位法の結果。

    軽負荷日を選ぶのは、需要が小さいほど「どれを止めるか」の自由度が
    増え、優先順位法の弱点（起動費を見ないこと）が出やすいためである。
    """
    demand = demand_profile(case, "light_load")
    return unit_commitment(case, demand), priority_list(case, demand)


def test_milp_is_never_more_expensive_than_priority_list(light_pair):
    """混合整数計画の総費用が優先順位法を上回らないこと。

    優先順位法が作る入切表は混合整数計画の実行可能解の 1 つなので、
    最適値がそれを上回ることは（相対ギャップの範囲を除いて）あり得ない。
    """
    optimal, heuristic = light_pair
    assert optimal.total_cost <= heuristic.total_cost * (1.0 + 1e-4)


def test_difference_from_priority_list_is_mostly_startup_cost(light_pair):
    """優先順位法との費用差の大半が起動費であること。

    全負荷平均費用の順に並べる優先順位法は「いま安い」ことしか見ておらず、
    起動費と最低運転時間を見ていない。その結果、混合整数計画なら止めずに
    済ませる号機を何度も入り切りする。0-1 の意思決定を最適化することの
    価値が、どの費目に現れるかを固定するテストである。
    """
    optimal, heuristic = light_pair
    gap = heuristic.total_cost - optimal.total_cost
    startup_gap = heuristic.cost_breakdown["startup"] - optimal.cost_breakdown["startup"]

    assert gap > 0.0, "この需要では差が出ていない（テストの前提が壊れている）"
    assert startup_gap >= 0.5 * gap, (
        f"差 {gap:,.0f} 円のうち起動費は {startup_gap:,.0f} 円しかない"
    )
    assert optimal.n_startups() < heuristic.n_startups()


# ======================================================================
# 6. 費用の内訳
# ======================================================================
@pytest.mark.parametrize("method", ["milp", "priority", "enumeration"])
def test_cost_breakdown_adds_up_to_total(case, tiny, method):
    """内訳の合計が総費用に一致すること（どの解法でも）。

    内訳は入切と出力から **数え直した**値、総費用はソルバが最適化した値
    （全列挙では貪欲法が積み上げた値）である。両者が一致することは、
    モデルの読み方が正しいことの検算になる。
    """
    demand = np.array([110.0, 85.0, 70.0, 95.0, 150.0, 175.0])
    if method == "milp":
        result = unit_commitment(tiny, demand, gap=EXACT_GAP)
    elif method == "priority":
        result = priority_list(tiny, demand)
    else:
        result = enumerate_commitment(tiny, demand)

    assert set(result.cost_breakdown) == {"fuel", "noload", "startup", "penalty"}
    assert sum(result.cost_breakdown.values()) == pytest.approx(
        result.total_cost, rel=1e-9
    )
    assert all(value >= 0.0 for value in result.cost_breakdown.values())


def test_noload_and_startup_costs_match_hand_calculation(case, summer_result):
    """無負荷費と起動費を、入切表から手で数えた値と突き合わせること。"""
    noload = sum(
        unit.noload_cost * float(summer_result.schedule[unit.name].sum())
        for unit in case.units
    )
    startup = 0.0
    for unit in case.units:
        commitment = summer_result.schedule[unit.name]
        previous = np.concatenate(([float(unit.u0)], commitment[:-1]))
        startup += unit.startup_cost * float(np.sum(commitment > previous))
    assert noload == pytest.approx(summer_result.cost_breakdown["noload"], rel=1e-12)
    assert startup == pytest.approx(summer_result.cost_breakdown["startup"], rel=1e-12)


# ======================================================================
# 7. 供給不足
# ======================================================================
def test_extreme_demand_produces_shortfall(case):
    """需要を極端に大きくすると、実行不可能にならず供給不足が立つこと。"""
    demand = demand_profile(case) * 3.0
    result = unit_commitment(case, demand)

    assert result.status == "Optimal"
    assert np.all(result.shortfall_mw > 0.0)
    assert result.cost_breakdown["penalty"] == pytest.approx(
        DEFAULT_VOLL * result.shortfall_mw.sum(), rel=1e-9
    )
    # 「12 時に何 MW 足りないか」が読める形になっていること。
    worst = int(np.argmax(result.shortfall_mw))
    assert result.shortfall_mw[worst] == pytest.approx(
        demand[worst] - result.committed_mw(worst) + result.reserve_mw()[worst], abs=1e-6
    )


def test_shortfall_can_be_forbidden_with_japanese_error(case):
    """``allow_shortfall=False`` なら日本語の ValueError で止まること。"""
    demand = demand_profile(case) * 3.0
    with pytest.raises(ValueError, match="実行不可能"):
        unit_commitment(case, demand, allow_shortfall=False)
    with pytest.raises(ValueError, match="不足"):
        unit_commitment(case, demand, allow_shortfall=False)


def test_reserve_requirement_beyond_capacity_is_diagnosed(case):
    """予備力要求が原理的に満たせないとき、その旨を日本語で言うこと。"""
    with pytest.raises(ValueError, match="未負荷容量の上限"):
        unit_commitment(case, demand_profile(case), reserve_rate=1.5)


# ======================================================================
# 8. 実用規模の求解時間
# ======================================================================
def test_full_case_solves_within_thirty_seconds(case, summer):
    """wscc9 の 7 号機 x 24 時刻が 30 秒以内に解けること。

    授業中に返ってこない計算を教材にしない、という制約そのものである。
    """
    start = time.perf_counter()
    result = unit_commitment(case, summer)
    elapsed = time.perf_counter() - start

    assert result.status == "Optimal"
    assert elapsed < 30.0, f"{elapsed:.1f} 秒かかった"
    assert len(result.schedule) == 7
    assert all(row.size == 24 for row in result.schedule.values())


# ======================================================================
# 9. 限界費用
# ======================================================================
def test_marginal_prices_match_incremental_cost_of_the_marginal_unit(tiny):
    """限界費用が、その時刻に出力を動かせる号機の増分費用と一致すること。

    突き合わせる基準は 2 つある。

    1. **貪欲法のオラクル**が最後に埋めたブロックの傾き（＝限界的な
       セグメントの増分費用）
    2. **数値微分**。需要を ±0.5 MW 動かしたときの費用の中心差分。
       区分線形なのでセグメントの内側では厳密に線形であり、打ち切り誤差は
       ゼロになる。

    ランプ率を非拘束にしたケースを使うのは、時間方向の結合があると
    「その時刻だけ需要を動かす」という操作が他の時刻に波及して、
    1 時刻の増分費用と比べられなくなるためである。
    """
    demand = np.array([110.0, 85.0, 70.0, 95.0, 150.0, 175.0])
    result = unit_commitment(tiny, demand, gap=EXACT_GAP)
    prices = marginal_prices(tiny, result)

    units = list(tiny.units)
    schedule = np.array([result.schedule[unit.name] for unit in units])
    horizon = len(demand)
    assert prices.shape == (horizon,)

    for t in range(horizon):
        on = schedule[:, t] > 0.5
        previous = (
            schedule[:, t - 1] > 0.5 if t > 0
            else np.array([bool(unit.u0) for unit in units])
        )
        following = (
            schedule[:, t + 1] > 0.5 if t + 1 < horizon else [None] * len(units)
        )
        reserve = 0.10 * demand[t]
        cost, _total, shed, spill, expected = hour_oracle(
            units, on, previous, following, demand[t], reserve, DEFAULT_SEGMENTS
        )
        assert shed == 0.0 and spill == 0.0, "この需要では緩和変数が立たないはず"
        assert prices[t] == pytest.approx(expected, rel=1e-9), f"t={t}"

        # 数値微分（中心差分）。区分線形の内側なので厳密に一致する。
        step = 0.5
        upper = hour_oracle(
            units, on, previous, following, demand[t] + step, reserve, DEFAULT_SEGMENTS
        )[0]
        lower = hour_oracle(
            units, on, previous, following, demand[t] - step, reserve, DEFAULT_SEGMENTS
        )[0]
        assert prices[t] == pytest.approx((upper - lower) / (2.0 * step), rel=1e-9)


def test_marginal_prices_are_segment_slopes_of_committed_units(case, summer_result):
    """24 時間ケースでも、価格は必ず運転中の号機のセグメントの傾きであること。

    起動費と無負荷費は入切を固定した線形計画には現れないので、価格は
    ``c_1 + c_2 (x_{k-1} + x_k)`` の形の値しか取らない。
    """
    prices = marginal_prices(case, summer_result)
    assert prices.shape == (24,)
    assert np.all(prices > 0.0)

    for t, price in enumerate(prices):
        slopes = [
            slope
            for unit in case.units
            if summer_result.schedule[unit.name][t] > 0.5
            for slope, _width in cost_blocks(unit, summer_result.options["n_segments"])
        ]
        assert any(abs(price - slope) < 1e-6 * max(1.0, slope) for slope in slopes), (
            f"t={t} の価格 {price} が、運転中の号機のどのセグメントとも一致しない"
        )
        # 価格は運転中の号機の増分費用の幅の中に収まる。
        assert min(slopes) - 1e-6 <= price <= max(slopes) + 1e-6


def tiny_result(tiny) -> CommitmentResult:
    """小さなケースの解（限界費用の比較に使う）。"""
    return unit_commitment(
        tiny, np.array([110.0, 85.0, 70.0, 95.0, 150.0, 175.0]), gap=EXACT_GAP
    )


def test_marginal_prices_do_not_settle_the_total_cost(case, summer_result, tiny):
    """限界費用で精算した金額が総費用に一致しないこと（どちらにもずれうる）。

    入切を固定した線形計画の双対には、起動費も無負荷費も現れない。
    価格収入 :math:`\\sum_t \\pi_t D_t` から燃料費を引いた残り（限界的でない
    号機が稼ぐ、いわゆる inframarginal rent）が固定費を賄えるかどうかは
    **ケース次第**である。

    * wscc9 の 24 時間ケース: 残りが固定費を上回る（回収できている）
    * 3 号機 6 時刻の小さなケース: 残りが固定費に届かない（missing money）

    「限界費用価格は固定費を回収できない」という言い方は片側しか見ていない。
    符号がどちらにも振れることを数字で押さえておく。
    """
    for result, covers_fixed_cost in ((summer_result, True), (tiny_result(tiny), False)):
        prices = marginal_prices(result.case, result)
        revenue = float(np.sum(prices * result.demand_mw))
        surplus = revenue - result.cost_breakdown["fuel"]
        fixed = result.cost_breakdown["noload"] + result.cost_breakdown["startup"]
        assert revenue != pytest.approx(result.total_cost, rel=1e-3)
        assert (surplus > fixed) is covers_fixed_cost, (
            f"残り {surplus:,.0f} 円 と固定費 {fixed:,.0f} 円 の大小が想定と違う"
        )


# ======================================================================
# 10. 変動性再生可能電源（ダックカーブ）
# ======================================================================
def test_vre_steepens_the_evening_ramp(case, summer):
    """VRE を入れると夕方の立ち上がりが急になること（需要の形は変えない）。"""
    net = net_demand(case, summer, vre_mw=120.0)
    assert np.max(np.diff(net)) > np.max(np.diff(summer))
    assert net.min() < summer.min()
    assert np.argmin(net) in range(9, 16)      # 谷は日中に来る


def test_vre_changes_the_commitment(case, summer, summer_result):
    """VRE を入れると起動回数か出力抑制が変わること。"""
    with_vre = unit_commitment(case, summer, vre_mw=120.0)
    verify_result(with_vre, reserve_rate=case.commitment["reserve_rate"])
    assert (
        with_vre.n_startups() != summer_result.n_startups()
        or with_vre.spill_mw.sum() > summer_result.spill_mw.sum()
    )
    assert with_vre.total_cost < summer_result.total_cost   # 燃料が浮く


def test_heavy_vre_forces_curtailment(case, summer):
    """VRE が大きいと出力抑制が立ち、その時刻は最低出力に張り付くこと。

    抑制の緩和変数が無ければ、この時刻で問題は実行不可能になる。
    「解けない」ではなく「抑制した」と答えるのが正しい、という設計の
    根拠そのものである。
    """
    result = unit_commitment(case, summer, vre_mw=300.0)
    verify_result(result, reserve_rate=case.commitment["reserve_rate"])
    assert result.spill_mw.sum() > 0.0

    for t in np.flatnonzero(result.spill_mw > 0.0):
        minimum = sum(
            unit.p_min_mw * result.schedule[unit.name][t] for unit in case.units
        )
        generated = sum(result.dispatch[unit.name][t] for unit in case.units)
        assert minimum > result.demand_mw[t], (
            f"t={t}: 抑制が立っているのに最低出力の合計が純需要を超えていない"
        )
        assert generated == pytest.approx(minimum, abs=1e-6)


# ======================================================================
# 対称性除去（適用してよい範囲）
# ======================================================================
def duplicate_case(case: Case, var_costs: tuple[float, ...]) -> Case:
    """費用だけを差し替えた同型の号機からなるケース。

    2 台とも運転中から始める。停止中から始めると起動時の上限
    :math:`SU = P^{min}` が効いて 1 時刻目に必ず 2 台とも要る形になり、
    見たい対称性の話が見えなくなる。
    """
    units = [
        Unit(
            name=f"D{i + 1}", bus=1, p_max_mw=50.0, p_min_mw=20.0,
            var_cost=cost, quadratic=1.0, noload_cost=20000.0,
            startup_cost=50000.0, min_up=1, min_down=1,
            ramp_up=math.inf, ramp_down=math.inf, u0=1, hours_in_state=24,
        )
        for i, cost in enumerate(var_costs)
    ]
    return replace(case, units=units, commitment={"reserve_rate": 0.0})


def test_symmetry_breaking_orders_identical_units(case):
    """費用も諸元も同じ号機なら、番号順に起動すること（費用は変わらない）。"""
    twins = duplicate_case(case, (10000.0, 10000.0))
    demand = np.full(3, 40.0)
    ordered = unit_commitment(twins, demand, symmetry_breaking=True, gap=EXACT_GAP)
    free = unit_commitment(twins, demand, symmetry_breaking=False, gap=EXACT_GAP)

    assert ordered.total_cost == pytest.approx(free.total_cost, rel=1e-9)
    assert np.all(ordered.schedule["D1"] >= ordered.schedule["D2"])
    assert np.all(ordered.schedule["D1"] == 1.0)
    assert np.all(ordered.schedule["D2"] == 0.0)


def test_symmetry_breaking_is_not_applied_to_units_with_different_costs(case):
    """費用が違う号機には対称性除去を入れないこと。

    入れてしまうと「安い方を後ろの番号に置いた解」が切り落とされる。
    ここでは 2 号機の順番に対して **安いのは後ろ**にしてあるので、
    誤って制約を入れると最適解に到達できない。
    """
    pair = duplicate_case(case, (20000.0, 10000.0))     # 後ろの号機が安い
    demand = np.full(3, 40.0)
    result = unit_commitment(pair, demand, symmetry_breaking=True, gap=EXACT_GAP)

    assert np.all(result.schedule["D2"] == 1.0)
    assert np.all(result.schedule["D1"] == 0.0)
    reference = unit_commitment(pair, demand, symmetry_breaking=False, gap=EXACT_GAP)
    assert result.total_cost == pytest.approx(reference.total_cost, rel=1e-9)


def test_symmetry_breaking_has_no_effect_on_wscc9(case, summer, summer_result):
    """wscc9 では対称性除去が 1 本も入らないこと（同一プラント内でも費用が違う）。"""
    free = unit_commitment(case, summer, symmetry_breaking=False)
    assert free.total_cost == pytest.approx(summer_result.total_cost, rel=1e-9)


# ======================================================================
# 需要の作り方
# ======================================================================
def test_demand_profile_scales_the_shape_by_the_peak(case):
    """需要形状に peak_mw を掛けたものが返ること。"""
    demand = demand_profile(case, "summer_weekday")
    shape = np.asarray(case.commitment["profiles"]["summer_weekday"])
    assert demand.shape == (24,)
    assert np.allclose(demand, shape * case.commitment["peak_mw"])
    assert demand.max() == pytest.approx(315.0)

    # light_load は形状の最大が 1.0 未満なので、系列の最大は peak_mw より小さい。
    light = demand_profile(case, "light_load")
    assert light.max() < case.commitment["peak_mw"]
    assert np.allclose(demand_profile(case, "light_load", peak_mw=100.0), light / 3.15)


def test_demand_profile_rejects_unknown_names(case):
    """知らない形状の名前は、使える名前を添えて日本語で断ること。"""
    with pytest.raises(ValueError, match="使えるのは"):
        demand_profile(case, "spring_weekend")


def test_net_demand_uses_the_case_vre_layer(case, summer):
    """``vre_mw`` を省くと、ケースの VRE 層を定格容量のまま使うこと。"""
    block = case.commitment["vre"]
    expected = summer - np.asarray(block["profile"]) * block["capacity_mw"]
    assert np.allclose(net_demand(case, summer), expected)
    # スカラーは設備容量、配列は出力そのもの。
    assert np.allclose(
        net_demand(case, summer, vre_mw=60.0),
        summer - np.asarray(block["profile"]) * 60.0,
    )
    assert np.allclose(net_demand(case, summer, vre_mw=np.zeros(24)), summer)


def test_net_demand_checks_the_length(case, summer):
    """VRE の長さが需要と合わなければ日本語で止まること。"""
    with pytest.raises(ValueError, match="長さ"):
        net_demand(case, summer, vre_mw=np.zeros(12))


# ======================================================================
# 結果オブジェクトの読み出し
# ======================================================================
def test_result_accessors_match_hand_calculation(case, summer_result):
    """``committed_mw`` / ``reserve_mw`` / ``n_startups`` を手計算と突き合わせる。"""
    committed = sum(
        unit.p_max_mw for unit in case.units if summer_result.schedule[unit.name][0] > 0.5
    )
    assert summer_result.committed_mw(0) == pytest.approx(committed)

    generated = sum(summer_result.dispatch[unit.name][0] for unit in case.units)
    assert summer_result.reserve_mw()[0] == pytest.approx(committed - generated)

    startups = 0
    for unit in case.units:
        commitment = summer_result.schedule[unit.name]
        previous = np.concatenate(([float(unit.u0)], commitment[:-1]))
        startups += int(np.sum(commitment > previous))
    assert summer_result.n_startups() == startups


def test_table_and_summary_are_printable(summer_result):
    """入切表と要約が例外なく作れ、号機名と時刻数が揃っていること。"""
    table = summer_result.to_table()
    lines = table.splitlines()
    assert lines[0].startswith("unit commitment schedule")
    for unit_name in summer_result.schedule:
        row = [line for line in lines if line.startswith(unit_name)][0]
        marks = row.split()[1]
        assert len(marks) == 24
        assert set(marks) <= {"#", "."}
    assert "起動停止計画" in summer_result.summary()


def test_price_is_the_value_of_lost_load_when_demand_is_shed(case):
    """供給不足が立っている時刻の限界費用が VOLL になること。

    需要が 1 MWh 増えても賄う手段がないので、増えた分はそのまま供給不足に
    なる。価格は「賄う費用」ではなく「賄えなかったときの費用」になる。
    """
    result = unit_commitment(case, demand_profile(case) * 3.0)
    prices = marginal_prices(case, result)
    assert np.all(result.shortfall_mw > 0.0)
    assert np.allclose(prices, DEFAULT_VOLL)


def test_enumeration_accepts_unit_names(case):
    """``units=`` に号機名を渡して部分集合を取れること。"""
    by_name = {unit.name: unit for unit in case.units}
    subset = replace(
        case,
        units=[relaxed(by_name[n], min_up=1, min_down=1) for n in ("G1-1", "G3-1")],
        commitment={"reserve_rate": 0.05},
    )
    result = enumerate_commitment(
        subset, np.array([60.0, 90.0, 70.0]), units=["G1-1", "G3-1"]
    )
    assert set(result.schedule) == {"G1-1", "G3-1"}
    verify_result(result, reserve_rate=0.05)

    with pytest.raises(ValueError, match="使えるのは"):
        enumerate_commitment(subset, np.array([60.0]), units=["G9-9"])


def test_case_without_units_layer_is_refused(case):
    """``units`` 層が無いケースは、どの回で必要になるかを添えて断ること。"""
    bare = replace(case, units=[], commitment={}, reliability={}, stability={})
    with pytest.raises(ValueError, match="units"):
        unit_commitment(bare, np.array([100.0]))


def test_net_demand_without_vre_layer_is_refused(case, summer):
    """VRE 層を持たないケースで ``vre_mw`` を省いたら日本語で断ること。"""
    bare = replace(case, commitment={"peak_mw": 315.0})
    with pytest.raises(ValueError, match="commitment.vre 層が無い"):
        net_demand(bare, summer)
