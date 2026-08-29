"""多機系統（Phase 6）の検証。

ケースファイルの潮流解とネットワークデータが整合していることを
最初に確かめる。ここが食い違っていると、以降の計算はすべて
「解けているが間違っている」状態になり、原因の切り分けが難しくなる。
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from genstab import FaultSchedule, eac, simulate
from genstab.events import Stage
from genstab.multimachine import electrical_power, load_case

CASE = Path(__file__).resolve().parents[1] / "cases" / "wscc9.yaml"

#: 事故前潮流解における各母線の複素注入電力 [p.u.]
EXPECTED_INJECTIONS = {
    1: 0.716 + 0.270j,
    2: 1.630 + 0.067j,
    3: 0.850 - 0.109j,
    4: 0.0 + 0.0j,
    5: -(1.25 + 0.50j),
    6: -(0.90 + 0.30j),
    7: 0.0 + 0.0j,
    8: -(1.00 + 0.35j),
    9: 0.0 + 0.0j,
}


@pytest.fixture(scope="module")
def system():
    return load_case(CASE)


# ----------------------------------------------------------------------
# データの整合性
# ----------------------------------------------------------------------
def test_power_flow_solution_is_consistent_with_network_data(system):
    """ケースファイルの潮流解が線路データと整合すること。

    母線電圧から計算した注入電力が、発電量および負荷と一致するかを
    確かめる。許容差は潮流解の記載桁数（位相 4 桁）で決まる。
    """
    injections = system.network.power_injections()
    for bus, expected in EXPECTED_INJECTIONS.items():
        assert abs(injections[bus] - expected) < 2e-3, f"母線 {bus} が不整合"


def test_internal_voltages_match_textbook_values(system):
    """内部起電力が Anderson & Fouad の値と一致すること。"""
    expected_magnitude = [1.0566, 1.0502, 1.0170]
    expected_angle_deg = [2.2717, 19.7315, 13.1752]
    assert system.emf_magnitude == pytest.approx(expected_magnitude, abs=1e-3)
    assert np.degrees(system._delta0) == pytest.approx(expected_angle_deg, abs=1e-2)


def test_reduced_matrix_has_machine_dimension(system):
    """Kron 縮約で発電機の数まで縮むこと。"""
    for stage in (Stage.PRE, Stage.FAULT, Stage.POST):
        assert system.reduced_matrix(stage).shape == (3, 3)
    # 3 つの状態はそれぞれ異なるネットワークを表す。
    assert not np.allclose(
        system.reduced_matrix(Stage.PRE), system.reduced_matrix(Stage.FAULT)
    )
    assert not np.allclose(
        system.reduced_matrix(Stage.PRE), system.reduced_matrix(Stage.POST)
    )


def test_reduced_matrix_is_symmetric(system):
    """縮約行列が対称であること（受動素子だけで構成されるため）。"""
    for stage in (Stage.PRE, Stage.FAULT, Stage.POST):
        Y = system.reduced_matrix(stage)
        assert np.allclose(Y, Y.T)


def test_faulted_network_transfers_no_power_from_faulted_bus(system):
    """母線 7 の地絡で、そこに直結する G2 の出力がほぼ失われること。"""
    power_pre = electrical_power(
        system.reduced_matrix(Stage.PRE), system.emf_magnitude, system._delta0
    )
    power_fault = electrical_power(
        system.reduced_matrix(Stage.FAULT), system.emf_magnitude, system._delta0
    )
    # G2 は母線 2 経由で地絡母線 7 に直結しているので出力が大きく落ちる。
    assert power_fault[1] < 0.05 * power_pre[1]


# ----------------------------------------------------------------------
# 物理的な不変性
# ----------------------------------------------------------------------
def test_electrical_power_depends_only_on_angle_differences(system):
    """全機の角度を一律にずらしても電気出力が変わらないこと。

    電気出力は角度差だけで決まる。この対称性が崩れていれば
    縮約行列か出力計算に誤りがある。
    """
    Y = system.reduced_matrix(Stage.PRE)
    base = electrical_power(Y, system.emf_magnitude, system._delta0)
    shifted = electrical_power(
        Y, system.emf_magnitude, system._delta0 + math.radians(37.0)
    )
    assert base == pytest.approx(shifted, abs=1e-12)


def test_operating_point_is_an_equilibrium(system):
    """事故前定常状態で全機の状態微分がゼロであること。

    ただしこれは弱い検証である。機械入力 Pm0 を縮約ネットワークから
    計算した電気出力として求めているため、発電機の宣言出力が誤っていても
    残差は必ずゼロになる。データの正しさは
    `test_broken_generation_data_is_rejected` が受け持つ。
    """
    dx = system.derivatives(0.0, system.initial_state(), Stage.PRE)
    assert np.max(np.abs(dx)) < 1e-12


def test_broken_generation_data_is_rejected():
    """発電機の宣言出力が潮流解と食い違うケースを拒否すること。

    Pm0 が Pe から再計算されるため、平衡点の残差だけを見ていると
    データの誤りに気づけない。実際、G2 の宣言出力を 1.63 から 0.30 に
    書き換えても残差はゼロのままだった。構築時に潮流解と突き合わせて
    検出する。
    """
    original = load_case(CASE)
    with pytest.raises(ValueError, match="潮流解とネットワーク"):
        replace(
            original,
            generators=[
                replace(g, P=0.30) if g.name == "G2" else g
                for g in original.generators
            ],
        )


def test_broken_load_data_is_rejected():
    """負荷が潮流解と食い違うケースを拒否すること。"""
    from genstab.multimachine import Load, MultiMachineNetwork

    original = load_case(CASE)
    network = MultiMachineNetwork(
        buses=original.network.buses,
        branches=original.network.branches,
        loads=[
            Load(bus=5, P=2.00, Q=0.50) if item.bus == 5 else item
            for item in original.network.loads
        ],
        voltages=original.network.voltages,
    )
    with pytest.raises(ValueError, match="母線 5"):
        replace(original, network=network)


def test_broken_line_data_is_rejected():
    """線路定数が潮流解と食い違うケースを拒否すること。"""
    from genstab.multimachine import Branch, MultiMachineNetwork

    original = load_case(CASE)
    network = MultiMachineNetwork(
        buses=original.network.buses,
        branches=[
            Branch(4, 5, 0.010, 0.12, 0.176) if item.key() == (4, 5) else item
            for item in original.network.branches
        ],
        loads=original.network.loads,
        voltages=original.network.voltages,
    )
    with pytest.raises(ValueError, match="潮流解とネットワーク"):
        replace(original, network=network)


def test_verify_power_flow_returns_injections(system):
    """検証メソッドが注入電力を返すこと（notebook で表示に使う）。"""
    injections = system.verify_power_flow()
    assert set(injections) == set(system.network.buses)
    for bus, expected in EXPECTED_INJECTIONS.items():
        assert abs(injections[bus] - expected) < 2e-3


def test_center_of_inertia_angles_sum_to_zero(system):
    """COI 基準の角度が慣性加重で打ち消し合うこと。"""
    result = simulate(system, t_end=3.0, dt=0.005)
    inertia = np.array([g.H for g in system.generators])
    weighted = sum(
        inertia[k] * result[f"delta_coi_{g.name}"]
        for k, g in enumerate(system.generators)
    )
    assert np.max(np.abs(weighted)) < 1e-9


# ----------------------------------------------------------------------
# 過渡安定性
# ----------------------------------------------------------------------
def test_standard_case_is_stable(system):
    """標準ケース（5 サイクル除去）で安定であること。"""
    result = simulate(system, t_end=5.0, dt=0.002)
    assert result.is_stable()
    # 機器間の角度差は 90° 程度までに収まる。
    assert np.degrees(result["max_separation"].max()) < 120.0


def test_slow_clearing_causes_loss_of_synchronism(system):
    """事故除去が遅れると脱調すること。"""
    slow = replace(system, fault=FaultSchedule(t_fault=1.0, t_clear=1.30))
    result = simulate(slow, t_end=5.0, dt=0.002)
    assert not result.is_stable()


def test_critical_clearing_time_is_in_expected_range(system):
    """CCT が文献で報告される範囲に入ること。"""
    cct = eac.critical_clearing_time(
        system, t_end=5.0, tolerance=1e-3, upper_bound=1.0
    )
    assert 0.12 < cct < 0.22
    # 標準ケースの除去時間は CCT より短い（だから安定）。
    assert system.fault.clearing_time < cct


def test_inter_machine_oscillation_is_in_local_mode_band(system):
    """事故後の機器間動揺が局所モードの周波数帯に入ること。"""
    result = simulate(system, t_end=6.0, dt=0.002)
    after = result.t > 1.5
    signal = result["delta_coi_G2"][after]
    signal = signal - signal.mean()

    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, d=0.002)
    peak = frequencies[np.argmax(spectrum[1:]) + 1]
    # 局所動揺モードは 0.7〜2 Hz が典型。
    assert 0.7 < peak < 2.0


def test_smallest_inertia_machine_swings_most(system):
    """慣性の小さい機ほど大きく振れること。"""
    result = simulate(system, t_end=5.0, dt=0.002)
    swings = {
        g.name: np.ptp(result[f"delta_coi_{g.name}"]) for g in system.generators
    }
    # H: G1=23.64 > G2=6.40 > G3=3.01、出力は G2 が最大。
    assert swings["G1"] < swings["G3"]
    assert swings["G1"] < swings["G2"]
