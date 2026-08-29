"""等面積法と臨界事故除去時間（Phase 2）の検証。

要点は「等面積法の解析解」と「時間領域シミュレーションの二分探索で
得た数値解」が一致することである。両者は独立に導かれるため、一致は
動揺方程式の実装と等面積法の実装の相互検証になる。
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from genstab import ClassicalMachine, FaultSchedule, SMIBNetwork, SMIBSystem, eac, simulate


def _undamped_system(D: float = 0.0) -> SMIBSystem:
    """等面積法の前提（制動なし・事故中は無電力）に合わせた系。"""
    machine = ClassicalMachine(H=5.0, D=D, x_d_prime=0.3, E=1.1)
    network = SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6, V_inf=1.0)
    return SMIBSystem(machine, network, FaultSchedule(t_fault=1.0, t_clear=1.1), Pe0=0.8)


def test_areas_balance_at_critical_angle():
    """臨界事故除去角では加速面積と減速面積が一致すること。"""
    outcome = eac.evaluate(_undamped_system())
    assert outcome.accelerating_area == pytest.approx(outcome.decelerating_area, rel=1e-10)
    assert outcome.margin == pytest.approx(0.0, abs=1e-10)


def test_unstable_equilibrium_angle_is_analytic():
    """δ_u = π - arcsin(Pm / Pmax_post) であること。"""
    system = _undamped_system()
    _, _, pmax_post, pm = eac._power_amplitudes(system)
    assert eac.unstable_equilibrium_angle(system) == pytest.approx(
        math.pi - math.asin(pm / pmax_post)
    )


def test_analytic_and_numerical_cct_agree():
    """解析 CCT と数値 CCT が一致すること（制動なし）。"""
    system = _undamped_system()
    t_analytic = eac.critical_clearing_time_analytic(system)
    t_numeric = eac.critical_clearing_time(system, tolerance=1e-5, t_end=20.0)
    assert t_numeric == pytest.approx(t_analytic, rel=2e-3)


def test_simulation_agrees_with_area_verdict():
    """臨界の前後で時間領域シミュレーションの判定が切り替わること。"""
    system = _undamped_system()
    t_cr = eac.critical_clearing_time_analytic(system)
    delta_u = eac.unstable_equilibrium_angle(system)
    limit = delta_u - system.operating_point.delta

    from dataclasses import replace

    stable = simulate(
        replace(system, fault=FaultSchedule(1.0, 1.0 + 0.9 * t_cr)), t_end=20.0
    )
    unstable = simulate(
        replace(system, fault=FaultSchedule(1.0, 1.0 + 1.1 * t_cr)), t_end=20.0
    )
    assert stable.is_stable(angle_limit=limit)
    assert not unstable.is_stable(angle_limit=limit)


def test_damping_extends_cct():
    """制動があると CCT が延びること（制動がエネルギーを吸収するため）。"""
    t_undamped = eac.critical_clearing_time(_undamped_system(D=0.0), tolerance=1e-5, t_end=20.0)
    t_damped = eac.critical_clearing_time(_undamped_system(D=2.0), tolerance=1e-5, t_end=20.0)
    assert t_damped > t_undamped


def test_analytic_cct_rejects_damped_system():
    """制動がある系では解析解が拒否されること。

    等面積法はエネルギー保存を前提にしているので、制動があると
    面積の釣り合いが崩れる。以前は警告だけで値を返していたが、
    誤った値をそのまま使われる危険があるため拒否する。
    """
    with pytest.raises(ValueError, match="制動係数"):
        eac.critical_clearing_time_analytic(_undamped_system(D=2.0))


def test_analytic_cct_allows_explicit_approximation():
    """近似と承知していれば警告つきで計算できること。"""
    with pytest.warns(UserWarning, match="適用条件を満たしていない"):
        value = eac.critical_clearing_time_analytic(
            _undamped_system(D=2.0), allow_approximation=True
        )
    assert value > 0.0


def test_analytic_cct_rejects_finite_fault_reactance():
    """事故中に電力を送れる場合、解析解は使えないと明示されること。"""
    machine = ClassicalMachine(H=5.0, D=0.0, x_d_prime=0.3, E=1.1)
    network = SMIBNetwork(x_pre=0.4, x_fault=1.5, x_post=0.6)
    system = SMIBSystem(machine, network, FaultSchedule(1.0, 1.1), Pe0=0.8)
    with pytest.raises(ValueError, match="Pmax_fault = 0"):
        eac.critical_clearing_time_analytic(system)
    # 数値解のほうは問題なく求まり、事故中も送電できるぶん CCT は長くなる。
    assert eac.critical_clearing_time(system, tolerance=1e-4, t_end=20.0) > 0.2


def test_no_post_fault_equilibrium_is_rejected():
    """事故後に平衡点が存在しない設定は明示的に拒否されること。"""
    machine = ClassicalMachine(H=5.0, D=0.0, x_d_prime=0.3, E=1.1)
    network = SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=5.0)
    system = SMIBSystem(machine, network, FaultSchedule(1.0, 1.1), Pe0=0.8)
    with pytest.raises(ValueError, match="平衡点が存在しない"):
        eac.unstable_equilibrium_angle(system)


# ----------------------------------------------------------------------
# 適用条件と観測時間（codex の指摘 #2, #3 への対応）
# ----------------------------------------------------------------------
def _one_axis_system():
    """1 軸モデル + AVR（等面積法の前提から外れる系）。"""
    import genstab

    machine = genstab.OneAxisMachine(
        H=3.5, D=0.0, xd=1.81, xd_prime=0.30, Td0_prime=8.0, Vt0=1.0
    )
    network = genstab.SMIBNetwork(x_pre=0.65, x_fault=np.inf, x_post=0.65, V_inf=0.995)
    return genstab.SMIBSystem(
        machine, network, FaultSchedule(1.0, 1.1), Pe0=0.9,
        controllers=[genstab.SimpleExciter(Ka=200.0, Ta=0.05)],
    )


def test_equal_area_rejects_non_classical_machine():
    """1 軸モデルには等面積法を適用させないこと。

    内部起電力 E'q が時間とともに変わるため、初期値から求めた面積は
    保存量を表さない。以前は数値を返していたが、初期 E'q から求めた
    不安定平衡点（130 度）と真の鞍点（75 度前後）が大きく食い違い、
    誤った CCT を返していた。
    """
    system = _one_axis_system()
    with pytest.raises(ValueError, match="古典モデルにのみ適用"):
        eac.critical_clearing_angle(system)
    with pytest.raises(ValueError, match="制御器が接続されている"):
        eac.evaluate(system)


def test_equal_area_allows_approximation_with_warning():
    """近似として明示すれば計算でき、その旨が警告されること。"""
    system = _one_axis_system()
    with pytest.warns(UserWarning, match="適用条件を満たしていない"):
        outcome = eac.evaluate(system, allow_approximation=True)
    assert outcome.delta_u > outcome.delta_0


def test_check_assumptions_reports_each_violation():
    """満たされていない条件が個別に報告されること。"""
    system = _one_axis_system()
    with pytest.warns(UserWarning, match="適用条件を満たしていない"):
        problems = eac.check_assumptions(system, allow_approximation=True)
    assert len(problems) == 2
    assert any("古典モデル" in p for p in problems)
    assert any("制御器" in p for p in problems)
    assert eac.check_assumptions(_undamped_system()) == []


def test_cct_warns_when_observation_window_is_too_short():
    """観測時間が短いと CCT を過大評価するので警告すること。

    脱調する軌道が角度しきい値に達する前に計算が終わると、二分探索は
    それを安定とみなす。実測で真値 206 ms に対して 390 ms（+90 %）が
    返ることを確認している。
    """
    system = _undamped_system()
    exact = eac.critical_clearing_time_analytic(system)

    with pytest.warns(UserWarning, match="過大評価"):
        too_short = eac.critical_clearing_time(system, t_end=1.0, tolerance=1e-5)
    assert too_short > 1.5 * exact, "この条件では実際に大きく過大評価される"


def test_cct_is_accurate_with_default_observation_window():
    """既定の観測時間なら警告なしで正しい値が得られること。"""
    system = _undamped_system()
    exact = eac.critical_clearing_time_analytic(system)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        value = eac.critical_clearing_time(system, tolerance=1e-5)
    assert value == pytest.approx(exact, rel=2e-3)
