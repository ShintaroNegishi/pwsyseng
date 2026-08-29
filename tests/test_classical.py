"""古典モデル（Phase 1）の検証。

既存の研究室コード ``GeneratorControl/simulation.py`` の定式化を参照実装
として再現し、genstab の標準形と数値的に一致することを確認する。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from genstab import (
    ClassicalMachine,
    FaultSchedule,
    SMIBNetwork,
    SMIBSystem,
    SystemBase,
    simulate,
)

# 既存コードのパラメータ
M, D_COEF, PM, EF, V_INF = 10.0, 2.0, 0.6, 1.0, 1.0
X_PRE, X_FAULT, X_POST = 0.7, np.inf, 1.0
T_FAULT, T_CLEAR, T_END = 10.0, 12.0, 30.0
#: 既存コードは dδ/dt = ω（速度偏差を rad/s で持つ形）なので ω_s = 1 とする
UNIT_OMEGA_BASE = SystemBase(frequency_hz=1.0 / (2.0 * np.pi))


def _reference_trajectory(grid: np.ndarray) -> np.ndarray:
    """既存コードの式をそのまま積分した参照解（不連続点で区間分割）。"""

    def rhs(t, y, x_total):
        delta, omega = y
        return [
            omega,
            (1.0 / M) * (PM - (EF * V_INF / x_total) * np.sin(delta))
            - (D_COEF / M) * omega,
        ]

    y = [np.arcsin(PM * X_PRE / (EF * V_INF)), 0.0]
    segments = [(0.0, T_FAULT, X_PRE), (T_FAULT, T_CLEAR, X_FAULT), (T_CLEAR, T_END, X_POST)]
    parts = []
    for a, b, x_total in segments:
        mask = (grid >= a) & (grid <= b) if b == T_END else (grid >= a) & (grid < b)
        sol = solve_ivp(rhs, (a, b), y, args=(x_total,), t_eval=grid[mask],
                        rtol=1e-10, atol=1e-12)
        parts.append(sol.y)
        end = solve_ivp(rhs, (a, b), y, args=(x_total,), rtol=1e-10, atol=1e-12)
        y = end.y[:, -1]
    return np.concatenate(parts, axis=1)


def _reference_system(x_d_prime: float = 0.1) -> SMIBSystem:
    machine = ClassicalMachine(H=M / 2.0, D=D_COEF, x_d_prime=x_d_prime, E=EF)
    network = SMIBNetwork(
        x_pre=X_PRE - x_d_prime,
        x_fault=X_FAULT,
        x_post=X_POST - x_d_prime,
        V_inf=V_INF,
    )
    return SMIBSystem(
        machine, network, FaultSchedule(T_FAULT, T_CLEAR),
        Pe0=PM, base=UNIT_OMEGA_BASE,
    )


def test_matches_reference_implementation():
    """既存コードの定式化と機械精度で一致すること。"""
    system = _reference_system()
    result = simulate(system, t_end=T_END, dt=0.005, rtol=1e-10, atol=1e-12)
    reference = _reference_trajectory(result.t)

    assert np.max(np.abs(result.delta - reference[0])) < 1e-10
    assert np.max(np.abs(result.omega - reference[1])) < 1e-10


def test_initial_angle_matches_analytic_value():
    """初期位相角が arcsin(Pe0 * X / (E * V)) と一致すること。"""
    system = _reference_system()
    expected = np.arcsin(PM * X_PRE / (EF * V_INF))
    assert system.operating_point.delta == pytest.approx(expected, abs=1e-12)
    # 定常では機械入力と電気出力が釣り合い、速度偏差はゼロ。
    assert system.operating_point.Pm == pytest.approx(system.operating_point.Pe)
    assert system.initial_state()[1] == pytest.approx(0.0)


def test_no_power_transfer_during_fault():
    """x_fault = inf のとき事故中の電気出力がゼロになること。"""
    system = _reference_system()
    result = simulate(system, t_end=T_END, dt=0.005)
    during = (result.t > T_FAULT) & (result.t < T_CLEAR)
    assert np.max(np.abs(result.Pe[during])) < 1e-12


def test_converges_to_post_fault_equilibrium():
    """制動があれば事故後平衡点 arcsin(Pm * X_post / (E * V)) に収束すること。"""
    system = _reference_system()
    result = simulate(system, t_end=200.0, dt=0.01)
    expected = np.arcsin(PM * X_POST / (EF * V_INF))
    assert result.delta[-1] == pytest.approx(expected, abs=1e-3)
    assert result.omega[-1] == pytest.approx(0.0, abs=1e-5)


def test_undamped_machine_oscillates_without_decay():
    """制動 D=0 なら振幅が減衰しないこと（エネルギー保存の確認）。"""
    machine = ClassicalMachine(H=5.0, D=0.0, x_d_prime=0.3, E=1.1)
    network = SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.4)
    system = SMIBSystem(machine, network, FaultSchedule(1.0, 1.05), Pe0=0.8)
    result = simulate(system, t_end=30.0, dt=0.005, rtol=1e-11, atol=1e-13)

    after = result.t > 1.05
    delta_after = result.delta[after]
    first_half = delta_after[: delta_after.size // 2]
    second_half = delta_after[delta_after.size // 2 :]
    amp1 = first_half.max() - first_half.min()
    amp2 = second_half.max() - second_half.min()
    assert amp2 == pytest.approx(amp1, rel=1e-3)


def test_system_without_controllers_reduces_to_swing_equation():
    """制御器を渡さなければ状態数が発電機モデルのままであること。"""
    system = _reference_system()
    assert system.n_states == 2
    assert system.state_names == ("delta", "omega")
    assert system.controllers == []


def test_permanent_fault_is_unstable():
    """事故を除去しなければ脱調すること。"""
    machine = ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
    network = SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.4)
    system = SMIBSystem(machine, network, FaultSchedule(1.0, np.inf), Pe0=0.8)
    result = simulate(system, t_end=10.0, dt=0.01)
    assert not result.is_stable()


def test_rejects_operating_point_beyond_transfer_limit():
    """送電可能な最大電力を超える動作点は明示的に拒否されること。"""
    machine = ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.0)
    network = SMIBNetwork(x_pre=0.7, x_fault=np.inf, x_post=0.7, V_inf=1.0)
    with pytest.raises(ValueError, match="送電可能な最大電力"):
        SMIBSystem(machine, network, FaultSchedule.none(), Pe0=1.5)
