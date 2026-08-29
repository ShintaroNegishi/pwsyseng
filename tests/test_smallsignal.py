"""定態安定性（Phase 3）の検証。

3 つの独立な経路が一致することを確かめる。

1. 数値微分で作った状態行列の固有値
2. 2 次系の特性方程式から得た解析解 ω_n, ζ
3. 非線形シミュレーションに微小擾乱を与えた応答
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from genstab import ClassicalMachine, FaultSchedule, SMIBNetwork, SMIBSystem, simulate
from genstab import linearize as lin
from genstab import smallsignal as ss


def _system(D: float = 2.0) -> SMIBSystem:
    machine = ClassicalMachine(H=5.0, D=D, x_d_prime=0.3, E=1.1)
    network = SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.4)
    return SMIBSystem(machine, network, FaultSchedule.none(), Pe0=0.8)


def test_numerical_eigenvalues_match_analytic_formula():
    """数値固有値が ω_n, ζ の解析式と一致すること。"""
    system = _system()
    numeric = np.sort_complex(ss.analyze(system).eigenvalues)
    analytic = np.sort_complex(ss.classical_mode_analytic(system).eigenvalues)
    assert np.max(np.abs(numeric - analytic)) < 1e-7


def test_synchronizing_coefficient_is_pmax_cos_delta():
    """K_s = Pmax * cos(δ0) であること。"""
    system = _system()
    mode = ss.classical_mode_analytic(system)
    emf = system.machine.internal_emf(system.initial_state())
    from genstab.events import Stage

    pmax = system.network.max_power(Stage.PRE, emf)
    expected = pmax * math.cos(system.operating_point.delta)
    assert mode.K_s == pytest.approx(expected, rel=1e-12)


def test_natural_frequency_formula():
    """ω_n = sqrt(K_s ω_s / 2H) であること。"""
    system = _system()
    mode = ss.classical_mode_analytic(system)
    expected = math.sqrt(mode.K_s * system.base.omega_s / (2.0 * system.machine.H))
    assert mode.natural_frequency == pytest.approx(expected, rel=1e-12)
    # 局所動揺モードは 0.7〜2 Hz に入るのが典型。
    assert 0.5 < mode.natural_frequency_hz < 2.5


def test_operating_point_is_an_equilibrium():
    """構築時に設定した動作点が本当に平衡点であること。"""
    system = _system()
    assert ss.residual_at_operating_point(system) < 1e-12
    solved = ss.equilibrium(system)
    assert solved == pytest.approx(system.initial_state(), abs=1e-9)


def test_linear_model_matches_nonlinear_small_disturbance():
    """線形モデルの初期値応答が非線形シミュレーションと一致すること。"""
    import control as ct

    system = _system()
    G = lin.state_space(system, inputs=("Pm",), outputs=("delta", "omega"))

    eps = 1e-4
    nonlinear = simulate(
        system, t_end=8.0, dt=0.005,
        x0=system.initial_state() + np.array([eps, 0.0]),
        rtol=1e-12, atol=1e-14,
    )
    _, y = ct.initial_response(G, T=nonlinear.t, X0=np.array([eps, 0.0]))
    linear_delta = system.operating_point.delta + y[0]

    assert np.max(np.abs(nonlinear.delta - linear_delta)) < 1e-6 * eps / 1e-4 + 1e-8


def test_damping_ratio_increases_with_D():
    """制動係数を上げると減衰比が上がること。"""
    weak = ss.classical_mode_analytic(_system(D=1.0)).damping_ratio
    strong = ss.classical_mode_analytic(_system(D=4.0)).damping_ratio
    assert strong > weak
    assert strong == pytest.approx(4.0 * weak, rel=1e-12)


def test_undamped_machine_is_marginally_stable():
    """D=0 なら固有値が虚軸上に乗ること（減衰比ゼロ）。"""
    modes = ss.analyze(_system(D=0.0))
    assert np.max(np.abs(modes.eigenvalues.real)) < 1e-8
    assert not modes.is_stable


def test_participation_factors_are_normalized():
    """参加係数が各モードで合計 1 になること。"""
    modes = ss.analyze(_system())
    assert np.allclose(modes.participation.sum(axis=0), 1.0)
    # 2 次の動揺モードでは δ と Δω が等しく寄与する。
    assert modes.participation[0, 0] == pytest.approx(0.5, abs=1e-6)


def test_state_space_has_named_signals():
    """python-control の系に信号名が設定されること。"""
    system = _system()
    G = lin.state_space(system, inputs=("Pm",), outputs=("delta", "omega", "Pe"))
    assert G.nstates == 2 and G.ninputs == 1 and G.noutputs == 3
    assert list(G.input_labels) == ["Pm"]
    assert list(G.output_labels) == ["delta", "omega", "Pe"]
    # B 行列は動揺方程式から 1/(2H) になるはず。
    assert G.B[1, 0] == pytest.approx(1.0 / (2.0 * system.machine.H), rel=1e-6)


def test_analytic_mode_rejects_higher_order_machines():
    """高次の発電機モデルでは解析解が拒否されること。"""
    from genstab import OneAxisMachine

    system = SMIBSystem(
        OneAxisMachine(H=3.5, D=0.0, xd=1.81, xd_prime=0.30, Td0_prime=8.0, Vt0=1.0),
        SMIBNetwork(x_pre=0.65, x_fault=np.inf, x_post=0.65, V_inf=0.995),
        Pe0=0.9,
    )
    with pytest.raises(ValueError, match="古典モデル専用"):
        ss.classical_mode_analytic(system)


def test_analytic_mode_rejects_systems_with_controllers():
    """状態を持たない制御器が付いた系でも拒否されること。

    状態数だけを見ると比例ガバナ付きの古典系は 2 次のままなので、
    以前はこの式が通ってしまい、真の固有値の実部が -1.1 なのに
    -0.1 を返していた。例外を出さずに誤った値を返す状態だった。
    """
    from genstab import ProportionalGovernor

    system = SMIBSystem(
        ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1),
        SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.4),
        Pe0=0.8,
        controllers=[ProportionalGovernor(K_gov=20.0)],
    )
    # 数値解はガバナの効果を正しく含む。
    numeric = ss.analyze(system).eigenvalues
    assert numeric.real.max() == pytest.approx(-1.1, abs=1e-6)

    with pytest.raises(ValueError, match="制御器を含まない系専用"):
        ss.classical_mode_analytic(system)
