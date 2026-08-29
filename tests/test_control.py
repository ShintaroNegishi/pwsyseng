"""制御器（Phase 5）の検証。

- PSS が AVR による負制動を回復させること
- 状態フィードバック（極配置・LQR）が指定どおりの閉ループ極を作ること
- 孤立系の周波数制御が既存の伝達関数モデルと一致すること
"""

from __future__ import annotations

import numpy as np
import pytest

import genstab
from genstab import (
    Governor,
    IsolatedSystem,
    LoadFrequencyControl,
    OneAxisMachine,
    ProportionalGovernor,
    SMIBNetwork,
    SMIBSystem,
    SimpleExciter,
    StepLoad,
    design_lqr,
    design_pole_placement,
    design_pss,
    simulate,
)
from genstab import smallsignal as ss
from genstab.controllers.pss import PowerSystemStabilizer, open_loop_gep


def _machine(D: float = 0.0) -> OneAxisMachine:
    return OneAxisMachine(H=3.5, D=D, xd=1.81, xd_prime=0.30, Td0_prime=8.0, Vt0=1.0)


def _network() -> SMIBNetwork:
    return SMIBNetwork(x_pre=0.65, x_fault=np.inf, x_post=0.65, V_inf=0.995)


def _system(controllers=None) -> SMIBSystem:
    return SMIBSystem(
        _machine(), _network(), controllers=list(controllers or []), Pe0=0.9
    )


def _swing_mode(modes: ss.ModalResult) -> int:
    candidates = [
        i
        for i in range(modes.eigenvalues.size)
        if modes.eigenvalues[i].imag > 1e-6
        and modes.participation[0, i] + modes.participation[1, i] > 0.4
    ]
    assert candidates, "動揺モードが見つからない"
    return candidates[0]


def _swing_damping(system: SMIBSystem) -> float:
    modes = ss.analyze(system)
    return float(modes.damping_ratios[_swing_mode(modes)])


# ----------------------------------------------------------------------
# PSS
# ----------------------------------------------------------------------
def test_open_loop_gep_removes_swing_states():
    """GEP は δ と Δω を取り除いた縮約系であること。"""
    system = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    gep = open_loop_gep(system)
    assert gep.nstates == system.n_states - 2

    # 動揺ループを開いたので、動揺モードの共振が消えているはず。
    swing_freq = abs(ss.analyze(system).eigenvalues[_swing_mode(ss.analyze(system))].imag)
    phase = np.degrees(np.angle(complex(np.squeeze(gep(1j * swing_freq)))))
    # 現実的な励磁系の位相遅れは数十度に収まる。
    assert -90.0 < phase < 0.0


def test_designed_pss_compensates_gep_phase():
    """設計された PSS が GEP の位相遅れをちょうど打ち消すこと。"""
    system = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    modes = ss.analyze(system)
    frequency = float(abs(modes.eigenvalues[_swing_mode(modes)].imag))

    gep = open_loop_gep(system)
    lag = np.angle(complex(np.squeeze(gep(1j * frequency))))

    pss = design_pss(system, Ks=10.0)
    assert pss.phase_lead(frequency) == pytest.approx(-lag, abs=1e-3)


def test_pss_restores_damping_destroyed_by_avr():
    """PSS が AVR による負制動を正に戻すこと（本教材の山場）。"""
    avr_only = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    zeta_avr = _swing_damping(avr_only)
    assert zeta_avr < 0.0, "この検証は AVR が負制動を作る条件で行う"

    pss = design_pss(avr_only, Ks=10.0)
    with_pss = _system([SimpleExciter(Ka=200.0, Ta=0.05), pss])
    zeta_pss = _swing_damping(with_pss)

    assert zeta_pss > 0.0
    # 実務上の目安である減衰比 5 % を上回ること。
    assert zeta_pss > 0.05
    assert ss.analyze(with_pss).is_stable


def test_pss_gain_increases_damping():
    """PSS ゲインを上げると動揺モードの制動が単調に強くなること。"""
    avr_only = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    damping = []
    for Ks in (1.0, 3.0, 5.0, 10.0):
        pss = design_pss(avr_only, Ks=Ks)
        damping.append(_swing_damping(_system([SimpleExciter(Ka=200.0, Ta=0.05), pss])))
    assert all(b > a for a, b in zip(damping, damping[1:]))


def test_pss_barely_shifts_oscillation_frequency():
    """PSS は制動だけを増やし、振動周波数はほとんど変えないこと。"""
    avr_only = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    modes_before = ss.analyze(avr_only)
    freq_before = modes_before.damped_frequencies_hz[_swing_mode(modes_before)]

    with_pss = _system([SimpleExciter(Ka=200.0, Ta=0.05), design_pss(avr_only, Ks=5.0)])
    modes_after = ss.analyze(with_pss)
    freq_after = modes_after.damped_frequencies_hz[_swing_mode(modes_after)]

    assert freq_after == pytest.approx(freq_before, rel=0.05)


def test_pss_requires_an_exciter():
    """PSS を単独で接続したら拒否されること。"""
    with pytest.raises(ValueError, match="AVR と一緒に接続"):
        _system([PowerSystemStabilizer()])
    with pytest.raises(ValueError, match="AVR を接続した系"):
        design_pss(_system())


def test_pss_is_initialized_to_zero_output():
    """定常状態で PSS の補助信号がゼロであること。"""
    avr = SimpleExciter(Ka=200.0, Ta=0.05)
    pss = design_pss(_system([SimpleExciter(Ka=200.0, Ta=0.05)]), Ks=10.0)
    system = _system([avr, pss])
    assert ss.residual_at_operating_point(system) < 1e-10


def test_invalid_pss_parameters_are_rejected():
    with pytest.raises(ValueError, match="時定数 Tw は正"):
        PowerSystemStabilizer(Tw=0.0)
    with pytest.raises(ValueError, match="input_signal"):
        PowerSystemStabilizer(input_signal="voltage")


# ----------------------------------------------------------------------
# 状態フィードバック
# ----------------------------------------------------------------------
def test_pole_placement_achieves_requested_poles():
    """極配置が指定した閉ループ極を正確に実現すること。"""
    zeta, wn = 0.25, 6.0
    target = -zeta * wn + 1j * wn * np.sqrt(1.0 - zeta**2)
    poles = [target, target.conjugate(), -0.2]

    plant = _system()
    controller = design_pole_placement(plant, poles, target="Pm")
    closed = _system([controller])

    achieved = np.sort_complex(ss.analyze(closed).eigenvalues)
    assert np.max(np.abs(achieved - np.sort_complex(np.array(poles)))) < 1e-6


def test_lqr_stabilizes_and_beats_open_loop():
    """LQR が開ループより強い制動を与えること。"""
    plant = _system()
    controller = design_lqr(plant, Q=np.diag([1.0, 1000.0, 1.0]), R=1.0, target="Pm")
    closed = _system([controller])

    assert ss.analyze(closed).is_stable
    assert _swing_damping(closed) > _swing_damping(plant)


def test_pole_placement_methods_agree():
    """place と place_varga が同じ極を実現すること。

    place_varga は slycot を必要とする。environment.yml で作った環境なら
    入っているので、ここで両方の経路を確認しておく。
    """
    pytest.importorskip("slycot", reason="place_varga は slycot を必要とする")

    poles = [-1.5 + 5.8j, -1.5 - 5.8j, -0.2]
    plant = _system()
    achieved = {}
    for method in ("place", "place_varga"):
        controller = design_pole_placement(plant, poles, target="Pm", method=method)
        closed = _system([controller])
        achieved[method] = np.sort_complex(ss.analyze(closed).eigenvalues)
        assert np.max(np.abs(achieved[method] - np.sort_complex(np.array(poles)))) < 1e-6
    assert np.allclose(achieved["place"], achieved["place_varga"])


def test_pole_placement_rejects_unknown_method():
    with pytest.raises(ValueError, match="'place' または 'place_varga'"):
        design_pole_placement(_system(), [-1, -2, -3], method="magic")


def test_state_feedback_rejects_systems_with_other_controllers():
    """他の制御器を含む系での設計が拒否されること。"""
    with pytest.raises(ValueError, match="他の制御器を含まない系"):
        design_lqr(_system([SimpleExciter(Ka=200.0, Ta=0.05)]))


def test_state_feedback_target_selects_controller_kind():
    """操作対象によって接続先の種別が変わること。"""
    from genstab.controllers.base import ControllerKind
    from genstab.controllers.statefb import StateFeedback

    assert StateFeedback(K=[1, 1, 1], target="Pm").kind is ControllerKind.GOVERNOR
    assert StateFeedback(K=[1, 1, 1], target="Efd").kind is ControllerKind.EXCITER
    with pytest.raises(ValueError, match="'Pm' または 'Efd'"):
        StateFeedback(K=[1, 1], target="Vt")


# ----------------------------------------------------------------------
# 孤立系の周波数制御
# ----------------------------------------------------------------------
def test_isolated_system_matches_transfer_function_model():
    """既存 main.py の伝達関数モデルと一致すること。

    差は `control.forced_response` がステップ入力を区分線形に補間する
    ことによるもので、時間刻みに比例して減る。実装の不一致ではない。
    """
    import control as ct

    M, D, K_gov, K_i, dPL, t_step, t_end = 10.0, 2.0, 20.0, 2.0, 0.1, 10.0, 60.0
    G = -ct.feedback(ct.tf(1, [M, D]), ct.tf(K_gov, 1) + ct.tf(K_i, [1, 0]))
    system = IsolatedSystem(
        H=M / 2.0, D=D,
        controllers=[ProportionalGovernor(K_gov=K_gov), LoadFrequencyControl(Ki=K_i)],
        load=StepLoad(magnitude=dPL, time=t_step),
    )

    errors = []
    for dt in (0.02, 0.01, 0.005):
        grid = np.arange(0.0, t_end, dt)
        _, reference = ct.forced_response(
            G, grid, np.where(grid < t_step, 0.0, dPL), X0=0.0
        )
        result = simulate(system, t_end=t_end, dt=dt, rtol=1e-12, atol=1e-14)
        errors.append(np.max(np.abs(np.interp(grid, result.t, result.omega) - reference)))

    assert errors[0] < 1e-3
    # 時間刻みを半分にすると誤差もおよそ半分になる（1 次収束）。
    assert errors[1] == pytest.approx(errors[0] / 2.0, rel=0.1)
    assert errors[2] == pytest.approx(errors[1] / 2.0, rel=0.1)


def test_governor_leaves_steady_state_error():
    """比例制御だけでは定常周波数偏差が残ること。"""
    system = IsolatedSystem(
        H=5.0, D=1.0, controllers=[Governor(R=0.05, Tg=0.2)], load=StepLoad(0.1, 1.0)
    )
    result = simulate(system, t_end=200.0, dt=0.01)
    # 定常偏差は -ΔPL / (D + 1/R)
    expected = -0.1 / (1.0 + 1.0 / 0.05)
    assert result.omega[-1] == pytest.approx(expected, rel=1e-4)
    assert system.steady_state_deviation(0.1) == pytest.approx(expected, rel=1e-12)


def test_lfc_removes_steady_state_error():
    """積分制御を加えると定常偏差が消えること。"""
    system = IsolatedSystem(
        H=5.0, D=1.0,
        controllers=[Governor(R=0.05, Tg=0.2), LoadFrequencyControl(Ki=0.3)],
        load=StepLoad(0.1, 1.0),
    )
    result = simulate(system, t_end=1000.0, dt=0.02)
    assert abs(result.omega[-1]) < 1e-6
    assert system.steady_state_deviation(0.1) == 0.0
    # ただし過渡的には落ち込む（一次調整のほうが速い）。
    assert result.omega.min() < -1e-3


def test_uncontrolled_isolated_system_drifts():
    """制御なしでは負荷の周波数特性だけで釣り合うこと。"""
    system = IsolatedSystem(H=5.0, D=1.0, controllers=[], load=StepLoad(0.1, 1.0))
    result = simulate(system, t_end=100.0, dt=0.01)
    assert result.omega[-1] == pytest.approx(-0.1, rel=1e-4)


def test_isolated_system_rejects_excitation_controllers():
    """孤立系の周波数モデルに励磁系を接続したら拒否されること。"""
    with pytest.raises(ValueError, match="励磁系や PSS は接続できない"):
        IsolatedSystem(controllers=[SimpleExciter()])


def test_isolated_result_has_no_rotor_angle():
    """孤立系の結果は回転子位相角を持たないこと。"""
    system = IsolatedSystem(H=5.0, D=1.0, load=StepLoad(0.1, 1.0))
    result = simulate(system, t_end=10.0, dt=0.01)
    assert not result.has_rotor_angle
    assert result.is_stable()
    with pytest.raises(AttributeError, match="状態 'delta' を持たない"):
        _ = result.delta


# ----------------------------------------------------------------------
# 使い回しと入力検証（codex の指摘 #4, #7, #9 への対応）
# ----------------------------------------------------------------------
def test_network_is_copied_so_it_can_be_reused():
    """同じネットワークを別の発電機で使い回しても先の系が壊れないこと。

    `attach()` は背後リアクタンスをネットワークに登録するので、複製せずに
    共有すると、後から作った系の x'd で先の系の動特性まで書き換わる。
    実測で残差が 1e-17 から 1.8e-2 へ悪化していた。
    """
    from genstab import ClassicalMachine

    shared = SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6)
    first = SMIBSystem(ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1),
                       shared, Pe0=0.8)
    before = ss.residual_at_operating_point(first)

    second = SMIBSystem(ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.5, E=1.1),
                        shared, Pe0=0.8)
    after = ss.residual_at_operating_point(first)

    assert after == pytest.approx(before, abs=1e-12)
    assert after < 1e-10
    # 系はそれぞれ自分の複製を持ち、渡した側は書き換えられない。
    assert first.network is not shared
    assert second.network is not shared
    assert first.network.x_internal == pytest.approx(0.3)
    assert second.network.x_internal == pytest.approx(0.5)
    assert shared.x_internal == pytest.approx(0.0)


def test_controller_reuse_is_reported():
    """同じ制御器インスタンスを別の系に接続したら警告すること。

    制御器は接続時に基準値（AVR なら v_ref）を自分の中に持つので、
    使い回すと先に作った系の動作点が壊れる。ネットワークと違って
    利用者が接続後に参照するため複製はせず、警告で知らせる。
    """
    avr = SimpleExciter(Ka=200.0, Ta=0.05)
    _ = _system([avr])
    with pytest.warns(UserWarning, match="既に別の系"):
        SMIBSystem(_machine(), _network(), Pe0=0.5, controllers=[avr])


def test_pss_design_rejects_power_input():
    """電気出力入力の PSS 自動設計が拒否されること。

    測定量から電気トルクまでの経路に約 90 度の位相差が加わるため、
    GEP(s) の位相だけを補償すると制動がかえって悪化する
    （同じ条件で減衰比が +0.167 から -0.084 に転じる）。
    """
    system = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    with pytest.raises(ValueError, match="速度偏差入力"):
        design_pss(system, Ks=10.0, input_signal="Pe")
    # 速度偏差入力なら従来どおり設計できる。
    assert design_pss(system, Ks=10.0, input_signal="omega").Ks == 10.0


@pytest.mark.parametrize("length", [0, 1, 5])
def test_derivatives_reject_wrong_state_length(length):
    """状態ベクトルの長さが合わなければ拒否されること。

    以前は長さの違う x を受理し、`np.empty_like` の未初期化領域を
    そのまま返していたため、偽のモードを持つ行列ができていた。
    """
    system = _system()
    with pytest.raises(ValueError, match="状態ベクトルの形状"):
        system.derivatives(0.0, np.zeros(length))


def test_isolated_and_multimachine_also_check_state_length():
    """孤立系と多機系統でも同じ検証が効くこと。"""
    from pathlib import Path

    from genstab import IsolatedSystem, StepLoad
    from genstab.multimachine import load_case

    isolated = IsolatedSystem(H=5.0, D=1.0, load=StepLoad(0.1, 1.0))
    with pytest.raises(ValueError, match="状態ベクトルの形状"):
        isolated.derivatives(0.0, np.zeros(4))

    case = Path(__file__).resolve().parents[1] / "cases" / "wscc9.yaml"
    multimachine = load_case(case)
    with pytest.raises(ValueError, match="状態ベクトルの形状"):
        multimachine.derivatives(0.0, np.zeros(4))
