"""1 軸モデルと励磁制御（Phase 4）の検証。

この段階の主題は「高応答 AVR が動揺モードの制動を悪化させる」という
古典的な現象を再現できることである。これは Heffron-Phillips モデルで
K5 < 0 となる重負荷・長距離送電条件で起きる。
"""

from __future__ import annotations

import numpy as np
import pytest

from genstab import (
    ClassicalMachine,
    FaultSchedule,
    OneAxisMachine,
    SMIBNetwork,
    SMIBSystem,
    SimpleExciter,
    IEEEType1Exciter,
    simulate,
)
from genstab import linearize as lin
from genstab import smallsignal as ss

#: 重負荷・長距離送電（AVR の負制動が現れる条件）
PE0 = 0.9


def _machine(D: float = 0.0) -> OneAxisMachine:
    return OneAxisMachine(H=3.5, D=D, xd=1.81, xd_prime=0.30, Td0_prime=8.0, Vt0=1.0)


def _network() -> SMIBNetwork:
    return SMIBNetwork(x_pre=0.65, x_fault=np.inf, x_post=0.65, V_inf=0.995)


def _system(controllers=None, D: float = 0.0) -> SMIBSystem:
    return SMIBSystem(
        _machine(D), _network(), FaultSchedule.none(),
        controllers=list(controllers or []), Pe0=PE0,
    )


def _oscillatory_mode(modes: ss.ModalResult) -> int:
    """δ と Δω が主役の振動モードのインデックスを返す。"""
    candidates = [
        i
        for i in range(modes.eigenvalues.size)
        if abs(modes.eigenvalues[i].imag) > 1e-6
        and modes.participation[0, i] + modes.participation[1, i] > 0.5
    ]
    assert candidates, "動揺モードが見つからない"
    return candidates[0]


# ----------------------------------------------------------------------
# 初期化
# ----------------------------------------------------------------------
def test_initial_state_reproduces_requested_operating_point():
    """指定した端子電圧と送電電力が厳密に再現されること。"""
    system = _system()
    assert system.operating_point.Vt == pytest.approx(1.0, abs=1e-10)
    assert system.operating_point.Pe == pytest.approx(PE0, abs=1e-10)
    assert ss.residual_at_operating_point(system) < 1e-12


def test_field_voltage_exceeds_internal_emf():
    """定常界磁電圧が内部起電力より大きいこと（電機子反作用による減磁）。"""
    system = _system()
    eq0 = system.initial_state()[2]
    assert system.operating_point.Efd > eq0
    # 差は (xd - x'd) * Id に等しい。
    from genstab.events import Stage

    solution = system.network.solve(Stage.PRE, eq0, system.operating_point.delta)
    machine = system.machine
    assert system.operating_point.Efd - eq0 == pytest.approx(
        (machine.xd - machine.xd_prime) * solution.Id, rel=1e-10
    )


def test_flux_decay_mode_appears():
    """1 軸モデルでは界磁磁束の遅いモードが 1 つ増えること。"""
    modes = ss.analyze(_system())
    assert modes.eigenvalues.size == 3
    real_modes = [i for i in range(3) if abs(modes.eigenvalues[i].imag) < 1e-9]
    assert len(real_modes) == 1
    # そのモードは E'q が支配的。
    assert modes.dominant_states(real_modes[0], top=1)[0][0] == "Eq_prime"


# ----------------------------------------------------------------------
# AVR
# ----------------------------------------------------------------------
def test_avr_is_initialized_without_transient():
    """AVR を接続しても定常状態が乱れないこと（bumpless 初期化）。

    時間応答で確かめるため、制動を入れて系全体が安定になる条件を使う。
    負制動の系（`test_high_gain_avr_degrades_damping` の条件）では
    丸め誤差そのものが指数的に成長するので、時間応答での検証には
    向かない。
    """
    avr = SimpleExciter(Ka=10.0, Ta=0.05)
    system = _system([avr], D=2.0)
    assert ss.residual_at_operating_point(system) < 1e-10
    assert ss.analyze(system).is_stable, "この検証は安定な系で行う必要がある"

    result = simulate(system, t_end=5.0, dt=0.01)
    # 事故がないので、どの状態も初期値から動かないはず。
    assert np.max(np.abs(result.delta - result.delta[0])) < 1e-9
    assert np.max(np.abs(result.Vt - result.Vt[0])) < 1e-8

    # 界磁電圧の許容差だけ緩く取る。動作点の残差（1e-10 程度）は完全な
    # ゼロではなく、それが励磁系のゲイン Ka/Ta = 200 倍されて Efd に
    # 現れるためである。定常値 1.89 p.u. に対して相対 1e-6 未満であれば
    # 「接続しても系が乱れない」ことの確認としては十分で、環境ごとの
    # 丸め誤差の差（macOS と Linux で BLAS の実装が異なる）も吸収できる。
    assert np.max(np.abs(result.Efd - result.Efd[0])) < 1e-6


def test_unstable_system_amplifies_numerical_noise():
    """負制動の系では丸め誤差が指数的に成長すること。

    これは実装の欠陥ではなく、系が不安定であることの帰結である。
    定態安定性が確保されていない運転点では、事故がなくても振動が
    育つという事実がそのまま数値に現れる。
    """
    system = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    modes = ss.analyze(system)
    growth_rate = modes.eigenvalues[_oscillatory_mode(modes)].real
    assert growth_rate > 0.0

    result = simulate(system, t_end=20.0, dt=0.01)
    early = np.max(np.abs(result.delta[result.t < 5.0] - result.delta[0]))
    late = np.max(np.abs(result.delta[result.t > 15.0] - result.delta[0]))
    assert late > early


def test_avr_reference_voltage_is_consistent():
    """基準電圧が Vt0 + Efd0/Ka に設定されること。"""
    avr = SimpleExciter(Ka=200.0, Ta=0.05)
    system = _system([avr])
    expected = system.operating_point.Vt + system.operating_point.Efd / avr.Ka
    assert avr.v_ref == pytest.approx(expected, rel=1e-12)


def test_high_gain_avr_degrades_damping():
    """AVR を入れると動揺モードの減衰比が下がること。

    これが PSS が必要になった理由であり、本教材の中心的な題材である。
    """
    without = ss.analyze(_system())
    with_avr = ss.analyze(_system([SimpleExciter(Ka=200.0, Ta=0.05)]))

    zeta_without = without.damping_ratios[_oscillatory_mode(without)]
    zeta_with = with_avr.damping_ratios[_oscillatory_mode(with_avr)]

    assert zeta_without > 0.0, "AVR なしでは弱いながら正の制動があるはず"
    assert zeta_with < 0.0, "高ゲイン AVR は負制動を生むはず"
    assert not with_avr.is_stable


def test_rate_feedback_exciter_is_less_destabilizing():
    """安定化変圧器付き励磁系は単純な高ゲイン AVR より制動が良いこと。"""
    simple = ss.analyze(_system([SimpleExciter(Ka=200.0, Ta=0.05)]))
    type1 = ss.analyze(_system([IEEEType1Exciter(Ka=200.0, Ta=0.05)]))

    zeta_simple = simple.damping_ratios[_oscillatory_mode(simple)]
    zeta_type1 = type1.damping_ratios[_oscillatory_mode(type1)]
    assert zeta_type1 > zeta_simple


# ----------------------------------------------------------------------
# 構成の検証
# ----------------------------------------------------------------------
def test_exciter_on_classical_machine_warns():
    """界磁回路を持たないモデルに AVR を付けたら警告すること。"""
    machine = ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
    with pytest.warns(UserWarning, match="界磁回路の状態量を持たない"):
        SMIBSystem(machine, _network(), FaultSchedule.none(),
                   controllers=[SimpleExciter()], Pe0=0.8)


def test_only_one_exciter_allowed():
    """励磁系を 2 台接続したら拒否されること。"""
    with pytest.raises(ValueError, match="1 台までしか接続できない"):
        _system([SimpleExciter(), SimpleExciter()])


def test_efd_input_rejected_when_avr_present():
    """AVR がある系では Efd ではなく Vref を入力に取るよう促されること。"""
    system = _system([SimpleExciter(Ka=200.0, Ta=0.05)])
    with pytest.raises(ValueError, match="'Vref'"):
        lin.state_space(system, inputs=("Efd",))

    G = lin.state_space(system, inputs=("Vref",), outputs=("Vt",))
    assert G.nstates == 4 and G.ninputs == 1
    # 基準電圧を上げれば端子電圧は上がる（直流ゲインが正）。
    import control as ct

    assert ct.dcgain(G) > 0.0


def test_invalid_exciter_parameters_are_rejected():
    with pytest.raises(ValueError, match="Ka は正"):
        SimpleExciter(Ka=0.0)
    with pytest.raises(ValueError, match="Ta は正"):
        SimpleExciter(Ta=-0.1)
    with pytest.raises(ValueError, match="下限, 上限"):
        SimpleExciter(efd_limits=(5.0, 1.0))


def test_invalid_machine_parameters_are_rejected():
    with pytest.raises(ValueError, match="0 < x'_d < x_d"):
        OneAxisMachine(xd=0.3, xd_prime=1.8)
    with pytest.raises(ValueError, match="T'd0 は正"):
        OneAxisMachine(Td0_prime=0.0)
