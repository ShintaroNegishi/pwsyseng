"""シミュレーション基盤の検証。

積分区間の分割、結果オブジェクトの取り扱い、作図関数が
落ちずに動くことを確認する。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 画面のない環境でも作図できるようにする

import matplotlib.pyplot as plt
import numpy as np
import pytest

import genstab
from genstab import (
    ClassicalMachine,
    FaultSchedule,
    IsolatedSystem,
    RampLoad,
    SMIBNetwork,
    SMIBSystem,
    StepLoad,
    simulate,
)
from genstab.controllers.governor import Governor
from genstab.events import Stage


def _system(t_fault: float = 1.0, t_clear: float = 1.15, D: float = 2.0) -> SMIBSystem:
    return SMIBSystem(
        ClassicalMachine(H=5.0, D=D, x_d_prime=0.3, E=1.1),
        SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6),
        FaultSchedule(t_fault=t_fault, t_clear=t_clear),
        Pe0=0.8,
    )


# ----------------------------------------------------------------------
# 積分区間の分割
# ----------------------------------------------------------------------
def test_switching_times_are_reported():
    """事故発生・除去の時刻が切替点として返ること。"""
    system = _system()
    assert system.switching_times(10.0) == [1.0, 1.15]
    # 終了時刻より後の切替点は含めない。
    assert system.switching_times(1.1) == [1.0]


@pytest.mark.parametrize("t_fault,t_clear", [(1.0, 1.15), (1.003, 1.1517), (0.7071, 0.9142)])
def test_handles_switching_times_off_the_output_grid(t_fault, t_clear):
    """切替時刻が出力の時間格子に乗らなくても正しく積分できること。

    事故時刻が格子点からずれると、区間の終端が t_eval に含まれない。
    このとき次区間の初期値を取りこぼすと、事故中の加速が失われて
    結果が大きく狂う。

    事故中の運動を閉形式と比べるため、制動 D = 0 の系を使う。
    """
    system = _system(t_fault, t_clear, D=0.0)
    result = simulate(system, t_end=8.0, dt=0.005, rtol=1e-11, atol=1e-13)

    # 事故中は電気出力がゼロで、位相角は単調に増える。
    during = (result.t > t_fault) & (result.t < t_clear)
    assert np.max(np.abs(result.Pe[during])) < 1e-12
    assert np.all(np.diff(result.delta[during]) > 0)

    # 制動がなければ事故中の運動は等加速度になり、
    # δ(t) = δ0 + (ω_s Pm / 4H) t^2 に厳密に一致する。
    elapsed = result.t[during] - t_fault
    expected = system.operating_point.delta + (
        system.base.omega_s * system.operating_point.Pm / (4.0 * system.machine.H)
    ) * elapsed**2
    assert np.max(np.abs(result.delta[during] - expected)) < 1e-8

    # 事故除去の受け渡しが正しければ、事故後の軌道は出力の時間刻みに
    # 依存しないはずである（t_eval は結果を書き出す間隔にすぎない）。
    finer = simulate(system, t_end=8.0, dt=0.0025, rtol=1e-11, atol=1e-13)
    after = result.t > t_clear
    assert np.max(np.abs(result.delta[after] - np.interp(result.t[after], finer.t, finer.delta))) < 1e-7


def test_time_grid_is_monotonic_and_unique():
    """出力の時刻が重複せず単調増加であること。"""
    result = simulate(_system(), t_end=5.0, dt=0.005)
    assert np.all(np.diff(result.t) > 0)
    assert result.t[0] == pytest.approx(0.0)
    assert result.t[-1] == pytest.approx(5.0)


def test_diverging_run_is_truncated():
    """脱調時に角度しきい値で打ち切られること。"""
    system = _system(t_fault=1.0, t_clear=np.inf)
    result = simulate(system, t_end=60.0, dt=0.01)
    assert result.diverged
    assert result.t[-1] < 60.0
    assert not result.is_stable()


def test_stage_is_fixed_within_each_segment():
    """区間の中ではネットワーク状態が変わらないこと。"""
    system = _system()
    assert system.stage_at(0.5) is Stage.PRE
    assert system.stage_at(1.05) is Stage.FAULT
    assert system.stage_at(2.0) is Stage.POST


# ----------------------------------------------------------------------
# 結果オブジェクト
# ----------------------------------------------------------------------
def test_result_access_by_name_and_attribute():
    """状態量・代数量が名前でも属性でも引けること。"""
    result = simulate(_system(), t_end=5.0, dt=0.01)
    assert np.allclose(result["delta"], result.delta)
    assert np.allclose(result["Pe"], result.Pe)
    assert np.allclose(result.delta_deg, np.degrees(result.delta))
    assert np.allclose(result.frequency_hz, 50.0 * (1.0 + result.omega))
    with pytest.raises(KeyError, match="存在しない"):
        _ = result["存在しない量"]


def test_result_to_dataframe():
    """DataFrame に変換できること。"""
    result = simulate(_system(), t_end=2.0, dt=0.01)
    frame = result.to_dataframe()
    assert list(frame.columns[:3]) == ["t", "delta", "omega"]
    assert len(frame) == result.t.size


def test_invalid_simulation_arguments_are_rejected():
    system = _system()
    with pytest.raises(ValueError, match="dt は正"):
        simulate(system, t_end=1.0, dt=0.0)
    with pytest.raises(ValueError, match="t_end は正"):
        simulate(system, t_end=-1.0)
    with pytest.raises(ValueError, match="初期状態の長さ"):
        simulate(system, t_end=1.0, x0=np.zeros(5))


# ----------------------------------------------------------------------
# 負荷プロファイル
# ----------------------------------------------------------------------
def test_ramp_load_profile():
    """ランプ負荷が指定した勾配で増え、停止時刻で頭打ちになること。"""
    load = RampLoad(rate=0.02, start=1.0, stop=6.0)
    assert load(0.5) == pytest.approx(0.0)
    assert load(3.0) == pytest.approx(0.04)
    assert load(6.0) == pytest.approx(0.10)
    assert load(20.0) == pytest.approx(0.10)
    assert set(load.switching_times) == {1.0, 6.0}


def test_ramp_load_drives_frequency_down():
    """ランプ負荷で周波数が徐々に下がること。"""
    system = IsolatedSystem(
        H=5.0, D=1.0, controllers=[Governor(R=0.05, Tg=0.2)],
        load=RampLoad(rate=0.02, start=1.0, stop=6.0),
    )
    result = simulate(system, t_end=30.0, dt=0.01)
    assert result.omega[-1] < 0.0
    # 停止後は 0.1 p.u. の負荷増に対する定常偏差に落ち着く。
    assert result.omega[-1] == pytest.approx(
        system.steady_state_deviation(0.10), rel=1e-3
    )


# ----------------------------------------------------------------------
# 作図
# ----------------------------------------------------------------------
def test_plotting_functions_run():
    """作図関数が例外なく動くこと（描画内容までは検証しない）。"""
    from genstab import eac
    from genstab.plotting import (
        compare_results,
        plot_eigenvalues,
        plot_phase_portrait,
        plot_power_angle,
        plot_swing,
        use_genstab_style,
    )
    from genstab import smallsignal as ss

    use_genstab_style()
    system = _system()
    result = simulate(system, t_end=5.0, dt=0.01)

    plot_swing(result, ("delta_deg", "omega", "Pe"))
    plot_power_angle(system, result)
    plot_phase_portrait([result], ["case"])
    compare_results([result, result], ["a", "b"], ("delta_deg",))
    plot_eigenvalues(ss.analyze(SMIBSystem(
        ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1),
        SMIBNetwork(0.4, np.inf, 0.4), Pe0=0.8,
    )).eigenvalues)
    eac.plot_equal_area(SMIBSystem(
        ClassicalMachine(H=5.0, D=0.0, x_d_prime=0.3, E=1.1),
        SMIBNetwork(0.4, np.inf, 0.6), FaultSchedule(1.0, 1.1), Pe0=0.8,
    ))
    plt.close("all")


def test_describe_outputs_are_readable():
    """構成の要約が主要な情報を含むこと。"""
    text = _system().describe()
    assert "ClassicalMachine" in text
    assert "制御器" in text

    isolated = IsolatedSystem(H=5.0, D=1.0, load=StepLoad(0.1, 1.0))
    assert "IsolatedSystem" in isolated.describe()
