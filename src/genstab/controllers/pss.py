"""電力系統安定化装置 (PSS)。

PSS は速度偏差（または電気出力）を入力として、励磁系の基準に補助信号
``V_s`` を重畳し、動揺モードに正の制動トルクを与える装置である。
AVR が悪化させた制動を回復させるために導入された。

構成は washout（直流分の除去）と 2 段の進み遅れ補償からなる::

           Δω      ┌──────────┐  ┌────────────┐  ┌────────────┐
        ──────────→│  K_s     │→ │ T_w s      │→ │ 1+T_1 s    │→ … → V_s
                   │ (gain)   │  │ ───────    │  │ ───────    │
                   └──────────┘  │ 1+T_w s    │  │ 1+T_2 s    │
                                 └────────────┘  └────────────┘

washout は定常的な速度偏差に PSS が反応して端子電圧を狂わせるのを防ぐ
ためのもので、時定数を十分大きく（5〜10 s）取れば動揺周波数帯の位相には
ほとんど影響しない。

進み遅れ補償は、``V_s`` から電気トルクまでの経路 GEP(s) が持つ位相遅れを
打ち消し、速度偏差と同相の電気トルク（＝正の制動トルク）を作るために
入れる。設計は :func:`design_pss` が自動化する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .base import Controller, ControllerKind, Measurement


@dataclass
class PowerSystemStabilizer(Controller):
    """washout + 2 段進み遅れ補償の PSS。

    Parameters
    ----------
    Ks:
        PSS ゲイン。大きいほど制動は強くなるが、大きすぎると別のモード
        （励磁系モードなど）を不安定にする。
    Tw:
        washout 時定数 [s]。5〜10 s が標準。
    T1, T2, T3, T4:
        進み遅れ補償の時定数 [s]。``T1 > T2`` かつ ``T3 > T4`` で位相を
        進める。:func:`design_pss` で自動設計できる。
    output_limits:
        補助信号の上下限 ``(min, max)`` [p.u.]。実機では ±0.05〜0.1 程度に
        制限され、PSS が端子電圧を大きく動かさないようにしている。
        ``None`` なら制限なし（線形解析と厳密に一致する）。
    input_signal:
        入力信号。``"omega"``（速度偏差）または ``"Pe"``（電気出力）。
        電気出力を使う場合は符号が反転する（出力が増えると減速するため）。
    """

    Ks: float = 10.0
    Tw: float = 10.0
    T1: float = 0.15
    T2: float = 0.03
    T3: float = 0.15
    T4: float = 0.03
    output_limits: tuple[float, float] | None = None
    input_signal: str = "omega"

    kind: ClassVar[ControllerKind] = ControllerKind.STABILIZER
    n_states: ClassVar[int] = 3
    state_names: ClassVar[tuple[str, ...]] = ("washout", "lead1", "lead2")

    _u0: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        for name in ("Tw", "T2", "T4"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"時定数 {name} は正でなければならない。")
        if self.input_signal not in ("omega", "Pe"):
            raise ValueError(
                f"input_signal は 'omega' または 'Pe' のいずれか (指定値: {self.input_signal})。"
            )
        if self.output_limits is not None and self.output_limits[0] >= self.output_limits[1]:
            raise ValueError(
                f"output_limits は (下限, 上限) の順で与えること: {self.output_limits}。"
            )

    # ------------------------------------------------------------------
    def _input(self, meas: Measurement) -> float:
        """入力信号（定常値からの偏差）。"""
        if self.input_signal == "omega":
            return meas.omega
        # 電気出力を使う場合は符号を反転する。出力が増えると回転子は
        # 減速するので、速度偏差と同じ向きに揃えるため。
        return -(meas.Pe - self._u0)

    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        # 定常状態では入力が 0（速度偏差ゼロ、電気出力は定常値）なので
        # すべての状態を 0 にすれば補助信号も 0 になる。
        self._u0 = meas.Pe
        return np.zeros(3, dtype=float)

    # ------------------------------------------------------------------
    def _stages(self, xc: np.ndarray, meas: Measurement) -> tuple[float, float, float]:
        """各段の出力を返す。"""
        u = self._input(meas)
        # washout: Tw s / (1 + Tw s) = 1 - 1/(1 + Tw s)
        y_washout = self.Ks * (u - float(xc[0]))
        # 1 段目の進み遅れ: (1 + T1 s)/(1 + T2 s)
        ratio_1 = self.T1 / self.T2
        y_lead1 = ratio_1 * y_washout + (1.0 - ratio_1) * float(xc[1])
        # 2 段目
        ratio_2 = self.T3 / self.T4
        y_lead2 = ratio_2 * y_lead1 + (1.0 - ratio_2) * float(xc[2])
        return y_washout, y_lead1, y_lead2

    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        _, _, y = self._stages(xc, meas)
        if self.output_limits is not None:
            y = float(np.clip(y, *self.output_limits))
        return float(y)

    def derivatives(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> np.ndarray:
        u = self._input(meas)
        y_washout, y_lead1, _ = self._stages(xc, meas)

        d_washout = (u - float(xc[0])) / self.Tw
        d_lead1 = (y_washout - float(xc[1])) / self.T2
        d_lead2 = (y_lead1 - float(xc[2])) / self.T4

        return np.array([d_washout, d_lead1, d_lead2], dtype=float)

    # ------------------------------------------------------------------
    def phase_lead(self, frequency_rad: float) -> float:
        """指定角周波数における進み遅れ補償の位相進み [rad]。"""
        s = 1j * frequency_rad
        transfer = ((1.0 + self.T1 * s) / (1.0 + self.T2 * s)) * (
            (1.0 + self.T3 * s) / (1.0 + self.T4 * s)
        )
        return float(np.angle(transfer))


def open_loop_gep(system, frequency_rad: float | None = None):
    r"""励磁系から電気トルクまでの伝達関数 GEP(s) を作る。

    PSS の位相補償を設計するとき、GEP は「回転子の動揺を止めた状態」で
    評価しなければならない。δ と Δω を状態に残したまま評価すると、
    動揺モードの共振がそのまま伝達関数に現れ、ちょうど設計したい
    動揺周波数で位相が急変してしまうためである。

    そこで状態行列から δ と Δω の行・列を取り除いた部分系

    .. math::

        GEP(s) = \left. \frac{\Delta P_e(s)}{\Delta V_{ref}(s)}
                 \right|_{\Delta\delta = \Delta\omega = 0}

    を使う。残るのは界磁磁束 E'q と励磁系のダイナミクスで、これらが
    作る位相遅れこそ PSS が補償すべき対象である。

    Returns
    -------
    control.StateSpace
        δ, Δω を除いた縮約系。
    """
    import control as ct

    from ..linearize import input_matrix, output_matrices, state_matrix

    A = state_matrix(system)
    B = input_matrix(system, ("Vref",))
    C, D = output_matrices(system, ("Pe",), ("Vref",))

    swing_states = [
        system.state_names.index(name)
        for name in ("delta", "omega")
        if name in system.state_names
    ]
    keep = [i for i in range(A.shape[0]) if i not in swing_states]
    if not keep:
        raise ValueError(
            "動揺ループを開くと状態が残らない。界磁回路を持つモデル"
            "（OneAxisMachine 以上）と励磁系が必要。"
        )

    index = np.ix_(keep, keep)
    return ct.ss(A[index], B[keep, :], C[:, keep], D)


def design_pss(
    system,
    *,
    Ks: float = 10.0,
    Tw: float = 10.0,
    n_stages: int = 2,
    frequency_rad: float | None = None,
    input_signal: str = "omega",
    output_limits: tuple[float, float] | None = None,
) -> PowerSystemStabilizer:
    """位相補償の考え方に従って PSS を設計する。

    手順
    ----
    1. PSS を外した系（AVR は接続したまま）を線形化し、
       基準電圧 ``V_ref`` から電気出力 ``P_e`` までの伝達関数 GEP(s) を作る
    2. 動揺モードの角周波数 ω_osc における位相遅れ ``∠GEP(jω_osc)`` を測る
    3. これを打ち消す位相進み ``φ = -∠GEP(jω_osc)`` を進み遅れ補償で作る
    4. ``n_stages`` 段で φ を等分し、各段について
       ``α = (1 - sin(φ/n)) / (1 + sin(φ/n))`` から時定数を決める

    こうして速度偏差と同相の電気トルク、すなわち正の制動トルクが得られる。

    Parameters
    ----------
    system:
        PSS を除いた（AVR は含む）:class:`~genstab.system.SMIBSystem`。
    Ks:
        PSS ゲイン。位相補償では決まらないので、根軌跡や固有値の
        再計算で調整すること。
    n_stages:
        進み遅れ補償の段数。1 または 2。必要な位相進みが大きいときは
        2 段に分けたほうが各段の負担が軽くなる。
    frequency_rad:
        補償の中心とする角周波数 [rad/s]。省略すると系の動揺モードを
        自動検出して使う。

    Returns
    -------
    PowerSystemStabilizer

    Notes
    -----
    PSS を接続したあとは必ず :func:`genstab.smallsignal.analyze` で
    固有値を再計算し、狙ったモードの制動が改善し、他のモードが
    悪化していないことを確認すること。位相補償は設計の出発点であって
    最終確認ではない。
    """
    from ..smallsignal import analyze

    if input_signal != "omega":
        raise ValueError(
            f"自動設計は速度偏差入力 (input_signal='omega') にのみ対応する"
            f"（指定値: '{input_signal}'）。電気出力を入力に使う場合、"
            " 測定量から電気トルクまでの経路に約 90 度の位相差と周波数依存の"
            " ゲインが加わるため、GEP(s) の位相だけを補償すると制動が"
            " かえって悪化する（同じ条件で減衰比が +0.167 から -0.084 に"
            " 転じることを確認している）。Pe 入力を使う場合は、測定経路の"
            " 位相を含めて時定数を手動で決めること。"
        )
    if n_stages not in (1, 2):
        raise ValueError(
            f"この実装は 1 段または 2 段の進み遅れ補償に対応している (n_stages={n_stages})。"
        )
    if system.exciter is None:
        raise ValueError(
            "PSS は励磁系への補助信号として働くため、AVR を接続した系に対して"
            " 設計する必要がある。"
        )

    modes = analyze(system)
    if frequency_rad is None:
        # δ と Δω の参加係数が大きい振動モードを動揺モードとみなす。
        candidates = [
            i
            for i in range(modes.eigenvalues.size)
            if modes.eigenvalues[i].imag > 1e-6
            and modes.participation[0, i] + modes.participation[1, i] > 0.5
        ]
        if not candidates:
            raise ValueError(
                "動揺モードを自動検出できなかった。frequency_rad を明示すること。"
            )
        frequency_rad = float(abs(modes.eigenvalues[candidates[0]].imag))

    # 動揺ループを開いた GEP(s) を使う。閉じたまま評価すると動揺モードの
    # 共振が位相に現れ、設計点がその急変点に重なってしまう。
    gep = open_loop_gep(system)
    # LTI システムは複素数で直接評価できる（GEP(jω)）。
    gep_value = complex(np.atleast_1d(np.squeeze(gep(1j * frequency_rad)))[0])
    phase_lag = float(np.angle(gep_value))

    # 打ち消すべき位相。GEP が遅れている（負の位相）ぶんを進める。
    required_lead = -phase_lag
    # 進み遅れ補償で作れるのは進み位相なので、範囲外なら折り返す。
    while required_lead > math.pi:
        required_lead -= 2.0 * math.pi
    while required_lead < -math.pi:
        required_lead += 2.0 * math.pi

    per_stage = required_lead / n_stages
    # 1 段あたり 55° を超える位相進みは進み遅れ補償として現実的でない
    # （時定数比が極端になり高周波雑音を増幅する）ので頭打ちにする。
    per_stage = float(np.clip(per_stage, -math.radians(55.0), math.radians(55.0)))

    if abs(per_stage) < 1e-9:
        lead, lag = 0.1, 0.1
    else:
        alpha = (1.0 - math.sin(per_stage)) / (1.0 + math.sin(per_stage))
        lag = math.sqrt(alpha) / frequency_rad
        lead = lag / alpha

    return PowerSystemStabilizer(
        Ks=Ks,
        Tw=Tw,
        T1=lead,
        T2=lag,
        T3=lead if n_stages > 1 else lag,
        T4=lag,
        input_signal=input_signal,
        output_limits=output_limits,
    )
