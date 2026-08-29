"""孤立系統の周波数制御（ガバナ・LFC）。

過渡安定性・定態安定性が「発電機が同期を保てるか」を問うのに対し、
周波数制御は「負荷が変化したときに系統周波数を保てるか」を問う。
無限大母線に繋がった SMIB では周波数が母線側に固定されてしまうので、
この問題を扱うには無限大母線を持たない系が必要になる。

本モジュールの :class:`IsolatedSystem` は、系統全体を 1 つの等価発電機
（慣性 2H と負荷の周波数特性 D）で表した最も単純な周波数モデルである。

.. math::

    2H \\frac{d\\Delta\\omega}{dt} = \\Delta P_m - \\Delta P_L - D\\,\\Delta\\omega

既存の研究室コード ``GeneratorControl/main.py`` が伝達関数
``1/(Ms + D)`` で表していたものと同じ系で、``M = 2H`` の対応関係にある。
違いは、こちらは制御器を状態空間で持つため飽和などの非線形要素を
そのまま組み込める点にある。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .controllers.base import Controller, ControllerKind, Measurement
from .events import Stage
from .units import DEFAULT_BASE, SystemBase


class LoadProfile(Protocol):
    """負荷変動 ΔP_L(t) [p.u.] を表すプロトコル。"""

    def __call__(self, t: float) -> float: ...

    @property
    def switching_times(self) -> Sequence[float]:
        """不連続点の時刻。積分区間の分割に使う。"""
        ...


@dataclass(frozen=True)
class StepLoad:
    """ステップ状の負荷変動。

    Parameters
    ----------
    magnitude:
        負荷の増分 [p.u.]。正なら負荷増（周波数は下がる）。
    time:
        変化する時刻 [s]。
    """

    magnitude: float = 0.1
    time: float = 1.0

    def __call__(self, t: float) -> float:
        return self.magnitude if t >= self.time else 0.0

    @property
    def switching_times(self) -> Sequence[float]:
        return (self.time,)


@dataclass(frozen=True)
class RampLoad:
    """一定勾配で増加する負荷変動。"""

    rate: float = 0.02       #: 増加率 [p.u./s]
    start: float = 1.0       #: 開始時刻 [s]
    stop: float = float("inf")  #: 増加を止める時刻 [s]

    def __call__(self, t: float) -> float:
        if t <= self.start:
            return 0.0
        return self.rate * (min(t, self.stop) - self.start)

    @property
    def switching_times(self) -> Sequence[float]:
        return tuple(t for t in (self.start, self.stop) if np.isfinite(t))


@dataclass(frozen=True)
class NoLoadChange:
    """負荷変動なし。"""

    def __call__(self, t: float) -> float:
        return 0.0

    @property
    def switching_times(self) -> Sequence[float]:
        return ()


@dataclass
class IsolatedSystem:
    """無限大母線を持たない単機等価系統（周波数制御の学習用）。

    Parameters
    ----------
    H:
        系統全体の等価慣性定数 [s]。動揺方程式には 2H として現れる。
    D:
        負荷の周波数特性 [p.u.]。周波数が下がると負荷も減る効果を表す。
    controllers:
        接続する制御器。周波数制御ではガバナ (``kind=GOVERNOR``) のみが
        意味を持つ。空なら制御なしで、負荷変動に対して周波数が下がりっぱなしになる。
    load:
        負荷変動 ΔP_L(t)。
    base:
        系統の基準値。

    Examples
    --------
    ガバナのみ（定常偏差が残る）と、ガバナ + LFC（偏差が消える）の比較::

        governor_only = IsolatedSystem(controllers=[Governor(R=0.05)],
                                       load=StepLoad(0.1, 1.0))
        with_lfc = IsolatedSystem(
            controllers=[Governor(R=0.05), LoadFrequencyControl(Ki=0.3)],
            load=StepLoad(0.1, 1.0),
        )
    """

    H: float = 5.0
    D: float = 1.0
    controllers: list[Controller] = field(default_factory=list)
    load: Callable[[float], float] = field(default_factory=NoLoadChange)
    base: SystemBase = DEFAULT_BASE

    #: 状態変数の名前（回転子位相角は持たない）。
    state_names: tuple[str, ...] = field(init=False, default=("omega",))

    def __post_init__(self) -> None:
        if self.H <= 0.0:
            raise ValueError(f"慣性定数 H は正でなければならない (H={self.H})。")
        non_governor = [
            c for c in self.controllers if c.kind is not ControllerKind.GOVERNOR
        ]
        if non_governor:
            names = ", ".join(type(c).__name__ for c in non_governor)
            raise ValueError(
                f"孤立系の周波数モデルは機械入力を操作する制御器のみを扱える。"
                f" 励磁系や PSS は接続できない（指定された制御器: {names}）。"
                " 励磁制御を扱うには SMIBSystem を使うこと。"
            )
        self._build_slices()
        self._initialize_controllers()

    # ------------------------------------------------------------------
    def _build_slices(self) -> None:
        start = 1  # Δω
        self._controller_slices: list[slice] = []
        names = ["omega"]
        for controller in self.controllers:
            stop = start + controller.n_states
            self._controller_slices.append(slice(start, stop))
            prefix = type(controller).__name__
            names.extend(f"{prefix}.{n}" for n in controller.state_names)
            start = stop
        self.n_states = start
        self.state_names = tuple(names)

    def _initialize_controllers(self) -> None:
        meas0 = Measurement(t=0.0, delta=0.0, omega=0.0, Pe=0.0, Vt=1.0)
        states = [
            np.atleast_1d(np.asarray(c.initialize(meas0, 0.0), dtype=float)).ravel()
            for c in self.controllers
        ]
        self._x0 = (
            np.concatenate([np.zeros(1), *states]) if states else np.zeros(1)
        )

    # ------------------------------------------------------------------
    def initial_state(self) -> np.ndarray:
        """定常状態（周波数偏差ゼロ）を返す。"""
        return self._x0.copy()

    def switching_times(self, t_end: float) -> list[float]:
        """負荷変動の不連続点を返す（積分区間の分割に使う）。"""
        times = getattr(self.load, "switching_times", ())
        return sorted({float(t) for t in times if 0.0 < float(t) < t_end})

    def stage_at(self, t: float) -> Stage:
        """孤立系はネットワークの切替を持たないので常に PRE を返す。"""
        return Stage.PRE

    # ------------------------------------------------------------------
    def _measurement(self, t: float, x: np.ndarray) -> Measurement:
        return Measurement(
            t=t,
            delta=0.0,
            omega=float(x[0]),
            Pe=float(self.load(t)),
            Vt=1.0,
            x_machine=x[:1],
        )

    def _mechanical_power(self, t: float, x: np.ndarray, meas: Measurement) -> float:
        total = 0.0
        for controller, sl in zip(self.controllers, self._controller_slices):
            total += controller.output(t, x[sl], meas)
        return total

    def derivatives(
        self, t: float, x: np.ndarray, stage: Stage | None = None
    ) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_states,):
            raise ValueError(
                f"状態ベクトルの形状 {x.shape} が孤立系の状態数 "
                f"({self.n_states},) と一致しない。"
            )
        meas = self._measurement(t, x)

        d_pm = self._mechanical_power(t, x, meas)
        d_pl = float(self.load(t))

        dx = np.empty_like(x)
        dx[0] = (d_pm - d_pl - self.D * x[0]) / (2.0 * self.H)
        for controller, sl in zip(self.controllers, self._controller_slices):
            if controller.n_states:
                dx[sl] = controller.derivatives(t, x[sl], meas)
        return dx

    # ------------------------------------------------------------------
    def algebraic_outputs(
        self, t: np.ndarray, x: np.ndarray
    ) -> dict[str, np.ndarray]:
        """代数量（ΔPm, ΔPL, 周波数）を軌道全体について求める。"""
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        out = {k: np.empty(t.size) for k in ("Pm", "Pe", "Vt", "Efd")}
        for i in range(t.size):
            meas = self._measurement(t[i], x[:, i])
            out["Pm"][i] = self._mechanical_power(t[i], x[:, i], meas)
            out["Pe"][i] = float(self.load(t[i]))
            out["Vt"][i] = 1.0
            out["Efd"][i] = 0.0
        return out

    # ------------------------------------------------------------------
    def steady_state_deviation(self, load_step: float) -> float:
        """負荷ステップに対する定常周波数偏差 [p.u.] の解析値。

        積分制御（LFC）が入っていれば 0、比例制御だけなら
        ``-ΔP_L / (D + Σ 1/R)`` になる。この値は「系統定数」と呼ばれ、
        周波数が 1 p.u. ずれたときに系統全体で何 p.u. の電力が
        自然に応答するかを表す。
        """
        from .controllers.governor import (
            Governor,
            LoadFrequencyControl,
            ProportionalGovernor,
        )

        if any(isinstance(c, LoadFrequencyControl) for c in self.controllers):
            return 0.0

        stiffness = self.D
        for controller in self.controllers:
            if isinstance(controller, Governor):
                stiffness += 1.0 / controller.R
            elif isinstance(controller, ProportionalGovernor):
                stiffness += controller.K_gov
        if stiffness == 0.0:
            return float("-inf") if load_step > 0 else float("inf")
        return -load_step / stiffness

    def describe(self) -> str:
        """構成の要約を返す。"""
        lines = [
            f"IsolatedSystem  (状態数 {self.n_states})",
            f"  等価発電機 : H={self.H} s (2H={2*self.H}), D={self.D} p.u.",
            f"  負荷変動   : {type(self.load).__name__}",
        ]
        if self.controllers:
            lines.append("  制御器     :")
            for c in self.controllers:
                lines.append(f"    - {type(c).__name__}")
        else:
            lines.append("  制御器     : なし（周波数は下がりっぱなしになる）")
        return "\n".join(lines)
