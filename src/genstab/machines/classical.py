"""古典モデル（2 次モデル）。

発電機を「一定の内部起電力 E' の背後に過渡リアクタンス x_d' を持つ
電圧源」で表す、最も単純な発電機モデルである。状態量は回転子位相角 δ と
速度偏差 Δω の 2 つだけで、界磁磁束の変化を無視する。

このモデルで扱えること
    - 過渡安定性の第 1 波動揺（first swing）
    - 等面積法との対応
    - 臨界事故除去時間 (CCT)
    - 同期化力係数と制動係数による固有振動モード

このモデルで扱えないこと
    - AVR（励磁制御）の効果。界磁回路の状態量 E'q を持たないため、
      界磁電圧 Efd を変えても内部起電力が変化しない。AVR を扱うには
      :class:`~genstab.machines.onaxis.OneAxisMachine` 以上が必要。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..network import ElectricalSolution, Network
from ..units import SystemBase
from .base import Machine


@dataclass
class ClassicalMachine(Machine):
    """古典モデルの同期発電機。

    Parameters
    ----------
    H:
        慣性定数 [s]。発電機定格容量基準。動揺方程式には 2H として現れる。
    D:
        制動係数 [p.u.]。速度偏差 1 p.u. あたりの制動トルク。
    x_d_prime:
        過渡リアクタンス（背後リアクタンス）[p.u.]。
    E:
        内部起電力の大きさ [p.u.]。古典モデルでは時間的に一定と仮定する。

    Examples
    --------
    >>> machine = ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
    >>> machine.n_states
    2
    """

    H: float = 5.0
    D: float = 0.0
    x_d_prime: float = 0.3
    E: float = 1.1

    n_states: ClassVar[int] = 2
    state_names: ClassVar[tuple[str, ...]] = ("delta", "omega")
    responds_to_excitation: ClassVar[bool] = False

    def __post_init__(self) -> None:
        if self.H <= 0.0:
            raise ValueError(f"慣性定数 H は正でなければならない (H={self.H})。")
        if self.x_d_prime <= 0.0:
            raise ValueError(
                f"過渡リアクタンス x_d_prime は正でなければならない "
                f"(x_d_prime={self.x_d_prime})。"
            )

    # ------------------------------------------------------------------
    @property
    def x_internal(self) -> float:
        return self.x_d_prime

    def rotor_angle(self, x: np.ndarray) -> float:
        return float(x[0])

    def speed(self, x: np.ndarray) -> float:
        return float(x[1])

    def internal_emf(self, x: np.ndarray) -> float:
        # 古典モデルでは内部起電力は状態に依存せず一定。
        return self.E

    # ------------------------------------------------------------------
    def derivatives(
        self,
        x: np.ndarray,
        solution: ElectricalSolution,
        Pm: float,
        Efd: float,
        base: SystemBase,
    ) -> np.ndarray:
        # Efd は古典モデルでは使わない（界磁回路を持たないため）。
        omega = float(x[1])
        d_delta, d_omega = self.swing_derivatives(omega, solution.Pe, Pm, base)
        return np.array([d_delta, d_omega], dtype=float)

    def initial_state(
        self, network: Network, Pe0: float
    ) -> tuple[np.ndarray, float, float]:
        delta0 = network.initial_angle(self.E, Pe0)
        x0 = np.array([delta0, 0.0], dtype=float)
        # 定常状態では機械入力と電気出力が釣り合い、速度偏差はゼロ。
        return x0, float(Pe0), self.E
