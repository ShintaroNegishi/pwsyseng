"""1 軸モデル（3 次モデル）。

古典モデルに界磁巻線の状態量 E'q を加えたもの。界磁磁束が時定数
``T'd0`` で変化するため、界磁電圧 Efd を操作する励磁制御 (AVR) の効果を
表現できる。AVR を扱うには最低限このモデルが必要である。

.. math::

    \\frac{d\\delta}{dt} = \\omega_s \\Delta\\omega, \\qquad
    2H \\frac{d\\Delta\\omega}{dt} = P_m - P_e - D\\,\\Delta\\omega, \\qquad
    T'_{d0}\\frac{dE'_q}{dt} = E_{fd} - E'_q - (x_d - x'_d) I_d

仮定
----
突極性を無視し ``x_q = x'_d`` としている。この仮定のもとでは発電機は
「大きさ E'q・位相 δ の電圧源の背後に x'_d」という古典モデルと同じ
等価回路で表せるので、ネットワーク側の計算を共通化できる。円筒形
回転子機（火力・原子力のタービン発電機）では良い近似だが、突極機
（水力発電機）では磁気トルクの寄与を落とすことになる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..events import Stage
from ..network import ElectricalSolution, Network
from ..units import SystemBase
from .base import Machine


@dataclass
class OneAxisMachine(Machine):
    """1 軸モデルの同期発電機。

    Parameters
    ----------
    H:
        慣性定数 [s]。
    D:
        制動係数 [p.u.]。
    xd:
        d 軸同期リアクタンス [p.u.]。
    xd_prime:
        d 軸過渡リアクタンス [p.u.]。``xd`` より小さくなければならない。
    Td0_prime:
        d 軸開放過渡時定数 [s]。界磁磁束が変化する速さを決める。
    Vt0:
        事故前定常状態における端子電圧の目標値 [p.u.]。この値と送電電力
        ``Pe0`` から δ0, E'q0, Efd0 が一意に決まる。

    Notes
    -----
    古典モデルと違い、内部起電力 E'q は状態量であって定数ではない。
    このため P-δ 曲線も時間とともに動く点に注意すること。
    """

    H: float = 5.0
    D: float = 0.0
    xd: float = 1.8
    xd_prime: float = 0.3
    Td0_prime: float = 8.0
    Vt0: float = 1.0

    n_states: ClassVar[int] = 3
    state_names: ClassVar[tuple[str, ...]] = ("delta", "omega", "Eq_prime")
    responds_to_excitation: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.H <= 0.0:
            raise ValueError(f"慣性定数 H は正でなければならない (H={self.H})。")
        if not 0.0 < self.xd_prime < self.xd:
            raise ValueError(
                f"リアクタンスは 0 < x'_d < x_d を満たす必要がある "
                f"(x'_d={self.xd_prime}, x_d={self.xd})。"
            )
        if self.Td0_prime <= 0.0:
            raise ValueError(
                f"過渡時定数 T'd0 は正でなければならない (T'd0={self.Td0_prime})。"
            )

    # ------------------------------------------------------------------
    @property
    def x_internal(self) -> float:
        return self.xd_prime

    def rotor_angle(self, x: np.ndarray) -> float:
        return float(x[0])

    def speed(self, x: np.ndarray) -> float:
        return float(x[1])

    def internal_emf(self, x: np.ndarray) -> float:
        # 古典モデルと違い、内部起電力そのものが状態量である。
        return float(x[2])

    # ------------------------------------------------------------------
    def derivatives(
        self,
        x: np.ndarray,
        solution: ElectricalSolution,
        Pm: float,
        Efd: float,
        base: SystemBase,
    ) -> np.ndarray:
        omega = float(x[1])
        eq_prime = float(x[2])

        d_delta, d_omega = self.swing_derivatives(omega, solution.Pe, Pm, base)
        # 界磁磁束の変化。第 3 項は電機子反作用による減磁で、
        # 負荷が重い（Id が大きい）ほど E'q を押し下げる。
        d_eq = (
            Efd - eq_prime - (self.xd - self.xd_prime) * solution.Id
        ) / self.Td0_prime
        return np.array([d_delta, d_omega, d_eq], dtype=float)

    # ------------------------------------------------------------------
    def initial_state(
        self, network: Network, Pe0: float
    ) -> tuple[np.ndarray, float, float]:
        """端子電圧 Vt0 と送電電力 Pe0 から定常状態を求める。

        手順は教科書どおり、端子から外へ向かって解く。

        1. 端子電圧の位相 θ を ``Pe0 = Vt0 V∞ sin θ / x_e`` から求める
        2. 電機子電流 ``I = (Vt∠θ - V∞) / (j x_e)`` を求める
        3. ``E'q∠δ = Vt∠θ + j x'_d I`` から δ0 と E'q0 を得る
        4. 電流を d-q 軸に分解して Id を求める
        5. 定常条件 ``dE'q/dt = 0`` から ``Efd0 = E'q0 + (x_d - x'_d) Id``
        """
        x_e = network.external_reactance(Stage.PRE)
        if not math.isfinite(x_e) or x_e <= 0.0:
            raise ValueError(
                f"事故前の外部リアクタンスが不正 (x_e={x_e})。正の有限値が必要。"
            )
        v_inf = network.V_inf

        sin_theta = Pe0 * x_e / (self.Vt0 * v_inf)
        if abs(sin_theta) > 1.0:
            raise ValueError(
                f"端子電圧 Vt0={self.Vt0} では送電電力 Pe0={Pe0} を送れない"
                f" (sin θ = {sin_theta:.4f})。Vt0 を上げるか Pe0 を下げること。"
            )
        theta = math.asin(sin_theta)

        vt_phasor = self.Vt0 * np.exp(1j * theta)
        current = (vt_phasor - v_inf) / (1j * x_e)
        e_phasor = vt_phasor + 1j * self.xd_prime * current

        delta0 = float(np.angle(e_phasor))
        eq0 = float(np.abs(e_phasor))

        i_dq = current * np.exp(-1j * delta0)
        Id = float(-np.imag(i_dq))

        efd0 = eq0 + (self.xd - self.xd_prime) * Id

        x0 = np.array([delta0, 0.0, eq0], dtype=float)
        return x0, float(Pe0), float(efd0)
