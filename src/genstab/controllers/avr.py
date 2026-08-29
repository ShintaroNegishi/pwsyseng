"""励磁制御装置 (AVR) と励磁系。

AVR は端子電圧を測って界磁電圧 Efd を操作し、端子電圧を基準値に保つ。
発電機の電圧品質と送電能力の確保に不可欠な装置である。

一方で、応答の速い高ゲイン AVR は動揺モードの制動を悪化させることが
知られている。端子電圧を素早く保とうとする動作が、回転子の動揺に対して
負の制動トルクを与えるためで、重負荷・長距離送電の条件で顕著になる。
この現象こそが PSS（電力系統安定化装置）が生まれた理由であり、
Phase 5 で :class:`~genstab.controllers.pss.PowerSystemStabilizer` を
追加して回復させる流れが本教材の山場になる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .base import Controller, ControllerKind, Measurement


@dataclass
class SimpleExciter(Controller):
    """1 次遅れで表した高応答励磁系（IEEE ST1A の簡略形）。

    .. math::

        T_a \\frac{dE_{fd}}{dt} = K_a\\,(V_{ref} - V_t + V_s) - E_{fd}

    ``V_s`` は PSS からの補助信号で、PSS を接続していなければ 0 である。

    Parameters
    ----------
    Ka:
        励磁系ゲイン。大きいほど端子電圧を強く保つが、動揺モードの
        制動を悪化させやすい。実機では 100〜400 程度。
    Ta:
        励磁系時定数 [s]。静止形励磁方式では 0.01〜0.1 s と小さい。
    efd_limits:
        界磁電圧の上下限 ``(min, max)`` [p.u.]。``None`` なら制限なし。
        制限を設けると系が非線形になり、線形化による固有値解析と
        時間応答が厳密には一致しなくなる点に注意すること。

    Attributes
    ----------
    v_ref:
        基準電圧 [p.u.]。:meth:`initialize` で「定常状態において
        界磁電圧が定常値と一致する」ように自動設定される。
        :func:`genstab.linearize.state_space` はこれを入力に取れる。
    """

    Ka: float = 200.0
    Ta: float = 0.05
    efd_limits: tuple[float, float] | None = None

    kind: ClassVar[ControllerKind] = ControllerKind.EXCITER
    n_states: ClassVar[int] = 1
    state_names: ClassVar[tuple[str, ...]] = ("Efd",)

    v_ref: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        if self.Ka <= 0.0:
            raise ValueError(f"励磁系ゲイン Ka は正でなければならない (Ka={self.Ka})。")
        if self.Ta <= 0.0:
            raise ValueError(f"励磁系時定数 Ta は正でなければならない (Ta={self.Ta})。")
        if self.efd_limits is not None and self.efd_limits[0] >= self.efd_limits[1]:
            raise ValueError(f"efd_limits は (下限, 上限) の順で与えること: {self.efd_limits}。")

    # ------------------------------------------------------------------
    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        # 定常状態で Ka (v_ref - Vt) = Efd0 となるように基準電圧を決める。
        # こうしないと接続した瞬間に界磁電圧が飛び、事故と無関係な
        # 過渡応答が現れてしまう。
        self.v_ref = meas.Vt + u_steady / self.Ka
        return np.array([u_steady], dtype=float)

    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        efd = float(xc[0])
        if self.efd_limits is not None:
            efd = float(np.clip(efd, *self.efd_limits))
        return efd

    def derivatives(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> np.ndarray:
        efd = float(xc[0])
        error = self.v_ref - meas.Vt + aux
        d_efd = (self.Ka * error - efd) / self.Ta

        if self.efd_limits is not None:
            # 簡易アンチワインドアップ。上下限に張り付いた状態でさらに
            # その向きへ積み上がるのを止める。
            low, high = self.efd_limits
            if (efd >= high and d_efd > 0.0) or (efd <= low and d_efd < 0.0):
                d_efd = 0.0

        return np.array([d_efd], dtype=float)


@dataclass
class IEEEType1Exciter(Controller):
    """安定化変圧器付き励磁系（IEEE Type-1 の簡略形）。

    .. math::

        T_a \\frac{dV_a}{dt} &= K_a (V_{ref} - V_t + V_s - V_f) - V_a \\\\
        T_e \\frac{dE_{fd}}{dt} &= V_a - (K_e + S_e) E_{fd} \\\\
        T_f \\frac{dV_f}{dt} &= K_f \\frac{dE_{fd}}{dt} - V_f

    第 3 式が安定化変圧器（rate feedback）で、界磁電圧の変化率を負帰還
    することで励磁系そのものの応答を穏やかにする。これは AVR による
    制動悪化をある程度和らげるが、PSS ほどの効果はない。両者を比較すると
    「なぜ PSS が別途必要になったのか」がはっきりする。

    Parameters
    ----------
    Ka, Ta:
        増幅器のゲインと時定数。
    Ke, Te:
        励磁機の自励定数と時定数。
    Kf, Tf:
        安定化変圧器のゲインと時定数。
    """

    Ka: float = 50.0
    Ta: float = 0.06
    Ke: float = 1.0
    Te: float = 0.5
    Kf: float = 0.05
    Tf: float = 1.0

    kind: ClassVar[ControllerKind] = ControllerKind.EXCITER
    n_states: ClassVar[int] = 3
    state_names: ClassVar[tuple[str, ...]] = ("Va", "Efd", "Vf")

    v_ref: float = field(init=False, default=1.0)

    def __post_init__(self) -> None:
        for name in ("Ta", "Te", "Tf"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"時定数 {name} は正でなければならない。")
        if self.Ke <= 0.0:
            raise ValueError(f"励磁機定数 Ke は正でなければならない (Ke={self.Ke})。")

    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        efd0 = u_steady
        va0 = self.Ke * efd0          # dEfd/dt = 0 より
        vf0 = 0.0                     # 定常では変化率帰還はゼロ
        self.v_ref = meas.Vt + va0 / self.Ka + vf0
        return np.array([va0, efd0, vf0], dtype=float)

    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        return float(xc[1])

    def derivatives(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> np.ndarray:
        va, efd, vf = float(xc[0]), float(xc[1]), float(xc[2])

        d_va = (self.Ka * (self.v_ref - meas.Vt + aux - vf) - va) / self.Ta
        d_efd = (va - self.Ke * efd) / self.Te
        d_vf = (self.Kf * d_efd - vf) / self.Tf

        return np.array([d_va, d_efd, d_vf], dtype=float)
