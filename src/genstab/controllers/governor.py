"""調速制御（ガバナ）と負荷周波数制御 (LFC)。

どちらも機械入力 ``P_m`` を操作して周波数を保つ制御である。役割は
はっきり分かれている。

ガバナ（一次調整）
    速度偏差に比例して機械入力を増減させる。応答は速いが比例制御なので
    定常偏差が残る。複数機が並列運転するとき、速度調定率 R によって
    負荷の分担比が決まる。

LFC（二次調整）
    速度偏差を積分して機械入力を動かす。応答は遅いが、定常偏差を
    ゼロにできる。

既存の研究室コード ``GeneratorControl/main.py`` は、この 2 つを
伝達関数 ``K_gov`` と ``K_I / s`` で表して周波数応答を計算していた。
ここでは同じ制御を状態空間で書き直し、非線形の動揺方程式に接続できる
形にしている。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .base import Controller, ControllerKind, Measurement, StatelessController


@dataclass
class Governor(Controller):
    """速度調定率（ドループ）付きガバナ。

    .. math::

        T_g \\frac{d\\Delta P_m}{dt} = -\\frac{1}{R}\\Delta\\omega - \\Delta P_m

    Parameters
    ----------
    R:
        速度調定率 [p.u.]。速度が 1 p.u. 変化したときに機械入力が
        ``1/R`` p.u. 変化する。実機では 0.03〜0.05（3〜5 %）。
        小さいほど周波数変化に敏感に応答する。
    Tg:
        調速機と原動機をまとめた等価時定数 [s]。蒸気タービンでは
        数秒のオーダーになる。
    limits:
        機械入力偏差の上下限 ``(min, max)`` [p.u.]。``None`` なら制限なし。

    Notes
    -----
    比例制御なので、負荷が変化したあとには必ず定常周波数偏差が残る。
    これをゼロにするには :class:`LoadFrequencyControl` を併用する。
    """

    R: float = 0.05
    Tg: float = 0.2
    limits: tuple[float, float] | None = None

    kind: ClassVar[ControllerKind] = ControllerKind.GOVERNOR
    n_states: ClassVar[int] = 1
    state_names: ClassVar[tuple[str, ...]] = ("dPm",)

    def __post_init__(self) -> None:
        if self.R <= 0.0:
            raise ValueError(f"速度調定率 R は正でなければならない (R={self.R})。")
        if self.Tg <= 0.0:
            raise ValueError(f"時定数 Tg は正でなければならない (Tg={self.Tg})。")
        if self.limits is not None and self.limits[0] >= self.limits[1]:
            raise ValueError(f"limits は (下限, 上限) の順で与えること: {self.limits}。")

    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        # 定常では速度偏差がゼロなので機械入力の偏差もゼロ。
        return np.array([u_steady], dtype=float)

    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        value = float(xc[0])
        if self.limits is not None:
            value = float(np.clip(value, *self.limits))
        return value

    def derivatives(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> np.ndarray:
        target = -meas.omega / self.R
        d_value = (target - float(xc[0])) / self.Tg

        if self.limits is not None:
            low, high = self.limits
            value = float(xc[0])
            if (value >= high and d_value > 0.0) or (value <= low and d_value < 0.0):
                d_value = 0.0

        return np.array([d_value], dtype=float)


@dataclass
class LoadFrequencyControl(Controller):
    """負荷周波数制御（積分制御による二次調整）。

    .. math::

        \\frac{dz}{dt} = -K_I \\Delta\\omega, \\qquad \\Delta P_m = z

    速度偏差を積分するので、定常状態では必ず ``Δω = 0`` になる。
    ガバナと併用するのが標準的な構成である。

    Parameters
    ----------
    Ki:
        積分ゲイン。大きくすると定常偏差の解消は速くなるが、
        大きすぎると振動的になる。
    limits:
        機械入力偏差の上下限 [p.u.]。
    """

    Ki: float = 0.5
    limits: tuple[float, float] | None = None

    kind: ClassVar[ControllerKind] = ControllerKind.GOVERNOR
    n_states: ClassVar[int] = 1
    state_names: ClassVar[tuple[str, ...]] = ("integral",)

    def __post_init__(self) -> None:
        if self.Ki <= 0.0:
            raise ValueError(f"積分ゲイン Ki は正でなければならない (Ki={self.Ki})。")
        if self.limits is not None and self.limits[0] >= self.limits[1]:
            raise ValueError(f"limits は (下限, 上限) の順で与えること: {self.limits}。")

    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        return np.array([u_steady], dtype=float)

    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        value = float(xc[0])
        if self.limits is not None:
            value = float(np.clip(value, *self.limits))
        return value

    def derivatives(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> np.ndarray:
        d_value = -self.Ki * meas.omega

        if self.limits is not None:
            low, high = self.limits
            value = float(xc[0])
            if (value >= high and d_value > 0.0) or (value <= low and d_value < 0.0):
                d_value = 0.0

        return np.array([d_value], dtype=float)


@dataclass
class ProportionalGovernor(StatelessController):
    """時定数を無視した比例ガバナ（最も単純な調速制御）。

    .. math::

        \\Delta P_m = -K_{gov}\\,\\Delta\\omega

    既存の研究室コードで ``tf(K_gov, 1)`` として表されていたものに
    対応する。原動機の遅れを持たないので現実的ではないが、
    ガバナの働きを最短で示すには分かりやすい。
    """

    K_gov: float = 20.0

    kind: ClassVar[ControllerKind] = ControllerKind.GOVERNOR

    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        return -self.K_gov * meas.omega
