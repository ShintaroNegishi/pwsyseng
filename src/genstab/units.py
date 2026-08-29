"""基準値と単位の取り扱い。

本パッケージは Kundur 流の標準形を採用する。すなわち動揺方程式を

    dδ/dt   = ω_s * Δω
    2H dΔω/dt = Pm - Pe - D * Δω

と書き、状態量を

    δ   : 発電機内部位相角 [rad]（同期回転座標系から見た角度）
    Δω  : 速度偏差 [p.u.]（(ω - ω_s) / ω_s）

とする。Δω を p.u. で持つのが要点で、これにより慣性定数 H [s] と
制動係数 D [p.u.] がそのまま教科書の値として使える。

よくある簡略形 `dδ/dt = ω` （速度偏差を rad/s で持つ形）とは
係数の置き方が違うだけで等価だが、標準形にしておくと固有振動数が

    ω_n = sqrt(K_s * ω_s / (2H))   [rad/s]

という解析式とそのまま一致し、教材として検算しやすい。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: 日本の東側 50 Hz を既定とする。西日本を扱う場合は 60.0 を指定する。
DEFAULT_FREQUENCY_HZ = 50.0


@dataclass(frozen=True)
class SystemBase:
    """系統の基準値。

    Parameters
    ----------
    frequency_hz:
        定格周波数 [Hz]。50 または 60。
    s_base_mva:
        皮相電力の基準値 [MVA]。p.u. 値のまま計算する限り結果には
        影響しないが、物理量へ戻すときに使う。
    """

    frequency_hz: float = DEFAULT_FREQUENCY_HZ
    s_base_mva: float = 100.0

    @property
    def omega_s(self) -> float:
        """同期角速度 [rad/s]。"""
        return 2.0 * math.pi * self.frequency_hz

    def to_hz(self, domega_pu: float) -> float:
        """速度偏差 [p.u.] を周波数 [Hz] に変換する。"""
        return self.frequency_hz * (1.0 + domega_pu)


#: 既定の基準値（50 Hz, 100 MVA）。
DEFAULT_BASE = SystemBase()


def deg(rad: float) -> float:
    """ラジアンを度に変換する（作図用）。"""
    return math.degrees(rad)


def rad(degree: float) -> float:
    """度をラジアンに変換する。"""
    return math.radians(degree)
