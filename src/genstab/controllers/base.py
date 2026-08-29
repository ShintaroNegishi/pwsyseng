"""制御器の抽象基底クラス。

制御器は「あとから付け外しできる部品」として設計されている。
発電機モデルと同様に自分の状態微分と出力を返すだけで、状態ベクトルの
連結は :class:`~genstab.system.SMIBSystem` が行う。制御器を 1 つも
渡さなければ機械入力と界磁電圧が定常値のまま固定され、素の動揺方程式に
そのまま縮退する。

接続の形
--------
制御器は操作対象によって 3 種類に分かれる::

    PSS (stabilizer) ──┐
                       ├─→ AVR (exciter) ──→ 界磁電圧 Efd ──→ 発電機
    端子電圧 Vt ───────┘

    ガバナ (governor) ─────→ 機械入力 Pm ────────────────→ 発電機

PSS は発電機に直接つながらず、AVR への補助信号として加わる。この
カスケード構造は実機の構成そのものであり、「AVR が悪化させた制動を
PSS が回復させる」という Phase 5 の題材はこの接続で初めて再現できる。

初期化について
--------------
制御器を接続した瞬間に系が乱れては困るので、各制御器は定常状態で
自分の出力が定常入力値に一致するように内部状態を初期化する
(:meth:`Controller.initialize`)。これを怠ると、シミュレーション開始
直後に事故と無関係な過渡応答が現れてしまう。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np


class ControllerKind(Enum):
    """制御器の種別（何を操作するか）。"""

    EXCITER = "exciter"        #: 界磁電圧 Efd を操作する（AVR）
    GOVERNOR = "governor"      #: 機械入力 Pm を操作する（ガバナ・LFC）
    STABILIZER = "stabilizer"  #: 励磁系への補助信号を出す（PSS）


@dataclass(frozen=True)
class Measurement:
    """制御器が参照できる測定量。

    実機の制御器が使えるのは端子で測れる量に限られる。ここでは教材の
    簡潔さを優先して回転子位相角 δ も渡しているが、δ は直接測れない量で
    あることに注意すること（PSS が速度偏差 Δω や電気出力 Pe を入力に
    使うのはこのためである）。
    """

    t: float       #: 時刻 [s]
    delta: float   #: 回転子位相角 [rad]
    omega: float   #: 速度偏差 [p.u.]
    Pe: float      #: 電気出力 [p.u.]
    Vt: float      #: 端子電圧 [p.u.]
    #: 発電機の状態ベクトル。状態フィードバックのように全状態を仮定する
    #: 手法のためだけに用意している。実機で全状態が測れるとは限らず、
    #: 通常はオブザーバ（状態推定器）が必要になる点に注意すること。
    x_machine: np.ndarray | None = None


class Controller(ABC):
    """制御器の抽象基底クラス。"""

    #: 何を操作する制御器か。
    kind: ClassVar[ControllerKind]
    #: 制御器自身が持つ状態変数の個数（比例制御のみなら 0）。
    n_states: ClassVar[int] = 0
    #: 状態変数の名前。
    state_names: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        """定常状態における内部状態を返す。

        Parameters
        ----------
        meas:
            事故前定常状態の測定量。
        u_steady:
            定常状態でこの制御器が出すべき値。励磁系なら定常界磁電圧
            Efd0、ガバナなら 0（機械入力の偏差として扱うため）。

        Returns
        -------
        長さ :attr:`n_states` の初期状態ベクトル。
        """

    @abstractmethod
    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        """制御器の出力を返す。

        Parameters
        ----------
        aux:
            励磁系のみが使う補助入力。PSS 出力の合計が渡される。
        """

    @abstractmethod
    def derivatives(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> np.ndarray:
        """制御器の状態微分 dxc/dt を返す。状態を持たない場合は空配列。"""

    # ------------------------------------------------------------------
    def _no_states(self) -> np.ndarray:
        """状態を持たない制御器のための空の微分ベクトル。"""
        return np.zeros(0, dtype=float)


class StatelessController(Controller):
    """状態を持たない制御器のための便宜的な基底クラス。

    比例ゲインのみの制御器はこれを継承し、:meth:`output` だけを実装
    すればよい。
    """

    n_states: ClassVar[int] = 0
    state_names: ClassVar[tuple[str, ...]] = ()

    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        return np.zeros(0, dtype=float)

    def derivatives(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> np.ndarray:
        return np.zeros(0, dtype=float)
