"""発電機モデルの抽象基底クラス。

本パッケージでは発電機モデルを詳細度の異なる複数クラスとして用意し、
授業の進度に合わせて差し替えられるようにしている。

===========================  ====  ==========================================
クラス                        次数  状態量
===========================  ====  ==========================================
:class:`~genstab.machines.classical.ClassicalMachine`      2     δ, Δω
:class:`~genstab.machines.onaxis.OneAxisMachine`           3     δ, Δω, E'q
===========================  ====  ==========================================

いずれのモデルも共通のインタフェースを実装するため、
:class:`~genstab.system.SMIBSystem` からは同じように扱える。
励磁制御 (AVR) は界磁電圧 ``Efd`` を、調速制御 (ガバナ) は機械入力
``Pm`` を操作する。古典モデルは界磁回路を持たないため ``Efd`` を
無視する点に注意すること（AVR の効果を見るには 1 軸モデル以上が必要）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

from ..network import ElectricalSolution, Network
from ..units import SystemBase


class Machine(ABC):
    """同期発電機モデルの抽象基底クラス。"""

    #: 状態変数の個数。
    n_states: ClassVar[int]
    #: 状態変数の名前（作図と結果アクセスに使う）。
    state_names: ClassVar[tuple[str, ...]]
    #: 界磁電圧 Efd が状態方程式に現れるか（AVR が効くか）。
    responds_to_excitation: ClassVar[bool] = False
    #: 慣性定数 [s]。全モデルが動揺方程式を持つため共通。
    H: float
    #: 制動係数 [p.u.]（速度偏差 p.u. あたりのトルク）。
    D: float

    # ------------------------------------------------------------------
    # 状態量へのアクセス
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def x_internal(self) -> float:
        """背後リアクタンス [p.u.]（古典モデル・1 軸モデルとも x_d'）。"""

    @abstractmethod
    def rotor_angle(self, x: np.ndarray) -> float:
        """状態ベクトルから回転子位相角 δ [rad] を取り出す。"""

    @abstractmethod
    def speed(self, x: np.ndarray) -> float:
        """状態ベクトルから速度偏差 Δω [p.u.] を取り出す。"""

    @abstractmethod
    def internal_emf(self, x: np.ndarray) -> float:
        """状態ベクトルから内部起電力の大きさ E [p.u.] を求める。"""

    # ------------------------------------------------------------------
    # 状態方程式
    # ------------------------------------------------------------------
    @abstractmethod
    def derivatives(
        self,
        x: np.ndarray,
        solution: ElectricalSolution,
        Pm: float,
        Efd: float,
        base: SystemBase,
    ) -> np.ndarray:
        """状態微分 dx/dt を返す。

        Parameters
        ----------
        x:
            この発電機の状態ベクトル。
        solution:
            現時刻のネットワーク解（電気出力・端子電圧・d-q 電流）。
        Pm:
            機械入力 [p.u.]。ガバナが接続されていればその出力。
        Efd:
            界磁電圧 [p.u.]。AVR が接続されていればその出力。
        base:
            系統の基準値（同期角速度 ω_s を使う）。
        """

    @abstractmethod
    def initial_state(
        self, network: Network, Pe0: float
    ) -> tuple[np.ndarray, float, float]:
        """事故前定常状態を求める。

        Returns
        -------
        x0:
            初期状態ベクトル。
        Pm0:
            定常状態の機械入力 [p.u.]（定常では電気出力に等しい）。
        Efd0:
            定常状態の界磁電圧 [p.u.]。
        """

    # ------------------------------------------------------------------
    # 補助
    # ------------------------------------------------------------------
    def state_index(self, name: str) -> int:
        """状態変数名からインデックスを引く。"""
        try:
            return self.state_names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"状態変数 '{name}' は存在しない。利用可能: {self.state_names}"
            ) from exc

    def swing_derivatives(
        self,
        omega: float,
        Pe: float,
        Pm: float,
        base: SystemBase,
    ) -> tuple[float, float]:
        """動揺方程式（全モデル共通の 2 式）を計算する。

        .. math::

            \\frac{d\\delta}{dt} = \\omega_s \\Delta\\omega, \\qquad
            2H \\frac{d\\Delta\\omega}{dt} = P_m - P_e - D\\,\\Delta\\omega

        ここで Δω は p.u. の速度偏差である。
        """
        d_delta = base.omega_s * omega
        d_omega = (Pm - Pe - self.D * omega) / (2.0 * self.H)
        return d_delta, d_omega
