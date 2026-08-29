r"""発電機が接続されるネットワーク（外部系統）のモデル。

Phase 1〜5 で使うのは 1 機無限大母線系統 (SMIB) である。等価回路は

        x_d'            x_e(stage)
    E∠δ ──/\/\/\── Vt ──/\/\/\── V∞∠0
    (発電機内部)   (端子)        (無限大母線)

で、事故前・事故中・事故後で外部リアクタンス ``x_e`` が切り替わる。
背後リアクタンス ``x_d'`` は発電機モデル側が保持する。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .events import Stage


@dataclass(frozen=True)
class ElectricalSolution:
    """ある時刻におけるネットワーク解。

    Attributes
    ----------
    Pe:
        発電機の電気出力 [p.u.]。
    Vt:
        発電機端子電圧の大きさ [p.u.]。
    Id, Iq:
        d 軸・q 軸電流 [p.u.]。1 軸モデル以降で界磁回路の
        微分方程式に必要になる。古典モデルでは使わない。
    """

    Pe: float
    Vt: float
    Id: float = 0.0
    Iq: float = 0.0


class Network(ABC):
    """ネットワークの抽象基底クラス。"""

    @abstractmethod
    def solve(self, stage: Stage, E: float, delta: float) -> ElectricalSolution:
        """内部起電力 ``E∠delta`` に対するネットワーク解を返す。"""

    @abstractmethod
    def transfer_reactance(self, stage: Stage) -> float:
        """内部起電力から無限大母線までの総リアクタンス [p.u.]。"""


@dataclass
class SMIBNetwork(Network):
    """1 機無限大母線系統。

    Parameters
    ----------
    x_pre, x_fault, x_post:
        事故前・事故中・事故後の外部リアクタンス [p.u.]。
        発電機端子から無限大母線までの値であり、背後リアクタンス x_d'
        は含まない。
    V_inf:
        無限大母線電圧の大きさ [p.u.]。
    x_internal:
        発電機の背後リアクタンス x_d' [p.u.]。通常は
        :meth:`attach` で発電機モデルから設定される。

    Notes
    -----
    **このモデルが正しく表せるのは電気出力 P_e だけである。**

    事故を「転送リアクタンスの変化」だけで表現し、どのステージでも
    内部起電力から無限大母線までが 1 本のリアクタンスでつながって
    いるとみなす。しかし実際の地絡では事故点に分路（地絡電流の経路）が
    でき、自己アドミタンスと相互アドミタンスが独立に変わる。1 本の
    転送リアクタンスではこの分路成分を表せないため、**事故中の端子電圧
    Vt と d-q 電流 Id, Iq は物理的に正しくない**。

    - ``x_fault = math.inf``: 電気出力はゼロになる。ただしこれは線路開放と
      同じ状態なので、端子電圧は内部起電力に等しくなる（実際の端子三相
      短絡なら Vt = 0 に落ちる）。
    - ``x_fault`` に有限値: 電気出力は減るが、端子電圧はむしろ上がることが
      ある（ある設定で事故前 1.055 → 事故中 1.097 になることを確認して
      いる）。地絡による電圧低下は再現できない。

    動揺方程式は P_e にしか依存しないので、過渡安定性を論じるうえでは
    この近似で十分であり、教科書でも広く使われている。一方、事故中の
    端子電圧に依存する量（たとえば AVR の事故中の応答）は、このモデルの
    適用範囲外である。定量的に扱うには、端子母線を残した Ybus を解くか、
    事故点の分路インピーダンスを陽に持つモデルが必要になる。
    """

    x_pre: float
    x_fault: float
    x_post: float
    V_inf: float = 1.0
    x_internal: float = 0.0

    def attach(self, x_internal: float) -> "SMIBNetwork":
        """発電機の背後リアクタンスを登録する。"""
        self.x_internal = float(x_internal)
        return self

    def external_reactance(self, stage: Stage) -> float:
        """外部リアクタンス [p.u.]（背後リアクタンスを含まない）。"""
        return {
            Stage.PRE: self.x_pre,
            Stage.FAULT: self.x_fault,
            Stage.POST: self.x_post,
        }[stage]

    def transfer_reactance(self, stage: Stage) -> float:
        return self.x_internal + self.external_reactance(stage)

    def max_power(self, stage: Stage, E: float) -> float:
        """P-δ 曲線の波高値 ``E * V∞ / X`` [p.u.]。"""
        x = self.transfer_reactance(stage)
        if not math.isfinite(x):
            return 0.0
        return E * self.V_inf / x

    def power_angle_curve(self, stage: Stage, E: float, delta) -> np.ndarray:
        """P-δ 曲線 ``Pmax * sin(delta)`` を返す（等面積法の作図用）。"""
        return self.max_power(stage, E) * np.sin(np.asarray(delta, dtype=float))

    def solve(self, stage: Stage, E: float, delta: float) -> ElectricalSolution:
        x_e = self.external_reactance(stage)
        x_total = self.x_internal + x_e

        if not math.isfinite(x_total):
            # 線路開放。電流が流れないので電気出力はゼロ、
            # 端子電圧は内部起電力に等しい。
            return ElectricalSolution(Pe=0.0, Vt=E, Id=0.0, Iq=0.0)

        E_phasor = E * np.exp(1j * delta)
        V_phasor = complex(self.V_inf, 0.0)

        current = (E_phasor - V_phasor) / (1j * x_total)
        Pe = float(np.real(E_phasor * np.conj(current)))
        Vt_phasor = V_phasor + 1j * x_e * current

        # d-q 座標（q 軸を内部起電力の向きに取る）へ電流を射影する。
        i_dq = current * np.exp(-1j * delta)
        Iq = float(np.real(i_dq))
        Id = float(-np.imag(i_dq))

        return ElectricalSolution(
            Pe=Pe, Vt=float(np.abs(Vt_phasor)), Id=Id, Iq=Iq
        )

    def initial_angle(self, E: float, Pe0: float) -> float:
        """事故前ネットワークで電気出力 ``Pe0`` を送る内部位相角 [rad]。

        ``Pe0 = Pmax * sin(delta0)`` を解く。安定平衡点（δ < 90°）を返す。
        """
        pmax = self.max_power(Stage.PRE, E)
        if pmax <= 0.0:
            raise ValueError("事故前ネットワークで電力を送れない（Pmax <= 0）。")
        ratio = Pe0 / pmax
        if abs(ratio) > 1.0:
            raise ValueError(
                f"要求出力 Pe0={Pe0} が送電可能な最大電力 Pmax={pmax:.4f} を超えている。"
                " 内部起電力 E を上げるか、リアクタンスを下げること。"
            )
        return float(np.arcsin(ratio))
