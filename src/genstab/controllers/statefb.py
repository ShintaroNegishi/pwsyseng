"""状態フィードバック制御（極配置・LQR）。

制御工学の授業で学ぶ状態フィードバック ``u = -K Δx`` を、そのまま
発電機系に適用するための制御器である。python-control の
``place`` と ``lqr`` で設計する。

前提と限界
----------
状態フィードバックは全状態が測定できることを前提にしている。実際の
発電機で回転子位相角 δ や界磁磁束 E'q を直接測ることはできないので、
実装するにはオブザーバ（状態推定器）が必要になる。本モジュールは
「理想的に全状態が使えたらどこまでできるか」という上限を示すもので、
実機に載せる制御則としては AVR・ガバナ・PSS の構成が現実的である。

この対比自体が教材として有用で、古典的な PSS（出力フィードバック）と
理想的な状態フィードバックの制動性能を比べられる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .base import Controller, ControllerKind, Measurement, StatelessController


@dataclass
class StateFeedback(StatelessController):
    """状態フィードバック制御器 ``u = u_0 - K (x - x_0)``。

    Parameters
    ----------
    K:
        フィードバックゲイン。長さは発電機モデルの状態数と一致すること。
    target:
        操作対象。``"Pm"``（機械入力）または ``"Efd"``（界磁電圧）。
    limits:
        操作量の上下限。``None`` なら制限なし。

    Notes
    -----
    ゲインは :func:`design_lqr` や :func:`design_pole_placement` で
    設計するのが簡単である。
    """

    K: np.ndarray = field(default_factory=lambda: np.zeros(2))
    target: str = "Pm"
    limits: tuple[float, float] | None = None

    _u0: float = field(init=False, default=0.0)
    _x0: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.K = np.atleast_1d(np.asarray(self.K, dtype=float)).ravel()
        if self.target not in ("Pm", "Efd"):
            raise ValueError(
                f"target は 'Pm' または 'Efd' のいずれか (指定値: {self.target})。"
            )
        # 操作対象によって接続先が変わるので、種別をインスタンスごとに決める。
        self.kind = (
            ControllerKind.GOVERNOR if self.target == "Pm" else ControllerKind.EXCITER
        )
        if self.limits is not None and self.limits[0] >= self.limits[1]:
            raise ValueError(f"limits は (下限, 上限) の順で与えること: {self.limits}。")

    def initialize(self, meas: Measurement, u_steady: float) -> np.ndarray:
        if meas.x_machine is None:
            raise ValueError("状態フィードバックには発電機の状態ベクトルが必要。")
        if meas.x_machine.size != self.K.size:
            raise ValueError(
                f"フィードバックゲインの長さ {self.K.size} が発電機の状態数 "
                f"{meas.x_machine.size} と一致しない。"
            )
        self._u0 = float(u_steady)
        self._x0 = np.array(meas.x_machine, dtype=float, copy=True)
        return np.zeros(0, dtype=float)

    def output(
        self, t: float, xc: np.ndarray, meas: Measurement, aux: float = 0.0
    ) -> float:
        if self._x0 is None or meas.x_machine is None:
            raise RuntimeError("initialize() が呼ばれていない。")
        deviation = np.asarray(meas.x_machine, dtype=float) - self._x0
        value = self._u0 - float(self.K @ deviation)
        if self.limits is not None:
            value = float(np.clip(value, *self.limits))
        return value


def _design_matrices(system, target: str):
    """設計に使う (A, B) を取り出す。"""
    from ..linearize import input_matrix, state_matrix

    if system.n_states != system.machine.n_states:
        raise ValueError(
            "状態フィードバックは発電機の状態だけを見る制御則なので、"
            "他の制御器を含まない系に対して設計すること"
            f"（現在の状態数 {system.n_states}、発電機の状態数 {system.machine.n_states}）。"
        )
    A = state_matrix(system)
    B = input_matrix(system, (target,))
    return A, B


def design_lqr(
    system,
    Q: np.ndarray | None = None,
    R: np.ndarray | float = 1.0,
    *,
    target: str = "Pm",
    limits: tuple[float, float] | None = None,
) -> StateFeedback:
    """LQR（最適レギュレータ）でゲインを設計する。

    評価関数 ``J = ∫ (Δxᵀ Q Δx + Δuᵀ R Δu) dt`` を最小化する。
    Q を大きくすると状態の収束を優先し、R を大きくすると操作量を
    節約する。

    Parameters
    ----------
    Q:
        状態の重み行列。省略すると単位行列。動揺の抑制を狙うなら
        速度偏差 Δω の重みを大きくするとよい。
    R:
        操作量の重み。スカラーでもよい。
    target:
        操作対象（``"Pm"`` または ``"Efd"``）。

    Notes
    -----
    ``control.lqr`` は内部で Riccati 方程式を解く。slycot が入っていれば
    その実装が、無ければ scipy の実装が使われる。どちらでも結果は同じで、
    授業で扱う規模の系では速度差も問題にならない。
    """
    import control as ct

    A, B = _design_matrices(system, target)
    Q = np.eye(A.shape[0]) if Q is None else np.asarray(Q, dtype=float)
    R = np.atleast_2d(np.asarray(R, dtype=float))

    K, _, _ = ct.lqr(A, B, Q, R)
    return StateFeedback(K=np.asarray(K).ravel(), target=target, limits=limits)


def design_pole_placement(
    system,
    poles: Sequence[complex],
    *,
    target: str = "Pm",
    limits: tuple[float, float] | None = None,
    method: str = "place",
) -> StateFeedback:
    """極配置でゲインを設計する。

    Parameters
    ----------
    poles:
        閉ループ極の希望値。発電機の状態数と同じ個数を指定し、
        複素極は共役対で与えること。
    method:
        極配置のアルゴリズム。``"place"`` は Kautsky らの手法、
        ``"place_varga"`` は Varga の手法（slycot が必要）。
        重根を含む配置では ``"place_varga"`` のほうが安定に解ける。

    Examples
    --------
    減衰比 0.2、固有角振動数 7 rad/s の動揺モードを狙う場合::

        zeta, wn = 0.2, 7.0
        s = -zeta * wn + 1j * wn * (1 - zeta**2) ** 0.5
        controller = design_pole_placement(system, [s, s.conjugate()])
    """
    import control as ct

    A, B = _design_matrices(system, target)
    poles = list(poles)
    if len(poles) != A.shape[0]:
        raise ValueError(
            f"極の個数 {len(poles)} が状態数 {A.shape[0]} と一致しない。"
        )

    if method == "place":
        K = ct.place(A, B, poles)
    elif method == "place_varga":
        try:
            K = ct.place_varga(A, B, poles)
        except Exception as error:  # slycot が無い場合など
            raise RuntimeError(
                "place_varga は slycot を必要とする。environment.yml で作った"
                " 環境なら入っているはずなので、conda env の指定を確認すること。"
                f" 元のエラー: {error}"
            ) from error
    else:
        raise ValueError(
            f"method は 'place' または 'place_varga' のいずれか (指定値: {method})。"
        )
    return StateFeedback(K=np.asarray(K).ravel(), target=target, limits=limits)
