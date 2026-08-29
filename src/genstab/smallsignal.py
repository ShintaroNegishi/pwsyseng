"""定態安定性（微小擾乱安定性）の解析。

動作点まわりで線形化した状態行列 A の固有値を調べる。すべての固有値の
実部が負であれば漸近安定であり、微小な擾乱を受けても動作点に戻る。

過渡安定性との違い
------------------
過渡安定性は「大きな事故に耐えられるか」を非線形シミュレーションで
調べるのに対し、定態安定性は「動作点が持続的に維持できるか」を線形
モデルの固有値で調べる。定態安定でなければ、事故がなくても振動が
成長する。両者は別の性質であり、片方だけでは系の安定性を語れない。

減衰比の目安
------------
電力系統の動揺モードは減衰比 ζ ≥ 0.05（5 %）が確保されていることが
一つの実務的な目安とされる。ζ が小さいと擾乱後の振動がいつまでも
収まらない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import fsolve

from .events import Stage
from .linearize import state_matrix
from .system import SMIBSystem


@dataclass
class ModalResult:
    """固有値解析の結果。"""

    eigenvalues: np.ndarray            #: 固有値 [1/s]
    participation: np.ndarray          #: 参加係数 (n_states, n_modes)
    state_names: tuple[str, ...]
    A: np.ndarray = field(repr=False)  #: 状態行列

    # ------------------------------------------------------------------
    @property
    def damping_ratios(self) -> np.ndarray:
        """各モードの減衰比 ζ = -Re(λ) / |λ|。"""
        magnitude = np.abs(self.eigenvalues)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(
                magnitude > 0.0, -self.eigenvalues.real / magnitude, np.nan
            )
        return ratios

    @property
    def damped_frequencies_hz(self) -> np.ndarray:
        """各モードの減衰振動周波数 [Hz]。実固有値では 0。"""
        return np.abs(self.eigenvalues.imag) / (2.0 * math.pi)

    @property
    def time_constants(self) -> np.ndarray:
        """各モードの時定数 [s]（実部の逆数の絶対値）。"""
        with np.errstate(divide="ignore"):
            return np.where(
                self.eigenvalues.real != 0.0,
                1.0 / np.abs(self.eigenvalues.real),
                np.inf,
            )

    @property
    def is_stable(self) -> bool:
        """すべての固有値の実部が負か。"""
        return bool(np.all(self.eigenvalues.real < 0.0))

    @property
    def dominant_index(self) -> int:
        """最も減衰の悪いモード（実部が最大）のインデックス。"""
        return int(np.argmax(self.eigenvalues.real))

    # ------------------------------------------------------------------
    def dominant_states(self, mode: int, top: int = 2) -> list[tuple[str, float]]:
        """指定モードに寄与の大きい状態変数を参加係数の順に返す。"""
        order = np.argsort(self.participation[:, mode])[::-1]
        return [
            (self.state_names[k], float(self.participation[k, mode]))
            for k in order[:top]
        ]

    def table(self) -> str:
        """固有値・減衰比・周波数・支配状態の一覧を文字列で返す。"""
        header = (
            f"{'#':>2}  {'eigenvalue':>26}  {'damping':>8}  "
            f"{'freq [Hz]':>9}  dominant states"
        )
        lines = [header, "-" * len(header)]
        for i, lam in enumerate(self.eigenvalues):
            dominant = ", ".join(
                f"{name} ({value:.2f})" for name, value in self.dominant_states(i)
            )
            lines.append(
                f"{i:>2}  {lam.real:>+11.5f}{lam.imag:>+11.5f}j  "
                f"{self.damping_ratios[i]:>8.4f}  "
                f"{self.damped_frequencies_hz[i]:>9.4f}  {dominant}"
            )
        verdict = "安定（すべての固有値が左半面）" if self.is_stable else "不安定（右半面に固有値あり）"
        lines.append("-" * len(header))
        lines.append(f"判定: {verdict}")
        return "\n".join(lines)

    def to_dataframe(self):
        """pandas.DataFrame に変換する。"""
        import pandas as pd

        return pd.DataFrame(
            {
                "eigenvalue": self.eigenvalues,
                "real": self.eigenvalues.real,
                "imag": self.eigenvalues.imag,
                "damping_ratio": self.damping_ratios,
                "frequency_hz": self.damped_frequencies_hz,
                "time_constant_s": self.time_constants,
            }
        )


def participation_factors(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """固有値と正規化した参加係数を返す。

    参加係数 ``p_ki = |Φ_ki · Ψ_ik|`` は状態 k がモード i にどれだけ
    関与しているかを表す無次元量である（Φ は右固有ベクトル行列、
    Ψ = Φ⁻¹ は左固有ベクトル行列）。単位系に依存しないため、
    「この振動モードは何の状態量が主役か」を判断するのに使える。
    """
    eigenvalues, right = np.linalg.eig(A)
    left = np.linalg.inv(right)
    factors = np.abs(right * left.T)
    column_sums = factors.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        factors = np.where(column_sums > 0.0, factors / column_sums, 0.0)
    return eigenvalues, factors


def analyze(
    system: SMIBSystem,
    *,
    x0: np.ndarray | None = None,
    stage: Stage = Stage.PRE,
) -> ModalResult:
    """動作点まわりの固有値解析を行う。

    Examples
    --------
    >>> import genstab
    >>> from genstab import smallsignal
    >>> machine = genstab.ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
    >>> network = genstab.SMIBNetwork(x_pre=0.4, x_fault=float("inf"), x_post=0.4)
    >>> system = genstab.SMIBSystem(machine, network, Pe0=0.8)
    >>> modes = smallsignal.analyze(system)
    >>> modes.is_stable
    True
    """
    A = state_matrix(system, x0=x0, stage=stage)
    eigenvalues, factors = participation_factors(A)
    return ModalResult(
        eigenvalues=eigenvalues,
        participation=factors,
        state_names=system.state_names,
        A=A,
    )


def synchronizing_coefficient(
    system: SMIBSystem, stage: Stage = Stage.PRE, x0: np.ndarray | None = None
) -> float:
    """同期化力係数 ``K_s = dPe/dδ`` [p.u./rad]。

    位相角がわずかに増えたとき電気出力がどれだけ増えるかを表す。
    正であれば位相角のずれを引き戻す向きに働き、これが同期運転を
    支えている。δ が 90° を超えると負になり、静的に脱調する。
    """
    x0 = system.initial_state() if x0 is None else np.asarray(x0, dtype=float)
    emf = system.machine.internal_emf(x0)
    delta0 = system.machine.rotor_angle(x0)
    pmax = system.network.max_power(stage, emf)
    return float(pmax * math.cos(delta0))


@dataclass(frozen=True)
class ClassicalMode:
    """古典モデルの動揺モードの解析解（数値解の検算用）。"""

    K_s: float                 #: 同期化力係数 [p.u./rad]
    natural_frequency: float   #: 非減衰固有角振動数 ω_n [rad/s]
    damping_ratio: float       #: 減衰比 ζ
    eigenvalues: np.ndarray    #: 固有値 [1/s]

    @property
    def natural_frequency_hz(self) -> float:
        """非減衰固有振動数 [Hz]。"""
        return self.natural_frequency / (2.0 * math.pi)


def classical_mode_analytic(
    system: SMIBSystem, stage: Stage = Stage.PRE
) -> ClassicalMode:
    """古典モデルの動揺モードを解析式で求める。

    2 次系の特性方程式

    .. math::

        \\lambda^2 + \\frac{D}{2H}\\lambda + \\frac{K_s \\omega_s}{2H} = 0

    から

    .. math::

        \\omega_n = \\sqrt{\\frac{K_s \\omega_s}{2H}}, \\qquad
        \\zeta = \\frac{D}{2\\sqrt{2 H K_s \\omega_s}}

    が得られる。この式と :func:`analyze` の数値固有値が一致することを
    確かめるのが Phase 3 の主眼である。

    Notes
    -----
    古典モデル（2 次）にのみ適用できる。制御器を接続した系や 1 軸
    モデルでは次数が上がるため、この式は使えない。
    """
    # 状態数だけを見ると、状態を持たない制御器（比例ガバナなど）が
    # 付いた系を見逃す。実際、ProportionalGovernor を付けた古典系では
    # 真の固有値の実部が -1.1 なのに、この式は -0.1 を返していた。
    # 数値は返るが誤っている、という最も危険な状態になるため、
    # 発電機の型と制御器の有無を直接確認する。
    from .machines.classical import ClassicalMachine

    if not isinstance(system.machine, ClassicalMachine):
        raise ValueError(
            f"この解析解は古典モデル専用（現在は {type(system.machine).__name__}）。"
            " 高次モデルでは analyze() の数値固有値を参照すること。"
        )
    if getattr(system, "controllers", None):
        names = ", ".join(type(c).__name__ for c in system.controllers)
        raise ValueError(
            f"この解析解は制御器を含まない系専用（接続されている制御器: {names}）。"
            " 制御器は状態を持たなくても動揺方程式の係数を変えるため、"
            " この式では表せない。analyze() の数値固有値を参照すること。"
        )
    if system.n_states != 2:
        raise ValueError(
            f"この解析解は 2 次の系専用（現在の状態数は {system.n_states}）。"
        )

    H = system.machine.H
    D = system.machine.D
    omega_s = system.base.omega_s
    K_s = synchronizing_coefficient(system, stage)

    if K_s <= 0.0:
        raise ValueError(
            f"同期化力係数 K_s = {K_s:.4f} が正でない。位相角が 90° を超えており"
            " 静的に安定な運転点になっていない。"
        )

    omega_n = math.sqrt(K_s * omega_s / (2.0 * H))
    zeta = D / (2.0 * math.sqrt(2.0 * H * K_s * omega_s))

    if zeta < 1.0:
        real = -zeta * omega_n
        imag = omega_n * math.sqrt(1.0 - zeta**2)
        eigenvalues = np.array([complex(real, imag), complex(real, -imag)])
    else:
        root = omega_n * math.sqrt(zeta**2 - 1.0)
        eigenvalues = np.array(
            [complex(-zeta * omega_n + root, 0.0), complex(-zeta * omega_n - root, 0.0)]
        )

    return ClassicalMode(
        K_s=K_s,
        natural_frequency=omega_n,
        damping_ratio=zeta,
        eigenvalues=eigenvalues,
    )


def equilibrium(
    system: SMIBSystem,
    stage: Stage = Stage.PRE,
    x_guess: np.ndarray | None = None,
) -> np.ndarray:
    """dx/dt = 0 を満たす平衡点を数値的に求める（動作点の検算用）。

    :class:`~genstab.system.SMIBSystem` は構築時に定常状態を解析的に
    設定するので通常は不要だが、制御器を追加したときに本当に平衡点に
    なっているかを確かめるのに使える。
    """
    x_guess = system.initial_state() if x_guess is None else np.asarray(x_guess, dtype=float)
    solution, info, status, message = fsolve(
        lambda x: system.derivatives(0.0, x, stage),
        x_guess,
        full_output=True,
    )
    if status != 1:
        raise RuntimeError(f"平衡点の求解に失敗した: {message}")
    return solution


def residual_at_operating_point(
    system: SMIBSystem, stage: Stage = Stage.PRE
) -> float:
    """動作点における dx/dt のノルム。0 に近ければ正しく初期化されている。"""
    return float(
        np.linalg.norm(system.derivatives(0.0, system.initial_state(), stage))
    )
