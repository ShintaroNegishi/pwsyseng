"""動作点まわりの線形化と python-control への橋渡し。

過渡安定性は非線形の動揺方程式をそのまま時間積分して調べるが、
定態安定性（微小擾乱安定性）は動作点まわりで線形化した

.. math::

    \\Delta\\dot{x} = A\\,\\Delta x + B\\,\\Delta u, \\qquad
    \\Delta y = C\\,\\Delta x + D\\,\\Delta u

を調べる。本モジュールは A, B, C, D を数値微分（中心差分）で求め、
``control.StateSpace`` に変換する。これにより固有値解析だけでなく、
ボード線図・根軌跡・極配置・LQR といった制御工学の道具をそのまま
発電機系に適用できる。

数値微分を使うのは、発電機モデルや制御器を追加してもコードを書き
換えずに線形化できるようにするためである。解析的な線形化
（Heffron-Phillips の K1〜K6 など）と結果が一致することは
:mod:`genstab.smallsignal` のテストで確認している。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import numpy as np

from .controllers.base import ControllerKind
from .events import Stage
from .system import SMIBSystem

#: 数値微分の既定刻み幅。状態量のスケール（δ は rad, Δω は p.u.）を
#: 考えると 1e-6 前後が中心差分の打切り誤差と丸め誤差の釣り合う点になる。
DEFAULT_STEP = 1e-6

#: 線形化で入力として選べる量。
AVAILABLE_INPUTS = ("Pm", "Efd", "Vref")

#: 線形化で出力として選べる代数量（状態変数名も指定できる）。
ALGEBRAIC_OUTPUTS = ("Pe", "Vt", "Pm", "Efd")


@contextmanager
def _input_offset(system: SMIBSystem, name: str, value: float):
    """入力を一時的にずらす（数値微分で B, D を作るために使う）。"""
    if name == "Pm":
        original = system._Pm0
        system._Pm0 = original + value
        try:
            yield
        finally:
            system._Pm0 = original
    elif name == "Efd":
        if system.exciter is not None:
            raise ValueError(
                "励磁系が接続されているので界磁電圧 Efd は直接の入力にならない。"
                " 代わりに 'Vref'（AVR の基準電圧）を入力に指定すること。"
            )
        original = system._Efd0
        system._Efd0 = original + value
        try:
            yield
        finally:
            system._Efd0 = original
    elif name == "Vref":
        if system.exciter is None:
            raise ValueError(
                "'Vref' を入力にするには励磁系 (exciter) が接続されている必要がある。"
            )
        if not hasattr(system.exciter, "v_ref"):
            raise ValueError(
                f"{type(system.exciter).__name__} は基準電圧 v_ref を持たないため"
                " 'Vref' を入力にできない。"
            )
        original = system.exciter.v_ref
        system.exciter.v_ref = original + value
        try:
            yield
        finally:
            system.exciter.v_ref = original
    else:
        raise ValueError(
            f"入力 '{name}' は未知。選べるのは {AVAILABLE_INPUTS}。"
        )


def _algebraic_vector(
    system: SMIBSystem, x: np.ndarray, stage: Stage, names: Sequence[str]
) -> np.ndarray:
    """指定した出力量の値を並べたベクトルを返す。"""
    solution, _, Pm, Efd, _aux = system._evaluate(0.0, x, stage)
    lookup = {
        "Pe": solution.Pe,
        "Vt": solution.Vt,
        "Pm": Pm,
        "Efd": Efd,
    }
    values = []
    for name in names:
        if name in lookup:
            values.append(lookup[name])
        elif name in system.state_names:
            values.append(float(x[system.state_names.index(name)]))
        else:
            raise ValueError(
                f"出力 '{name}' は未知。状態 {system.state_names} または"
                f" 代数量 {ALGEBRAIC_OUTPUTS} から選ぶこと。"
            )
    return np.asarray(values, dtype=float)


def state_matrix(
    system: SMIBSystem,
    x0: np.ndarray | None = None,
    stage: Stage = Stage.PRE,
    step: float = DEFAULT_STEP,
) -> np.ndarray:
    """状態行列 A = ∂f/∂x を中心差分で求める。"""
    x0 = system.initial_state() if x0 is None else np.asarray(x0, dtype=float)
    n = x0.size
    A = np.empty((n, n))
    for j in range(n):
        h = step * max(1.0, abs(x0[j]))
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        A[:, j] = (
            system.derivatives(0.0, xp, stage) - system.derivatives(0.0, xm, stage)
        ) / (2.0 * h)
    return A


def input_matrix(
    system: SMIBSystem,
    inputs: Sequence[str],
    x0: np.ndarray | None = None,
    stage: Stage = Stage.PRE,
    step: float = 1e-6,
) -> np.ndarray:
    """入力行列 B = ∂f/∂u を中心差分で求める。"""
    x0 = system.initial_state() if x0 is None else np.asarray(x0, dtype=float)
    B = np.empty((x0.size, len(inputs)))
    for j, name in enumerate(inputs):
        with _input_offset(system, name, step):
            f_plus = system.derivatives(0.0, x0, stage)
        with _input_offset(system, name, -step):
            f_minus = system.derivatives(0.0, x0, stage)
        B[:, j] = (f_plus - f_minus) / (2.0 * step)
    return B


def output_matrices(
    system: SMIBSystem,
    outputs: Sequence[str],
    inputs: Sequence[str],
    x0: np.ndarray | None = None,
    stage: Stage = Stage.PRE,
    step: float = DEFAULT_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """出力行列 C = ∂y/∂x と直達行列 D = ∂y/∂u を中心差分で求める。"""
    x0 = system.initial_state() if x0 is None else np.asarray(x0, dtype=float)
    n_y, n_x = len(outputs), x0.size

    C = np.empty((n_y, n_x))
    for j in range(n_x):
        h = step * max(1.0, abs(x0[j]))
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h
        xm[j] -= h
        C[:, j] = (
            _algebraic_vector(system, xp, stage, outputs)
            - _algebraic_vector(system, xm, stage, outputs)
        ) / (2.0 * h)

    D = np.empty((n_y, len(inputs)))
    for j, name in enumerate(inputs):
        with _input_offset(system, name, 1e-6):
            y_plus = _algebraic_vector(system, x0, stage, outputs)
        with _input_offset(system, name, -1e-6):
            y_minus = _algebraic_vector(system, x0, stage, outputs)
        D[:, j] = (y_plus - y_minus) / (2.0 * 1e-6)

    return C, D


def state_space(
    system: SMIBSystem,
    *,
    inputs: Sequence[str] = ("Pm",),
    outputs: Sequence[str] | None = None,
    x0: np.ndarray | None = None,
    stage: Stage = Stage.PRE,
    step: float = DEFAULT_STEP,
):
    """動作点まわりで線形化し ``control.StateSpace`` を返す。

    Parameters
    ----------
    inputs:
        入力に取る量。``"Pm"``（機械入力）、``"Efd"``（界磁電圧、
        AVR 非接続時のみ）、``"Vref"``（AVR の基準電圧）から選ぶ。
    outputs:
        出力に取る量。状態変数名または ``"Pe"``, ``"Vt"``, ``"Pm"``,
        ``"Efd"``。省略すると全状態量を出力にする。
    stage:
        線形化するネットワーク状態。定態安定性は通常 ``Stage.PRE``。
    x0:
        線形化する動作点。省略すると事故前定常状態。

    Returns
    -------
    control.StateSpace

    Examples
    --------
    >>> import genstab, genstab.linearize as lin
    >>> machine = genstab.ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
    >>> network = genstab.SMIBNetwork(x_pre=0.4, x_fault=float("inf"), x_post=0.4)
    >>> system = genstab.SMIBSystem(machine, network, Pe0=0.8)
    >>> ss = lin.state_space(system, inputs=("Pm",), outputs=("delta", "omega"))
    >>> ss.nstates, ss.ninputs, ss.noutputs
    (2, 1, 2)
    """
    import control as ct

    outputs = tuple(system.state_names) if outputs is None else tuple(outputs)
    inputs = tuple(inputs)

    A = state_matrix(system, x0=x0, stage=stage, step=step)
    B = input_matrix(system, inputs, x0=x0, stage=stage)
    C, D = output_matrices(system, outputs, inputs, x0=x0, stage=stage, step=step)

    ss = ct.ss(A, B, C, D)
    ss.set_states(list(system.state_names))
    ss.set_inputs(list(inputs))
    ss.set_outputs(list(outputs))
    return ss


def describe_controllers(system: SMIBSystem) -> str:
    """線形化に使える入力の一覧を返す（notebook の補助）。"""
    lines = ["線形化に使える入力:", "  Pm  : 機械入力（常に使える）"]
    if system.exciter is None:
        lines.append("  Efd : 界磁電圧（励磁系が未接続なので使える）")
    else:
        lines.append(
            f"  Vref: {type(system.exciter).__name__} の基準電圧"
            "（励磁系が接続されているため Efd の代わりにこちらを使う）"
        )
    stabilizers = [c for c in system.controllers if c.kind is ControllerKind.STABILIZER]
    if stabilizers:
        lines.append(
            f"  （PSS {len(stabilizers)} 台は AVR への補助信号として A 行列に含まれる）"
        )
    return "\n".join(lines)
