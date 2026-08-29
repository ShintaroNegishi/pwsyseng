"""教材用の作図ヘルパ。

軸ラベルはすべて英語で書いている。日本語フォントの有無は OS によって
異なり、学生の環境で豆腐（□）になる事故が起きやすいためである。
図の説明は notebook の markdown セルに日本語で書くこと。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .events import Stage
from .simulate import SimulationResult
from .system import SMIBSystem

#: 教材で共通に使う描画設定。
GENSTAB_RCPARAMS = {
    "figure.figsize": (9.0, 4.5),
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "legend.framealpha": 0.9,
}


def use_genstab_style() -> None:
    """統一スタイルを適用する（notebook の冒頭で 1 回呼ぶ）。"""
    plt.rcParams.update(GENSTAB_RCPARAMS)


def _shade_fault(ax, system: SMIBSystem) -> None:
    """事故期間を網掛けして示す。"""
    fault = system.fault
    if not np.isfinite(fault.t_fault):
        return
    t_left, t_right = ax.get_xlim()
    t_clear = fault.t_clear if np.isfinite(fault.t_clear) else t_right
    ax.axvspan(
        max(fault.t_fault, t_left),
        min(t_clear, t_right),
        color="tab:red",
        alpha=0.12,
        label="fault",
        zorder=0,
    )


def plot_swing(
    result: SimulationResult,
    quantities: Sequence[str] = ("delta_deg", "frequency_hz", "Pe"),
    *,
    title: str | None = None,
    axes=None,
):
    """時間応答を縦に並べて描く。

    Parameters
    ----------
    result:
        シミュレーション結果。
    quantities:
        描く量。``"delta_deg"``, ``"omega"``, ``"frequency_hz"``,
        ``"Pe"``, ``"Vt"``, ``"Pm"``, ``"Efd"`` から選ぶ。
    """
    labels = {
        "delta_deg": "Rotor angle $\\delta$ [deg]",
        "delta": "Rotor angle $\\delta$ [rad]",
        "omega": "Speed deviation $\\Delta\\omega$ [p.u.]",
        "frequency_hz": "Frequency [Hz]",
        "Pe": "Electrical power $P_e$ [p.u.]",
        "Pm": "Mechanical power $P_m$ [p.u.]",
        "Vt": "Terminal voltage $V_t$ [p.u.]",
        "Efd": "Field voltage $E_{fd}$ [p.u.]",
    }

    n = len(quantities)
    if axes is None:
        _, axes = plt.subplots(n, 1, figsize=(9.0, 2.4 * n), sharex=True, squeeze=False)
        axes = axes.ravel()
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, quantities):
        values = getattr(result, name, None)
        if values is None:
            values = result[name]
        ax.plot(result.t, values)
        ax.set_ylabel(labels.get(name, name))
        ax.set_xlim(result.t[0], result.t[-1])
        _shade_fault(ax, result.system)

    axes[-1].set_xlabel("Time [s]")
    if title:
        axes[0].set_title(title)
    plt.tight_layout()
    return axes


def plot_power_angle(
    system: SMIBSystem,
    result: SimulationResult | None = None,
    *,
    ax=None,
    show_stages: Iterable[Stage] = (Stage.PRE, Stage.FAULT, Stage.POST),
    title: str | None = None,
):
    """P-δ 曲線と（あれば）運動の軌跡を描く。

    等面積法の説明図そのものになる。加速面積・減速面積を塗り分けたい
    場合は :func:`genstab.eac.plot_equal_area` を使うこと。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))

    emf = system.machine.internal_emf(system.initial_state())
    delta_grid = np.linspace(0.0, np.pi, 400)
    styles = {
        Stage.PRE: ("Pre-fault", "tab:blue", "-"),
        Stage.FAULT: ("During fault", "tab:red", "--"),
        Stage.POST: ("Post-fault", "tab:green", "-."),
    }
    for stage in show_stages:
        label, color, ls = styles[stage]
        curve = system.network.power_angle_curve(stage, emf, delta_grid)
        ax.plot(np.degrees(delta_grid), curve, color=color, ls=ls, label=label)

    ax.axhline(
        system.operating_point.Pm,
        color="k",
        ls=":",
        lw=1.4,
        label="$P_m$",
    )
    if result is not None:
        ax.plot(
            result.delta_deg,
            result.Pe,
            color="tab:orange",
            lw=1.2,
            alpha=0.9,
            label="trajectory",
        )
    ax.plot(
        np.degrees(system.operating_point.delta),
        system.operating_point.Pe,
        "ko",
        ms=6,
        label="operating point",
    )

    ax.set_xlabel("Rotor angle $\\delta$ [deg]")
    ax.set_ylabel("Power [p.u.]")
    ax.set_xlim(0.0, 180.0)
    ax.legend(loc="best", fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_phase_portrait(
    results: SimulationResult | Sequence[SimulationResult],
    labels: Sequence[str] | None = None,
    *,
    ax=None,
    title: str | None = None,
):
    """位相面 (δ, Δω) の軌跡を描く。安定・不安定の違いが一目で分かる。"""
    if isinstance(results, SimulationResult):
        results = [results]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.0))

    for i, result in enumerate(results):
        label = labels[i] if labels else None
        ax.plot(result.delta_deg, result.omega, label=label)
        ax.plot(result.delta_deg[0], result.omega[0], "o", ms=5, color="k")

    ax.set_xlabel("Rotor angle $\\delta$ [deg]")
    ax.set_ylabel("Speed deviation $\\Delta\\omega$ [p.u.]")
    ax.axhline(0.0, color="gray", lw=0.8)
    if labels:
        ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax


def compare_results(
    results: Sequence[SimulationResult],
    labels: Sequence[str],
    quantities: Sequence[str] = ("delta_deg", "frequency_hz"),
    *,
    title: str | None = None,
):
    """複数の結果を重ね描きする（制御あり・なしの比較用）。"""
    axes = None
    for result, label in zip(results, labels):
        axes = plot_swing(result, quantities, axes=axes)
        for ax in np.atleast_1d(axes):
            ax.lines[-1].set_label(label)
    for ax in np.atleast_1d(axes):
        handles, lbls = ax.get_legend_handles_labels()
        keep = [(h, l) for h, l in zip(handles, lbls) if l in labels]
        if keep:
            ax.legend(*zip(*keep), fontsize=9, loc="best")
    if title:
        np.atleast_1d(axes)[0].set_title(title)
    plt.tight_layout()
    return axes


def plot_eigenvalues(
    eigenvalues,
    labels: Sequence[str] | None = None,
    *,
    ax=None,
    title: str | None = None,
):
    """固有値を複素平面に描く（定態安定性の判定図）。

    虚軸より左にすべての固有値があれば漸近安定である。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 5.0))

    groups = eigenvalues if isinstance(eigenvalues, (list, tuple)) else [eigenvalues]
    if np.ndim(groups[0]) == 0:
        groups = [np.asarray(eigenvalues)]

    markers = ["x", "o", "s", "^", "v", "D"]
    for i, group in enumerate(groups):
        group = np.asarray(group)
        label = labels[i] if labels else None
        ax.plot(
            group.real, group.imag,
            markers[i % len(markers)], ms=9, mew=2,
            ls="none", label=label,
        )

    ax.axvline(0.0, color="k", lw=1.0)
    ax.axhline(0.0, color="gray", lw=0.8)
    ax.set_xlabel("Real part [1/s]")
    ax.set_ylabel("Imaginary part [rad/s]")
    if labels:
        ax.legend(fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax
