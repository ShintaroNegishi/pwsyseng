"""教材用の作図ヘルパ（第 00 回〜第 11 回）。

**このモジュールの文字列リテラルは、docstring とコメントを除いてすべて
ASCII で書く。** 軸ラベル・凡例・タイトル・注記に日本語を書くと、日本語
フォントの入っていない環境（学生の PC、研究室の Linux サーバ、CI）で
豆腐（□）になり、その場で授業が止まる。フォントの有無に依存しない図に
しておけば、誰がどこで実行しても同じものが出る。

規約を「軸ラベルは英語で書く」ではなく **「非 ASCII の文字列リテラルを
ゼロにする」** と機械的に決めたのは、目視の規律は必ず漏れるからである。
``tests/test_plotting.py`` が :mod:`ast` で文字列リテラルだけを走査して
これを検査する（docstring とコメントは除外して判定する）。

その代償として、**このモジュールに限り例外メッセージも英語**である。
パッケージの他のモジュールは「例外メッセージは日本語」という規約に従って
いるので、ここだけが例外であることを承知しておくこと。図についての説明・
考察・注意書きは notebook の markdown セルに日本語で書く。

作法は ``genstab.plotting`` に合わせてある。

- すべての関数が ``ax`` を受け取り ``ax`` を返す。``ax=None`` なら新しい
  Figure と Axes を作る。返された ``ax`` にさらに描き足せるので、図の
  組み立てを notebook 側で続けられる。
- 色は matplotlib の既定の循環（``tab:*``）から選ぶ。系統の量については
  「健全・警戒・逸脱」の 3 色を :data:`SEVERITY_COLORS` に固定してあり、
  どの図でも同じ色が同じ意味を持つ。
- ``use_gridops_style()`` を notebook の冒頭で 1 回呼ぶ。
- カラーバーと twin 軸は使わない。図に Axes が 2 つあると、片方だけ軸
  ラベルが無い図が混ざる。凡例（``legend``）で代用する。

このモジュールは :mod:`gridops.case` 以外の gridops のモジュールを
**関数の中でだけ** import する。作図のためだけに PuLP を読み込ませない
ためと、将来 :mod:`gridops.voltage` や :mod:`gridops.security` から
作図を呼びたくなったときに循環 import にしないためである。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from .case import BusType, Case

if TYPE_CHECKING:   # 型注釈のためだけの import（実行時には読み込まない）。
    from .adequacy import CapacityOutageTable, MonteCarloResult
    from .commitment import CommitmentResult
    from .dc import DCSolution
    from .dispatch import DCOPFResult
    from .powerflow import PowerFlowSolution
    from .security import SecurityReport
    from .voltage import PVCurve

__all__ = [
    "GRIDOPS_RCPARAMS",
    "SEVERITY_COLORS",
    "WARN_LOADING",
    "BUS_TYPE_COLORS",
    "COURSE_THEMES",
    "TIME_MARKS",
    "COURSE_QUESTION",
    "use_gridops_style",
    "timescale_map",
    "plot_voltage_profile",
    "plot_convergence",
    "plot_pv_curve",
    "plot_network_flows",
    "plot_merit_order",
    "plot_lambda_search",
    "plot_lmp",
    "plot_commitment",
    "plot_commitment_schedule",
    "plot_duck_curve",
    "plot_capacity_outage_table",
    "plot_lolp_convergence",
    "plot_contingency_ranking",
]


# ======================================================================
# スタイルと色
# ======================================================================

#: 教材で共通に使う描画設定（genstab と同じ値にしてある）。
GRIDOPS_RCPARAMS = {
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

#: 熱容量・電圧の逸脱を表す 3 色。**どの図でも同じ色が同じ意味**を持つ。
SEVERITY_COLORS = {
    "ok": "tab:blue",
    "warn": "tab:orange",
    "violation": "tab:red",
}

#: 「警戒」と塗る負荷率のしきい値。定格の 9 割で色が変わる。
WARN_LOADING = 0.90

#: 母線種別の色。潮流計算で「何を与えて何を求めるか」の違いを図でも保つ。
BUS_TYPE_COLORS = {
    BusType.SLACK: "tab:purple",
    BusType.PV: "tab:green",
    BusType.PQ: "tab:blue",
}

#: :func:`timescale_map` に描く 5 テーマ。
#: ``(テーマ名, その回で問うこと, 対応する notebook, 開始 [s], 終了 [s])``
#:
#: 時間の範囲は ``docs/course_map.md`` の表（ミリ秒〜秒 / 秒〜分 / 分〜時 /
#: 時〜日 / 年）をそのまま秒に直したものである。**帯が隣と接している**のは
#: 時間スケールが連続だからで、テーマの境目に物理的な壁があるわけではない。
COURSE_THEMES = (
    (
        "Transient stability",
        "stay in step\nthrough a fault?",
        "genstab 01-03, 08",
        1.0e-3,
        1.0e1,
    ),
    (
        "Steady-state stability,\nfrequency control",
        "hold the operating point?\nrestore frequency?",
        "genstab 04-07",
        1.0e0,
        1.0e2,
    ),
    (
        "Power flow, security",
        "flows within limits,\neven after N-1?",
        "gridops 01-04, 09",
        6.0e1,
        3.6e3,
    ),
    (
        "Economic dispatch,\nunit commitment",
        "which units,\nat what output?",
        "gridops 05-08",
        3.6e3,
        8.64e4,
    ),
    (
        "Adequacy",
        "enough capacity\nat all?",
        "gridops 10",
        8.64e4,
        3.1536e8,
    ),
)

#: 時間軸に立てる目盛の目印 ``(秒, 表示名)``。
TIME_MARKS = (
    (1.0 / 60.0, "1 cycle"),
    (1.0, "1 s"),
    (60.0, "1 min"),
    (3600.0, "1 h"),
    (86400.0, "1 day"),
    (3.1536e7, "1 year"),
)

#: :func:`timescale_map` が科目全体に掲げる 1 つの問い。
COURSE_QUESTION = "Can supply and demand stay balanced -- at every time scale?"


def use_gridops_style() -> None:
    """統一スタイルを適用する（notebook の冒頭で 1 回呼ぶ）。

    ``matplotlib.rcParams`` を書き換えるだけなので、呼ばなくても図は出る。
    フォントの種類は **指定しない**。日本語フォントを指定すると環境に
    よっては見つからず警告が出るし、そもそもこのモジュールは非 ASCII の
    文字を一切描かないので指定する必要がない。
    """
    plt.rcParams.update(GRIDOPS_RCPARAMS)


# ======================================================================
# 内部ヘルパ（下流から使わないこと）
# ======================================================================

def _prepare(ax: Axes | None, figsize: tuple[float, float]) -> tuple[Axes, bool]:
    """``ax`` が ``None`` なら新しい Figure と Axes を作る。

    第 2 要素は「この関数が Figure を作ったか」で、``tight_layout`` を
    呼んでよいかの判定に使う。人から渡された Axes の載る Figure には
    手を出さない（subplot の一部として渡されることがあるため）。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        return ax, True
    return ax, False


def _finish(ax: Axes, created: bool) -> Axes:
    """自分で作った Figure だけ体裁を整えてから ``ax`` を返す。"""
    if created:
        ax.figure.tight_layout()
    return ax


def _palette(n: int) -> list:
    """区別しやすい色を ``n`` 個返す（10 を超えると循環する）。"""
    cmap = mpl.colormaps["tab10"]
    return [cmap(i % 10) for i in range(n)]


def _severity_color(loading: float) -> str:
    """負荷率から「健全・警戒・逸脱」の色を選ぶ。"""
    if not math.isfinite(loading):
        return SEVERITY_COLORS["violation"]
    if loading > 1.0:
        return SEVERITY_COLORS["violation"]
    if loading >= WARN_LOADING:
        return SEVERITY_COLORS["warn"]
    return SEVERITY_COLORS["ok"]


def _severity_legend() -> list[Patch]:
    """3 色の意味を示す凡例の代理ハンドル。"""
    return [
        Patch(color=SEVERITY_COLORS["ok"], label="below 90%"),
        Patch(color=SEVERITY_COLORS["warn"], label="90% to 100%"),
        Patch(color=SEVERITY_COLORS["violation"], label="above 100%"),
    ]


def _labelled(
    solutions: Mapping[str, object] | Iterable[object] | object,
) -> list[tuple[str, object]]:
    """``{名前: 解}`` / 解の列 / 解 1 つ、のどれで渡されても対にして返す。"""
    if isinstance(solutions, Mapping):
        return [(str(key), value) for key, value in solutions.items()]
    if hasattr(solutions, "mismatch_history"):
        solutions = [solutions]
    pairs = []
    for i, solution in enumerate(solutions):
        name = str(getattr(solution, "method", "") or f"solution {i + 1}")
        pairs.append((name, solution))
    return pairs


def _plant_colors(case: Case) -> dict[str, object]:
    """発電所名から色への対応。号機ではなく **発電所**で色を分ける。"""
    plants: list[str] = []
    for unit in case.units:
        name = unit.plant or unit.name
        if name not in plants:
            plants.append(name)
    palette = _palette(len(plants))
    return {name: palette[i] for i, name in enumerate(plants)}


def _unit_colors(case: Case) -> dict[str, tuple]:
    """号機名から色への対応。

    発電所で色相を変え、同じ発電所の号機は明度で分ける。積み上げ図で
    「どの発電所か」と「何号機か」を同時に読めるようにするためである。
    発電所ごとに 1 色だけにすると、同じ発電所の号機の境目が消える。
    """
    from matplotlib.colors import to_rgb

    plants = _plant_colors(case)
    grouped: dict[str, list[str]] = {}
    for unit in case.units:
        grouped.setdefault(unit.plant or unit.name, []).append(unit.name)
    colors: dict[str, tuple] = {}
    for plant, names in grouped.items():
        base = np.array(to_rgb(plants[plant]))
        for i, name in enumerate(names):
            weight = 0.0 if len(names) == 1 else 0.42 * i / (len(names) - 1)
            colors[name] = tuple(base + (1.0 - base) * weight)
    return colors


def _branch_values(
    solution: PowerFlowSolution | DCSolution, limit: str
) -> tuple[np.ndarray, np.ndarray]:
    """枝ごとの ``(負荷率, 潮流の大きさ [p.u.])`` を枝の並び順で返す。

    交流の :class:`~gridops.powerflow.PowerFlowSolution`（辞書を返す）と
    直流の :class:`~gridops.dc.DCSolution`（配列を返す）の両方を受ける。
    """
    case = solution.case
    loading = solution.loading(limit)
    if isinstance(loading, Mapping):
        values = np.array([float(loading[b.key()]) for b in case.branches])
    else:
        values = np.asarray(loading, dtype=float)

    apparent = getattr(solution, "apparent_flows", None)
    if callable(apparent):
        flows = apparent()
        magnitude = np.array([float(flows[b.key()]) for b in case.branches])
    else:
        magnitude = np.abs(np.asarray(solution.flows, dtype=float))
    return values, magnitude


def _bus_column(curve: PVCurve, bus_id: int) -> int:
    """P-V 曲線の ``voltages`` から母線 ``bus_id`` の列番号を得る。

    :class:`~gridops.voltage.PVCurve` は契約上ケースを持っていないので、
    ``case`` / ``bus_ids`` のどちらかが付いていればそれを使い、無ければ
    列が 1 本のときだけ推測する。**黙って別の母線を描くことはしない。**
    """
    case = getattr(curve, "case", None)
    if case is not None:
        return int(case.index_of(bus_id))
    bus_ids = getattr(curve, "bus_ids", None)
    if bus_ids is not None:
        return int(list(bus_ids).index(bus_id))
    voltages = np.atleast_2d(np.asarray(curve.voltages, dtype=float))
    if voltages.shape[1] == 1:
        return 0
    raise ValueError(
        "cannot map bus id to a column of PVCurve.voltages: "
        "attach 'case' or 'bus_ids' to the curve object"
    )


def _hour_axis(ax: Axes, horizon: int) -> None:
    """24 時間前後の系列に読みやすい時刻目盛を付ける。"""
    step = 1 if horizon <= 12 else (2 if horizon <= 26 else max(1, horizon // 12))
    ticks = np.arange(0, horizon, step)
    ax.set_xticks(ticks, [str(int(t)) for t in ticks])
    ax.set_xlim(-0.5, horizon - 0.5)


# ======================================================================
# 第 00 回 — 科目の地図
# ======================================================================

def timescale_map(ax: Axes | None = None) -> Axes:
    """5 テーマの時間スケールを 1 枚に並べた地図（第 00 回）。

    横軸はミリ秒から年までの **対数**時間である。過渡安定度・定態安定度と
    周波数制御・潮流とセキュリティ・経済負荷配分と起動停止計画・アデカシー
    の 5 つが、それぞれどの時間スケールの話なのかを帯で示す。

    この図の要点は、5 つが別々の科目ではなく **1 つの問いを違う時間
    スケールで見たもの**だということである。すなわち「需要と供給を、どの
    時間スケールでも釣り合わせ続けられるか」。ミリ秒では同期を保てるか、
    秒〜分では周波数を戻せるか、分〜時では制約を守れるか、時〜日ではどの
    発電機を動かすか、年では設備が足りるか。問いは同じで、釣り合わせる
    手段と時定数だけが違う。

    各テーマの帯は隣と接している。時間スケールは連続であって、テーマの
    境目に物理的な壁があるわけではないからである（周波数制御と潮流計算は
    「分」のあたりで重なる）。

    Parameters
    ----------
    ax:
        描画先。``None`` なら新しい Figure と Axes を作る。

    Returns
    -------
    matplotlib.axes.Axes
        描画した Axes。

    Notes
    -----
    数値は ``docs/course_map.md`` の表と同じである。表を変えたら
    :data:`COURSE_THEMES` も変えること（両方に同じ数字が書いてある）。
    """
    ax, created = _prepare(ax, (11.0, 5.2))
    n = len(COURSE_THEMES)
    colors = _palette(n)

    labels: list[str] = []
    for i, (name, question, lessons, t_start, t_end) in enumerate(COURSE_THEMES):
        y = n - 1 - i
        ax.barh(
            y,
            width=t_end - t_start,
            left=t_start,
            height=0.66,
            color=colors[i],
            alpha=0.75,
            edgecolor="k",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            math.sqrt(t_start * t_end),
            y,
            question,
            ha="center",
            va="center",
            fontsize=8.5,
            color="k",
            zorder=4,
        )
        labels.append(f"{name}\n[{lessons}]")

    # 時間の目印。1 サイクル (60 Hz) から 1 年まで。
    for seconds, name in TIME_MARKS:
        ax.axvline(seconds, color="0.45", ls=":", lw=1.0, zorder=1)
        ax.text(
            seconds,
            n - 0.42,
            name,
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.3",
        )

    # 科目全体を貫く 1 つの問い。図の下に矢印で通す。
    left, right = 4.0e-4, 7.0e8
    ax.annotate(
        "",
        xy=(right, -0.78),
        xytext=(left, -0.78),
        arrowprops={"arrowstyle": "-|>", "color": "0.35", "lw": 1.4},
        annotation_clip=False,
    )
    ax.text(
        math.sqrt(left * right),
        -0.50,
        COURSE_QUESTION,
        ha="center",
        va="bottom",
        fontsize=10,
        color="0.25",
    )

    ax.set_xscale("log")
    ax.set_xlim(left, right)
    ax.set_ylim(-1.05, n - 0.1)
    ax.set_yticks(np.arange(n), labels[::-1])
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("Time scale [s]")
    ax.set_ylabel("Course theme")
    ax.set_title("One question, five time scales")
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    return _finish(ax, created)


# ======================================================================
# 第 02 回・第 03 回 — 潮流計算
# ======================================================================

def plot_voltage_profile(
    solution: PowerFlowSolution, ax: Axes | None = None, *, limits: bool = True
) -> Axes:
    """母線ごとの電圧の大きさを並べる（電圧プロファイル）。

    棒ではなく点と線で描くのは、縦軸を 1.0 付近に拡大すると棒の根元が
    切れて「ゼロから伸びていない棒」になり、面積を読み違えるからである。

    色は母線種別（slack / PV / PQ）で分ける。PV と slack の電圧が設定値
    ちょうどに並び、PQ 母線だけが解として動く、という潮流計算の構造が
    そのまま見える。運用上の上下限（``Bus.v_min`` / ``Bus.v_max``）を
    外れた母線には赤い縁取りを付ける。

    Parameters
    ----------
    solution:
        :class:`~gridops.powerflow.PowerFlowSolution`。``case`` と ``v``
        を持つオブジェクトであれば何でもよい。
    ax:
        描画先。
    limits:
        ``True`` なら電圧の上下限を帯で示す。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (9.0, 4.2))
    case = solution.case
    v = np.asarray(solution.v, dtype=float)
    x = np.arange(case.n_bus)

    if limits:
        v_min = np.array([bus.v_min for bus in case.buses], dtype=float)
        v_max = np.array([bus.v_max for bus in case.buses], dtype=float)
        ax.fill_between(
            x, v_min, v_max, step="mid", color="tab:green", alpha=0.10, zorder=0
        )
        ax.step(x, v_min, where="mid", color="tab:green", ls="--", lw=1.2, zorder=1)
        ax.step(x, v_max, where="mid", color="tab:green", ls="--", lw=1.2, zorder=1)

    ax.plot(x, v, "-", color="0.6", lw=1.2, zorder=2)
    for i, bus in enumerate(case.buses):
        outside = limits and (v[i] < bus.v_min - 1e-9 or v[i] > bus.v_max + 1e-9)
        ax.plot(
            x[i],
            v[i],
            marker="o",
            ms=9,
            color=BUS_TYPE_COLORS.get(bus.type, "tab:blue"),
            mec=SEVERITY_COLORS["violation"] if outside else "k",
            mew=2.2 if outside else 0.7,
            ls="none",
            zorder=3,
        )

    handles = [
        Patch(color=BUS_TYPE_COLORS[kind], label=kind.value)
        for kind in (BusType.SLACK, BusType.PV, BusType.PQ)
    ]
    if limits:
        handles.append(Patch(color="tab:green", alpha=0.3, label="voltage limits"))
    ax.legend(handles=handles, loc="best", fontsize=9, ncol=2)

    ax.set_xticks(x, [str(bus.id) for bus in case.buses])
    ax.set_xlabel("Bus")
    ax.set_ylabel("Voltage magnitude $|V|$ [p.u.]")
    lower = min(0.93, float(v.min()) - 0.02)
    upper = max(1.07, float(v.max()) + 0.02)
    ax.set_ylim(lower, upper)
    ax.set_title(f"Voltage profile ({getattr(solution, 'method', 'power flow')})")
    return _finish(ax, created)


def plot_convergence(
    solutions: Mapping[str, PowerFlowSolution] | Iterable[PowerFlowSolution]
    | PowerFlowSolution,
    ax: Axes | None = None,
) -> Axes:
    """反復ごとのミスマッチを対数軸で描く（第 02 回の主教材）。

    Newton は下に折れ曲がり（二次収束: 誤差の桁が倍々に増える）、
    Gauss-Seidel はほぼ直線になる（一次収束: 誤差が一定の比で減る）。
    Fast Decoupled はその中間である。**「Gauss-Seidel が遅いのは実装が
    悪いからではなく収束次数の帰結である」**ことが、この 1 枚で見える。

    横軸は反復回数だが、``mismatch_history[0]`` は **初期値での**値なので
    点の数は ``iterations + 1`` になる。

    Parameters
    ----------
    solutions:
        解 1 つ、解の列、または ``{凡例名: 解}`` の辞書。名前を省くと
        ``PowerFlowSolution.method`` を凡例に使う。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (8.0, 4.8))
    markers = ["o", "s", "^", "D", "v", "*"]
    for i, (name, solution) in enumerate(_labelled(solutions)):
        history = np.asarray(list(solution.mismatch_history), dtype=float)
        # 対数軸に載らない値（厳密なゼロ）は描かずに飛ばす。
        history = np.where(history > 0.0, history, np.nan)
        iterations = int(getattr(solution, "iterations", history.size - 1))
        ax.plot(
            np.arange(history.size),
            history,
            marker=markers[i % len(markers)],
            ms=5,
            label=f"{name} ({iterations} iterations)",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max mismatch $|\\Delta S|_\\infty$ [p.u.]")
    ax.set_title("Convergence of the power flow iterations")
    ax.legend(loc="best", fontsize=9)
    return _finish(ax, created)


def plot_pv_curve(curve: PVCurve, bus_id: int, ax: Axes | None = None) -> Axes:
    """P-V 曲線（ノーズカーブ）を描く（第 03 回）。

    横軸は負荷倍率、縦軸はその母線の電圧である。上半分（運転側）を
    たどって右端で折り返す形になり、折り返し点が **ノーズ点**、そこまでの
    余裕が負荷余裕（``loading_margin``）である。

    ノーズ点より右で Newton が収束しないのは、ソルバが弱いからではなく
    **その倍率では潮流方程式に解が無い**からである。収束しなかった倍率の
    範囲は網掛けにして、「解が無い領域」であることを図でも区別する。

    Parameters
    ----------
    curve:
        :class:`~gridops.voltage.PVCurve`。``factors`` / ``voltages`` /
        ``converged`` を持つオブジェクトなら何でもよい。母線番号から列を
        引くために ``case`` か ``bus_ids`` が付いていること。
    bus_id:
        描く母線の番号。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (7.0, 5.0))
    column = _bus_column(curve, bus_id)
    factors = np.asarray(curve.factors, dtype=float)
    voltages = np.atleast_2d(np.asarray(curve.voltages, dtype=float))[:, column]
    converged = np.asarray(
        getattr(curve, "converged", np.ones(factors.size, dtype=bool))
    ).astype(bool)

    ax.plot(
        factors[converged],
        voltages[converged],
        "o-",
        ms=4,
        color=SEVERITY_COLORS["ok"],
        label=f"bus {bus_id}",
    )

    if (~converged).any():
        ax.axvspan(
            float(factors[~converged].min()),
            float(factors.max()),
            color=SEVERITY_COLORS["violation"],
            alpha=0.10,
            zorder=0,
            label="no solution",
        )

    nose = getattr(curve, "nose", None)
    if callable(nose):
        nose_factor, nose_voltage = nose(bus_id)
        ax.plot(
            [nose_factor], [nose_voltage], marker="*", ms=16, color="k", ls="none"
        )
        margin = getattr(curve, "loading_margin", nose_factor - 1.0)
        ax.annotate(
            f"nose: factor {nose_factor:.3f}, $|V|$ {nose_voltage:.3f}\n"
            f"loading margin {100.0 * margin:.1f}%",
            xy=(nose_factor, nose_voltage),
            xytext=(-10, 28),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.0},
        )

    ax.axvline(1.0, color="0.4", ls=":", lw=1.2)
    ax.set_xlabel("Load scaling factor [-]")
    ax.set_ylabel("Voltage magnitude $|V|$ [p.u.]")
    ax.set_title(f"P-V curve at bus {bus_id}")
    ax.legend(loc="best", fontsize=9)
    return _finish(ax, created)


def plot_network_flows(
    solution: PowerFlowSolution | DCSolution,
    ax: Axes | None = None,
    *,
    limit: str = "rate_a",
) -> Axes:
    """枝ごとの潮流と熱容量の余裕を横棒で並べる。

    単線結線図を描かないのは、ケースファイルが母線の座標を持っていない
    からである。座標を作図側に埋め込むと wscc9 専用の図になり、学生が
    自分でケースを作ったとたんに描けなくなる。**どのケースでも同じ
    コードで描けること**を優先し、枝を並べた棒グラフにしてある。

    棒の長さは負荷率（1.0 で定格）で、色は :data:`SEVERITY_COLORS` の
    規約に従う。棒の右に潮流の大きさ [p.u.] を添える。交流の解では
    皮相電力 :math:`|S|`、直流の解では有効電力 :math:`P` である。
    **直流の P を熱容量と比べてはいけない**（wscc9 の枝 4-5 では交流の
    :math:`|S|` が直流の :math:`P` より 47.6% 大きい）。

    Parameters
    ----------
    solution:
        :class:`~gridops.powerflow.PowerFlowSolution` または
        :class:`~gridops.dc.DCSolution`。
    ax:
        描画先。
    limit:
        ``"rate_a"``（常時）または ``"rate_b"``（緊急時）。N-1 の事故後は
        ``"rate_b"`` で見るのが規約である。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (8.0, 5.0))
    case = solution.case
    values, magnitude = _branch_values(solution, limit)
    n = len(case.branches)
    y = np.arange(n)[::-1]

    ax.barh(
        y,
        values,
        height=0.66,
        color=[_severity_color(value) for value in values],
        edgecolor="k",
        linewidth=0.6,
    )
    for position, value, flow in zip(y, values, magnitude):
        ax.text(
            value + 0.02,
            position,
            f"{flow:.3f} p.u.",
            va="center",
            ha="left",
            fontsize=8,
            color="0.25",
        )

    ax.axvline(1.0, color="k", ls="--", lw=1.4)
    ax.set_yticks(y, [branch.label for branch in case.branches])
    ax.set_xlim(0.0, max(1.35, float(np.nanmax(values)) * 1.30))
    ax.set_xlabel(f"Loading (flow / {limit}) [-]")
    ax.set_ylabel("Branch")
    ax.set_title(f"Branch loading against {limit}")
    ax.legend(handles=_severity_legend(), loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    return _finish(ax, created)


# ======================================================================
# 第 05 回・第 06 回 — 経済負荷配分とノード価格
# ======================================================================

def plot_merit_order(
    case: Case, ax: Axes | None = None, *, demand_mw: float | None = None
) -> Axes:
    """メリットオーダー（全負荷平均費用の安い順の階段）を描く（第 05 回）。

    横軸は累積容量、縦軸は全負荷平均費用 :math:`C(P^{max})/P^{max}` である。
    ``demand_mw`` を与えると需要の位置に縦線を引く。線が横切る号機が
    「最後に呼ばれる号機」であり、優先順位法の順位づけはこの並びである。

    **この階段は限界費用の順ではない。** 全負荷平均費用は無負荷費を
    出力で割った量を含むので、順位が増分費用の順と一致するとは限らない。
    等 λ 法（:func:`plot_lambda_search`）が決める限界的な号機と、この
    階段が示す順位が食い違いうることを確かめさせるとよい。

    Parameters
    ----------
    case:
        ケース。``units`` 層が要る。
    ax:
        描画先。
    demand_mw:
        需要 [MW]。``None`` なら線を引かない。

    Returns
    -------
    matplotlib.axes.Axes
    """
    from .dispatch import merit_order  # 循環 import と pulp 依存を関数内に閉じる

    ax, created = _prepare(ax, (9.0, 4.6))
    units = merit_order(case)
    colors = _plant_colors(case)

    left = 0.0
    for unit in units:
        width = float(unit.p_max_mw)
        cost = float(unit.full_load_average_cost())
        ax.bar(
            left + width / 2.0,
            cost,
            width=width * 0.98,
            color=colors[unit.plant or unit.name],
            edgecolor="k",
            linewidth=0.7,
        )
        ax.text(
            left + width / 2.0,
            cost * 1.01,
            unit.name,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            color="0.2",
        )
        left += width

    if demand_mw is not None:
        ax.axvline(
            float(demand_mw),
            color="k",
            ls="--",
            lw=1.6,
            label=f"demand = {float(demand_mw):.1f} MW",
        )

    handles = [
        Patch(color=color, label=plant) for plant, color in colors.items()
    ]
    if demand_mw is not None:
        handles.append(
            plt.Line2D([], [], color="k", ls="--", label="demand")
        )
    ax.legend(handles=handles, loc="upper left", fontsize=9, ncol=2)

    ax.set_xlim(0.0, left)
    ax.set_ylim(0.0, None)
    ax.set_xlabel("Cumulative capacity [MW]")
    ax.set_ylabel("Full-load average cost [JPY/MWh]")
    ax.set_title(f"Merit order ({left:.0f} MW installed)")
    return _finish(ax, created)


def plot_lambda_search(
    case: Case, demand_mw: float, ax: Axes | None = None
) -> Axes:
    """等 λ 法の探索を描く（第 05 回）。

    横軸を :math:`\\lambda` [円/MWh]、縦軸を各号機の出力の合計
    :math:`\\sum_i P_i(\\lambda)` [MW] にする。ここで

    .. math::

        P_i(\\lambda) = \\mathrm{clip}\\!\\left(
            \\frac{\\lambda - b_i}{2 c_i},\\; P_i^{min},\\; P_i^{max}\\right)

    である。この曲線が **単調非減少**であることが、二分法が必ず効く根拠で
    ある。需要の水平線と交わる :math:`\\lambda` が答えで、そこを
    :func:`gridops.dispatch.economic_dispatch` の結果と突き合わせている。

    2 次係数 :math:`c_i = 0` の号機があると曲線は段差を持ち、段差の高さの
    範囲にある需要に対して :math:`\\lambda` は一意に決まらない（縦の
    区間のどこでも需給が合う）。これは第 06 回の「線形計画の縮退」の
    伏線である。

    需要が :math:`[\\sum P^{min}, \\sum P^{max}]` の外にあるときは
    :func:`~gridops.dispatch.economic_dispatch` が例外になるが、この関数は
    図を描いたうえでその旨を注記する。**需要の線が階段の届かない高さに
    あることが目で見える**ほうが、例外の文面より分かりやすいからである。

    Parameters
    ----------
    case:
        ケース。``units`` 層が要る。
    demand_mw:
        需要 [MW]。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    from .dispatch import economic_dispatch

    ax, created = _prepare(ax, (8.5, 4.8))
    units = [unit for unit in case.units if unit.p_max_mw > 0.0]
    lam_lo = min(unit.var_cost for unit in units)
    lam_hi = max(
        unit.var_cost + 2.0 * unit.quadratic * unit.p_max_mw for unit in units
    )
    span = max(lam_hi - lam_lo, max(abs(lam_hi), 1.0) * 0.1)
    grid = np.linspace(lam_lo - 0.08 * span, lam_hi + 0.08 * span, 1201)

    total = np.zeros_like(grid)
    for unit in units:
        if unit.quadratic > 0.0:
            output = (grid - unit.var_cost) / (2.0 * unit.quadratic)
        else:
            # 線形費用の号機は lambda = b_i で下限から上限へ跳ぶ。
            output = np.where(grid < unit.var_cost, unit.p_min_mw, unit.p_max_mw)
        total += np.clip(output, unit.p_min_mw, unit.p_max_mw)

    ax.plot(grid, total, color=SEVERITY_COLORS["ok"], label="$\\sum_i P_i(\\lambda)$")
    ax.axhline(
        float(demand_mw),
        color="k",
        ls="--",
        lw=1.5,
        label=f"demand = {float(demand_mw):.1f} MW",
    )

    try:
        result = economic_dispatch(case, float(demand_mw))
    except ValueError:
        # 需要が可動範囲の外。曲線と需要線の位置関係がそのまま説明になる。
        ax.text(
            0.5,
            0.06,
            "demand is outside [sum Pmin, sum Pmax]: no lambda solves it",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color=SEVERITY_COLORS["violation"],
        )
    else:
        ax.axvline(result.lam, color=SEVERITY_COLORS["warn"], ls=":", lw=1.6)
        ax.plot([result.lam], [float(demand_mw)], "o", ms=9, color="k")
        marginal = ", ".join(result.marginal_units) or "none (all at a bound)"
        ax.annotate(
            f"$\\lambda$ = {result.lam:,.1f} JPY/MWh\nmarginal: {marginal}",
            xy=(result.lam, float(demand_mw)),
            xytext=(12, -34),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.0},
        )

    ax.set_xlabel("$\\lambda$ [JPY/MWh]")
    ax.set_ylabel("Total output $\\sum_i P_i(\\lambda)$ [MW]")
    ax.set_title("Equal incremental cost: the bisection target")
    ax.legend(loc="upper left", fontsize=9)
    return _finish(ax, created)


def plot_lmp(result: DCOPFResult, ax: Axes | None = None) -> Axes:
    """母線ごとのノード価格（LMP）を並べる（第 06 回）。

    混雑が無ければ全母線が同じ高さになり、その値は等 λ 法の
    :math:`\\lambda` に一致する。**1 本でも線路制約が拘束した瞬間に価格が
    母線ごとに割れる**。価格差の原因は費用ではなく送電制約であることが、
    熱容量を無限大にした複製との比較で確かめられる。

    拘束した枝とその混雑料金は図の中に注記する。混雑時の LMP は最も高い
    号機の限界費用を **上回ることがある**（再給電に安い機の減出力が要る
    ため）。これはバグではない。

    Parameters
    ----------
    result:
        :class:`~gridops.dispatch.DCOPFResult`。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (8.5, 4.6))
    buses = sorted(result.lmp)
    values = np.array([float(result.lmp[bus]) for bus in buses])
    x = np.arange(len(buses))

    ax.bar(x, values, color=SEVERITY_COLORS["ok"], edgecolor="k", linewidth=0.7)
    reference = float(values.min())
    ax.axhline(
        reference,
        color="0.35",
        ls="--",
        lw=1.3,
        label=f"lowest LMP = {reference:,.0f} JPY/MWh",
    )

    congested = {
        key: price
        for key, price in getattr(result, "congestion_price", {}).items()
        if price > 0.0
    }
    if congested:
        lines = [
            f"{key[0]}-{key[1]}: {price:,.0f} JPY/MWh" for key, price in congested.items()
        ]
        ax.text(
            0.02,
            0.97,
            "congested branches\n" + "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"boxstyle": "round", "fc": "w", "ec": "0.6", "alpha": 0.9},
        )
        spread = float(values.max() - values.min())
        ax.set_title(f"Locational marginal prices (spread {spread:,.0f} JPY/MWh)")
    else:
        ax.set_title("Locational marginal prices (no congestion)")

    ax.set_xticks(x, [str(bus) for bus in buses])
    ax.set_ylim(0.0, float(values.max()) * 1.24)
    ax.set_xlabel("Bus")
    ax.set_ylabel("LMP [JPY/MWh]")
    ax.legend(loc="upper right", fontsize=9)
    return _finish(ax, created)


# ======================================================================
# 第 07 回・第 08 回 — 起動停止計画と変動性電源
# ======================================================================

def _committed_units(result: CommitmentResult) -> list:
    """入切表に載っている号機を、ケースの並び順で返す。"""
    return [unit for unit in result.case.units if unit.name in result.schedule]


def plot_commitment(result: CommitmentResult, ax: Axes | None = None) -> Axes:
    """時刻ごとの出力を積み上げ、需要曲線と重ねる（第 07 回）。

    積み上げの高さが需要にちょうど届いていること、その上に同期並列容量
    （破線）との差が残っていることを見る。その差が **運転予備力**である。
    停止中の号機はこの図に 1 MW も現れない。「起動に何時間もかかる容量を
    予備力に数えない」という定義が、図の形として出ている。

    供給不足（``shortfall_mw``）があれば積み上げの上に赤い斜線で載せ、
    出力抑制（``spill_mw``）はゼロより下に描く。抑制は「発電できるのに
    出さない量」なので、需要の積み上げには入れない。

    Notes
    -----
    描いている需要は :attr:`CommitmentResult.demand_mw`、すなわち VRE を
    差し引いた **純需要**である。もとの需要は
    ``result.options["gross_demand_mw"]`` にある。

    Parameters
    ----------
    result:
        :class:`~gridops.commitment.CommitmentResult`。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (10.0, 5.0))
    demand = np.asarray(result.demand_mw, dtype=float)
    horizon = demand.size
    hours = np.arange(horizon)
    units = _committed_units(result)
    colors = _unit_colors(result.case)

    bottom = np.zeros(horizon)
    for unit in units:
        output = np.asarray(result.dispatch[unit.name], dtype=float)
        ax.bar(
            hours,
            output,
            bottom=bottom,
            width=1.0,
            color=colors[unit.name],
            edgecolor="w",
            linewidth=0.3,
            label=unit.name,
        )
        bottom = bottom + output

    shortfall = np.asarray(getattr(result, "shortfall_mw", np.zeros(horizon)), dtype=float)
    if float(shortfall.sum()) > 0.0:
        ax.bar(
            hours,
            shortfall,
            bottom=bottom,
            width=1.0,
            color=SEVERITY_COLORS["violation"],
            hatch="//",
            edgecolor="w",
            linewidth=0.3,
            label="unserved energy",
        )

    spill = np.asarray(getattr(result, "spill_mw", np.zeros(horizon)), dtype=float)
    if float(spill.sum()) > 0.0:
        ax.bar(
            hours,
            -spill,
            width=1.0,
            color=SEVERITY_COLORS["warn"],
            hatch="\\\\",
            edgecolor="w",
            linewidth=0.3,
            label="curtailed (spill)",
        )

    committed = np.array([result.committed_mw(t) for t in range(horizon)])
    ax.step(
        hours, committed, where="mid", color="0.25", ls="--", lw=1.6,
        label="committed capacity",
    )
    ax.step(hours, demand, where="mid", color="k", lw=2.4, label="net demand")

    _hour_axis(ax, horizon)
    # 凡例（号機の数だけ項目がある）が積み上げに重ならないよう上を空ける。
    top = max(float((bottom + shortfall).max()), float(committed.max()))
    ax.set_ylim(min(0.0, float(-spill.max()) * 1.1), top * 1.34)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power [MW]")
    ax.set_title(
        f"Unit commitment [{result.method}] "
        f"total cost {result.total_cost:,.0f} JPY"
    )
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    return _finish(ax, created)


def plot_commitment_schedule(
    result: CommitmentResult, ax: Axes | None = None
) -> Axes:
    """入切表 :math:`u_{it}` をヒートマップで描く（第 07 回）。

    運転中の連なりの長さが最低運転時間、停止の連なりの長さが最低停止時間
    の下限に効く。起動（0 から 1 への遷移）には上向きの印を付けてある。
    ``u0`` からの遷移も 1 回の起動として数えるので、**先頭の時刻に印が
    付く号機がある**。

    横に長く連なる石炭機（``min_up`` が長い）と、細かく出入りするピーク機
    の違いがそのまま模様になる。優先順位法と混合整数計画の解を 2 枚
    並べると、起動回数の差（wscc9 の light_load では 5 回対 1 回）が
    模様の細かさとして見える。

    Parameters
    ----------
    result:
        :class:`~gridops.commitment.CommitmentResult`。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.colors import ListedColormap

    ax, created = _prepare(ax, (10.0, 4.0))
    units = _committed_units(result)
    matrix = np.array(
        [np.asarray(result.schedule[unit.name], dtype=float) for unit in units]
    )
    horizon = matrix.shape[1] if matrix.size else 0

    ax.imshow(
        matrix,
        aspect="auto",
        cmap=ListedColormap(["0.92", SEVERITY_COLORS["ok"]]),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=(-0.5, horizon - 0.5, len(units) - 0.5, -0.5),
    )

    # 起動の印。u0 からの遷移も 1 回と数える（先頭の時刻にも付きうる）。
    xs: list[float] = []
    ys: list[float] = []
    for row, unit in enumerate(units):
        state = np.asarray(result.schedule[unit.name], dtype=float)
        previous = np.concatenate(([float(unit.u0)], state[:-1]))
        for t in np.nonzero(state - previous > 0.5)[0]:
            xs.append(float(t))
            ys.append(float(row))
    if xs:
        ax.plot(xs, ys, marker="^", ms=6, color="k", ls="none", label="start-up")

    ax.set_yticks(np.arange(len(units)), [unit.name for unit in units])
    _hour_axis(ax, horizon)
    ax.set_xticks(np.arange(-0.5, horizon, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(units), 1.0), minor=True)
    ax.grid(which="minor", color="w", lw=0.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    handles = [
        Patch(color="0.92", label="offline ($u=0$)"),
        Patch(color=SEVERITY_COLORS["ok"], label="online ($u=1$)"),
    ]
    if xs:
        handles.append(
            plt.Line2D([], [], marker="^", color="k", ls="none", label="start-up")
        )
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Unit")
    ax.set_title(
        f"Commitment schedule [{result.method}] ({result.n_startups()} start-ups)"
    )
    return _finish(ax, created)


def plot_duck_curve(
    case: Case, ax: Axes | None = None, *, profile: str = "summer_weekday"
) -> Axes:
    """需要・VRE 出力・純需要を重ねたダックカーブ（第 08 回）。

    太陽光は昼の需要を押し下げるが夕方に急速に消える。押し下げの深さ
    （下げ代の問題）と、日没に向かう **立ち上がりの急峻さ**（ランプの
    問題）の 2 つが、この 1 枚に同時に出る。図には最大ランプ [MW/h] と
    純需要の最小値を注記してある。

    純需要の最小値が運転中号機の最低出力の合計を下回ると、出力抑制なしでは
    実行不可能になる。第 08 回で ``vre_mw`` を増やしていくと、この図の谷が
    下がっていって抑制が立つ。

    Parameters
    ----------
    case:
        ケース。``commitment`` 層（需要形状と VRE）が要る。
    ax:
        描画先。
    profile:
        需要形状の名前（``"summer_weekday"`` / ``"winter_weekday"`` /
        ``"light_load"``）。

    Returns
    -------
    matplotlib.axes.Axes
    """
    from .commitment import demand_profile, net_demand

    ax, created = _prepare(ax, (9.0, 4.8))
    gross = np.asarray(demand_profile(case, profile), dtype=float)
    net = np.asarray(net_demand(case, gross), dtype=float)
    vre = gross - net
    hours = np.arange(gross.size)

    ax.fill_between(
        hours, net, gross, color=SEVERITY_COLORS["warn"], alpha=0.30, label="VRE output"
    )
    ax.plot(hours, gross, color="0.35", lw=1.6, ls="--", label="gross demand")
    ax.plot(hours, net, color=SEVERITY_COLORS["ok"], lw=2.6, label="net demand")

    if net.size > 1:
        ramp = np.diff(net)
        k = int(np.argmax(ramp))
        ax.annotate(
            f"max ramp {ramp[k]:.1f} MW/h",
            xy=(k + 1, net[k + 1]),
            xytext=(-14, 26),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            bbox={"boxstyle": "round", "fc": "w", "ec": "none", "alpha": 0.75},
            arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.0},
        )
    # 注記が軸の外に出ないよう、谷の下に余白を作ってから書き込む。
    low, high = float(net.min()), float(gross.max())
    pad = 0.12 * max(high - low, 1.0)
    ax.set_ylim(low - 2.2 * pad, high + pad)
    trough = int(np.argmin(net))
    ax.annotate(
        f"minimum net demand {net[trough]:.1f} MW",
        xy=(trough, net[trough]),
        xytext=(0, -34),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "k", "lw": 1.0},
    )

    _hour_axis(ax, gross.size)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power [MW]")
    ax.set_title(f"Net demand with VRE ({profile}, {float(vre.max()):.0f} MW peak PV)")
    ax.legend(loc="lower left", fontsize=9)
    return _finish(ax, created)


# ======================================================================
# 第 10 回 — アデカシー
# ======================================================================

def plot_capacity_outage_table(
    copt: CapacityOutageTable, ax: Axes | None = None
) -> Axes:
    """容量停止確率表を描く（第 10 回）。

    縦軸は対数である。確率は 1e-20 の桁まで落ちるので、線形軸では
    「先頭以外は全部ゼロ」に見えてしまい、**稀な多重停止こそが供給支障を
    決める**という要点が消える。

    階段は累積確率 :math:`P(\\text{outage} \\ge x)`、点はその停止容量
    ちょうどの確率である。期待停止容量 :math:`\\sum_i P_i^{max}
    \\mathrm{FOR}_i` を破線で示す。丸め（``rounding_mw``）を使った表でも
    この破線の位置は変わらない（期待値が保存されるため）。

    Parameters
    ----------
    copt:
        :class:`~gridops.adequacy.CapacityOutageTable`。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (8.5, 5.0))
    outage = np.asarray(copt.outage_mw, dtype=float)
    probability = np.asarray(copt.probability, dtype=float)
    cumulative = np.asarray(copt.cumulative, dtype=float)

    positive = probability[probability > 0.0]
    floor = float(positive.min()) * 0.3 if positive.size else 1e-12
    floor = max(floor, 1e-30)

    ax.step(
        outage,
        np.maximum(cumulative, floor),
        where="post",
        color=SEVERITY_COLORS["ok"],
        lw=2.0,
        label="$P(\\mathrm{outage} \\geq x)$",
    )
    ax.vlines(
        outage,
        floor,
        np.maximum(probability, floor),
        color=SEVERITY_COLORS["warn"],
        lw=1.0,
    )
    ax.plot(
        outage,
        np.maximum(probability, floor),
        "o",
        ms=4,
        color=SEVERITY_COLORS["warn"],
        label="$P(\\mathrm{outage} = x)$",
    )

    expected = float(copt.expected_outage_mw())
    ax.axvline(
        expected,
        color="k",
        ls="--",
        lw=1.4,
        label=f"expected outage = {expected:.1f} MW",
    )

    ax.set_yscale("log")
    ax.set_ylim(floor, 2.0)
    ax.set_xlabel("Capacity outage [MW]")
    ax.set_ylabel("Probability [-]")
    ax.set_title(
        f"Capacity outage probability table "
        f"({outage.size} states, {copt.installed_mw:.0f} MW installed)"
    )
    ax.legend(loc="lower left", fontsize=9)
    return _finish(ax, created)


def plot_lolp_convergence(
    results: Sequence[MonteCarloResult],
    ax: Axes | None = None,
    *,
    reference: float | None = None,
) -> Axes:
    """モンテカルロの LOLP を標本数に対して描く（第 10 回）。

    点推定だけを並べても収束は判断できない。**95% 信頼区間が解析解を
    含むか**で見ること。誤差は :math:`1/\\sqrt{N}` でしか縮まないので、
    標本数を 4 倍にして誤差の幅が半分になることを確かめさせるとよい。

    Parameters
    ----------
    results:
        :class:`~gridops.adequacy.MonteCarloResult` の列（標本数の違う
        ものを並べる）。
    ax:
        描画先。
    reference:
        解析解の LOLP。与えると水平線で示す（契約に無い追加の引数。
        既定は ``None`` なので契約どおりの呼び方でも動く）。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (8.0, 4.8))
    items = list(results)
    samples = np.array([float(item.n_samples) for item in items])
    estimate = np.array([float(item.lolp) for item in items])
    intervals = np.array([item.lolp_interval() for item in items], dtype=float)
    yerr = np.vstack([estimate - intervals[:, 0], intervals[:, 1] - estimate])

    ax.errorbar(
        samples,
        estimate,
        yerr=yerr,
        fmt="o-",
        ms=6,
        capsize=5,
        color=SEVERITY_COLORS["ok"],
        label="Monte Carlo (95% CI)",
    )
    if reference is not None:
        ax.axhline(
            float(reference),
            color="k",
            ls="--",
            lw=1.5,
            label=f"analytic LOLP = {float(reference):.5f}",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of samples $N$ [-]")
    ax.set_ylabel("LOLP [-]")
    ax.set_title("Monte Carlo convergence of LOLP")
    ax.legend(loc="best", fontsize=9)
    return _finish(ax, created)


# ======================================================================
# 第 09 回 — セキュリティ
# ======================================================================

def plot_contingency_ranking(
    report: SecurityReport, ax: Axes | None = None
) -> Axes:
    """N-1 の事故を性能指数で並べる（第 09 回）。

    棒の長さは性能指数 :math:`PI = \\sum_l w_l (f_l/f_l^{max})^{2n}/(2n)`、
    色は **実際に安全かどうか**である。この 2 つがずれること、すなわち
    「PI の上位が危険な事故とは限らない」ことがこの図の主題である。

    :math:`n=1` の PI は、軽い過負荷が多数ある事故に大きな値を与え、
    重い過負荷が 1 本だけの事故を下位に沈める（masking）。**PI の順位は
    順位を誤るものである。** スクリーニングは候補の絞り込みであって
    判断ではない、という第 09 回の結論がここに出る。

    さらに、直流の潮流だけを見る指標は電圧を原理的に見ない。wscc9 の
    枝 4-6 の開放は熱容量では 75.7% で健全なのに、母線 6 の電圧が
    0.9418 p.u. まで落ちて下限 0.95 を割る。色（安全か否か）に電圧を
    含めておくと、PI が小さいのに赤い棒として現れる。

    Parameters
    ----------
    report:
        :class:`~gridops.security.SecurityReport`、または
        ``ContingencyResult`` の列。
    ax:
        描画先。

    Returns
    -------
    matplotlib.axes.Axes
    """
    ax, created = _prepare(ax, (8.5, 5.0))
    results = list(getattr(report, "results", report))
    ranked = sorted(results, key=lambda item: float(item.performance_index), reverse=True)
    n = len(ranked)
    y = np.arange(n)[::-1]

    values = np.array([float(item.performance_index) for item in ranked])
    colors = [
        SEVERITY_COLORS["ok"] if bool(getattr(item, "is_secure", True))
        else SEVERITY_COLORS["violation"]
        for item in ranked
    ]
    ax.barh(y, values, height=0.66, color=colors, edgecolor="k", linewidth=0.6)

    limit = float(values.max()) if n else 1.0
    for position, item, value in zip(y, ranked, values):
        worst = getattr(item, "worst_branch", None)
        loading = float(getattr(item, "worst_loading", 0.0))
        note = f"{100.0 * loading:.1f}%"
        if worst is not None:
            note = f"{worst[0]}-{worst[1]} {note}"
        ax.text(
            value + 0.02 * limit,
            position,
            note,
            va="center",
            ha="left",
            fontsize=8,
            color="0.25",
        )

    labels = [f"{item.outage[0]}-{item.outage[1]}" for item in ranked]
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, limit * 1.35 if limit > 0.0 else 1.0)

    handles = [
        Patch(color=SEVERITY_COLORS["ok"], label="secure"),
        Patch(color=SEVERITY_COLORS["violation"], label="insecure"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    skipped = list(getattr(report, "skipped", ()))
    if skipped:
        lines = [f"{key[0]}-{key[1]}: {reason}" for key, reason in skipped]
        ax.text(
            0.98,
            0.97,
            "skipped\n" + "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox={"boxstyle": "round", "fc": "w", "ec": "0.6", "alpha": 0.9},
        )

    ax.set_xlabel("Performance index $PI$ [-]")
    ax.set_ylabel("Outage (branch)")
    ax.set_title("N-1 ranking by performance index (worst branch and loading shown)")
    ax.grid(axis="y", visible=False)
    return _finish(ax, created)
