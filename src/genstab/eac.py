"""等面積法 (Equal Area Criterion) と臨界事故除去時間 (CCT)。

等面積法は、制動を無視した古典モデルの SMIB に限って成り立つ図式的な
安定判別法である。事故中に回転子が得た運動エネルギー（加速面積）を、
事故除去後に減速面積として返せるかどうかで判定する。

.. math::

    A_1 = \\int_{\\delta_0}^{\\delta_c} (P_m - P_{e,\\text{fault}}) \\, d\\delta,
    \\qquad
    A_2 = \\int_{\\delta_c}^{\\delta_u} (P_{e,\\text{post}} - P_m) \\, d\\delta

``A_1 = A_2`` となる事故除去角が臨界事故除去角 δ_c である。

本モジュールは解析解と、時間領域シミュレーションを二分探索して得る
数値解の両方を提供する。両者が一致することを確かめるのが教材の要点で、
制動 D > 0 を入れると数値解のほうが長くなる（制動がエネルギーを
吸収するぶん余裕が増える）ことも確認できる。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace

import numpy as np

from .events import FaultSchedule, Stage
from .simulate import simulate
from .system import SMIBSystem


@dataclass(frozen=True)
class EqualAreaResult:
    """等面積法による判定結果。"""

    delta_0: float           #: 事故前の安定平衡点 [rad]
    delta_u: float           #: 事故後の不安定平衡点 [rad]
    delta_c: float           #: 臨界事故除去角 [rad]
    accelerating_area: float #: 与えられた除去角までの加速面積 [p.u.·rad]
    decelerating_area: float #: 同じ除去角以降に使える減速面積 [p.u.·rad]
    Pmax_pre: float
    Pmax_fault: float
    Pmax_post: float
    Pm: float

    @property
    def margin(self) -> float:
        """安定余裕 ``(A2 - A1) / A2``。正なら安定、負なら脱調。"""
        if self.decelerating_area == 0.0:
            return float("-inf")
        return (self.decelerating_area - self.accelerating_area) / self.decelerating_area

    @property
    def is_stable(self) -> bool:
        """減速面積が加速面積以上か。"""
        return self.decelerating_area >= self.accelerating_area

    def summary(self) -> str:
        return (
            f"等面積法\n"
            f"  δ0 = {math.degrees(self.delta_0):7.3f} deg  (事故前安定平衡点)\n"
            f"  δu = {math.degrees(self.delta_u):7.3f} deg  (事故後不安定平衡点)\n"
            f"  δc = {math.degrees(self.delta_c):7.3f} deg  (臨界事故除去角)\n"
            f"  A1 = {self.accelerating_area:7.4f}  A2 = {self.decelerating_area:7.4f}"
            f"  余裕 = {self.margin:+.3f}\n"
            f"  判定: {'安定' if self.is_stable else '脱調'}"
        )


def check_assumptions(
    system: SMIBSystem,
    *,
    require_undamped: bool = False,
    allow_approximation: bool = False,
) -> list[str]:
    """等面積法の適用条件を確認する。

    等面積法が成り立つのは「古典モデル・内部起電力一定・機械入力一定・
    制動なし」の場合に限られる。1 軸モデルでは内部起電力 E'q が時間と
    ともに変わり、制御器を付ければ機械入力や界磁電圧も動くため、
    面積の計算そのものが意味を失う。

    実際、1 軸モデル + AVR の系に等面積法を適用すると、初期の E'q から
    求めた不安定平衡点は 130 度、時間領域で求めた真の鞍点は 75 度前後と
    大きく食い違う。数値は返るが正しくない、という最も危険な状態になる。

    Parameters
    ----------
    require_undamped:
        制動 D = 0 も条件に含めるか。解析 CCT のように制動があると
        明確に誤差が出る計算では ``True`` にする。
    allow_approximation:
        条件を満たさない場合に、例外の代わりに警告を出して続行するか。

    Returns
    -------
    満たされていない条件の説明。すべて満たしていれば空リスト。

    Raises
    ------
    ValueError
        条件を満たさず ``allow_approximation=False`` の場合。
    """
    from .machines.classical import ClassicalMachine

    problems: list[str] = []
    if not isinstance(system.machine, ClassicalMachine):
        problems.append(
            f"発電機モデルが {type(system.machine).__name__} である"
            "（等面積法は内部起電力が一定の古典モデルにのみ適用できる）"
        )
    if system.controllers:
        names = ", ".join(type(c).__name__ for c in system.controllers)
        problems.append(
            f"制御器が接続されている（{names}）。機械入力や界磁電圧が"
            "時間とともに変わるため、面積が保存量を表さなくなる"
        )
    if require_undamped and system.machine.D != 0.0:
        problems.append(
            f"制動係数が D = {system.machine.D} である"
            "（制動があると運動エネルギーが散逸し、面積の釣り合いが崩れる）"
        )

    if problems and not allow_approximation:
        detail = "\n  - ".join(problems)
        raise ValueError(
            "等面積法の適用条件を満たしていない:\n  - "
            + detail
            + "\n近似として承知のうえで使う場合は allow_approximation=True を"
            " 指定すること。時間領域の critical_clearing_time() なら"
            " これらの制約なしに使える。"
        )
    if problems:
        warnings.warn(
            "等面積法の適用条件を満たしていないため、結果は近似にすぎない: "
            + "; ".join(problems),
            UserWarning,
            stacklevel=3,
        )
    return problems


def _power_amplitudes(system: SMIBSystem) -> tuple[float, float, float, float]:
    """各ステージの P-δ 曲線の波高値と機械入力を返す。"""
    emf = system.machine.internal_emf(system.initial_state())
    return (
        system.network.max_power(Stage.PRE, emf),
        system.network.max_power(Stage.FAULT, emf),
        system.network.max_power(Stage.POST, emf),
        system.operating_point.Pm,
    )


def unstable_equilibrium_angle(
    system: SMIBSystem, *, allow_approximation: bool = True
) -> float:
    """事故後ネットワークの不安定平衡点 δ_u = π - arcsin(Pm / Pmax_post) [rad]。

    Notes
    -----
    内部起電力が一定であることを前提にした静的な境界である。1 軸モデルや
    制御器付きの系では E'q が動くため、この角度は真の安定境界ではない。
    既定では警告のみで計算を続ける。
    """
    check_assumptions(system, allow_approximation=allow_approximation)
    _, _, pmax_post, pm = _power_amplitudes(system)
    if pmax_post <= pm:
        raise ValueError(
            f"事故後の送電可能最大電力 Pmax_post={pmax_post:.4f} が機械入力 "
            f"Pm={pm:.4f} 以下なので、事故後に平衡点が存在しない（必ず脱調する）。"
        )
    return math.pi - math.asin(pm / pmax_post)


def accelerating_area(system: SMIBSystem, delta_c: float) -> float:
    """δ0 から δ_c までの加速面積 [p.u.·rad]。"""
    _, pmax_fault, _, pm = _power_amplitudes(system)
    delta_0 = system.operating_point.delta
    return pm * (delta_c - delta_0) + pmax_fault * (
        math.cos(delta_c) - math.cos(delta_0)
    )


def decelerating_area(system: SMIBSystem, delta_c: float) -> float:
    """δ_c から δ_u までに使える減速面積 [p.u.·rad]。"""
    _, _, pmax_post, pm = _power_amplitudes(system)
    delta_u = unstable_equilibrium_angle(system)
    return pmax_post * (math.cos(delta_c) - math.cos(delta_u)) - pm * (
        delta_u - delta_c
    )


def critical_clearing_angle(
    system: SMIBSystem, *, allow_approximation: bool = False
) -> float:
    """臨界事故除去角 δ_c [rad]（解析解）。

    .. math::

        \\cos\\delta_c =
        \\frac{P_m(\\delta_u - \\delta_0) + P_{2}\\cos\\delta_u
               - P_{1}\\cos\\delta_0}{P_{2} - P_{1}}

    ここで P_1, P_2 はそれぞれ事故中・事故後の P-δ 曲線の波高値。
    """
    check_assumptions(system, allow_approximation=allow_approximation)
    _, pmax_fault, pmax_post, pm = _power_amplitudes(system)
    delta_0 = system.operating_point.delta
    delta_u = unstable_equilibrium_angle(system)

    denominator = pmax_post - pmax_fault
    if abs(denominator) < 1e-12:
        raise ValueError(
            "事故中と事故後の P-δ 曲線が同じため、等面積法では臨界角を定義できない。"
        )
    cos_dc = (
        pm * (delta_u - delta_0)
        + pmax_post * math.cos(delta_u)
        - pmax_fault * math.cos(delta_0)
    ) / denominator

    if not -1.0 <= cos_dc <= 1.0:
        raise ValueError(
            f"cos(δc) = {cos_dc:.4f} が [-1, 1] の外に出た。"
            " 事故前から安定に運転できない設定になっていないか確認すること。"
        )
    return math.acos(cos_dc)


def evaluate(
    system: SMIBSystem,
    delta_c: float | None = None,
    *,
    allow_approximation: bool = False,
) -> EqualAreaResult:
    """等面積法の各量をまとめて計算する。

    Parameters
    ----------
    delta_c:
        評価したい事故除去角 [rad]。省略すると臨界事故除去角を使う
        （このとき加速面積と減速面積が一致する）。
    """
    check_assumptions(system, allow_approximation=allow_approximation)
    pmax_pre, pmax_fault, pmax_post, pm = _power_amplitudes(system)
    delta_critical = critical_clearing_angle(system, allow_approximation=True)
    target = delta_critical if delta_c is None else float(delta_c)

    return EqualAreaResult(
        delta_0=system.operating_point.delta,
        delta_u=unstable_equilibrium_angle(system),
        delta_c=target,
        accelerating_area=accelerating_area(system, target),
        decelerating_area=decelerating_area(system, target),
        Pmax_pre=pmax_pre,
        Pmax_fault=pmax_fault,
        Pmax_post=pmax_post,
        Pm=pm,
    )


def critical_clearing_time_analytic(
    system: SMIBSystem, *, allow_approximation: bool = False
) -> float:
    """臨界事故除去時間 CCT [s] の解析解。

    事故中に電力を送れない（Pmax_fault = 0）場合に限り、事故中の運動は

    .. math::

        \\delta(t) = \\delta_0 + \\frac{\\omega_s P_m}{4H} t^2

    という単純な等加速度運動になるので、臨界事故除去角に達する時刻が
    閉形式で求まる。

    .. math::

        t_{cr} = \\sqrt{\\frac{4H(\\delta_c - \\delta_0)}{\\omega_s P_m}}

    Raises
    ------
    ValueError
        事故中に電力を送れる場合（Pmax_fault > 0）。このとき事故中の
        運動は非線形になり閉形式では書けないので、
        :func:`critical_clearing_time` を使うこと。
    """
    check_assumptions(
        system, require_undamped=True, allow_approximation=allow_approximation
    )
    _, pmax_fault, _, pm = _power_amplitudes(system)
    if pmax_fault > 1e-12:
        raise ValueError(
            f"この解析解は事故中に電力を送れない場合 (Pmax_fault = 0) にのみ"
            f" 成り立つ。現在 Pmax_fault = {pmax_fault:.4f} なので"
            " critical_clearing_time() による数値解を使うこと。"
        )
    if pm <= 0.0:
        raise ValueError(
            f"機械入力が Pm = {pm:.4f} で正でないため、事故中に加速せず"
            " 臨界事故除去時間を定義できない。"
        )

    delta_c = critical_clearing_angle(system, allow_approximation=True)
    delta_0 = system.operating_point.delta
    return math.sqrt(
        4.0 * system.machine.H * (delta_c - delta_0) / (system.base.omega_s * pm)
    )


def critical_clearing_time(
    system: SMIBSystem,
    *,
    t_end: float | None = None,
    tolerance: float = 1e-4,
    upper_bound: float = 5.0,
    angle_limit: float | None = None,
    **simulate_kwargs,
) -> float:
    """臨界事故除去時間 CCT [s] を二分探索で求める（数値解）。

    事故除去時間を変えながらシミュレーションを繰り返し、安定・不安定の
    境界を挟み込む。解析解と違って制動や制御器の効果がそのまま反映される
    ので、AVR や PSS が CCT をどれだけ延ばすかを定量的に見られる。

    Parameters
    ----------
    t_end:
        各シミュレーションの終了時刻 [s]。省略すると事故発生から 5 秒後。
    tolerance:
        二分探索の打ち切り幅 [s]。
    upper_bound:
        探索する事故除去時間の上限 [s]。
    angle_limit:
        安定判定に使う位相角偏差のしきい値 [rad]。省略すると
        ``δ_u - δ_0``（事故後の不安定平衡点までの余裕）を使う。
        回転子が δ_u を越えると電気出力が機械入力を下回ったまま
        加速し続けるため、これが物理的に正しい脱調判定であり、
        等面積法の判定とも一致する。
    **simulate_kwargs:
        :func:`~genstab.simulate.simulate` にそのまま渡される。

    Returns
    -------
    臨界事故除去時間 [s]。上限まで安定なら ``upper_bound`` を返す。

    Warns
    -----
    UserWarning
        観測時間が短く CCT を過大評価している疑いがあるとき。脱調する
        軌道が角度しきい値へ達する前に計算が終わると、二分探索はそれを
        安定とみなす。求まった値の 95 % の点を 3 倍の観測時間で再確認して
        検出しているので、数 % 程度の小さな過大評価までは捉えられない。
        既定の観測時間（事故発生から 5 秒）では正しい値が得られる。
    """
    t_fault = system.fault.t_fault
    if not math.isfinite(t_fault):
        raise ValueError("事故が設定されていない系では CCT を定義できない。")
    horizon = t_end if t_end is not None else t_fault + 5.0

    # 定態不安定な運転点で CCT を求めても意味を持たない。数値としては
    # 何らかの値が返るので、気づかずに使ってしまう危険がある。
    try:
        from .smallsignal import analyze

        modes = analyze(system)
        growth = float(np.max(modes.eigenvalues.real))
        if growth > 1e-6:
            warnings.warn(
                f"動作点が定態不安定（固有値の実部が最大 {growth:+.4f}）なので、"
                " ここで得られる CCT は意味を持たない。事故がなくても振動が"
                " 成長する運転点では、まず定態安定性を確保すること。",
                UserWarning,
                stacklevel=2,
            )
    except Exception:  # 線形化できない系では判定を省く
        pass

    if angle_limit is None:
        try:
            with warnings.catch_warnings():
                # ここでは静的な角度しきい値が欲しいだけなので、
                # 等面積法の適用条件に関する警告は出さない。
                warnings.simplefilter("ignore", UserWarning)
                angle_limit = (
                    unstable_equilibrium_angle(system)
                    - system.operating_point.delta
                )
        except (ValueError, AttributeError):
            # SMIB 以外の系（多機系統など）では不安定平衡点が一意に
            # 定まらないので、既定のしきい値を使う。多機系統では
            # MultiMachineSystem.assess_stability が機器間の角度差で
            # 判定するため、しきい値の意味もそちらに従う。
            angle_limit = math.pi

    def stable_for(clearing: float, observation: float | None = None) -> bool:
        """事故除去時間 `clearing` で安定かどうかを時間領域で判定する。"""
        window = horizon if observation is None else observation
        trial = replace(
            system, fault=FaultSchedule(t_fault=t_fault, t_clear=t_fault + clearing)
        )
        result = simulate(trial, t_end=window + clearing, **simulate_kwargs)
        return result.is_stable(angle_limit=angle_limit)

    if not stable_for(0.0):
        # 原因を切り分けて伝える。ここで詰まる理由はほぼ次の 2 つに限られる。
        reasons = []
        try:
            _, _, pmax_post, pm = _power_amplitudes(system)
            if pmax_post <= pm:
                reasons.append(
                    f"事故後の送電可能最大電力 Pmax_post={pmax_post:.4f} が"
                    f" 機械入力 Pm={pm:.4f} 以下で、事故後に平衡点が存在しない"
                )
        except (ValueError, AttributeError):
            pass
        try:
            from .smallsignal import analyze

            modes = analyze(system)
            if not modes.is_stable:
                worst = modes.eigenvalues[modes.dominant_index]
                reasons.append(
                    f"動作点そのものが定態不安定（固有値 {worst:.4f} が右半面）で、"
                    " 事故がなくても振動が成長する"
                )
        except Exception:  # 線形化できない系では判定を省く
            pass

        detail = (
            "\n  - ".join(reasons)
            if reasons
            else (
                "事故後のネットワーク構成では現在の運転点を維持できない。"
                " 1 軸モデルでは、界磁電圧が固定のままだと内部起電力 E'q が"
                " 電機子反作用で下がり、事故後の平衡点に届かないことがある"
                "（AVR を接続すると解消する場合がある）"
            )
        )
        raise ValueError(
            "事故除去時間 0 でも安定にならないため CCT を定義できない。\n"
            f"  - {detail}\n"
            "定態安定でない運転点では、そもそも過渡安定性を論じる意味がない点に注意すること。"
        )
    if stable_for(upper_bound):
        warnings.warn(
            f"探索上限 {upper_bound} s まで安定だった。upper_bound を大きくすること。",
            UserWarning,
            stacklevel=2,
        )
        return upper_bound

    low, high = 0.0, upper_bound
    while high - low > tolerance:
        mid = 0.5 * (low + high)
        if stable_for(mid):
            low = mid
        else:
            high = mid

    # 観測時間が足りているかを事後に検証する。脱調する軌道が角度しきい値に
    # 達する前にシミュレーションを打ち切ってしまうと、二分探索はそれを
    # 「安定」とみなし、CCT を過大評価する。実際、既定より短い観測時間では
    # 真値 206 ms に対して 390 ms（+90 %）という値が返ることを確認している。
    # 数値としては返るので、検証しなければ誤りに気づけない。
    if low > tolerance:
        # 事故後を観察する時間を 3 倍に延ばす。t_end が t_fault と同じか
        # それより短い場合でも必ず延びるよう、最低 1 秒は確保する。
        extended = t_fault + 3.0 * max(horizon - t_fault, 1.0)
        # 二分探索の打ち切り幅ぶん、low は真の臨界をわずかに超えうる。
        # 5 % 内側の明らかに安定なはずの点で検証し、そこでも脱調するなら
        # 観測時間が足りていないと判断する。
        probe = 0.95 * low
        if not stable_for(probe, observation=extended):
            warnings.warn(
                f"観測時間 t_end={horizon:g} s では CCT を過大評価している可能性が高い。"
                f" 除去時間 {probe:.4f} s（求まった {low:.4f} s の 95 %）は、"
                f" t_end={extended:g} s まで見ると脱調する。"
                " t_end を長くして計算し直すこと。",
                UserWarning,
                stacklevel=2,
            )
    return low


def plot_equal_area(
    system: SMIBSystem,
    delta_c: float | None = None,
    *,
    ax=None,
    title: str | None = None,
):
    """加速面積と減速面積を塗り分けた等面積法の図を描く。"""
    import matplotlib.pyplot as plt

    from .plotting import plot_power_angle

    outcome = evaluate(system, delta_c)
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.8))
    plot_power_angle(system, ax=ax)

    d0, dc, du = outcome.delta_0, outcome.delta_c, outcome.delta_u
    pm = outcome.Pm

    grid_acc = np.linspace(d0, dc, 200)
    ax.fill_between(
        np.degrees(grid_acc),
        outcome.Pmax_fault * np.sin(grid_acc),
        pm,
        color="tab:red",
        alpha=0.25,
        label=f"$A_1$ = {outcome.accelerating_area:.4f}",
    )

    grid_dec = np.linspace(dc, du, 200)
    ax.fill_between(
        np.degrees(grid_dec),
        pm,
        outcome.Pmax_post * np.sin(grid_dec),
        color="tab:green",
        alpha=0.25,
        label=f"$A_2$ = {outcome.decelerating_area:.4f}",
    )

    for angle, name in ((d0, "$\\delta_0$"), (dc, "$\\delta_c$"), (du, "$\\delta_u$")):
        ax.axvline(np.degrees(angle), color="gray", ls=":", lw=1.0)
        ax.annotate(
            name,
            xy=(np.degrees(angle), ax.get_ylim()[1]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=11,
        )

    ax.legend(loc="lower left", fontsize=9)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax
