"""経済負荷配分（等 λ 法）と直流最適潮流。

本モジュールは「1 時点の発電機出力をどう決めるか」だけを扱う。時間方向の
拘束（最低運転時間・ランプ率・起動費）は :mod:`gridops.commitment` の
仕事であり、ここには入れない。

なぜ解法を 2 つ持つのか
-----------------------
:func:`economic_dispatch` は **2 次費用のまま** λ の二分法で解き、
:func:`dc_opf` は **1 次費用**（:attr:`Unit.var_cost`）で線形計画を解く。
同じ「一番安い出力の組」を求めるのに 2 通り書くのは冗長に見えるが、
これは PuLP と CBC が線形計画と混合整数計画しか扱えないという道具の
制約から来ている。2 次費用を線形計画に載せるには区分線形近似が要り、
その近似誤差の議論は起動停止計画（第 07 回）の主題である。一方で
等 λ 法は 2 次費用を厳密に、しかも 30 行で解ける。**同じ問題に対して
「厳密に解ける定式化」と「制約を足せる定式化」の 2 つを持ち、
どちらを使うかを選べること自体**が教材の中身である。

============================  ==================  ====================
関数                          費用の形            扱える制約
============================  ==================  ====================
:func:`economic_dispatch`     2 次（厳密）        出力の上下限だけ
:func:`dispatch_with_losses`  2 次（厳密）        上下限 + 送電損失
:func:`dc_opf`                1 次（線形計画）    上下限 + 線路の熱容量
============================  ==================  ====================

二分法が必ず効く理由
--------------------
各号機の最適出力は λ の関数として

.. math::

    P_i(\\lambda) = \\mathrm{clip}\\!\\left(
        \\frac{\\lambda - b_i}{2 c_i},\\; P_i^{min},\\; P_i^{max}\\right)

と書ける。:math:`c_i > 0` なら :math:`P_i` は λ の単調非減少な連続関数で
あり、:math:`c_i = 0`（線形費用）なら :math:`\\lambda = b_i` で
:math:`P^{min}` から :math:`P^{max}` へ飛ぶ **階段関数**になる。いずれに
せよ :math:`\\sum_i P_i(\\lambda)` は単調非減少なので、需要と交わる λ を
二分法で必ず挟み込める。ニュートン法ではなく二分法を使うのは、上下限に
張り付く点で微分が不連続になり、階段の段差では微分が存在しないためである。

**需要が :math:`[\\sum P^{min}, \\sum P^{max}]` の外にあるときの検査を、
二分法より先に必ず行う。** この検査が無いと、二分法は交点の無い区間を
律儀に半分にし続け、区間幅が許容差を下回った時点で「収束した」と報告
する。返ってくる λ はブラケットの端に張り付いた無意味な値であり、
:math:`\\sum P \\ne D` のまま下流へ流れる。**収束したふりが最も危険な
故障モード**であって、例外で止めるほうがはるかに親切である。

符号の規約
----------
:mod:`gridops.solvers` の規約に従い、需給の等式は **右辺に需要を正の符号で
置く**向きで書く。:func:`dc_opf` の母線ごとの注入等式は

.. math:: \\sum_{i \\in b} p_i - (B \\theta)_b = d_b

の向きであり、この制約の双対がそのまま母線 :math:`b` の限界価格（LMP）
:math:`\\partial(\\text{総費用})/\\partial d_b` [円/MWh] になる。一方、
線路の熱容量制約 :math:`f_k \\le \\bar f_k` の双対は最小化問題では **負**
（容量が増えれば費用が減る）なので、:attr:`DCOPFResult.congestion_price`
には **符号を反転して正の量**として格納する。混雑レントが負になる事故を
防ぐためであり、反転をここ 1 箇所に閉じ込めておく。

単位
----
出力・費用は MW と 円（:attr:`DispatchResult.dispatch` は号機名 -> MW）、
ネットワーク量は p.u.（:attr:`DCOPFResult.theta` は rad、
:attr:`DCOPFResult.flows` は p.u.）である。:class:`gridops.case.Case` の
規約をそのまま引き継いでいる。混雑レントを組み立てるときは
p.u. の潮流に :attr:`Case.base_mva` を掛けて MW に直してから
[円/MWh] を掛けること。この掛け忘れが 100 倍の誤差として出る。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import solvers
from .case import Branch, BusType, Case, Unit
from .ybus import build_ybus, incidence_matrix

__all__ = [
    "DispatchResult",
    "DCOPFResult",
    "merit_order",
    "economic_dispatch",
    "penalty_factors",
    "dispatch_with_losses",
    "dc_opf",
]


# ======================================================================
# 既定値
# ======================================================================
#: 上下限に「張り付いている」と判定する幅 [MW]（容量に対する相対値）。
#: CBC の許容誤差ではなく倍精度の丸めを見込んだ値で、λ 法の解は
#: 解析的に決まるのでこの程度で十分に分離できる。
BOUND_TOLERANCE = 1e-9

#: 混雑していると判定する混雑料金の下限 [円/MWh]。
CONGESTION_TOLERANCE = 1e-6

#: ペナルティファクタの分母 :math:`1 - \\partial P_{loss}/\\partial P_i` が
#: これより小さいときに警告する。極に近いと出力がいくらでも大きくなる。
PENALTY_POLE_TOLERANCE = 1e-3

#: 損失の数値微分の刻み [p.u.]（0.1 MW 相当）。中心差分の打切り誤差
#: :math:`O(\\delta^2)` と潮流の収束誤差 :math:`O(10^{-11}/\\delta)` の
#: 釣り合いで決めてある。
LOSS_DERIVATIVE_STEP = 1e-3


# ======================================================================
# 号機の選択
# ======================================================================
def _select_units(
    case: Case,
    units: Sequence[Unit] | None = None,
    committed: Iterable[str] | Mapping[str, object] | None = None,
) -> list[Unit]:
    """計算に載せる号機を決める。

    Parameters
    ----------
    case:
        系統ケース。``units`` を明示しないときの号機の出どころ。
    units:
        号機を直接与える（テストや小さな例題で使う）。``None`` なら
        :attr:`Case.units`。
    committed:
        起動している号機。号機名の並び、または ``{号機名: 真偽}`` の
        対応。``None`` なら全機が起動しているとみなす。

    Raises
    ------
    ValueError
        号機が 1 台も無いとき、``committed`` に未知の号機名があるとき、
        起動している号機が 1 台も無いとき。
    """
    pool = list(case.units if units is None else units)
    if not pool:
        raise ValueError(
            f"ケース '{case.name}' に号機がない。units 層を持つケース"
            "（同梱の 'wscc9'）を使うか、units 引数で号機を直接渡すこと。"
        )
    if committed is None:
        return pool

    if isinstance(committed, Mapping):
        wanted = {str(name) for name, on in committed.items() if on}
    else:
        wanted = {str(name) for name in committed}

    known = {unit.name for unit in pool}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(
            f"committed に未知の号機名がある: {unknown}。"
            f"ケース '{case.name}' の号機は {sorted(known)}。"
        )
    chosen = [unit for unit in pool if unit.name in wanted]
    if not chosen:
        raise ValueError(
            "起動している号機が 1 台もない。committed が空でないか、"
            "起動停止計画の結果を取り違えていないかを確認すること。"
        )
    return chosen


# ======================================================================
# 優先順位
# ======================================================================
def merit_order(case: Case, *, units: Sequence[Unit] | None = None) -> list[Unit]:
    """全負荷平均費用 :math:`C(P^{max})/P^{max}` の安い順に号機を並べる。

    優先順位法（:func:`gridops.commitment.priority_list`）の順位づけであり、
    「どの号機から起動するか」を決める。**限界費用（増分燃料費）の順とは
    別物である**ことに注意する。全負荷平均費用は無負荷費 :math:`c_0` を
    出力で割った分を含むので、無負荷費の重い大容量機は限界費用が安くても
    順位が下がることがある。等 λ 法（:func:`economic_dispatch`）が使うのは
    増分費用のほうであって、この順位ではない。

    Parameters
    ----------
    case:
        系統ケース。
    units:
        並べ替える号機。``None`` なら :attr:`Case.units`。

    Returns
    -------
    list[gridops.case.Unit]
        安い順。同値のときは号機名の辞書順で決める（ソルバの版や辞書の
        並び順で結果が変わらないようにするため）。
    """
    pool = _select_units(case, units, None)
    return sorted(pool, key=lambda u: (u.full_load_average_cost(), u.name))


# ======================================================================
# 経済負荷配分の結果
# ======================================================================
@dataclass
class DispatchResult:
    """等 λ 法（およびその損失込みの拡張）の結果。

    Parameters
    ----------
    dispatch:
        号機名 -> 出力 [MW]。
    lam:
        系統 λ [円/MWh]。損失を考慮しない場合は各号機の増分燃料費
        :math:`dC_i/dP_i` そのもの、考慮する場合は
        :math:`L_i \\, dC_i/dP_i`（:math:`L_i` はペナルティファクタ）。
    total_cost:
        総燃料費 [円/h]。:meth:`Unit.fuel_cost` を使うので **無負荷費
        :math:`c_0` を含む**。:attr:`DCOPFResult.total_cost` は線形費用の
        分だけなので、2 つを直接比べてはいけない。
    iterations:
        二分法の反復回数（:func:`dispatch_with_losses` では外側の反復回数）。
    converged:
        :math:`|\\sum P - D|` が許容差に収まったか。
    marginal_units:
        上下限のどちらにも張り付いていない号機の名前。ここに入る号機だけが
        「需要が 1 MW 増えたときに出力を変える号機」であり、その増分費用が
        系統 λ に等しい。
    """

    dispatch: dict[str, float]
    lam: float
    total_cost: float
    iterations: int = 0
    converged: bool = True
    marginal_units: tuple[str, ...] = ()
    #: 号機名 -> ペナルティファクタ。損失を考慮しない場合はすべて 1。
    #: （契約に無い追加フィールド。既定値付きなので構築側の互換は保たれる。）
    penalty: dict[str, float] = field(default_factory=dict)
    #: 送電損失 [MW]。:func:`economic_dispatch` では 0。
    losses_mw: float = 0.0
    #: 需要 [MW]。:meth:`summary` の表示に使う。
    demand_mw: float = 0.0

    def total_mw(self) -> float:
        """全号機の出力の合計 [MW]。"""
        return float(sum(self.dispatch.values()))

    def summary(self) -> str:
        """人が読む要約。"""
        lines = [
            "economic dispatch",
            f"  demand      : {self.demand_mw:10.3f} MW",
            f"  generation  : {self.total_mw():10.3f} MW",
            f"  losses      : {self.losses_mw:10.3f} MW",
            f"  lambda      : {self.lam:10.2f} 円/MWh",
            f"  total cost  : {self.total_cost:12.1f} 円/h",
            f"  iterations  : {self.iterations} ({'converged' if self.converged else 'NOT converged'})",
            f"  marginal    : {', '.join(self.marginal_units) or '(none)'}",
        ]
        return "\n".join(lines)


# ======================================================================
# 等 λ 法
# ======================================================================
def _output_at(unit: Unit, lam: float, factor: float = 1.0) -> float:
    """λ における号機の最適出力 [MW]。

    最適性条件は :math:`L_i \\, (2 c_i P_i + b_i) = \\lambda`
    （:math:`L_i` はペナルティファクタ。損失を考えないときは 1）である。

    :math:`c_i = 0` の号機ではこの式が :math:`P_i` を決めない。増分費用が
    出力によらず :math:`b_i` で一定なので、:math:`\\lambda > L_i b_i` なら
    上限まで、:math:`\\lambda < L_i b_i` なら下限まで出すのが最適であり、
    :math:`\\lambda = L_i b_i` のときは **区間 :math:`[P^{min}, P^{max}]` の
    どこでも最適**になる。ここでは下限を返しておき、二分法が収束したあとの
    ブラケット :math:`[\\lambda_{lo}, \\lambda_{hi}]` に段差が入っている
    号機に残りの需要を配分する（:func:`_lambda_dispatch` を参照）。
    """
    if unit.quadratic > 0.0:
        target = lam / factor
        p = (target - unit.var_cost) / (2.0 * unit.quadratic)
    else:
        p = unit.p_max_mw if lam > factor * unit.var_cost else unit.p_min_mw
    return min(max(p, unit.p_min_mw), unit.p_max_mw)


def _bracket(units: Sequence[Unit], factor: Mapping[str, float]) -> tuple[float, float]:
    """二分法のブラケット :math:`[\\lambda_{lo}, \\lambda_{hi}]`。

    :math:`\\lambda_{lo} = \\min_i L_i b_i` では全号機が下限、
    :math:`\\lambda_{hi} = \\max_i L_i (b_i + 2 c_i P_i^{max})` では全号機が
    上限に達する。後者が全号機について成り立つのは、``max`` が個々の
    :math:`L_i (b_i + 2 c_i P_i^{max})` 以上だからである。したがって
    :math:`\\sum P(\\lambda_{lo}) = \\sum P^{min}`、
    :math:`\\sum P(\\lambda_{hi}) = \\sum P^{max}` となり、需要が
    その間にある限り交点はブラケットの中にある。
    """
    lo = min(factor[u.name] * u.var_cost for u in units)
    hi = max(
        factor[u.name] * (u.var_cost + 2.0 * u.quadratic * u.p_max_mw) for u in units
    )
    return float(lo), float(hi)


def _check_demand_range(units: Sequence[Unit], demand_mw: float) -> None:
    """需要が供給可能な範囲にあるかを二分法より **先に** 検査する。

    Raises
    ------
    ValueError
        需要が :math:`[\\sum P^{min}, \\sum P^{max}]` の外にあるとき。
    """
    if not math.isfinite(demand_mw):
        raise ValueError(f"需要が有限の値でない: {demand_mw!r}")

    p_min = sum(u.p_min_mw for u in units)
    p_max = sum(u.p_max_mw for u in units)
    names = ", ".join(u.name for u in units)

    if demand_mw > p_max:
        raise ValueError(
            f"需要 {demand_mw:.6g} MW が運転中号機の最大出力の合計 "
            f"{p_max:.6g} MW を上回っている。"
            f"供給力が {demand_mw - p_max:.6g} MW 不足している。\n"
            f"  運転中の号機: {names}\n"
            "  この状態で二分法を回すと、λ_hi に張り付いたまま区間幅だけが"
            "縮んで『収束したふり』をする。\n"
            "  号機を追加起動する（committed に加える）か、需要を下げること。"
        )
    if demand_mw < p_min:
        raise ValueError(
            f"需要 {demand_mw:.6g} MW が運転中号機の最低出力の合計 "
            f"{p_min:.6g} MW を下回っている。"
            f"最低出力が {p_min - demand_mw:.6g} MW 超過している。\n"
            f"  運転中の号機: {names}\n"
            "  この状態で二分法を回すと、λ_lo に張り付いたまま区間幅だけが"
            "縮んで『収束したふり』をする。\n"
            "  号機を解列する（committed から外す）か、揚水・出力抑制などの"
            "下げ調整を入れること。"
        )


def _lambda_dispatch(
    units: Sequence[Unit],
    demand_mw: float,
    *,
    factor: Mapping[str, float] | None = None,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> tuple[dict[str, float], float, int, bool, tuple[str, ...]]:
    """λ の二分法の中身。``(dispatch, lam, iterations, converged, marginal)``。

    :func:`economic_dispatch` と :func:`dispatch_with_losses` の共通部分で
    ある。需要の範囲検査は呼び出し側で済ませてあることを前提にする。
    """
    weights = {u.name: 1.0 for u in units} if factor is None else dict(factor)
    for unit in units:
        if weights.get(unit.name, 1.0) <= 0.0:
            raise ValueError(
                f"号機 {unit.name} のペナルティファクタが非正 "
                f"({weights.get(unit.name)})。1/(1 - ∂P_loss/∂P_i) の分母が"
                "負に落ちている。基準潮流解が壊れていないか確認すること。"
            )

    lo, hi = _bracket(units, weights)

    def total(lam: float) -> float:
        return sum(_output_at(u, lam, weights[u.name]) for u in units)

    iterations = 0
    for iterations in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        # 倍精度でこれ以上分割できなくなったら打ち切る。ブラケットは
        # 常に λ* を挟んだままなので、階段の段差もこの時点で確定する。
        if mid <= lo or mid >= hi:
            break
        if total(mid) < demand_mw:
            lo = mid
        else:
            hi = mid

    lam = 0.5 * (lo + hi)

    # --- λ から出力を決める。段差にいる号機だけ後回しにする -------------
    dispatch: dict[str, float] = {}
    flat: list[Unit] = []
    for unit in units:
        w = weights[unit.name]
        if unit.quadratic == 0.0 and lo <= w * unit.var_cost <= hi:
            # 増分費用が一定で、その値がちょうどブラケットの中にある。
            # λ = L b の区間ではどの出力も最適なので、残余で決める。
            flat.append(unit)
            continue
        dispatch[unit.name] = _output_at(unit, lam, w)

    residual = demand_mw - sum(dispatch.values())
    if flat:
        span = sum(u.p_max_mw - u.p_min_mw for u in flat)
        floor = sum(u.p_min_mw for u in flat)
        share = 0.0 if span <= 0.0 else (residual - floor) / span
        share = min(max(share, 0.0), 1.0)
        for unit in flat:
            dispatch[unit.name] = unit.p_min_mw + share * (unit.p_max_mw - unit.p_min_mw)

    served = sum(dispatch.values())
    scale = max(1.0, abs(demand_mw))
    converged = abs(served - demand_mw) <= max(tol, 1e-12) * scale

    marginal = tuple(
        unit.name
        for unit in units
        if _is_marginal(unit, dispatch[unit.name])
    )
    return dispatch, lam, iterations, converged, marginal


def _is_marginal(unit: Unit, p_mw: float) -> bool:
    """出力が上下限のどちらにも張り付いていないか。"""
    width = BOUND_TOLERANCE * max(1.0, abs(unit.p_max_mw))
    return (unit.p_min_mw + width) < p_mw < (unit.p_max_mw - width)


def economic_dispatch(
    case: Case,
    demand_mw: float,
    *,
    units: Sequence[Unit] | None = None,
    committed: Iterable[str] | Mapping[str, object] | None = None,
    tol: float = 1e-9,
    max_iter: int = 200,
) -> DispatchResult:
    """等増分燃料費（等 λ 法）による経済負荷配分。

    総費用 :math:`\\sum_i C_i(P_i)` を :math:`\\sum_i P_i = D` と
    :math:`P_i^{min} \\le P_i \\le P_i^{max}` のもとで最小化する。ラグランジュ
    関数の停留条件（KKT 条件）は

    .. math::

        \\frac{dC_i}{dP_i} = \\lambda \\quad (P_i^{min} < P_i < P_i^{max}), \\qquad
        \\frac{dC_i}{dP_i} \\le \\lambda \\quad (P_i = P_i^{max}), \\qquad
        \\frac{dC_i}{dP_i} \\ge \\lambda \\quad (P_i = P_i^{min})

    である。**上限に張り付いた号機の増分費用は λ 以下、下限に張り付いた
    号機の増分費用は λ 以上**であって、等号ではない。「等 λ 法」という
    名前から全号機の増分費用が等しくなると思い込むのが最も多い誤解で、
    等しくなるのは :attr:`DispatchResult.marginal_units` に入る号機だけで
    ある。

    解法は λ の二分法である。:math:`\\sum_i P_i(\\lambda)` が λ について
    単調非減少であることが根拠で、ブラケットは
    :math:`\\lambda_{lo} = \\min_i b_i`,
    :math:`\\lambda_{hi} = \\max_i (b_i + 2 c_i P_i^{max})` から取る。

    2 次係数がゼロの号機の扱い
    --------------------------
    :math:`c_i = 0`（線形費用）の号機があると :math:`P_i(\\lambda)` は
    :math:`\\lambda = b_i` で :math:`P^{min}` から :math:`P^{max}` へ跳ぶ
    **階段関数**になり、:math:`\\sum_i P_i(\\lambda)` は連続でなくなる。
    それでも単調非減少なので二分法は交点を挟み込めるが、収束した λ に
    おける出力は **一意に決まらない**（その λ では区間内のどの出力も同じ
    総費用を与える）。本実装は二分法が返すブラケット
    :math:`[\\lambda_{lo}, \\lambda_{hi}]` に :math:`b_i` が入っている号機を
    「段差にいる号機」とみなし、残余需要をその可動幅に比例して配分する。
    許容差との比較ではなくブラケットの包含で判定するので、λ の桁数に
    依存しない。配分の仕方は最適解の中での選び方であって、総費用は
    どの配分でも同じである。

    Parameters
    ----------
    case:
        系統ケース（``units`` 層が必要）。
    demand_mw:
        満たすべき需要 [MW]。送電損失は含まない
        （含めたい場合は :func:`dispatch_with_losses`）。
    units:
        号機を直接与える。``None`` なら :attr:`Case.units`。
    committed:
        起動している号機の名前。``None`` なら全機。
    tol:
        収束判定の相対許容差。:math:`|\\sum P - D| \\le tol \\cdot \\max(1, D)`
        を :attr:`DispatchResult.converged` の判定に使う。二分法そのものは
        倍精度でこれ以上分割できなくなるまで回すので、``tol`` を緩めても
        解の精度は落ちない。
    max_iter:
        二分法の上限回数。倍精度なら 60 回程度で機械精度に達する。

    Returns
    -------
    DispatchResult

    Raises
    ------
    ValueError
        需要が :math:`[\\sum P^{min}, \\sum P^{max}]` の外にあるとき。
        **この検査を二分法より先に置くことが本関数の要点である。**
        検査が無いと、交点の無い区間を二分し続けて区間幅だけが縮み、
        「収束した」と報告しながら :math:`\\sum P \\ne D` の答えを返す。

    Notes
    -----
    費用は 2 次関数のまま厳密に扱う。線形計画（:func:`dc_opf`、
    :func:`gridops.commitment.unit_commitment`）は 1 次費用か区分線形近似
    しか扱えないので、2 次費用の厳密解が要るときは必ずこちらを使う。

    Examples
    --------
    >>> from gridops import load_case
    >>> case = load_case("wscc9")
    >>> result = economic_dispatch(case, 315.0)
    >>> sorted(result.marginal_units)
    ['G2-1', 'G2-2']
    """
    pool = _select_units(case, units, committed)
    demand = float(demand_mw)
    _check_demand_range(pool, demand)

    dispatch, lam, iterations, converged, marginal = _lambda_dispatch(
        pool, demand, tol=tol, max_iter=max_iter
    )
    total_cost = float(sum(u.fuel_cost(dispatch[u.name]) for u in pool))
    return DispatchResult(
        dispatch=dispatch,
        lam=float(lam),
        total_cost=total_cost,
        iterations=iterations,
        converged=converged,
        marginal_units=marginal,
        penalty={u.name: 1.0 for u in pool},
        losses_mw=0.0,
        demand_mw=demand,
    )


# ======================================================================
# 送電損失（ペナルティファクタ）
# ======================================================================
def _injections(Y: np.ndarray, v_mag: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """複素注入電力 :math:`\\bar S = \\bar V (Y \\bar V)^{*}` [p.u.]。"""
    voltage = v_mag * np.exp(1j * theta)
    return voltage * np.conj(Y @ voltage)


def _state_from(case: Case, solution: object | None) -> tuple[np.ndarray, np.ndarray]:
    """潮流解らしきものから ``(|V|, θ[rad])`` を取り出す。

    :class:`gridops.powerflow.PowerFlowSolution`（``v`` と ``theta`` を
    持つ）、:class:`gridops.case.ReferenceSolution`（``v`` と ``angle_deg``
    を持つ）、複素電圧の配列のいずれでも受ける。``None`` ならケースの
    参照解を使う。潮流計算のモジュールに依存しないのは、ペナルティ
    ファクタの計算が「解がどう得られたか」に依存しないためである。
    """
    if solution is None:
        if case.reference is None:
            raise ValueError(
                f"ケース '{case.name}' に参照解がないので基準潮流を決められない。"
                "潮流計算の解を solution 引数で渡すこと。"
            )
        solution = case.reference

    if isinstance(solution, np.ndarray):
        voltage = np.asarray(solution, dtype=complex)
        return np.abs(voltage), np.angle(voltage)

    v_mag = getattr(solution, "v", None)
    if v_mag is None:
        raise TypeError(
            "solution から電圧を取り出せない。|V| を持つ 'v' 属性か、"
            "複素電圧の配列を渡すこと。"
        )
    theta = getattr(solution, "theta", None)
    if theta is None:
        angle_deg = getattr(solution, "angle_deg", None)
        if angle_deg is None:
            raise TypeError("solution が 'theta' も 'angle_deg' も持っていない。")
        theta = np.radians(np.asarray(angle_deg, dtype=float))

    v_mag = np.asarray(v_mag, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if v_mag.shape != (case.n_bus,) or theta.shape != (case.n_bus,):
        raise ValueError(
            f"潮流解の長さ {v_mag.shape} が母線数 {case.n_bus} と合わない。"
            "別のケースの解を渡していないか確認すること。"
        )
    return v_mag.copy(), theta.copy()


def _numeric_jacobian(func, x: np.ndarray, *, step: float = 1e-6) -> np.ndarray:
    """中心差分でヤコビアンを作る。

    解析形（極座標の 4 ブロック）は :func:`gridops.powerflow.jacobian_blocks`
    の仕事なので、ここでは書かない。ペナルティファクタに要るのは
    「基準解の近傍で方程式を解き直す」ことだけで、9 母線なら 14x14 の
    差分ヤコビアンでも一瞬である。**同じ式を 2 箇所に書かない**ほうが
    ドリフトの危険が小さい。
    """
    base = func(x)
    jac = np.zeros((base.size, x.size))
    for k in range(x.size):
        h = step * max(1.0, abs(x[k]))
        forward, backward = x.copy(), x.copy()
        forward[k] += h
        backward[k] -= h
        jac[:, k] = (func(forward) - func(backward)) / (2.0 * h)
    return jac


def _solve_state(
    case: Case,
    Y: np.ndarray,
    p_spec: np.ndarray,
    q_spec: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    v0: np.ndarray | None = None,
    theta0: np.ndarray | None = None,
    tol: float = 1e-11,
    max_iter: int = 40,
    context: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """指定注入に合う ``(|V|, θ)`` を Newton 法で求める。

    本モジュールに閉じた最小限の交流潮流である。:mod:`gridops.powerflow`
    の一般的な解法（初期値の選び方・Q 制限・複数手法の比較）とは目的が
    違い、ここでは **基準解のすぐ近くで方程式を解き直す**ことしかしない。
    ペナルティファクタは「注入を少し動かしたときに損失がどれだけ動くか」
    であって、その微分は基準解の近傍だけで決まるからである。

    Raises
    ------
    RuntimeError
        収束しなかったとき。
    """
    slack_idx, pv_idx, pq_idx = indices
    non_slack = np.array(
        [i for i in range(case.n_bus) if i not in set(slack_idx.tolist())], dtype=int
    )

    v_mag = np.ones(case.n_bus) if v0 is None else np.asarray(v0, dtype=float).copy()
    theta = np.zeros(case.n_bus) if theta0 is None else np.asarray(theta0, dtype=float).copy()
    n_theta = non_slack.size

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v_new = v_mag.copy()
        t_new = theta.copy()
        t_new[non_slack] = x[:n_theta]
        if pq_idx.size:
            v_new[pq_idx] = x[n_theta:]
        return v_new, t_new

    def residual(x: np.ndarray) -> np.ndarray:
        v_new, t_new = unpack(x)
        s = _injections(Y, v_new, t_new)
        return np.concatenate(
            [s.real[non_slack] - p_spec[non_slack], s.imag[pq_idx] - q_spec[pq_idx]]
        )

    x = np.concatenate([theta[non_slack], v_mag[pq_idx]])
    for _ in range(max_iter):
        g = residual(x)
        if np.max(np.abs(g)) < tol:
            break
        jac = _numeric_jacobian(residual, x)
        try:
            x = x - np.linalg.solve(jac, g)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"潮流のヤコビアンが特異になった{f'（{context}）' if context else ''}。"
                "系統が島に分かれていないか（gridops.ybus.islands）、"
                "電圧崩壊の近傍にいないかを確認すること。"
            ) from exc
    else:
        raise RuntimeError(
            f"損失計算のための潮流が収束しなかった"
            f"{f'（{context}）' if context else ''}。"
            f"最大ミスマッチ {np.max(np.abs(residual(x))):.3e} p.u.。"
            "需要が過大でないか、Case.check() が通るかを確認すること。"
        )

    return unpack(x)


def _itl_from_state(
    case: Case,
    Y: np.ndarray,
    v_mag: np.ndarray,
    theta: np.ndarray,
    indices: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    step: float = LOSS_DERIVATIVE_STEP,
) -> dict[int, float]:
    """基準状態から :math:`\\partial P_{loss}/\\partial P_i` を中心差分で求める。"""
    slack_idx, pv_idx, pq_idx = indices
    slack = int(slack_idx[0])

    s0 = _injections(Y, v_mag, theta)
    p_spec = s0.real.copy()
    q_spec = s0.imag.copy()

    def losses(p_vector: np.ndarray) -> float:
        v_new, t_new = _solve_state(
            case, Y, p_vector, q_spec, indices,
            v0=v_mag, theta0=theta, context="ペナルティファクタの数値微分",
        )
        return float(np.sum(_injections(Y, v_new, t_new).real))

    itl: dict[int, float] = {}
    for i, bus in enumerate(case.buses):
        if i == slack:
            # slack は損失の受け皿そのものであり、増分損失を測る基準点で
            # ある。基準点の増分損失は定義により 0（ペナルティ 1）になる。
            itl[bus.id] = 0.0
            continue
        plus, minus = p_spec.copy(), p_spec.copy()
        plus[i] += step
        minus[i] -= step
        itl[bus.id] = (losses(plus) - losses(minus)) / (2.0 * step)
    return itl


def penalty_factors(case: Case, solution: object | None = None) -> dict[int, float]:
    """ペナルティファクタ :math:`L_i = 1/(1 - \\partial P_{loss}/\\partial P_i)`。

    送電損失があると、等 λ 法の最適性条件は

    .. math:: L_i \\frac{dC_i}{dP_i} = \\lambda

    になる。負荷から遠い号機は自分の出力が損失を増やすので
    :math:`\\partial P_{loss}/\\partial P_i > 0`、したがって
    :math:`L_i > 1` となり、**実効的な増分費用が押し上げられて出力が
    減る**。これがペナルティファクタの意味である。

    増分損失は **slack 母線を基準として測る**。:math:`P_{loss}` は
    「母線 :math:`i` の注入を 1 増やし、その分 slack の注入を減らしたとき」
    の損失の変化率であり、slack 自身の増分損失は定義により 0、
    ペナルティファクタは 1 になる。基準の取り方を変えると個々の
    :math:`L_i` は変わるが、最適な出力の組は変わらない（λ が同じ比率で
    ずれるだけである）。

    Parameters
    ----------
    case:
        系統ケース。
    solution:
        基準となる潮流解。``v``/``theta`` または ``v``/``angle_deg`` を持つ
        オブジェクト、複素電圧の配列、``None``（ケースの参照解を使う）。

    Returns
    -------
    dict[int, float]
        母線番号 -> ペナルティファクタ。負荷母線も含む全母線を返す。

    Warns
    -----
    UserWarning
        分母 :math:`1 - \\partial P_{loss}/\\partial P_i` が
        :data:`PENALTY_POLE_TOLERANCE` より小さいとき。極に近づくと
        ペナルティファクタが発散し、その母線から 1 MW 送るために
        必要な発電が無限大に近づくことを意味する。重負荷で電圧が
        低い系統では実際に起こるので、黙って大きな値を返さない。

    Notes
    -----
    :math:`\\partial P_{loss}/\\partial P_i` は基準潮流解のまわりの
    **数値微分**（中心差分、刻み :data:`LOSS_DERIVATIVE_STEP`）で求める。
    B 係数法（:math:`P_{loss} = \\sum B_{ij} P_i P_j`）のような近似式を
    使わないのは、係数の同定そのものが別の近似であり、教材としては
    「損失を測って微分する」ほうが手続きが透明だからである。
    """
    case.require("network")
    v_mag, theta = _state_from(case, solution)
    Y = build_ybus(case)
    indices = case.type_indices()

    itl = _itl_from_state(case, Y, v_mag, theta, indices)

    factors: dict[int, float] = {}
    for bus_id, value in itl.items():
        denominator = 1.0 - value
        if abs(denominator) < PENALTY_POLE_TOLERANCE:
            warnings.warn(
                f"母線 {bus_id} のペナルティファクタが極に近い"
                f"（1 - ∂P_loss/∂P = {denominator:.3e}）。"
                "その母線から送電するために必要な発電が発散しつつある。"
                "負荷倍率を下げるか、基準潮流解が電圧崩壊の近傍にないかを"
                "確認すること。",
                UserWarning,
                stacklevel=2,
            )
        factors[bus_id] = 1.0 / denominator
    return factors


def dispatch_with_losses(
    case: Case,
    demand_mw: float,
    *,
    solution: object | None = None,
    committed: Iterable[str] | Mapping[str, object] | None = None,
    tol: float = 1e-6,
    max_iter: int = 50,
) -> DispatchResult:
    """送電損失を含む経済負荷配分。

    2 つの反復を入れ子にする。

    1. **内側**: ペナルティファクタ :math:`L_i` を固定して、
       :math:`L_i \\, dC_i/dP_i = \\lambda` と
       :math:`\\sum P_i = D + P_{loss}` を満たす λ を二分法で求める。
    2. **外側**: 得られた出力で交流潮流を解き直し、損失
       :math:`P_{loss}` とペナルティファクタを更新する。

    損失は出力の関数、出力は損失の関数なので、この不動点反復以外に
    素直な解き方がない。**λ 法そのものは内側にそのまま入っている**ことが
    要点で、損失を考えることで新しい最適性条件が生まれるのではなく、
    増分費用が :math:`L_i` 倍されるだけである。

    Parameters
    ----------
    case:
        系統ケース。
    demand_mw:
        **負荷の合計** [MW]。損失はここに含めない（含めるのは本関数である）。
    solution:
        初回のペナルティファクタを計算する潮流解。``None`` なら損失ゼロ・
        ペナルティ 1 から始める（外側の反復が自分で潮流を解き直す）。
    committed:
        起動している号機。
    tol:
        損失の収束判定（相対）。
    max_iter:
        外側の反復回数の上限。

    Returns
    -------
    DispatchResult
        :attr:`DispatchResult.losses_mw` に収束した損失、
        :attr:`DispatchResult.penalty` に号機ごとのペナルティファクタが入る。
        :attr:`DispatchResult.dispatch` の合計は **需要 + 損失**である。

    Raises
    ------
    ValueError
        需要（+損失）が供給可能な範囲の外にあるとき。
    RuntimeError
        途中の交流潮流が収束しなかったとき。

    Notes
    -----
    収束後も :attr:`DispatchResult.lam` は **系統 λ** であって、個々の
    号機の増分費用ではない。号機 :math:`i` の増分費用は
    :math:`\\lambda / L_i` である。損失を考えない場合に比べて λ が上がるのは
    「同じ負荷を賄うのに余分に発電しなければならない」ためであり、
    バグではない。
    """
    case.require("network", "units")
    pool = _select_units(case, None, committed)
    demand = float(demand_mw)
    _check_demand_range(pool, demand)

    Y = build_ybus(case)
    # 母線種別は「起動している号機がその母線にあるか」で決める。反復の
    # 途中で出力がたまたま 0 になっても種別が揺れないよう、committed から
    # 一度だけ決めて固定する。
    indices = case.type_indices({unit.name: 1.0 for unit in pool})

    # 初期値。参照解があればそこから、無ければ設定電圧の平坦開始から入る。
    # solution を明示的に渡された場合は、その誤りを黙って飲み込まない。
    factors = {unit.name: 1.0 for unit in pool}
    if solution is not None:
        v_start, theta_start = _state_from(case, solution)
        by_bus = penalty_factors(case, solution)
        factors = {unit.name: by_bus[unit.bus] for unit in pool}
    elif case.reference is not None:
        v_start, theta_start = _state_from(case, case.reference)
    else:
        v_start = np.array(
            [1.0 if bus.type is BusType.PQ else bus.v_set for bus in case.buses]
        )
        theta_start = np.zeros(case.n_bus)

    losses = 0.0
    dispatch: dict[str, float] = {}
    lam = 0.0
    marginal: tuple[str, ...] = ()
    converged = False
    iterations = 0

    for iterations in range(1, max_iter + 1):
        _check_demand_range(pool, demand + losses)
        dispatch, lam, _, inner_ok, marginal = _lambda_dispatch(
            pool, demand + losses, factor=factors, tol=1e-12, max_iter=200
        )

        p_spec, q_spec = case.bus_injection(dispatch)
        v_mag, theta = _solve_state(
            case, Y, p_spec, q_spec, indices,
            v0=v_start, theta0=theta_start,
            context=f"損失込み経済負荷配分 (需要 {demand:.1f} MW)",
        )
        v_start, theta_start = v_mag, theta

        new_losses = float(case.to_mw(np.sum(_injections(Y, v_mag, theta).real)))
        itl = _itl_from_state(case, Y, v_mag, theta, indices)
        factors = {}
        for unit in pool:
            denominator = 1.0 - itl[unit.bus]
            if abs(denominator) < PENALTY_POLE_TOLERANCE:
                warnings.warn(
                    f"母線 {unit.bus} のペナルティファクタが極に近い"
                    f"（1 - ∂P_loss/∂P = {denominator:.3e}）。",
                    UserWarning,
                    stacklevel=2,
                )
            factors[unit.name] = 1.0 / denominator

        if inner_ok and abs(new_losses - losses) <= tol * max(1.0, demand):
            losses = new_losses
            converged = True
            break
        losses = new_losses

    total_cost = float(sum(u.fuel_cost(dispatch[u.name]) for u in pool))
    return DispatchResult(
        dispatch=dispatch,
        lam=float(lam),
        total_cost=total_cost,
        iterations=iterations,
        converged=converged,
        marginal_units=marginal,
        penalty=factors,
        losses_mw=losses,
        demand_mw=demand,
    )


# ======================================================================
# 直流最適潮流
# ======================================================================
def _susceptance(case: Case) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """直流潮流の :math:`B'`、枝サセプタンス、接続行列、位相調整の注入を返す。

    枝 :math:`k` の潮流は :math:`f_k = b_k (\\theta_f - \\theta_t - \\phi_k)`、
    :math:`b_k = 1/(x_k \\tau_k)` で、母線注入は
    :math:`P = A^{T} f = B' \\theta - A^{T} \\mathrm{diag}(b)\\phi` になる。
    抵抗を無視するので損失はゼロであり、**直流潮流の世界に混雑レント以外の
    お金は存在しない**。

    :math:`b_k` と :math:`B'` は :mod:`gridops.dc` に委譲している（二重実装
    にすると片方だけタップの扱いを直したときに気づけないため）。位相調整の
    注入項 :math:`-A^{T}\\mathrm{diag}(b)\\phi` だけはここで組む。import を
    関数の中で行うのは、作図や経済負荷配分だけを使うときに直流潮流の層まで
    読み込ませないためである。
    """
    from .dc import _susceptances, susceptance_matrix

    A = incidence_matrix(case)
    b = _susceptances(case)
    phi = np.array([math.radians(branch.shift_deg) for branch in case.branches])
    B = susceptance_matrix(case)
    shift_injection = A.T @ (b * phi)
    return B, b, A, shift_injection


def _bus_loads(
    case: Case,
    *,
    demand_mw: float | None = None,
    loads: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """母線ごとの負荷 [MW] を決める。

    ``loads`` があればそれを使い、``demand_mw`` があればケースの負荷分布を
    保ったまま合計をその値に合わせる。どちらも無ければケースの
    :attr:`Bus.pd` をそのまま MW に直す。
    """
    if loads is not None:
        unknown = sorted(set(loads) - set(case.bus_ids))
        if unknown:
            raise ValueError(
                f"loads に存在しない母線番号がある: {unknown}。"
                f"ケース '{case.name}' の母線は {case.bus_ids}。"
            )
        return {bus.id: float(loads.get(bus.id, 0.0)) for bus in case.buses}

    base = {bus.id: float(case.to_mw(bus.pd)) for bus in case.buses}
    if demand_mw is None:
        return base

    total = sum(base.values())
    if total <= 0.0:
        raise ValueError(
            f"ケース '{case.name}' の基準負荷がゼロなので demand_mw を"
            "母線に配分できない。loads で母線ごとの負荷を直接渡すこと。"
        )
    scale = float(demand_mw) / total
    return {bus_id: value * scale for bus_id, value in base.items()}


@dataclass
class DCOPFResult:
    """直流最適潮流の結果。

    Parameters
    ----------
    dispatch:
        号機名 -> 出力 [MW]。
    theta:
        母線位相 [rad]。並びは :attr:`Case.buses` の順。
    flows:
        枝の有効電力 [p.u.]。並びは :attr:`Case.branches` の順で、
        from -> to の向きを正とする。**p.u. である**（費用は MW 基準なので、
        混雑レントを組むときは :attr:`Case.base_mva` を掛けること）。
    lmp:
        母線番号 -> 限界価格 [円/MWh]。母線ごとの注入等式の双対。
    congestion_price:
        枝 -> 混雑料金 [円/MWh]。熱容量制約の双対の **絶対値**（正で保持）。
        拘束していない枝は 0。
    total_cost:
        線形費用の合計 :math:`\\sum_i b_i P_i` [円/h]。無負荷費も 2 次項も
        含まない（:attr:`DispatchResult.total_cost` とは中身が違う）。
    """

    dispatch: dict[str, float]
    theta: np.ndarray
    flows: np.ndarray
    lmp: dict[int, float]
    congestion_price: dict[tuple[int, int], float]
    total_cost: float
    #: 以下は契約に無い追加フィールド（既定値付き）。混雑レントを 2 通りで
    #: 計算するには枝の両端と熱容量が要るので、結果に持たせておく。
    case: Case | None = None
    limit: str = "rate_a"
    loads_mw: dict[int, float] = field(default_factory=dict)
    status: str = "Optimal"
    seconds: float = 0.0

    # ------------------------------------------------------------------
    def congestion_rent(self, *, method: str = "price") -> float:
        """混雑レント（振替収支）[円/h]。

        送電系統の運用者が受け取る額、すなわち **負荷の支払いと発電の
        受取りの差**である。損失のない直流最適潮流では、この差はすべて
        混雑から生じる。

        2 通りの計算ができ、**両者は厳密に一致する**。

        ``method="price"`` : 価格差形式

        .. math::

            R = \\sum_k \\bigl(\\pi_{t(k)} - \\pi_{f(k)}\\bigr) f_k

        枝の両端の価格差に潮流を掛けて足す。「安い側で買って高い側で売る」
        差額そのものである。

        ``method="shadow"`` : 影値形式

        .. math:: R = \\sum_k |\\mu_k| \\, \\bar f_k

        拘束した枝の混雑料金に、その枝の熱容量を掛けて足す。混雑して
        いない枝は :math:`\\mu_k = 0` なので寄与しない。

        一致するのは、価格差が拘束した枝の影値から生じているからである
        （:math:`\\pi = \\pi_{ref} \\mathbf{1} - H^{T}\\mu` の関係）。
        **教材では必ず 2 通り計算して突き合わせること。** 片方だけを
        信じると、双対の符号を取り違えたまま気づかない。

        Parameters
        ----------
        method:
            ``"price"``（価格差形式）または ``"shadow"``（影値形式）。

        Returns
        -------
        float
            [円/h]。混雑が無ければ 0。
        """
        if self.case is None:
            raise ValueError(
                "混雑レントの計算には枝の情報が要るが case が入っていない。"
                "dc_opf が返した DCOPFResult をそのまま使うこと。"
            )
        base = self.case.base_mva
        if method == "price":
            total = 0.0
            for k, branch in enumerate(self.case.branches):
                delta = self.lmp[branch.to_bus] - self.lmp[branch.from_bus]
                total += delta * float(self.flows[k]) * base
            return float(total)
        if method == "shadow":
            total = 0.0
            for branch in self.case.branches:
                price = self.congestion_price.get(branch.key(), 0.0)
                if price == 0.0:
                    continue
                cap = getattr(branch, self.limit)
                total += price * float(cap) * base
            return float(total)
        raise ValueError(
            f"method は 'price' か 'shadow' のいずれかであること（渡された値: {method!r}）"
        )

    def is_congested(self, *, tolerance: float = CONGESTION_TOLERANCE) -> bool:
        """混雑している枝があるか。"""
        return any(abs(v) > tolerance for v in self.congestion_price.values())

    def summary(self) -> str:
        """人が読む要約（英語見出し。学生環境の日本語フォント欠如対策）。"""
        lines = [
            "DC optimal power flow",
            f"  total cost  : {self.total_cost:12.1f} 円/h",
            f"  congested   : {'yes' if self.is_congested() else 'no'}",
            f"  rent(price) : {self.congestion_rent(method='price'):12.1f} 円/h"
            if self.case is not None
            else "",
            "  bus      LMP [円/MWh]",
        ]
        for bus_id in sorted(self.lmp):
            lines.append(f"  {bus_id:>4d}   {self.lmp[bus_id]:12.2f}")
        if self.is_congested():
            lines.append("  congested branches")
            for key, price in sorted(self.congestion_price.items()):
                if abs(price) > CONGESTION_TOLERANCE:
                    lines.append(f"  {key[0]}-{key[1]}   mu = {price:12.2f} 円/MWh")
        return "\n".join(line for line in lines if line)


def dc_opf(
    case: Case,
    *,
    demand_mw: float | None = None,
    committed: Iterable[str] | Mapping[str, object] | None = None,
    limit: str = "rate_a",
    loads: Mapping[int, float] | None = None,
) -> DCOPFResult:
    """直流最適潮流（線形費用）。

    定式化は次のとおり（出力と負荷は MW、位相は rad）。

    .. math::

        \\min \\sum_i b_i p_i
        \\quad \\text{s.t.} \\quad
        \\sum_{i \\in b} p_i - S_{base}(B\\theta)_b = d_b \\;\\; (\\pi_b), \\qquad
        -\\bar f_k \\le f_k \\le \\bar f_k \\;\\; (\\mu_k^{\\pm}), \\qquad
        P_i^{min} \\le p_i \\le P_i^{max}

    **母線ごとに 1 本ずつ注入等式を書く**のが要点である。系統全体で 1 本の
    需給バランスにしてしまうと、双対が 1 つしか出ず、母線ごとの限界価格
    （LMP）が定義できない。等式は右辺に負荷を正の符号で置く向き
    （:mod:`gridops.solvers` の規約）なので、双対 :math:`\\pi_b` は
    :math:`\\partial(\\text{総費用})/\\partial d_b`、つまり LMP そのものになる。

    LMP の読み方
    ------------
    混雑が無ければ全母線の LMP は等しく、系統 λ に一致する。混雑すると
    分かれ、しかも **最も高い号機の限界費用を上回ることがある**。
    ループ系統で、ある母線の負荷を 1 MW 増やすために、混雑した枝を避けて
    「安い機を下げ、高い機を上げる」再給電が必要になる場合、1 MW あたりの
    費用増は最も高い号機の限界費用を超える。負の LMP も同様に起こりうる。
    **これはバグではなく、送電制約のある系統の価格の性質である。**

    Parameters
    ----------
    case:
        系統ケース（``network`` と ``units`` 層が必要）。
    demand_mw:
        系統全体の需要 [MW]。ケースの負荷分布を保ったまま合計を合わせる。
        ``loads`` を与えたときは無視される。
    committed:
        起動している号機。``None`` なら全機。
    limit:
        使う熱容量の名前（``"rate_a"`` / ``"rate_b"``）。無限大の枝には
        制約を付けない。
    loads:
        母線番号 -> 負荷 [MW]。母線ごとに直接指定する。LMP の意味を
        確かめる（ある母線の負荷だけ 1 MW 増やす）ときに使う。

    Returns
    -------
    DCOPFResult

    Raises
    ------
    ValueError
        実行不可能・非有界のとき（:func:`gridops.solvers.solve` が投げる）。
        ``limit`` が :class:`~gridops.case.Branch` に無い属性名のとき。

    Notes
    -----
    **費用は 1 次（:attr:`Unit.var_cost`）だけを使う。** PuLP と CBC は
    線形計画しか扱えないので 2 次項 :attr:`Unit.quadratic` は落ちる。
    2 次費用のまま解きたいときは :func:`economic_dispatch`（等 λ 法）を
    使う。区分線形近似で 2 次費用を線形計画に載せる話は
    :func:`gridops.commitment.unit_commitment` の担当である。

    直流近似なので **損失はゼロ**である。したがって
    :math:`\\sum p_i = \\sum d_b` が厳密に成り立ち、負荷の支払いと発電の
    受取りの差はすべて混雑レントになる。
    """
    case.require("network", "units")
    pool = _select_units(case, None, committed)
    if not hasattr(Branch, limit):
        raise ValueError(
            f"limit='{limit}' は Branch の属性にない。"
            "'rate_a'（常時許容容量）か 'rate_b'（緊急時許容容量）を指定すること。"
        )

    load_mw = _bus_loads(case, demand_mw=demand_mw, loads=loads)
    B, b_vec, A, shift = _susceptance(case)
    base = case.base_mva
    slack_idx, _, _ = case.type_indices()
    slack_bus = case.buses[int(slack_idx[0])].id

    problem = solvers.problem("dc_opf")

    p = {
        unit.name: solvers.variable(
            f"p_{unit.name}", unit.p_min_mw, unit.p_max_mw
        )
        for unit in pool
    }
    theta = {
        bus.id: solvers.variable(
            f"theta_{bus.id}",
            0.0 if bus.id == slack_bus else None,
            0.0 if bus.id == slack_bus else None,
        )
        for bus in case.buses
    }

    problem += solvers.lp_sum(unit.var_cost * p[unit.name] for unit in pool), "fuel_cost"

    # --- 母線ごとの注入等式。右辺に負荷を正で置く（双対 = LMP）----------
    for i, bus in enumerate(case.buses):
        generation = solvers.lp_sum(
            p[unit.name] for unit in pool if unit.bus == bus.id
        )
        network = solvers.lp_sum(
            base * B[i, j] * theta[case.buses[j].id]
            for j in range(case.n_bus)
            if B[i, j] != 0.0
        )
        constant = base * shift[i]
        problem += (
            generation - network + constant == load_mw[bus.id],
            f"balance-bus-{bus.id}",
        )

    # --- 線路の熱容量。双対は負で返るので、あとで符号を反転する ----------
    limited: list[int] = []
    for k, branch in enumerate(case.branches):
        cap = float(getattr(branch, limit))
        if not math.isfinite(cap):
            continue
        limited.append(k)
        flow = base * b_vec[k] * (theta[branch.from_bus] - theta[branch.to_bus])
        offset = base * b_vec[k] * math.radians(branch.shift_deg)
        problem += (flow - offset <= cap * base, f"flow-pos-{k}-{branch.from_bus}-{branch.to_bus}")
        problem += (-(flow - offset) <= cap * base, f"flow-neg-{k}-{branch.from_bus}-{branch.to_bus}")

    solution = solvers.solve(
        problem,
        context=f"直流最適潮流 (ケース '{case.name}', 負荷 {sum(load_mw.values()):.1f} MW)",
    )

    dispatch = {unit.name: float(solution.values[f"p_{unit.name}"]) for unit in pool}
    theta_value = np.array(
        [float(solution.values[f"theta_{bus.id}"]) for bus in case.buses]
    )
    flows = b_vec * (A @ theta_value - np.array(
        [math.radians(branch.shift_deg) for branch in case.branches]
    ))

    lmp = {
        bus.id: float(solution.duals[f"balance-bus-{bus.id}"]) for bus in case.buses
    }

    congestion: dict[tuple[int, int], float] = {branch.key(): 0.0 for branch in case.branches}
    for k in limited:
        branch = case.branches[k]
        pos = solution.duals.get(
            f"flow-pos-{k}-{branch.from_bus}-{branch.to_bus}", 0.0
        )
        neg = solution.duals.get(
            f"flow-neg-{k}-{branch.from_bus}-{branch.to_bus}", 0.0
        )
        # 最小化問題の '<=' 制約の双対は負である。混雑レントを正の量として
        # 扱うため、ここで符号を反転して足し合わせる（同時に両向きが
        # 拘束することはないので、実際にはどちらか一方だけが非ゼロ）。
        congestion[branch.key()] += -(pos + neg)

    total_cost = float(sum(unit.var_cost * dispatch[unit.name] for unit in pool))

    return DCOPFResult(
        dispatch=dispatch,
        theta=theta_value,
        flows=flows,
        lmp=lmp,
        congestion_price=congestion,
        total_cost=total_cost,
        case=case,
        limit=limit,
        loads_mw=dict(load_mw),
        status=solution.status,
        seconds=solution.seconds,
    )
