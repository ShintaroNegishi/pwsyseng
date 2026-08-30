"""電圧安定性 — P-V 曲線、ノーズ点、そして「収束しない」の読み方。

このモジュールが答えようとしている問いは 1 つだけである。

    **潮流計算が収束しなかったとき、それは解が無いからなのか、
    それともソルバの都合なのか。**

第 02 回で学生は「Newton は速い」を体験する。第 03 回でその同じ Newton が
負荷を増やしただけで止まる。そこで「反復回数を増やそう」「初期値を変えよう」
「tol を緩めよう」と手を動かし始めるのが自然な反応だが、負荷がノーズ点を
超えていれば **どれも効かない**。潮流方程式に実数解が存在しないからである。

この 2 つを区別する道具立てが本モジュールである。

=================================  ===================================
道具                               何が分かるか
=================================  ===================================
:func:`two_bus_nose`               2 母線なら限界は **閉形式で書ける**
:func:`pv_curve`                   多母線でも数値的に限界を挟める
:func:`min_singular_value`         限界に近づくとゼロに向かう連続量
:func:`voltage_sensitivity`        どの母線が無効電力に敏感か
=================================  ===================================

なぜ解析解から始めるのか
------------------------
本モジュールの負荷倍率追跡は「収束しなくなった点」を限界の下側推定とする簡易手続きであり、
予測子・修正子と弧長パラメータを用いる本格的な Continuation Power Flow とは異なる。
そのままでは **手続きが自分の答えを定義してしまっている**。答え合わせの
足場が要る。2 母線・無限大母線・力率一定という最小の系統なら限界は
4 次方程式の判別式から閉じた式で書けるので、これを基準に据える。
:func:`pv_curve` が :func:`two_bus_nose` と一致することを確かめて初めて、
9 母線の結果を信用してよい、という順序になっている。

「収束しない」と「解が無い」は違う
----------------------------------
本モジュールは **この 2 つを同一視していない**。同一視できないことを
むしろ主題にしている。

* ノーズ点を **超えた**倍率では解が本当に存在しない。どんなソルバでも解けない。
* ノーズ点の **手前**でも、初期値が悪ければ Newton は失敗しうる。
* そしてノーズ点の **近く**では、ヤコビアンが特異に近づくので
  この 2 つが数値的に見分けられなくなる。

したがって :func:`pv_curve` が返すノーズ点は「解が存在する倍率の上限」の
**下からの推定**である。二分法で挟み撃ちにするのは、この推定を限界に
近づけるためであって、限界を「発見」しているのではない。前の解を初期値に
使う（warm start）のも同じ理由で、初期値の悪さに由来する失敗を減らし、
残った失敗を「解が無い」ことの証拠に近づけるためである。

補助として :func:`min_singular_value` を用意してある。こちらは 0/1 の
判定ではなく **連続量**なので、限界にどれだけ近いかが読める。WSCC 9 母線
では負荷倍率 1.0 で 0.96、ノーズ点の直前で 1e-5 の桁まで落ちる。

発電はどう扱うか
----------------
:meth:`Case.scaled` は負荷だけを倍にし、参照解を落とす（電圧の解が無効に
なるため）。しかし参照解は :meth:`Case.bus_injection` が **発電**を読む
場所でもあるので、そのまま渡すと「発電ゼロの別系統」を解いてしまう。
本モジュールは倍率をかけたケースに元の参照解を付け直し、**発電を基準値に
据え置いたまま負荷だけを増やす**。増えた分と損失は slack 母線が引き受ける。
これは「slack が無限に強い」という仮定であり、実系統の限界（発電機の
無効電力上限で先に頭打ちになる）より楽観側であることに注意すること。
:func:`gridops.powerflow.solve` に ``enforce_q_limits=True`` を与えて
自分で掃引すると、ノーズ点がもっと手前に来る様子を見せられる。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .case import Case
from .powerflow import PowerFlowSolution, jacobian, jacobian_blocks, solve
from .ybus import build_ybus

__all__ = [
    "REFINE_RTOL",
    "REFINE_MAX_ITER",
    "PVCurve",
    "two_bus_nose",
    "two_bus_voltages",
    "pv_curve",
    "voltage_sensitivity",
    "min_singular_value",
]

#: 二分法でノーズ点を挟むときの、区間幅の相対許容差。
#:
#: 1e-9 まで詰めても電圧の相対誤差は 1e-5 の桁までしか下がらない。
#: ノーズ点では :math:`dV/d\\lambda \\to \\infty` で、電圧の誤差が倍率の
#: 誤差の **平方根**の速さでしか縮まないからである。倍率をこれ以上細かく
#: 詰めても意味がない、という判断がこの値の根拠である。
REFINE_RTOL = 1e-9

#: 二分法の最大反復回数。刻み 0.02 から相対 1e-9 まで詰めるのに 25 回、
#: 余裕を見て倍以上を取ってある。
REFINE_MAX_ITER = 60


# ======================================================================
# 2 母線の解析解 — 答え合わせの足場
# ======================================================================
def _phase(power_factor: float) -> tuple[float, float]:
    """力率から :math:`(\\cos\\phi, \\sin\\phi)` を作る。

    符号の規約: ``power_factor`` が **正なら遅れ**（誘導性、:math:`Q > 0`）、
    **負なら進み**（容量性、:math:`Q < 0`）である。大きさが
    :math:`\\cos\\phi` になる。0.95 と -0.95 はどちらも
    :math:`|\\cos\\phi| = 0.95` だが、前者は負荷が無効電力を消費し、
    後者は供給する。
    """
    magnitude = abs(float(power_factor))
    if magnitude == 0.0:
        raise ValueError(
            "power_factor がゼロ。有効電力を運ばない純無効負荷では "
            "P のノーズ点が定義できない（P_max = 0）。"
            "0 < |power_factor| <= 1 の値を与えること。"
        )
    if magnitude > 1.0:
        raise ValueError(
            f"power_factor={power_factor} の絶対値が 1 を超えている。"
            "力率は cos(phi) であって百分率ではない（95% なら 0.95）。"
        )
    cos_phi = magnitude
    sin_phi = math.copysign(math.sqrt(max(0.0, 1.0 - cos_phi**2)), float(power_factor))
    return cos_phi, sin_phi


def two_bus_nose(
    e: float, x: float, p_factor: float = 1.0, *, power_factor: float = 1.0
) -> tuple[float, float]:
    """2 母線系統のノーズ点の **解析解**。

    無限大母線（電圧 :math:`E` 固定）とリアクタンス :math:`X` だけの
    線路の先に、力率一定の負荷 :math:`S = P + jQ` をつないだ系統を考える。
    負荷母線の電圧 :math:`|V|` は 4 次方程式

    .. math::

        |V|^4 + (2QX - E^2)\\,|V|^2 + X^2 (P^2 + Q^2) = 0

    を満たす。:math:`u = |V|^2` の 2 次方程式と見れば、実数解が存在する
    条件は判別式

    .. math::

        D = (2QX - E^2)^2 - 4 X^2 (P^2 + Q^2) \\ge 0

    である。**ノーズ点とはこの判別式がちょうどゼロになる点**であり、
    そこで上枝（運用解）と下枝（低電圧解）が合流する。

    :math:`Q = P \\tan\\phi` を代入して :math:`D = 0` を :math:`P` について
    解くと（正の根を取る）

    .. math::

        P_{max} = \\frac{E^2}{2X} \\cdot \\frac{1 - \\sin\\phi}{\\cos\\phi},
        \\qquad
        |V|_{crit} = \\frac{E}{\\sqrt{2 (1 + \\sin\\phi)}}

    が得られる。力率 1（:math:`\\phi = 0`）なら
    :math:`P_{max} = E^2/(2X)`、:math:`|V|_{crit} = E/\\sqrt{2}` という
    教科書の値になる。

    Parameters
    ----------
    e:
        無限大母線の電圧 :math:`E` [p.u.]。
    x:
        線路の直列リアクタンス :math:`X` [p.u.]（抵抗はゼロとする）。
    p_factor:
        **負荷倍率 1.0 に対応する有効電力** :math:`P_0` [p.u.]。戻り値の
        第 1 要素はこれで割った **倍率**になる。既定の 1.0 では
        :math:`P_{max}` そのものが返る。:func:`pv_curve` の返す倍率と
        直接比べたいときに、その基準負荷を渡す。
    power_factor:
        負荷の力率 :math:`\\cos\\phi`。**正なら遅れ、負なら進み**。
        大きさが 1 以下でなければならない。

    Returns
    -------
    tuple of float
        ``(倍率, 臨界電圧)``。倍率は :math:`P_{max} / P_0`、臨界電圧は
        :math:`|V|_{crit}` [p.u.]。

    Raises
    ------
    ValueError
        ``e`` または ``x`` が非正のとき、``p_factor`` が非正のとき、
        ``power_factor`` がゼロまたは絶対値が 1 を超えるとき。

    Notes
    -----
    **進み力率のほうがノーズが伸びる。** :math:`\\sin\\phi < 0` なので
    :math:`(1 - \\sin\\phi)` が大きくなるからである。これが調相設備の
    効能そのものであり、:attr:`Bus.bs` を足すと WSCC 9 母線のノーズ点が
    実際に伸びる（テストで固定してある）。逆に遅れ力率の負荷は自分で
    電圧を下げに行くので限界が手前に来る。

    この式は **抵抗をゼロと仮定している**。抵抗があると
    :math:`P_{max}` は少し下がる。9 母線ケースの答え合わせに 2 母線の
    式をそのまま持ち込んではいけない理由がこれで、本モジュールでは
    :math:`r = 0` の 2 母線ケースを別に組み立てて突き合わせている。

    Examples
    --------
    >>> two_bus_nose(1.0, 0.1)
    (5.0, 0.7071067811865475)
    >>> factor, v = two_bus_nose(1.0, 0.1, 0.5, power_factor=-0.95)
    >>> round(factor, 6), round(v, 6)
    (13.813157, 0.852648)
    """
    e = float(e)
    x = float(x)
    p_factor = float(p_factor)
    if e <= 0.0:
        raise ValueError(f"e={e} が非正。無限大母線の電圧は正でなければならない。")
    if x <= 0.0:
        raise ValueError(
            f"x={x} が非正。この解析解は誘導性リアクタンスだけの線路を前提にしている。"
        )
    if p_factor <= 0.0:
        raise ValueError(
            f"p_factor={p_factor} が非正。倍率 1.0 に対応する基準負荷 [p.u.] を"
            "正の値で与えること（既定の 1.0 なら P_max がそのまま返る）。"
        )
    cos_phi, sin_phi = _phase(power_factor)
    p_max = e**2 * (1.0 - sin_phi) / (2.0 * x * cos_phi)
    v_crit = e / math.sqrt(2.0 * (1.0 + sin_phi))
    return p_max / p_factor, v_crit


def two_bus_voltages(
    e: float, x: float, p: float, *, power_factor: float = 1.0
) -> tuple[float, float]:
    """2 母線系統の負荷母線電圧の 2 つの解（上枝と下枝）。

    :func:`two_bus_nose` と同じ 4 次方程式

    .. math:: u^2 + (2QX - E^2) u + X^2 (P^2 + Q^2) = 0, \\qquad u = |V|^2

    を解いて :math:`\\sqrt{u}` を返す。**P-V 曲線そのものの厳密解**で
    あり、:func:`pv_curve` が描く数値解に重ねると継続法の答え合わせが
    目で見える形になる。

    Parameters
    ----------
    e, x:
        無限大母線の電圧 [p.u.] とリアクタンス [p.u.]。
    p:
        負荷の有効電力 [p.u.]。
    power_factor:
        力率。正なら遅れ、負なら進み（:func:`two_bus_nose` と同じ規約）。

    Returns
    -------
    tuple of float
        ``(上枝の |V|, 下枝の |V|)`` [p.u.]。判別式が負（ノーズ点を
        超えた負荷）なら ``(nan, nan)`` を返す。**例外にはしない。**
        「その負荷では解が存在しない」ことは計算の失敗ではなく答えの
        一部だからである。

    Notes
    -----
    上枝と下枝はノーズ点で合流する。運用点は上枝にあり、下枝は
    同じ電力を **低い電圧・大きい電流**で運ぶ解で、実際には不安定である。
    Newton がノーズ点の近くで下枝へ飛び移ることがあるのは、この 2 つの
    解が近づくためで、:func:`pv_curve` が前の解を初期値に使うのは
    飛び移りを減らすためでもある。
    """
    cos_phi, sin_phi = _phase(power_factor)
    p = float(p)
    q = p * sin_phi / cos_phi
    b = 2.0 * q * float(x) - float(e) ** 2
    c = float(x) ** 2 * (p**2 + q**2)
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return (math.nan, math.nan)
    root = math.sqrt(disc)
    upper = (-b + root) / 2.0
    lower = (-b - root) / 2.0
    return (math.sqrt(max(0.0, upper)), math.sqrt(max(0.0, lower)))


# ======================================================================
# P-V 曲線
# ======================================================================
@dataclass
class PVCurve:
    """負荷倍率を掃引して得た P-V 曲線。

    Parameters
    ----------
    factors:
        試した負荷倍率（昇順）。基準ケースの 1.0 から始まる。
    voltages:
        ``(n_point, n_bus)`` の電圧の大きさ [p.u.]。列の並びは
        :attr:`Case.buses` の順。収束しなかった点の行は ``nan`` である。
    converged:
        各点で潮流が収束したか。``nan`` の行と対応する。
    critical_index:
        収束した点のうち **倍率が最大**のものの添字。これがノーズ点の
        推定であり、真のノーズ点を **下から**押さえた値である。
    case:
        掃引に使った基準ケース（倍率 1.0 のもの）。母線番号から列の
        添字を引くために持っている。
    scaled_buses:
        倍率をかけた母線の番号。``None`` なら全母線。
    min_singular_values:
        各点での潮流ヤコビアンの最小特異値。収束しなかった点は ``nan``。
        ノーズ点に向かって単調に減り、ゼロに近づく。
    iterations:
        各点の Newton の反復回数。収束しなかった点は ``-1``。

    Notes
    -----
    ``critical_index`` の点は「解が存在すると **確かめられた**最大の
    倍率」であって、「解が存在する最大の倍率」ではない。両者の差は
    二分法の刻み（:data:`REFINE_RTOL`）以下だが、原理的にゼロにはできない。
    この区別が本モジュールの主題である。
    """

    factors: np.ndarray
    voltages: np.ndarray
    converged: np.ndarray
    critical_index: int
    case: Case | None = None
    scaled_buses: tuple[int, ...] | None = None
    min_singular_values: np.ndarray | None = None
    iterations: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.factors = np.asarray(self.factors, dtype=float)
        self.voltages = np.atleast_2d(np.asarray(self.voltages, dtype=float))
        self.converged = np.asarray(self.converged, dtype=bool)
        self.critical_index = int(self.critical_index)

    # ------------------------------------------------------------------
    def _column(self, bus_id: int) -> int:
        """母線番号から :attr:`voltages` の列番号を引く。"""
        if self.case is None:
            raise ValueError(
                "この PVCurve は case を持っていないので母線番号から列を引けない。"
                "voltages の列は Case.buses の順に並んでいるので、"
                "case.index_of(bus_id) で自分で添字を作ること。"
            )
        return self.case.index_of(bus_id)

    def voltage_of(self, bus_id: int) -> np.ndarray:
        """指定した母線の電圧の列 [p.u.]（点の並びは :attr:`factors` と同じ）。"""
        return self.voltages[:, self._column(bus_id)]

    def nose(self, bus_id: int) -> tuple[float, float]:
        """ノーズ点の ``(負荷倍率, その母線の電圧 [p.u.])``。

        Notes
        -----
        返る倍率は母線によらず同じ（系統全体の限界だから）で、電圧だけが
        母線ごとに違う。曲線の折り返し点であって「電圧が下限を割る点」では
        **ない**。運用上の限界は下限電圧 :attr:`Bus.v_min` で先に来るのが
        普通で、ノーズ点はその外側にある物理的な限界である。
        """
        index = self.critical_index
        return float(self.factors[index]), float(self.voltages[index, self._column(bus_id)])

    @property
    def loading_margin(self) -> float:
        """負荷余裕（ノーズ点の倍率 - 1）。

        基準負荷の何倍まで余裕があるかを表す無次元量である。0.5 なら
        「基準の 50% 増しまで解が存在する」の意味になる。MW で言いたい
        ときは基準の総負荷を掛けること。
        """
        return float(self.factors[self.critical_index]) - 1.0

    def critical_bus(self) -> tuple[int, float]:
        """ノーズ点で電圧が最も低い ``(母線番号, |V|)``。

        電圧崩壊が「どこから始まるか」を指す量。WSCC 9 母線では負荷の
        最も大きい母線 5 になる。
        """
        if self.case is None:
            raise ValueError("case を持っていない PVCurve では母線番号を返せない。")
        row = self.voltages[self.critical_index]
        index = int(np.argmin(row))
        return self.case.buses[index].id, float(row[index])

    def summary(self) -> str:
        """日本語の要約。"""
        n_ok = int(np.count_nonzero(self.converged))
        lines = [
            f"P-V 曲線: {self.factors.size} 点（収束 {n_ok} 点）",
            f"  ノーズ点の負荷倍率 : {self.factors[self.critical_index]:.6f}",
            f"  負荷余裕           : {self.loading_margin:.6f}"
            + (
                f" (= 基準負荷の {100 * self.loading_margin:.1f}% 増)"
                if self.loading_margin >= 0
                else ""
            ),
        ]
        if self.case is not None:
            bus_id, v = self.critical_bus()
            lines.append(f"  最低電圧の母線     : {bus_id} ({v:.4f} p.u.)")
        if self.min_singular_values is not None:
            sigma = self.min_singular_values[self.critical_index]
            lines.append(f"  最小特異値         : {sigma:.3e}（ノーズ点でゼロに向かう）")
        if bool(self.converged.all()):
            lines.append(
                "  注意: 掃引した範囲では最後まで収束した。max_factor を"
                "上げないとノーズ点は挟めていない。"
            )
        return "\n".join(lines)


# ----------------------------------------------------------------------
def _scaled_case(case: Case, factor: float, buses: Iterable[int] | None) -> Case:
    """負荷を ``factor`` 倍し、**発電は基準値に据え置いた**ケースを作る。

    :meth:`Case.scaled` は既定で ``reference=None`` にする（倍率をかけた系統
    では参照解の電圧が答えではなくなるので当然の処置である）。ところが
    :meth:`Case.bus_injection` は参照解から **発電**も読むので、そのまま
    渡すと発電ゼロの別系統を解いてしまう。``keep_generation=True`` は
    発電だけを引き継ぐための明示的な指定である。参照解の電圧は使わない。
    """
    return case.scaled(factor, buses=buses, keep_generation=True)


def _solve_point(
    case: Case,
    factor: float,
    buses: Iterable[int] | None,
    warm: PowerFlowSolution | None,
    tol: float,
    max_iter: int | None,
) -> PowerFlowSolution | None:
    """1 点だけ解く。収束しなければ ``None`` を返す（例外にしない）。"""
    scaled = _scaled_case(case, factor, buses)
    try:
        return solve(
            scaled,
            tol=tol,
            max_iter=max_iter,
            v0=None if warm is None else warm.v,
            theta0=None if warm is None else warm.theta,
        )
    except RuntimeError:
        # ノーズ点の外側では解が存在しないので、収束しないのが正しい。
        # ここで例外を握りつぶすのは「失敗そのものが測定値」だからである。
        return None


def pv_curve(
    case: Case,
    *,
    buses: Sequence[int] | None = None,
    step: float = 0.02,
    max_factor: float = 4.0,
    refine: bool = True,
) -> PVCurve:
    """負荷倍率を上げながら潮流を解き、ノーズ点を挟み撃ちする。

    手順は 3 段である。

    1. 倍率 1.0 から ``step`` 刻みで負荷を増やし、そのつど潮流を解く。
       **前の点の解を初期値に使う**（warm start）。曲線に沿って初期値が
       動くので、フラットスタートより深くまで追える。
    2. 初めて収束しなかった倍率を見つけたら、そこで刻み進みを止める。
    3. ``refine=True`` なら、最後に収束した倍率と最初に失敗した倍率の
       あいだを **二分法**で挟み、区間幅が相対 :data:`REFINE_RTOL` を
       切るまで詰める。

    Parameters
    ----------
    case:
        基準ケース。倍率 1.0 でまず解けなければならない。
    buses:
        倍率をかける母線の番号。``None`` なら全母線（総負荷を一律に
        増やす）。特定の母線だけを重くしたいときに使う。
    step:
        倍率の刻み。既定の 0.02 は「基準負荷の 2% 刻み」。
    max_factor:
        掃引の上限。ここまで収束し続けた場合は ``UserWarning`` を出し、
        上限の点をノーズ点として返す（**ノーズ点は挟めていない**）。
    refine:
        二分法による細かい挟み撃ちを行うか。``False`` なら刻みの精度
        （最悪 ``step``）でしかノーズ点が決まらない。

    Returns
    -------
    PVCurve
        掃引の全記録。

    Raises
    ------
    ValueError
        ``step`` が非正、``max_factor`` が 1 以下、または **基準ケース
        （倍率 1.0）が解けない**とき。基準が解けないのはノーズ点の話
        ではなくデータの問題なので、区別して止める。

    Notes
    -----
    **「収束しない」と「解が無い」は違う。** 本関数はこの 2 つを
    区別できていない。区別できないことを承知のうえで、次の 2 つの手当てで
    「収束しない = 解が無い」に近づけている。

    * warm start（前の解を初期値にする）で、初期値の悪さによる失敗を減らす
    * 二分法で、失敗した倍率と成功した倍率の距離を詰める

    それでも残る誤差は **必ず安全側**（真のノーズ点より手前）である。
    ノーズ点のごく近くではヤコビアンが特異に近づき、Newton の修正量が
    発散するか、上枝から下枝へ飛び移る。前者は失敗として、後者は
    「電圧が急に落ちた点」として現れる。:attr:`PVCurve.min_singular_values`
    を一緒に見ると、判定が 0/1 ではなく連続量として読める。

    2 母線ケースでは本関数の結果が :func:`two_bus_nose` の解析解と
    相対 1e-9 の水準で一致する（倍率について。電圧は 1e-5 の水準で、
    これは :math:`dV/d\\lambda \\to \\infty` の帰結である）。この一致が
    9 母線の結果を信用してよい根拠になっている。

    **発電の扱い**については本モジュールの docstring を読むこと。
    負荷だけが増え、増分と損失は slack 母線が引き受ける。発電機の無効
    電力上限は既定では効かない（``enforce_q_limits`` を使っていない）ので、
    ここで得られる余裕は実系統より **楽観側**である。

    Examples
    --------
    >>> from gridops import load_case
    >>> from gridops.voltage import pv_curve
    >>> curve = pv_curve(load_case("wscc9"))       # doctest: +SKIP
    >>> round(curve.loading_margin, 3)             # doctest: +SKIP
    1.374
    """
    if step <= 0.0:
        raise ValueError(f"step={step} が非正。負荷倍率の刻みは正でなければならない。")
    if max_factor <= 1.0:
        raise ValueError(
            f"max_factor={max_factor} が 1 以下。掃引は基準ケース（倍率 1.0）から"
            "始まるので、上限は 1 より大きくなければならない。"
        )

    bus_list = None if buses is None else tuple(int(b) for b in buses)
    tol = 1e-10
    max_iter = None

    factors: list[float] = []
    rows: list[np.ndarray] = []
    flags: list[bool] = []
    sigmas: list[float] = []
    iters: list[int] = []
    nan_row = np.full(case.n_bus, np.nan)

    def record(factor: float, solution: PowerFlowSolution | None) -> None:
        factors.append(float(factor))
        if solution is None:
            rows.append(nan_row.copy())
            flags.append(False)
            sigmas.append(math.nan)
            iters.append(-1)
        else:
            rows.append(np.asarray(solution.v, dtype=float).copy())
            flags.append(True)
            sigmas.append(min_singular_value(solution.case, solution))
            iters.append(int(solution.iterations))

    # --- 1. 基準ケース ------------------------------------------------
    try:
        base = solve(_scaled_case(case, 1.0, bus_list), tol=tol)
    except RuntimeError as error:
        raise ValueError(
            f"ケース '{case.name}' は倍率 1.0（基準負荷）で既に潮流が解けない。"
            "これはノーズ点の話ではなくデータの問題である。"
            "Case.check() を通し、参照解（発電）が付いているかを確かめること。"
            "Case.scaled が参照解を落とすので、自分で倍率をかけたケースを"
            "渡している場合は発電がゼロになっていないか疑うこと。"
        ) from error
    record(1.0, base)
    warm = base

    # --- 2. 刻み進み --------------------------------------------------
    last_ok_factor = 1.0
    first_bad_factor: float | None = None
    n_step = 1
    while True:
        factor = 1.0 + n_step * step
        if factor > max_factor + 1e-12:
            break
        solution = _solve_point(case, factor, bus_list, warm, tol, max_iter)
        record(factor, solution)
        if solution is None:
            first_bad_factor = factor
            break
        warm = solution
        last_ok_factor = factor
        n_step += 1

    # --- 3. 二分法 ----------------------------------------------------
    if first_bad_factor is None:
        warnings.warn(
            f"負荷倍率 {max_factor} まで潮流が解け続けたので、ノーズ点を"
            "挟めていない。max_factor を上げて掃引し直すこと。"
            "返す critical_index は掃引の最後の点であって、ノーズ点ではない。",
            UserWarning,
            stacklevel=2,
        )
    elif refine:
        low, high = last_ok_factor, first_bad_factor
        for _ in range(REFINE_MAX_ITER):
            if high - low <= REFINE_RTOL * max(1.0, low):
                break
            middle = 0.5 * (low + high)
            solution = _solve_point(case, middle, bus_list, warm, tol, max_iter)
            record(middle, solution)
            if solution is None:
                high = middle
            else:
                low = middle
                warm = solution

    order = np.argsort(np.asarray(factors, dtype=float), kind="stable")
    factor_array = np.asarray(factors, dtype=float)[order]
    voltage_array = np.asarray(rows, dtype=float)[order]
    converged_array = np.asarray(flags, dtype=bool)[order]
    sigma_array = np.asarray(sigmas, dtype=float)[order]
    iter_array = np.asarray(iters, dtype=int)[order]

    ok = np.flatnonzero(converged_array)
    critical = int(ok[int(np.argmax(factor_array[ok]))])

    return PVCurve(
        factors=factor_array,
        voltages=voltage_array,
        converged=converged_array,
        critical_index=critical,
        case=case,
        scaled_buses=bus_list,
        min_singular_values=sigma_array,
        iterations=iter_array,
    )


# ======================================================================
# 感度と余裕の指標
# ======================================================================
def _reduced_jacobian(
    case: Case, solution: PowerFlowSolution
) -> tuple[np.ndarray, np.ndarray]:
    """縮約した V-Q ヤコビアンと、PQ 母線の電圧を返す。

    :math:`J = [[H, N], [M, L]]` に対して :math:`\\Delta P = 0` を課すと

    .. math::

        \\Delta\\theta = -H^{-1} N \\frac{\\Delta |V|}{|V|}, \\qquad
        (L - M H^{-1} N) \\frac{\\Delta |V|}{|V|} = \\Delta Q

    となる。この :math:`J_R = L - M H^{-1} N`（L ブロックの Schur 補元）が
    「有効電力の方程式を消去したあとの L ブロック」である。
    """
    Y = build_ybus(case)
    H, N, M, L = jacobian_blocks(case, Y, solution.v, solution.theta)
    if L.size == 0:
        return np.zeros((0, 0)), np.zeros(0)
    try:
        reduced = L - M @ np.linalg.solve(H, N)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "有効電力のヤコビアン H が特異で、V-Q の縮約ができない。"
            "系統が島に分かれていないか（gridops.ybus.islands）、"
            "ノーズ点を超えていないか（gridops.voltage.pv_curve）を確かめること。"
        ) from error
    _, _, pq = case.type_indices(solution.dispatch)
    return reduced, np.asarray(solution.v, dtype=float)[pq]


def voltage_sensitivity(case: Case, solution: PowerFlowSolution) -> np.ndarray:
    """PQ 母線の電圧-無効電力感度 :math:`\\partial |V| / \\partial Q` [p.u./p.u.]。

    .. math::

        \\frac{\\partial |V|}{\\partial Q}
        = \\mathrm{diag}(|V|) \\, \\bigl(L - M H^{-1} N\\bigr)^{-1}

    Parameters
    ----------
    case:
        系統ケース。母線種別と Ybus の出どころ。
    solution:
        その点の潮流解（:func:`gridops.powerflow.solve` の返り値）。

    Returns
    -------
    numpy.ndarray
        ``(n_PQ, n_PQ)`` の行列。行が「電圧が動く母線」、列が
        「無効電力を注入する母線」である。並びはどちらも
        :meth:`Case.type_indices` の ``pq``（= :attr:`Case.buses` の順の
        うち PQ 母線）と同じ。母線番号は
        ``[case.buses[i].id for i in case.type_indices()[2]]`` で得られる。

    Raises
    ------
    ValueError
        H ブロックが特異で縮約できないとき。

    Notes
    -----
    **「L ブロックの逆行列」ではなく「縮約した L ブロックの逆行列」である。**
    素の :math:`L^{-1}` は :math:`\\Delta P = 0` ではなく
    :math:`\\Delta\\theta = 0` を課したことになり、有効電力の方程式を
    無視した減結合近似になってしまう。WSCC 9 母線の基準解では両者が
    最大 12% 違い、数値微分（無効電力を少し変えて潮流を解き直す）と
    一致するのは **縮約したほう**である（テストで両方向を固定してある）。
    減結合近似のほうを見たければ

    >>> from gridops.powerflow import jacobian_blocks     # doctest: +SKIP
    >>> H, N, M, L = jacobian_blocks(case, Y, s.v, s.theta)   # doctest: +SKIP
    >>> naive = np.diag(s.v[pq]) @ np.linalg.inv(L)           # doctest: +SKIP

    と書けるので、両者の差を測らせると Fast Decoupled 法の前提
    （:math:`|N|, |M| \\ll |H|, |L|`）が「小さいがゼロではない」ことが
    数字で見える。

    :math:`\\mathrm{diag}(|V|)` を左から掛けているのは、ヤコビアンの
    :math:`N, L` が :math:`\\partial / \\partial |V| \\cdot |V|` の形
    （修正量が :math:`\\Delta |V| / |V|` の無次元量）で定義されているので、
    その正規化を戻すためである。これを忘れると電圧の高い母線で数 % ずれる。

    対角成分が大きい母線ほど「無効電力を入れれば電圧が上がりやすい =
    無効電力が足りていない」母線である。ノーズ点に近づくと縮約ヤコビアンが
    特異に近づくので、感度は **発散**する。感度の逆数（:math:`dQ/dV`）を
    余裕の指標に使う流儀もあるが、本モジュールでは連続量として
    :func:`min_singular_value` を推している。

    Examples
    --------
    >>> from gridops import load_case                     # doctest: +SKIP
    >>> from gridops.powerflow import solve               # doctest: +SKIP
    >>> case = load_case("wscc9")                         # doctest: +SKIP
    >>> s = voltage_sensitivity(case, solve(case))        # doctest: +SKIP
    >>> s.shape                                           # doctest: +SKIP
    (6, 6)
    """
    reduced, v_pq = _reduced_jacobian(case, solution)
    if reduced.size == 0:
        return np.zeros((0, 0))
    try:
        inverse = np.linalg.inv(reduced)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "縮約した V-Q ヤコビアンが特異で逆行列が作れない。"
            "この点がノーズ点そのものである可能性が高い"
            "（gridops.voltage.min_singular_value で確かめられる）。"
        ) from error
    return np.diag(v_pq) @ inverse


def min_singular_value(case: Case, solution: PowerFlowSolution) -> float:
    """潮流ヤコビアンの最小特異値。電圧安定余裕の連続的な指標。

    Parameters
    ----------
    case:
        系統ケース。**こちらが正**で、Ybus と母線種別はここから作る
        （``solution.case`` とは別物でもよい）。
    solution:
        その点の潮流解。

    Returns
    -------
    float
        :math:`\\sigma_{min}(J)`。:math:`J` は
        :func:`gridops.powerflow.jacobian` が返す
        :math:`[[H, N], [M, L]]`。

    Notes
    -----
    ノーズ点ではヤコビアンが特異になる。すなわち最小特異値がゼロになる。
    その手前では正の値を取り、限界に近づくほど小さくなるので、
    **「解けた／解けない」の 0/1 判定ではなく、どれだけ近いかが読める**。
    WSCC 9 母線では負荷倍率 1.0 で 0.96、2.0 で 0.67、2.2 で 0.49、
    ノーズ点（2.374）の直前で 1e-5 の桁まで落ちる。

    **絶対値そのものには意味がない。** ヤコビアンは
    :math:`\\Delta |V| / |V|` を未知数に取る規約で組んであり、基準容量
    （100 MVA）にも依存する。比較してよいのは **同じケースの同じ規約で
    測った値どうし**であり、系統をまたいで「0.3 だから危ない」とは
    言えない。この点は文献でも繰り返し注意されている。

    最小特異値は 1 本の曲線に沿って単調に減るが、母線種別が切り替わる
    （PV 母線が Q 上限に達して PQ になる）と行列の大きさが変わるので
    **不連続に飛ぶ**。``enforce_q_limits=True`` で掃引するときは
    この飛びを見落とさないこと。

    Examples
    --------
    >>> from gridops import load_case                       # doctest: +SKIP
    >>> from gridops.powerflow import solve                 # doctest: +SKIP
    >>> case = load_case("wscc9")                           # doctest: +SKIP
    >>> round(min_singular_value(case, solve(case)), 4)     # doctest: +SKIP
    0.9614
    """
    matrix = jacobian(
        case,
        build_ybus(case),
        solution.v,
        solution.theta,
        dispatch=solution.dispatch,
    )
    if matrix.size == 0:
        return 0.0
    return float(np.linalg.svd(matrix, compute_uv=False)[-1])
