"""発電設備のアデカシー（供給信頼度）評価。

「設備は足りているか」を確率で答えるのが本モジュールの仕事である。決定論的
な設備予備率（設備容量 / 最大需要 - 1）は台数と停止率の情報を捨ててしまう。
WSCC 9 母線の教材ケースは設備容量 460 MW / 最大需要 315 MW で予備率 46% だが、
90 MW 機が 2 台同時に止まる確率は :math:`0.05^2 = 0.0025` あり、
それが起きた時刻の需要が高ければ供給支障になる。**同じ予備率でも、大きな
号機が少数ある系統のほうが危ない。** その差を数値にするのが容量停止確率表
(COPT) である。

設計の骨格
----------
中心は :func:`capacity_outage_table` の **再帰的な畳み込み**である。号機を
1 台ずつ足していき、各段で「健全（確率 :math:`1-FOR`）」と「停止（確率
:math:`FOR`）」の 2 分岐を既存の分布に合成する。号機ごとの停止が独立なら
これが厳密な分布であり、全号機が同一容量のときは二項分布に **厳密に**
一致する（テストでその一致を 1e-14 で固定してある）。

得られた分布を需要と突き合わせると 3 つの指標が出る。

=========  =====================================  ==================
指標       定義                                   拾うもの
=========  =====================================  ==================
LOLP       :math:`P(\\text{供給力} < \\text{需要})`   ある時点の確率
LOLE       LOLP の期間合計 [h/期間, d/期間]       不足の **頻度**
EUE        不足電力量の期待値 [MWh/期間]          不足の **深さ**
=========  =====================================  ==================

LOLE と EUE は別のものを測る。**LOLE が同じでも EUE が桁で違う 2 つの系統を
作れる**（テスト ``test_same_lole_but_different_eue`` がその実演で、比は
ちょうど 9 倍になる）。「何時間足りないか」だけを見て設備計画を決めると、
1 時間に 100 MW 足りない系統と 1 時間に 5 MW 足りない系統を同じ品質だと
判定してしまう。指標を 1 つに絞らないこと。

規約（文献と突き合わせる前に必ず読むこと）
------------------------------------------
**LOLP は等号を含めない。** :func:`lolp` は :math:`P(\\text{供給力} <
\\text{需要})` であって :math:`\\le` ではない。連続分布ならどちらでも同じ
だが、COPT は離散なので **等号は実際に起こる**（60 MW 機 5 台の系統に
ちょうど 300 MW の需要を与えれば起こる）。文献の値と合わないときは、まず
相手がどちらの規約かを確かめること。数値の突き合わせでは、浮動小数点の
丸めで等号が偶然壊れるのを防ぐため相対 1e-9 の許容差を置いている。

**LOLE の "days" は "hours" の 1/24 ではない。** ``unit="hours"`` は毎時の
需要に対する LOLP の和（時間/期間）、``unit="days"`` は **日ごとの最大需要**
に対する LOLP の和（日/期間）である。後者は日負荷持続曲線 (daily peak load
variation curve) の考え方で、1 日を「その日のピークで代表させる」。LOLP は
需要について単調非減少だから日ピークの LOLP は日平均の LOLP 以上であり、
``lole(days) >= lole(hours) / 24`` が常に成り立つ（教材ケースでは 3.1 倍
違う。実測 4.1527 h/年 に対し 0.5409 日/年）。北米の "LOLE 0.1 day/year" (10 年に 1 日) を "2.4 h/year" と読み替える
のは誤りである。

**丸め (rounding_mw) は確率と期待停止容量を保存するが LOLP はずらす。**
理由と偏りの向きは :func:`capacity_outage_table` の Notes に書いた。

**非逐次モンテカルロは「頻度と継続時間」を区別できない。** FOR = 0.05 は
「年に 1 回 438 時間止まる」でも「年に 50 回 8.8 時間止まる」でも成り立つ。
状態サンプリングはどちらも同じ確率としてしか見ないので、起動停止の応援可能
時間や必要な予備力の性格は出せない。それを出すには ``mttf`` / ``mttr`` を
使う逐次法が要る（本モジュールでは実装しない。:func:`monte_carlo_adequacy`
の Notes を参照）。

年間需要のデータについて
------------------------
第三者の実需要データは再配布しない方針なので、:func:`annual_load` は
seed 固定の **合成**系列を返す。誰が実行しても同じ 8760 点が得られ、
再現性の議論を「データが手元にあるか」から切り離せる。合成である以上、
絶対値そのものに意味はない。指標の**比較**（丸めの有無、号機構成の違い、
モンテカルロと解析解）に使うこと。
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import numpy as np

from .case import Case, Unit

__all__ = [
    "CapacityOutageTable",
    "MonteCarloResult",
    "capacity_outage_table",
    "lolp",
    "lole",
    "eue",
    "load_duration_curve",
    "monte_carlo_adequacy",
    "annual_load",
    "elcc",
]

#: MW の比較に使う相対許容差。等号を含めない規約を、浮動小数点の丸めで
#: 壊さないための緩衝である（``available == load`` のつもりの状態が
#: 1e-13 MW ずれて「不足」と判定されるのを防ぐ）。
_MW_TOL = 1e-9

#: 状態の停止容量をまとめるときの丸め桁数 [MW]。1e-9 MW = 1 mW 未満の
#: 差しかない状態は同じ状態とみなす。畳み込みの足し込み順序による
#: 浮動小数点の誤差で、同じ容量の状態が 2 つに割れるのを防ぐ。
_STATE_DECIMALS = 9

#: 1 日の時間数。``lole(unit="days")`` の日ごとの区切りに使う。
HOURS_PER_DAY = 24

#: モンテカルロを分割して回すときの 1 ブロックの標本数。標本数を大きく
#: しても使用メモリが増えないようにするための刻みであり、値を固定して
#: あるので seed が同じなら結果も同じになる。
_MC_BLOCK = 100_000

#: ``commitment`` 層を持たないケースで :func:`annual_load` が使う日内形状。
#: 合成データであり、出典はない。
_DEFAULT_DAILY_SHAPE = np.array([
    0.72, 0.68, 0.66, 0.65, 0.67, 0.72, 0.82, 0.90,
    0.95, 0.97, 0.96, 0.93, 0.90, 0.92, 0.95, 0.97,
    1.00, 0.99, 0.95, 0.90, 0.86, 0.82, 0.78, 0.75,
])


# ======================================================================
# 内部ヘルパ
# ======================================================================
def _capacities_and_rates(units: Sequence[Unit]) -> tuple[np.ndarray, np.ndarray]:
    """号機の並びから ``(容量 [MW], FOR)`` の配列を取り出して検査する。

    Raises
    ------
    TypeError
        :class:`~gridops.case.Case` をそのまま渡したとき。
    ValueError
        号機が 1 台もないとき。容量が負のとき。FOR が ``[0, 1]`` の外のとき。
    """
    if isinstance(units, Case):
        raise TypeError(
            "アデカシーの関数には号機の並びを渡すこと（Case ではない）。"
            "case.units をそのまま渡せばよい。号機の部分集合を評価したい"
            "ときは [u for u in case.units if ...] のように絞って渡す。"
        )

    unit_list = list(units)
    if not unit_list:
        raise ValueError(
            "号機が 1 台もない。容量停止確率表は号機の集合に対して定義される。"
            "case.units が空でないか（units 層があるか）を確かめること。"
        )

    capacities = np.empty(len(unit_list))
    rates = np.empty(len(unit_list))
    for i, unit in enumerate(unit_list):
        capacity = float(unit.p_max_mw)
        rate = float(unit.forced_outage_rate)
        if capacity < 0.0:
            raise ValueError(
                f"号機 {unit.name} の p_max_mw={capacity} が負である。"
                "容量は MW の正の値で与えること。"
            )
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"号機 {unit.name} の FOR={rate} が [0, 1] の外にある。"
                "FOR は確率であって百分率ではない（4% なら 0.04）。"
            )
        capacities[i] = capacity
        rates[i] = rate
    return capacities, rates


def _convolve_one(
    states: dict[float, float], capacity: float, rate: float
) -> dict[float, float]:
    """既存の状態分布に号機を 1 台足す（再帰的畳み込みの 1 段）。

    停止容量 :math:`X` の分布に容量 :math:`C`、強制停止率 :math:`p` の
    号機を足すと

    .. math::

        P'(x) = (1 - p) P(x) + p P(x - C)

    となる。確率ゼロの分岐は状態を作らない（``FOR = 0`` の号機を足しても
    状態数が増えないので、「必ず動く容量」を 1 台の号機として書ける）。
    """
    up = 1.0 - rate
    new: dict[float, float] = {}
    for outage, probability in states.items():
        if up > 0.0:
            key = round(outage, _STATE_DECIMALS)
            new[key] = new.get(key, 0.0) + probability * up
        if rate > 0.0:
            key = round(outage + capacity, _STATE_DECIMALS)
            new[key] = new.get(key, 0.0) + probability * rate
    return new


def _round_states(states: dict[float, float], step: float) -> dict[float, float]:
    """状態を ``step`` [MW] の格子に載せ替える（Billinton の丸め）。

    格子の間にある状態 :math:`C_i`（:math:`C_j < C_i < C_k`、両隣は格子点）
    の確率を、**内分**して両隣に配る。

    .. math::

        P_j \\mathrel{+}= \\frac{C_k - C_i}{C_k - C_j} P_i, \\qquad
        P_k \\mathrel{+}= \\frac{C_i - C_j}{C_k - C_j} P_i

    近い方に丸める（最近傍丸め）のではなく内分にするのが要点である。
    内分なら確率の総和だけでなく **期待停止容量も厳密に保存される**
    （重み付き平均が :math:`C_i` に戻るため）。最近傍丸めでは期待値が
    ずれ、設備の平均的な不足量という一番基本的な量が壊れる。
    """
    new: dict[float, float] = {}
    for outage, probability in states.items():
        index = math.floor(outage / step + 1e-9)
        low = round(index * step, _STATE_DECIMALS)
        weight = (outage - low) / step
        if weight <= 1e-12:          # もともと格子の上にある（丸め誤差を含む）
            new[low] = new.get(low, 0.0) + probability
            continue
        high = round((index + 1) * step, _STATE_DECIMALS)
        new[low] = new.get(low, 0.0) + probability * (1.0 - weight)
        new[high] = new.get(high, 0.0) + probability * weight
    return new


def _load_levels(load_profile_mw) -> np.ndarray:
    """需要系列を 1 次元の :class:`numpy.ndarray` に整える。"""
    levels = np.asarray(load_profile_mw, dtype=float).ravel()
    if levels.size == 0:
        raise ValueError(
            "需要系列が空である。長さ 1 以上の系列（[MW] の並び）を渡すこと。"
        )
    if not np.all(np.isfinite(levels)):
        raise ValueError("需要系列に有限でない値が含まれている。")
    return levels


def _shortfall_matrix(copt: "CapacityOutageTable", levels: np.ndarray) -> np.ndarray:
    """``(時刻, 状態)`` の不足電力 [MW]（不足がなければ 0）。"""
    available = copt.available_mw()
    return np.maximum(levels[:, None] - available[None, :], 0.0)


def _lolp_array(copt: "CapacityOutageTable", levels: np.ndarray) -> np.ndarray:
    """需要系列の各点に対する LOLP。"""
    available = copt.available_mw()
    tolerance = _MW_TOL * np.maximum(1.0, np.abs(levels))
    short = available[None, :] < (levels - tolerance)[:, None]
    return short.astype(float) @ copt.probability


# ======================================================================
# 容量停止確率表
# ======================================================================
@dataclass(frozen=True)
class CapacityOutageTable:
    """容量停止確率表 (Capacity Outage Probability Table, COPT)。

    「何 MW が止まっている状態が、どれだけの確率で起きるか」の一覧である。
    需要と突き合わせる前の、**系統側だけで決まる量**であることが要点で、
    同じ表を任意の需要系列に使い回せる。

    Parameters
    ----------
    outage_mw:
        停止容量 [MW]。**昇順**に並んでいる。
    probability:
        その停止容量 **ちょうど** の確率（累積ではない）。総和は 1。
    installed_mw:
        設備容量 [MW]。号機容量の単純合計。

    Notes
    -----
    ``rounding_mw`` を使った表では、最大の停止容量が ``installed_mw`` を
    超えることがある（156 MW の全停止状態が 150/160 MW に内分されるため）。
    丸めは号機を 1 台足すごとに行うので、この行き過ぎは台数の分だけ
    積み上がりうる（12 台なら最大で刻みの 12 倍）。その状態の
    :meth:`available_mw` は負になるが、確率は表の末尾（教材ケースで
    1e-20 の桁）にしか乗らない。期待停止容量を保存するための代償で
    あって物理的な状態ではない。
    """

    outage_mw: np.ndarray
    probability: np.ndarray
    installed_mw: float

    def __post_init__(self) -> None:
        # notebook で小さな表を手で組めるように、配列に直してから検査する。
        object.__setattr__(self, "outage_mw", np.asarray(self.outage_mw, dtype=float))
        object.__setattr__(
            self, "probability", np.asarray(self.probability, dtype=float)
        )
        object.__setattr__(self, "installed_mw", float(self.installed_mw))
        if self.outage_mw.shape != self.probability.shape:
            raise ValueError(
                "outage_mw と probability の長さが違う: "
                f"{self.outage_mw.shape} と {self.probability.shape}"
            )

    # ------------------------------------------------------------------
    @property
    def cumulative(self) -> np.ndarray:
        """累積確率 :math:`P(\\text{停止容量} \\ge x)`。

        各要素が :attr:`outage_mw` の同じ位置の値に対応する。先頭は必ず 1
        （停止容量が 0 以上である確率）であり、末尾は最大停止状態の確率に
        なる。文献の COPT はこの列を載せていることが多い。
        """
        return np.cumsum(self.probability[::-1])[::-1].copy()

    def probability_of_at_least(self, outage_mw: float) -> float:
        """:math:`P(\\text{停止容量} \\ge x)` を任意の :math:`x` [MW] で。

        表にない値でも引ける（その値以上の状態の確率を足す）。
        こちらは **等号を含む**。LOLP の「等号を含めない」規約とは
        向きが逆なので、直接 LOLP を計算するのに使ってはいけない
        （:func:`lolp` を使うこと）。
        """
        threshold = float(outage_mw)
        tolerance = _MW_TOL * max(1.0, abs(threshold))
        mask = self.outage_mw >= threshold - tolerance
        return float(self.probability[mask].sum())

    def available_mw(self) -> np.ndarray:
        """利用可能容量 [MW] = ``installed_mw`` - 停止容量。

        :attr:`outage_mw` が昇順なので、こちらは **降順**である。
        :attr:`probability` との対応は添字で保たれている。
        """
        return self.installed_mw - self.outage_mw

    def expected_outage_mw(self) -> float:
        """期待停止容量 [MW]。

        独立性から :math:`\\sum_i P_{max,i} \\cdot FOR_i` に厳密に一致する。
        畳み込みの実装が壊れたかどうかを一番安く検出できる不変量なので、
        テストで固定してある。丸め (``rounding_mw``) を入れても保存される。
        """
        return float(np.dot(self.outage_mw, self.probability))

    def summary(self) -> str:
        """表の要約（先頭の数状態と不変量）を返す。"""
        lines = [
            "Capacity Outage Probability Table",
            f"  設備容量     : {self.installed_mw:.1f} MW",
            f"  状態数       : {self.outage_mw.size}",
            f"  確率の総和   : {float(self.probability.sum()):.12f}",
            f"  期待停止容量 : {self.expected_outage_mw():.4f} MW",
            "  停止容量 [MW]   確率        累積 P(>= x)",
        ]
        cumulative = self.cumulative
        shown = min(self.outage_mw.size, 8)
        for i in range(shown):
            lines.append(
                f"  {self.outage_mw[i]:12.2f}  {self.probability[i]:11.3e}"
                f"  {cumulative[i]:11.3e}"
            )
        if shown < self.outage_mw.size:
            lines.append(f"  ... 残り {self.outage_mw.size - shown} 状態")
        return "\n".join(lines)


def capacity_outage_table(
    units: Sequence[Unit], *, rounding_mw: float | None = None
) -> CapacityOutageTable:
    """号機の集合から容量停止確率表を作る（再帰的畳み込み）。

    号機を 1 台ずつ足していき、各段で「健全（確率 :math:`1 - FOR`）」と
    「停止（確率 :math:`FOR`）」の 2 分岐を既存の分布に合成する。

    .. math::

        P_{n}(x) = (1 - p_n) P_{n-1}(x) + p_n P_{n-1}(x - C_n)

    :math:`2^n` 通りを全列挙するのと **厳密に同じ**分布が得られるが、
    同じ停止容量の状態がその場でまとまるので状態数が抑えられる
    （7 号機の教材ケースでは 128 通りが 33 状態になる）。

    Parameters
    ----------
    units:
        号機の並び。:attr:`Unit.p_max_mw` と :attr:`Unit.forced_outage_rate`
        だけを使う。
    rounding_mw:
        状態を丸める刻み [MW]。``None`` なら丸めない。指定すると 1 台
        足すごとに格子へ載せ替えるので、状態数が
        ``installed_mw / rounding_mw + 1`` 程度で頭打ちになる。

    Returns
    -------
    CapacityOutageTable
        停止容量の昇順に並んだ表。

    Raises
    ------
    TypeError
        :class:`~gridops.case.Case` を渡したとき（``case.units`` を渡すこと）。
    ValueError
        号機が空、容量が負、FOR が ``[0, 1]`` の外、``rounding_mw`` が非正。

    Notes
    -----
    **同一容量の号機だけなら二項分布に厳密に一致する。** 容量 :math:`C`、
    強制停止率 :math:`p` の号機が :math:`n` 台なら、停止容量 :math:`kC` の
    確率は :math:`\\binom{n}{k} p^k (1-p)^{n-k}` である。この一致
    （``scipy.stats.binom.pmf`` と 1e-14）が本実装の一番強い独立基準であり、
    テストで固定してある。

    **丸めについて。** 格子の間の状態は近い方に丸めるのではなく、両隣に
    **内分**して配る（:func:`_round_states` 参照）。そのため
    ``rounding_mw`` を入れても

    * 確率の総和は 1 のまま
    * 期待停止容量は :math:`\\sum_i P_{max,i} \\cdot FOR_i` のまま

    である。**しかし LOLP は保存されない。** しきい値
    :math:`x = \\text{設備容量} - \\text{需要}` が格子点の上にあるとき、
    :math:`x` と次の格子点の間にあった状態は確率の一部を :math:`x` 自身へ
    落とすので、その分だけ :math:`P(\\text{停止容量} > x)` が減る。
    すなわち **号機容量がすべて格子の倍数で、しきい値が格子点の上にある
    とき**、丸めは LOLP を過小評価側にずらす。この条件つきの向きは
    ``tests/test_adequacy.py`` が固定している。

    **条件が崩れると向きは保証されない。** 同梱ケースは 60 / 90 / 50 MW の
    7 号機なので、``rounding_mw=25`` を与えると 60 と 90 が格子から外れ、
    偏りはむしろ逆を向く。年間 LOLE は 4.1527 h から 5.1636 h へ
    **24% 増える**（過大評価側）。なおケースファイルの ``rounding_mw: 5.0``
    は全号機の容量が 10 の倍数なので **何も丸めない**（状態数は 33 のまま）。
    丸めの効果を授業で見せるときは ``rounding_mw=25`` を明示的に渡すこと。

    丸めは「状態数を減らすための近似」であって「精度を上げる整理」では
    ない。設備計画の数値を丸めた表から読むときは必ず丸めなしと比べること。

    Examples
    --------
    >>> from gridops import load_case
    >>> from gridops.adequacy import capacity_outage_table
    >>> copt = capacity_outage_table(load_case("wscc9").units)
    >>> copt.installed_mw
    460.0
    """
    capacities, rates = _capacities_and_rates(units)

    if rounding_mw is not None:
        rounding_mw = float(rounding_mw)
        if rounding_mw <= 0.0:
            raise ValueError(
                f"rounding_mw={rounding_mw} が非正である。"
                "丸めの刻みは正の MW で与えること（丸めないなら None）。"
            )

    states: dict[float, float] = {0.0: 1.0}
    for capacity, rate in zip(capacities, rates):
        states = _convolve_one(states, capacity, rate)
        if rounding_mw is not None:
            states = _round_states(states, rounding_mw)

    outage = np.array(sorted(states), dtype=float)
    probability = np.array([states[key] for key in outage], dtype=float)
    return CapacityOutageTable(
        outage_mw=outage,
        probability=probability,
        installed_mw=float(capacities.sum()),
    )


# ======================================================================
# 指標
# ======================================================================
def lolp(copt: CapacityOutageTable, load_mw: float) -> float:
    """供給支障確率 :math:`P(\\text{利用可能容量} < \\text{需要})`。

    Parameters
    ----------
    copt:
        容量停止確率表。
    load_mw:
        その時点の需要 [MW]。

    Returns
    -------
    float
        確率 :math:`[0, 1]`。

    Notes
    -----
    **等号は不足に数えない。** 利用可能容量が需要にちょうど等しい状態は
    「足りている」とする。連続分布なら気にしなくてよい違いだが、COPT は
    離散なので **等号は実際に起こる**。60 MW 機 5 台の系統に 300 MW の
    需要を与えれば、全機健全の状態がちょうど等号になる。文献やほかの
    ツールの LOLP と合わないときは、まず相手の規約（:math:`<` か
    :math:`\\le` か）を確かめること。ここでは浮動小数点の丸めで等号が
    偶然壊れないよう、相対 1e-9 の許容差を置いている。

    LOLP は「停電の確率」ではない。**発電設備が需要に足りない確率**で
    あって、送電線の事故も、周波数を保つための予備力の質も入っていない。
    ここで足りていても系統が停電しないとは言えない。
    """
    load = float(load_mw)
    tolerance = _MW_TOL * max(1.0, abs(load))
    available = copt.available_mw()
    return float(copt.probability[available < load - tolerance].sum())


def lole(
    copt: CapacityOutageTable,
    load_profile_mw,
    *,
    unit: str = "hours",
) -> float:
    """期待供給支障時間（``"hours"``）または日数（``"days"``）。

    Parameters
    ----------
    copt:
        容量停止確率表。
    load_profile_mw:
        需要の系列 [MW]。1 点が 1 時間を表す。
    unit:
        ``"hours"`` なら時間/期間、``"days"`` なら日/期間。

    Returns
    -------
    float
        ``"hours"`` なら :math:`\\sum_t LOLP(D_t)`、``"days"`` なら
        :math:`\\sum_d LOLP(\\max_{t \\in d} D_t)`。

    Raises
    ------
    ValueError
        ``unit`` が 2 つのどちらでもないとき。``"days"`` を指定したのに
        系列の長さが 24 の倍数でないとき。

    Notes
    -----
    **"days" は "hours" の 1/24 ではない。** ``"hours"`` は毎時の LOLP を
    足すが、``"days"`` は **その日の最大需要**に対する LOLP を 1 日 1 回
    足す（日負荷持続曲線の考え方で、1 日をピークで代表させる）。LOLP は
    需要について単調非減少なので、日ピークの LOLP はその日の LOLP の平均
    以上であり、

    .. math::

        \\mathrm{LOLE}_{days} \\ge \\frac{\\mathrm{LOLE}_{hours}}{24}

    が常に成り立つ。教材ケースでは 3.1 倍開く。北米で使われる
    「LOLE = 0.1 day/year（10 年に 1 日）」を「2.4 h/year」と読み替えては
    ならない。**単位が違えば指標そのものが違う。**

    ``"days"`` は時間系列を日に切って使うので、長さは 24 の倍数でなければ
    ならない。日ごとの最大需要の系列を自分で用意した場合は、それを
    ``unit="hours"`` に渡すこと（1 点 1 日と読み替えれば同じ計算になる）。
    """
    levels = _load_levels(load_profile_mw)

    if unit == "hours":
        pass
    elif unit == "days":
        if levels.size % HOURS_PER_DAY != 0:
            raise ValueError(
                f"unit='days' には 24 の倍数の長さの時間系列が要る"
                f"（渡されたのは {levels.size} 点）。"
                "日ごとの最大需要の系列を自分で作った場合は、"
                "1 点 1 日と読み替えて unit='hours' に渡すこと。"
            )
        levels = levels.reshape(-1, HOURS_PER_DAY).max(axis=1)
    else:
        raise ValueError(
            f"unit={unit!r} は 'hours' か 'days' のいずれかでなければならない。"
            "'days' は 'hours' の 1/24 ではなく、日ごとの最大需要に対する"
            "LOLP の和である。"
        )

    return float(_lolp_array(copt, levels).sum())


def eue(copt: CapacityOutageTable, load_profile_mw) -> float:
    """期待供給支障電力量 [MWh/期間]。

    .. math::

        \\mathrm{EUE} = \\sum_t \\sum_i P_i \\max(D_t - A_i, 0)

    :math:`A_i` は状態 :math:`i` の利用可能容量、:math:`P_i` はその確率。
    文献では EENS (expected energy not served) や LOEE とも呼ばれる。

    Parameters
    ----------
    copt:
        容量停止確率表。
    load_profile_mw:
        需要の系列 [MW]。**1 点を 1 時間**とみなすので、戻り値の単位は
        MWh になる。30 分値を渡したいなら結果を 2 で割ること。

    Notes
    -----
    **EUE は不足の「深さ」を拾い、LOLE は拾わない。** LOLE が同じでも
    EUE が桁で違う系統を作れる（本モジュールのテストがその実演）。
    設備計画の判断を LOLE 単独で下すと、めったに起きないが起きたときに
    深い不足になる構成を見落とす。
    """
    levels = _load_levels(load_profile_mw)
    shortfall = _shortfall_matrix(copt, levels)
    return float((shortfall @ copt.probability).sum())


def load_duration_curve(load_mw) -> np.ndarray:
    """需要持続曲線（降順に並べ替えた需要）[MW]。

    横軸を「その値以上になる時間数」と読む古典的な図である。並べ替える
    だけなので時刻の情報は失われる。**太陽光のように時刻と強く結びつく
    電源を扱うときは、この曲線だけで判断してはいけない**（同じ持続曲線
    でも、昼に高いか夜に高いかで必要な設備が変わる）。

    Parameters
    ----------
    load_mw:
        需要の系列 [MW]。

    Returns
    -------
    numpy.ndarray
        降順に並べ替えた需要。入力は変更しない。
    """
    levels = _load_levels(load_mw)
    return np.sort(levels)[::-1].copy()


# ======================================================================
# モンテカルロ
# ======================================================================
@dataclass(frozen=True)
class MonteCarloResult:
    """非逐次モンテカルロの結果。

    Parameters
    ----------
    lolp:
        供給支障確率の点推定（不足が起きた標本の割合）。
    eue:
        期待供給支障電力量 [MWh/期間] の点推定。
    n_samples:
        標本数。
    lolp_stderr, eue_stderr:
        点推定の標準誤差。**点推定だけを見ないための値**である。
    """

    lolp: float
    eue: float
    n_samples: int
    lolp_stderr: float
    eue_stderr: float

    def lolp_interval(self, level: float = 0.95) -> tuple[float, float]:
        """LOLP の信頼区間（正規近似）。

        Parameters
        ----------
        level:
            信頼水準（0 < level < 1）。既定は 0.95。

        Returns
        -------
        tuple[float, float]
            ``(下限, 上限)``。確率なので :math:`[0, 1]` に切り詰める。

        Notes
        -----
        モンテカルロの答え合わせは **点推定の一致では書けない**。標本ごとに
        値が動くのが正しい振る舞いだからである。「解析解がこの区間に入るか」
        で検証すること（本モジュールのテストがその形になっている）。
        正規近似なので、不足がほとんど起きない系統（:math:`Np < 10` 程度）
        では区間が信用できない。そのときは標本数を増やすしかない。
        """
        if not 0.0 < level < 1.0:
            raise ValueError(
                f"level={level} が (0, 1) の外にある。95% 信頼区間なら 0.95。"
            )
        z = NormalDist().inv_cdf(0.5 + level / 2.0)
        half_width = z * self.lolp_stderr
        return (max(0.0, self.lolp - half_width), min(1.0, self.lolp + half_width))

    def coefficient_of_variation(self) -> float:
        """変動係数 :math:`\\beta = \\sqrt{(1-p)/(pN)}`。

        収束の判定に使う量である。要点は :math:`\\beta` が
        :math:`1/\\sqrt{N}` でしか減らないこと、そして **稀な事象ほど
        分母の :math:`p` が小さく、同じ精度に必要な標本数が増える**こと
        である。:math:`p = 10^{-3}` の LOLP を 5% の精度で出すには
        :math:`N \\approx 4 \\times 10^{5}` 要る。信頼度の高い系統ほど
        モンテカルロが重くなるという、この分野の基本的な事情がここに出る。
        """
        if self.lolp <= 0.0:
            return math.inf
        return math.sqrt((1.0 - self.lolp) / (self.lolp * self.n_samples))

    def summary(self) -> str:
        """結果の要約を返す。"""
        low, high = self.lolp_interval()
        return "\n".join([
            "Monte Carlo adequacy (non-sequential)",
            f"  標本数   : {self.n_samples}",
            f"  LOLP     : {self.lolp:.6f} +/- {self.lolp_stderr:.6f}"
            f"  (95% CI [{low:.6f}, {high:.6f}])",
            f"  EUE      : {self.eue:.3f} MWh"
            f"  +/- {self.eue_stderr:.3f}",
            f"  変動係数 : {self.coefficient_of_variation():.4f}",
        ])


def monte_carlo_adequacy(
    units: Sequence[Unit],
    load_profile_mw,
    *,
    n_samples: int = 100_000,
    seed: int = 0,
) -> MonteCarloResult:
    """非逐次モンテカルロ（状態サンプリング）でアデカシーを評価する。

    1 標本は「系統の状態」1 つである。すなわち

    1. 需要系列から時刻を 1 つ一様に引く
    2. 号機ごとに独立に :math:`U < FOR` なら停止、そうでなければ健全とする
    3. 不足 :math:`\\max(D - A, 0)` を記録する

    を ``n_samples`` 回繰り返す。LOLP は不足が起きた割合、EUE は不足の
    平均に期間の時間数を掛けた値である。

    Parameters
    ----------
    units:
        号機の並び。
    load_profile_mw:
        需要の系列 [MW]。1 点 1 時間。
    n_samples:
        標本数。
    seed:
        :func:`numpy.random.default_rng` に渡す種。**同じ種なら同じ結果**
        になる（標本を 10 万点ずつのブロックに切って回すので、種と標本数が
        同じなら分割の仕方も同じになる）。

    Returns
    -------
    MonteCarloResult
        点推定と標準誤差。

    Notes
    -----
    畳み込み (:func:`capacity_outage_table`) が厳密に解ける問題を、あえて
    サンプリングでも解くのは **答え合わせができる題材でモンテカルロの
    振る舞いを学ぶため**である。標本数を 10 倍にしても誤差は
    :math:`\\sqrt{10}` 倍しか縮まないことを、解析解と並べて確かめられる。

    **逐次法は実装しない（発展課題）。** ここで使っているのは強制停止率
    :math:`FOR` だけで、:attr:`Unit.mttf` / :attr:`Unit.mttr` は使っていない。
    FOR = 0.05 は「年に 1 回 438 時間止まる」でも「年に 50 回 8.8 時間
    止まる」でも成り立つ。**どちらであるかは非逐次法では原理的に出せない。**
    区別するには時間軸に沿って稼働と修復を追う逐次モンテカルロが要り、
    そこで初めて 1 回の停止の継続時間、連続する不足の長さ、必要な貯蔵の
    容量といった量が扱える。実装する場合は指数分布の稼働時間
    :math:`\\mathrm{Exp}(1/\\mathrm{MTTF})` と修復時間
    :math:`\\mathrm{Exp}(1/\\mathrm{MTTR})` を交互に生成し、定常状態で
    :math:`FOR = \\mathrm{MTTR}/(\\mathrm{MTTF} + \\mathrm{MTTR})` に
    一致することを確かめるところから始めるとよい（教材ケースの G2 は
    :math:`50/(950+50) = 0.05` になっている）。
    """
    capacities, rates = _capacities_and_rates(units)
    levels = _load_levels(load_profile_mw)

    total = int(n_samples)
    if total <= 0:
        raise ValueError(
            f"n_samples={n_samples} が非正である。標本数は正の整数で与えること。"
        )

    rng = np.random.default_rng(seed)
    n_short = 0
    sum_deficit = 0.0
    sum_deficit_sq = 0.0

    drawn = 0
    while drawn < total:
        block = min(_MC_BLOCK, total - drawn)
        hours = rng.integers(0, levels.size, size=block)
        healthy = (rng.random((block, capacities.size)) >= rates).astype(float)
        available = healthy @ capacities
        deficit = np.maximum(levels[hours] - available, 0.0)
        n_short += int(np.count_nonzero(deficit > 0.0))
        sum_deficit += float(deficit.sum())
        sum_deficit_sq += float(np.dot(deficit, deficit))
        drawn += block

    p_hat = n_short / total
    lolp_stderr = math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / total)

    mean_deficit = sum_deficit / total
    if total > 1:
        variance = max(sum_deficit_sq - total * mean_deficit**2, 0.0) / (total - 1)
    else:
        variance = 0.0
    hours_in_period = float(levels.size)

    return MonteCarloResult(
        lolp=p_hat,
        eue=mean_deficit * hours_in_period,
        n_samples=total,
        lolp_stderr=lolp_stderr,
        eue_stderr=math.sqrt(variance / total) * hours_in_period,
    )


# ======================================================================
# 年間需要の合成
# ======================================================================
def _daily_shape(case: Case, day: np.ndarray, hour_of_day: np.ndarray) -> np.ndarray:
    """日内形状。冬型と夏型を季節位置でなめらかに混ぜる。

    ``commitment`` 層の 24 時間形状（第 07 回で使ったもの）をそのまま
    使い回す。同じ形が別の回にも出てくることで、学生が系統を覚え直さずに
    済むようにするのが狙いである。層が無いケースでは既定の形状を使う。
    """
    profiles = case.commitment.get("profiles") or {}
    summer = np.asarray(
        profiles.get("summer_weekday", _DEFAULT_DAILY_SHAPE), dtype=float
    )
    winter = np.asarray(
        profiles.get("winter_weekday", _DEFAULT_DAILY_SHAPE), dtype=float
    )
    if summer.size != HOURS_PER_DAY or winter.size != HOURS_PER_DAY:
        summer = winter = _DEFAULT_DAILY_SHAPE

    # 1 月中旬で 1（冬型）、7 月中旬で 0（夏型）になる重み。
    weight = 0.5 * (1.0 + np.cos(2.0 * np.pi * (day - 15.0) / 365.0))
    return weight * winter[hour_of_day] + (1.0 - weight) * summer[hour_of_day]


def annual_load(
    case: Case, *, peak_mw: float | None = None, hours: int = 8760
) -> np.ndarray:
    """年間の時間需要 [MW] を合成する。

    ``reliability.annual`` の ``seed`` / ``seasonal_amplitude`` /
    ``weekend_factor`` / ``noise_sigma`` を使い、

    * 日内形状（``commitment`` 層の 24 時間形状を季節でなめらかに混ぜる）
    * 季節変動（冬に最大、夏に次の山、春秋が谷になる 2 調波）
    * 週末係数（週の 6, 7 日目に掛ける）
    * 対数正規ではない単純な乗法性の雑音

    を重ねてから、**最大値がちょうど ``peak_mw`` になるように**規格化する。

    Parameters
    ----------
    case:
        ``reliability`` 層を持つケース。
    peak_mw:
        年間最大需要 [MW]。``None`` なら ``commitment.peak_mw``、それも
        無ければ基準潮流の総負荷を使う。
    hours:
        生成する点数。既定は 8760（平年 1 年分）。

    Returns
    -------
    numpy.ndarray
        長さ ``hours`` の需要 [MW]。先頭は月曜 0 時とみなす。

    Raises
    ------
    ValueError
        ``reliability`` 層が無いとき（:meth:`Case.require` が投げる）。
        ``hours`` が非正、``peak_mw`` が非正のとき。

    Notes
    -----
    **これは実データではない。** 第三者の実需要データを同梱しない方針を
    採っているので、生成関数で作る。その代わり誰が実行しても同じ 8760 点が
    得られ、「先生の手元のデータでは」という状況が起きない。合成である以上
    絶対値そのものに意味はなく、指標の **比較**（丸めの有無、号機構成の
    違い、モンテカルロと解析解の突き合わせ）に使うこと。実務の値を出したい
    ときは、この関数を実測の 8760 時間値で置き換えればよい。以降の
    :func:`lole` / :func:`eue` / :func:`elcc` は需要系列の出所を問わない。

    規格化を最後に置いているので、雑音の実現によらず最大値は厳密に
    ``peak_mw`` になる。「年間最大需要」を条件に固定して設備を比べたい
    ときに、この性質があると議論が単純になる。
    """
    case.require("reliability")

    annual = dict(case.reliability.get("annual", {}) or {})
    seed = int(annual.get("seed", 0))
    amplitude = float(annual.get("seasonal_amplitude", 0.0))
    weekend_factor = float(annual.get("weekend_factor", 1.0))
    noise_sigma = float(annual.get("noise_sigma", 0.0))

    count = int(hours)
    if count <= 0:
        raise ValueError(f"hours={hours} が非正である。生成する点数は正の整数で。")

    if peak_mw is None:
        peak = float(case.commitment.get("peak_mw", 0.0) or 0.0)
        if peak <= 0.0:
            peak = float(case.to_mw(sum(bus.pd for bus in case.buses)))
    else:
        peak = float(peak_mw)
    if peak <= 0.0:
        raise ValueError(
            "年間最大需要が決まらない。peak_mw を明示的に渡すか、"
            "ケースの commitment.peak_mw か母線負荷を確かめること。"
        )

    index = np.arange(count)
    day = index // HOURS_PER_DAY
    hour_of_day = index % HOURS_PER_DAY

    daily = _daily_shape(case, day, hour_of_day)

    # 季節変動: 1 月中旬が最大、7 月中旬が第 2 の山、春秋が谷。
    phase = 2.0 * np.pi * (day - 15.0) / 365.0
    seasonal = 1.0 + amplitude * (0.6 * np.cos(phase) + 0.4 * np.cos(2.0 * phase))

    # 週末係数: 先頭を月曜とみなし、6 日目と 7 日目に掛ける。
    weekend = np.where(day % 7 >= 5, weekend_factor, 1.0)

    rng = np.random.default_rng(seed)
    noise = 1.0 + rng.normal(0.0, noise_sigma, size=count)
    noise = np.maximum(noise, 0.1)      # 雑音で需要が負になるのを防ぐ

    raw = daily * seasonal * weekend * noise
    # 先に最大値で割ってから掛けると、最大値が厳密に peak になる。
    return (raw / raw.max()) * peak


# ======================================================================
# 等価容量価値
# ======================================================================
def elcc(
    units: Sequence[Unit],
    load_profile_mw,
    new_unit: Unit,
    *,
    tol: float = 0.1,
) -> float:
    """等価容量価値 ELCC (effective load carrying capability) [MW]。

    新電源を 1 台足したとき、**信頼度を元の水準に保ったまま何 MW の需要を
    追加で背負えるか**を返す。すなわち

    .. math::

        \\mathrm{LOLE}(\\text{既設} + \\text{新設},\\ D + \\Delta)
        = \\mathrm{LOLE}(\\text{既設},\\ D)

    を満たす :math:`\\Delta` である。LOLE は :math:`\\Delta` について単調
    非減少なので、:math:`[0, P_{max}]` を挟んで二分法で求められる。

    Parameters
    ----------
    units:
        既設の号機。
    load_profile_mw:
        需要の系列 [MW]。1 点 1 時間。
    new_unit:
        追加する号機。
    tol:
        二分法の打ち切り幅 [MW]。既定 0.1 MW。

    Returns
    -------
    float
        等価容量価値 [MW]。挟み込んだ区間の下端、すなわち
        **元の LOLE を悪化させない最大の需要増分**（真の値との差は
        ``tol`` 以内で、必ず安全側）。

    Raises
    ------
    ValueError
        追加する号機の容量が非正のとき。``tol`` が非正のとき。

    Warns
    -----
    UserWarning
        既設系統の LOLE がゼロのとき。この場合は「元の水準」が
        「一度も不足しない」なので、答えは需要のわずかな増加で不足が
        始まる点になり、新電源の性能をほとんど反映しない。需要をもっと
        高くして評価すること。

    Notes
    -----
    容量 :math:`P_{max}`、強制停止率 :math:`FOR` の在来型電源なら
    :math:`\\mathrm{ELCC} \\approx P_{max}(1 - FOR)` になる。**この近似が
    成り立たないものを測るのが ELCC の本来の使いどころ**である。太陽光や
    風力のように出力が時刻に強く依存する電源では、ELCC は設備容量の数割
    にしかならないことが多く、しかも導入量が増えるほど 1 MW あたりの価値が
    下がる（夕方の不足が支配的になり、昼に発電しても需要を背負えない）。
    ここでは需要と独立な号機だけを扱っているので、時刻依存の電源を評価
    するには利用可能容量の側に時系列を持ち込む拡張が要る。

    LOLE は離散な階段関数なので、二分法が収束する先は「段が切り替わる
    需要増分」である。``tol`` を極端に小さくしても意味のある桁は増えない。
    """
    profile = _load_levels(load_profile_mw)
    if float(new_unit.p_max_mw) <= 0.0:
        raise ValueError(
            f"追加する号機 {new_unit.name} の p_max_mw が非正である。"
            "等価容量価値は正の容量に対してしか定義できない。"
        )
    if tol <= 0.0:
        raise ValueError(f"tol={tol} が非正である。二分法の打ち切り幅は正の MW で。")

    base = capacity_outage_table(units)
    target = lole(base, profile)
    if target <= 0.0:
        warnings.warn(
            "既設系統の LOLE がゼロなので、等価容量価値が新電源の性能を"
            "ほとんど反映しない。需要系列を高くして（不足が起きる水準にして）"
            "評価すること。",
            UserWarning,
            stacklevel=2,
        )

    expanded = capacity_outage_table(list(units) + [new_unit])

    def excess(delta: float) -> float:
        """需要を ``delta`` MW 一律に上げたときの LOLE の増分。"""
        return lole(expanded, profile + delta) - target

    low, high = 0.0, float(new_unit.p_max_mw)
    # 号機を足して LOLE が悪化することはないので excess(0) <= 0 は常に成立。
    if excess(high) <= 0.0:
        return high

    while high - low > tol:
        middle = 0.5 * (low + high)
        if excess(middle) <= 0.0:
            low = middle
        else:
            high = middle
    # 挟み込んだ区間の **下端**を返す。中点を返すと「元の LOLE に戻る」
    # という定義そのものを tol の幅だけ破ることがある（LOLE は階段関数
    # なので、中点が段の向こう側に落ちうる）。下端なら
    # LOLE(既設+新設, D+ELCC) <= LOLE(既設, D) が必ず成り立つ。
    return low
