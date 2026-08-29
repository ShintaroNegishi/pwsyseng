"""起動停止計画（unit commitment）。

経済負荷配分（第 05 回）は「どの機がどれだけ出すか」を連続量の最適化で
決める。そこには **どの機を運転させるか** という 0-1 の意思決定が無い。
起動費・無負荷費・最低出力・最低運転停止時間はいずれも「運転しているか
どうか」に紐づく量で、連続量の最適化では表せない。ここが混合整数計画に
なる理由であり、本モジュールが扱う問題である。

定式化の要点
------------
変数は号機 :math:`i` と時刻 :math:`t` について

======================  ====================================================
:math:`u_{it}`          運転状態（0-1）
:math:`v_{it}`          起動（0-1）
:math:`w_{it}`          停止（0-1 だが連続変数でよい。後述）
:math:`p_{it}`          出力 [MW]
:math:`d_{itk}`         区分線形費用の第 :math:`k` セグメントの出力 [MW]
:math:`\\mathit{shed}_t`  供給不足 [MW]
:math:`\\mathit{spill}_t` 出力抑制 [MW]
======================  ====================================================

制約は次のとおり。

.. math::

    u_{it} - u_{i,t-1} &= v_{it} - w_{it} \\\\
    v_{it} + w_{it} &\\le 1 \\\\
    P^{min}_i u_{it} \\le p_{it} &\\le P^{max}_i u_{it} \\\\
    p_{it} &= P^{min}_i u_{it} + \\sum_k d_{itk} \\\\
    \\sum_i p_{it} + \\mathit{shed}_t - \\mathit{spill}_t &= D_t \\\\
    \\sum_i (P^{max}_i u_{it} - p_{it}) &\\ge R_t \\\\
    \\sum_{s=t-TU_i+1}^{t} v_{is} &\\le u_{it} \\\\
    \\sum_{s=t-TD_i+1}^{t} w_{is} &\\le 1 - u_{it} \\\\
    p_{it} - p_{i,t-1} &\\le RU_i u_{i,t-1} + SU_i v_{it} \\\\
    p_{i,t-1} - p_{it} &\\le RD_i u_{it} + SD_i w_{it}

予備力の書き方
--------------
予備力制約は **同期並列している未負荷容量** :math:`\\sum_i (P^{max}_i u_{it}
- p_{it}) \\ge R_t` として書く。よく見る

.. math:: \\sum_i P^{max}_i u_{it} \\ge (1 + r) D_t

という書き方は、需給が :math:`\\sum_i p_{it} = D_t` で閉じているときだけ
上の式と一致する。供給不足 :math:`\\mathit{shed}` や出力抑制
:math:`\\mathit{spill}` を入れた瞬間に両者はずれ、**需要を捨てているのに
予備力を満たしている**という無意味な解が出る。予備力は需要ではなく
「いま出していない容量」に対する要求である。

最低運転停止時間の初期条件
--------------------------
:math:`t < TU` の範囲を「窓和の切り詰め」だけで済ませてはいけない。
切り詰めた窓和は計画期間の中で起きた起動しか見ないので、**期間が始まる
前から運転している号機の拘束が消える**。最低運転時間 8 時間の石炭機が
1 時間目に止められる、という形で必ず現れる。本モジュールは
:meth:`~gridops.case.Unit.remaining_min_up` /
:meth:`~gridops.case.Unit.remaining_min_down` を使い、期間の頭の
``u`` を強制的に固定する。窓和はそれとは別に、期間の内側で起きる起動停止の
ために（切り詰めた形で）置く。切り詰めは「窓が期間からはみ出す部分では
拘束が弱くなる」ことを意味するのではなく、はみ出す部分の起動は物理的に
存在しない、という意味である。

区分線形費用
------------
PuLP は線形の問題しか解けないので、2 次の燃料費
:math:`C(P) = c_2 P^2 + c_1 P + c_0` はそのままでは扱えない。
:math:`[P^{min}, P^{max}]` を等間隔に分け、各区間を割線で置き換える。

* 分割した各セグメントの傾きは **狭義単調増加**である（:math:`C` が凸）。
  したがって安いセグメントから自動的に埋まる。**SOS2 も追加の 0-1 変数も
  要らない。** 「セグメント k を使うなら k-1 は満杯」を強制する制約を
  書きたくなるが、凸性がその制約を自動的に満たしてくれる。
* 割線は 2 次曲線の **上側**に来るので、近似は必ず **過大評価側**である。
  分割数を増やすと誤差は :math:`O(1/K^2)` で減る（1 セグメントの誤差の
  上界は :math:`c_2 L^2 / 4`、:math:`L` はセグメント幅）。
* 割線の傾きは :math:`c_1 + c_2 (x_{k-1} + x_k)`、すなわち **セグメント
  中点における増分費用**に等しい。区分線形近似は「増分費用を階段関数で
  近似する」ことと同じである。

緩和変数 shed / spill
---------------------
``allow_shortfall=True``（既定）で供給不足 ``shed`` を許す。実行不可能で
授業が止まるより、「12 時に 80 MW 足りない」と返るほうが学生には有益で
ある。出力抑制 ``spill`` は **既定で常に**入れる。これが無いと、太陽光を
差し引いた純需要が起動中の号機の :math:`\\sum P^{min}` を下回った瞬間に
問題が実行不可能になる。ダックカーブの正午前後で普通に起こる状況であり、
「解けない」ではなく「抑制した」と答えるのが正しい。

対称性除去
----------
同一諸元の号機が複数あると、入切表を入れ替えただけの同じ費用の解が
:math:`n!` 個できて分枝限定が無駄に長引く。``symmetry_breaking=True``
（既定）は :math:`u_{it} \\ge u_{i+1,t}` を入れて番号順の起動だけを残す。
**適用するのは費用も諸元も完全に同じ号機だけ**である。費用がわずかでも
違う号機に入れると、安い方を後ろの番号に置いた最適解を切り落としてしまう。
同梱ケース wscc9 は同一プラント内でも燃料費をわずかにずらしてあるので、
**既定でもこの制約は 1 本も入らない**。効果を見たい場合は
:func:`dataclasses.replace` で費用を揃えた号機を作ること。

限界費用は 2 段階で
-------------------
混合整数計画の双対は取れない（:mod:`gridops.solvers` の Notes を参照）。
時間別限界費用は、得られた入切 :math:`u` を **固定して線形計画に落とし
直し**、需給バランス制約の双対から取る（:func:`marginal_prices`）。

その他の規約
------------
* :mod:`pulp` は :mod:`gridops.solvers` 経由でだけ使う（本モジュールは
  ``import pulp`` を書かない）。
* 計画期間の直前の出力 :math:`p_{i,-1}` はケースデータに無い。停止して
  いた号機 (:math:`u^0 = 0`) だけは 0 と分かるので、1 時刻目のランプ制約は
  その号機にしか課さない（:func:`unit_commitment` の Notes を参照）。
* :attr:`CommitmentResult.demand_mw` には **最適化に使った需要**、すなわち
  VRE を差し引いた純需要が入る。元の需要は ``result.options`` にある。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import solvers
from .case import Case, Unit

__all__ = [
    "CommitmentResult",
    "demand_profile",
    "net_demand",
    "unit_commitment",
    "priority_list",
    "enumerate_commitment",
    "marginal_prices",
]

#: 区分線形近似の既定の分割数。:func:`unit_commitment` の既定値と、
#: 引数を持たない :func:`priority_list` / :func:`enumerate_commitment` が
#: 使う値。**総費用を直接比べられるように 3 者で同じ値にしてある。**
DEFAULT_SEGMENTS = 4

#: 供給支障費用（VOLL）の既定値 [円/MWh]。燃料費の 2 桁上に置き、
#: 「他のあらゆる手段を使い切ってから初めて需要を切る」ことを表す。
DEFAULT_VOLL = 1_000_000.0

#: 出力抑制に置く極小の係数（VOLL に対する比）。抑制に価格を主張して
#: いるのではない。費用が同じ解が複数あるときに **抑制量が最小の解**を
#: 選ばせるためのタイブレークである。既定では 1 円/MWh になり、混合整数
#: 計画の相対ギャップ（1e-4）よりはるかに小さいので最適性の判定を変えない。
SPILL_PRICE_RATIO = 1e-6

#: 全列挙が扱える候補数の上限。これを超えると日本語 ValueError を投げる。
ENUMERATION_LIMIT = 2 ** 18

#: 数値の丸め落ちを吸収する許容差 [MW]。
_TOL_MW = 1e-6


# ======================================================================
# 需要
# ======================================================================
def demand_profile(
    case: Case, profile: str = "summer_weekday", *, peak_mw: float | None = None
) -> np.ndarray:
    """時系列需要 [MW] を作る。

    ケースの ``commitment.profiles`` は最大需要に対する比率の列である。
    これに ``peak_mw`` を掛けたものを返す。

    Parameters
    ----------
    case:
        ``commitment`` 層を持つケース。
    profile:
        ``"summer_weekday"`` / ``"winter_weekday"`` / ``"light_load"`` など、
        ケースが持つ形状の名前。
    peak_mw:
        基準にする最大需要 [MW]。``None`` ならケースの ``peak_mw``。

    Returns
    -------
    numpy.ndarray
        ``(T,)`` の需要 [MW]。

    Raises
    ------
    ValueError
        その名前の形状がケースに無いとき（使える名前を並べて示す）。

    Notes
    -----
    形状の最大値が 1.0 とは限らない。``light_load`` の最大は 0.78 なので、
    返る系列の最大は ``peak_mw`` ではなく ``0.78 * peak_mw`` になる。
    ``peak_mw`` は「その系統の年間最大需要」であって「この日の最大需要」
    ではない、という読み方をすること。
    """
    case.require("commitment")
    profiles = dict(case.commitment.get("profiles", {}) or {})
    if profile not in profiles:
        raise ValueError(
            f"需要形状 '{profile}' はケース '{case.name}' に無い。"
            f"使えるのは {sorted(profiles)}。"
        )
    shape = np.asarray(profiles[profile], dtype=float)
    peak = case.commitment.get("peak_mw") if peak_mw is None else peak_mw
    if peak is None:
        raise ValueError(
            f"ケース '{case.name}' に peak_mw が無い。"
            "demand_profile(case, peak_mw=315.0) のように明示すること。"
        )
    return shape * float(peak)


def net_demand(
    case: Case, demand_mw, *, vre_mw: float | Sequence[float] | None = None
) -> np.ndarray:
    """変動性再生可能電源（VRE）を差し引いた **純需要** [MW]。

    起動停止計画が見るのは需要そのものではなく、この純需要である。
    太陽光は昼に需要を押し下げるが夕方に急速に消えるので、純需要の形は
    もとの需要より **夕方の立ち上がりが急**になる（ダックカーブ）。
    起動停止計画で効いてくるのは需要の大きさではなく、この傾きである。

    Parameters
    ----------
    case:
        ケース。``vre_mw`` を省いたときは ``commitment.vre`` 層を使う。
    demand_mw:
        ``(T,)`` の需要 [MW]。
    vre_mw:
        ``None`` なら **ケースの VRE 層を定格容量のまま**使う。
        スカラーなら「その設備容量 [MW] でケースの出力形状を使う」。
        配列なら VRE 出力 [MW] そのものとして使う。

    Returns
    -------
    numpy.ndarray
        ``(T,)`` の純需要 [MW]。**負になりうる**（VRE が需要を上回る時刻）。

    Raises
    ------
    ValueError
        ``vre_mw`` を省いたのにケースが VRE 層を持たないとき、
        または配列の長さが需要と合わないとき。

    Notes
    -----
    :func:`unit_commitment` の ``vre_mw=None`` は意味が違い、そちらは
    **VRE 無し**（純需要 = 需要）である。最適化の既定を「ケースにたまたま
    VRE 層があるかどうか」で変えたくないためである。VRE を入れた計画が
    欲しいときは ``unit_commitment(case, d, vre_mw=120.0)`` のように
    明示すること。
    """
    demand = np.asarray(demand_mw, dtype=float).ravel()
    return demand - _vre_output(case, demand.size, vre_mw, required=True)


def _vre_output(
    case: Case, horizon: int, vre_mw, *, required: bool = False
) -> np.ndarray:
    """VRE 出力 [MW] の ``(T,)`` 配列を組む。``vre_mw`` が None なら零。"""
    if vre_mw is None and not required:
        return np.zeros(horizon)

    if vre_mw is None or np.ndim(vre_mw) == 0:
        block = dict(case.commitment.get("vre", {}) or {})
        shape = block.get("profile")
        if shape is None:
            raise ValueError(
                f"ケース '{case.name}' に commitment.vre 層が無いので、"
                "VRE の出力形状を決められない。"
                "vre_mw に (T,) の出力 [MW] を直接渡すこと。"
            )
        profile = np.asarray(shape, dtype=float)
        capacity = block.get("capacity_mw", 0.0) if vre_mw is None else float(vre_mw)
        output = profile * float(capacity)
    else:
        output = np.asarray(vre_mw, dtype=float).ravel()

    if output.size != horizon:
        raise ValueError(
            f"VRE 出力の長さ {output.size} が需要の長さ {horizon} と合わない。"
            "同じ時間刻み・同じ時間数で渡すこと。"
        )
    if np.any(output < -_TOL_MW):
        raise ValueError("VRE 出力に負の値がある。発電を正の値で渡すこと。")
    return output


# ======================================================================
# 結果
# ======================================================================
@dataclass
class CommitmentResult:
    """起動停止計画の結果。

    ``frozen`` にしていないのは、notebook で一部を差し替えて作図し直す
    使い方（抑制を手で埋めてみる等）を許すためである。中身の配列を書き換え
    たら :meth:`summary` の値も変わる、という素直な振る舞いにしてある。

    Parameters
    ----------
    case:
        もとのケース。
    demand_mw:
        ``(T,)`` の **最適化に使った需要** [MW]。VRE を差し引いた純需要で
        あり、元の需要は ``options["gross_demand_mw"]`` にある。
    status:
        ソルバの状態（``"Optimal"`` など）。時間切れで打ち切られた場合は
        ``"Not Solved"`` になり、**実行可能だが最適とは限らない**。
    schedule:
        号機名 -> ``(T,)`` の 0/1。
    dispatch:
        号機名 -> ``(T,)`` の出力 [MW]。
    shortfall_mw, spill_mw:
        ``(T,)`` の供給不足と出力抑制 [MW]。
    total_cost:
        総費用 [円]。**モデルが最適化した値**、すなわち区分線形近似の費用
        である（2 次曲線そのものの費用ではない。過大評価側になる）。
    cost_breakdown:
        ``{"fuel", "noload", "startup", "penalty"}`` [円]。合計は
        :attr:`total_cost` に一致する（区分線形の評価が同じなので、
        丸め誤差の範囲で厳密に一致する）。
    seconds:
        求解に要した時間 [s]。
    method:
        ``"milp"`` / ``"priority"`` / ``"enumeration"``。
    options:
        再現に必要な設定（契約に無い追加のフィールド）。
        ``reserve_rate`` / ``reserve_mw`` / ``gross_demand_mw`` / ``vre_mw``
        / ``n_segments`` / ``voll`` / ``spill_price`` / ``allow_shortfall``
        / ``symmetry_breaking`` / ``unit_names``。
        :func:`marginal_prices` が線形計画を組み直すのに使う。
    """

    case: Case
    demand_mw: np.ndarray
    status: str
    schedule: dict[str, np.ndarray]
    dispatch: dict[str, np.ndarray]
    shortfall_mw: np.ndarray
    spill_mw: np.ndarray
    total_cost: float
    cost_breakdown: dict[str, float]
    seconds: float
    method: str
    options: Mapping[str, object] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def _units(self) -> list[Unit]:
        """入切表に載っている号機を、ケースの並び順で返す。"""
        return [unit for unit in self.case.units if unit.name in self.schedule]

    @property
    def _horizon(self) -> int:
        return int(np.asarray(self.demand_mw).size)

    def committed_mw(self, t: int) -> float:
        """時刻 ``t`` に同期並列している容量 :math:`\\sum_i P^{max}_i u_{it}` [MW]。"""
        return float(
            sum(unit.p_max_mw * self.schedule[unit.name][t] for unit in self._units())
        )

    def reserve_mw(self) -> np.ndarray:
        """時刻ごとの運転予備力 :math:`\\sum_i (P^{max}_i u_{it} - p_{it})` [MW]。

        「同期並列しているが出していない容量」である。停止中の号機は
        :math:`u = 0` なので 1 MW も数えない。**起動に何時間もかかる容量を
        予備力に数えないこと**が、この式の要点である。
        """
        reserve = np.zeros(self._horizon)
        for unit in self._units():
            reserve += unit.p_max_mw * self.schedule[unit.name] - self.dispatch[unit.name]
        return reserve

    def n_startups(self) -> int:
        """計画期間中の起動回数の合計。

        期間の直前の状態 :attr:`~gridops.case.Unit.u0` からの遷移も 1 回と
        数える。停止中だった号機を 1 時間目に起動すれば 1 回である。
        """
        total = 0
        for unit in self._units():
            u = np.asarray(self.schedule[unit.name], dtype=float)
            previous = np.concatenate(([float(unit.u0)], u[:-1]))
            total += int(np.sum(np.maximum(u - previous, 0.0) > 0.5))
        return total

    def summary(self) -> str:
        """要約を返す（日本語）。"""
        horizon = self._horizon
        reserve = self.reserve_mw()
        breakdown = self.cost_breakdown
        lines = [
            f"起動停止計画 [{self.method}] status={self.status}",
            f"  期間       : {horizon} 時刻 / 号機 {len(self.schedule)} 台",
            f"  需要(純)   : {self.demand_mw.min():.1f} 〜 {self.demand_mw.max():.1f} MW",
            f"  総費用     : {self.total_cost:,.0f} 円",
            f"    燃料費   : {breakdown.get('fuel', 0.0):,.0f} 円",
            f"    無負荷費 : {breakdown.get('noload', 0.0):,.0f} 円",
            f"    起動費   : {breakdown.get('startup', 0.0):,.0f} 円"
            f"（{self.n_startups()} 回）",
            f"    ペナルティ: {breakdown.get('penalty', 0.0):,.0f} 円",
            f"  予備力     : {reserve.min():.1f} 〜 {reserve.max():.1f} MW",
            f"  供給不足   : {self.shortfall_mw.sum():.1f} MWh"
            f"（最大 {self.shortfall_mw.max():.1f} MW）",
            f"  出力抑制   : {self.spill_mw.sum():.1f} MWh"
            f"（最大 {self.spill_mw.max():.1f} MW）",
            f"  求解時間   : {self.seconds:.3f} s",
        ]
        if self.status != "Optimal":
            lines.append("  ※ 最適性は保証されていない（時間切れの可能性）。")
        return "\n".join(lines)

    def to_table(self) -> str:
        """入切表を ASCII で返す（``#`` が運転、``.`` が停止）。

        列見出しは英語（学生環境の日本語フォント欠如で崩れないため）。
        """
        horizon = self._horizon
        units = self._units()
        width = max([len(unit.name) for unit in units] + [9])
        tens = "".join(str((t // 10) % 10) for t in range(horizon))
        ones = "".join(str(t % 10) for t in range(horizon))
        lines = [
            "unit commitment schedule ( # = on, . = off )",
            f"{'hour':<{width}}  {tens}",
            f"{'':<{width}}  {ones}",
        ]
        for unit in units:
            marks = "".join(
                "#" if value > 0.5 else "." for value in self.schedule[unit.name]
            )
            lines.append(f"{unit.name:<{width}}  {marks}")
        demand = " ".join(f"{value:.0f}" for value in self.demand_mw)
        lines.append(f"{'demand MW':<{width}}  {demand}")
        return "\n".join(lines)


# ======================================================================
# 費用のモデル（区分線形近似）
# ======================================================================
def _variable_cost(unit: Unit, p_mw: float) -> float:
    """無負荷費を除いた燃料費 :math:`c_2 P^2 + c_1 P` [円/h]。

    :meth:`~gridops.case.Unit.fuel_cost` から :math:`c_0` を抜いたもの。
    無負荷費は「運転しているだけでかかる費用」で、出力に依らない。
    起動停止計画では 0-1 変数 :math:`u` に掛かるので分けて持つ。
    """
    return unit.quadratic * p_mw * p_mw + unit.var_cost * p_mw


def _segments(unit: Unit, n_segments: int) -> list[tuple[float, float]]:
    """区分線形近似の ``(幅 [MW], 傾き [円/MWh])`` の列。

    :math:`[P^{min}, P^{max}]` を等分し、各区間を割線で置き換える。
    傾きは

    .. math:: s_k = \\frac{f(x_k) - f(x_{k-1})}{x_k - x_{k-1}}
                  = c_1 + c_2 (x_{k-1} + x_k)

    となり、**セグメント中点における増分費用**に等しい。:math:`c_2 > 0`
    なら :math:`s_1 < s_2 < \\cdots` と狭義単調増加なので、線形計画は
    放っておいても安いセグメントから埋める。ここが SOS2 も追加の 0-1
    変数も要らない理由である。
    """
    span = unit.p_max_mw - unit.p_min_mw
    count = max(1, int(n_segments))
    if span <= 0.0:
        return []
    width = span / count
    out: list[tuple[float, float]] = []
    for k in range(count):
        low = unit.p_min_mw + k * width
        high = low + width
        slope = (_variable_cost(unit, high) - _variable_cost(unit, low)) / width
        out.append((width, slope))
    return out


def _piecewise_cost(unit: Unit, p_mw: float, n_segments: int) -> float:
    """区分線形近似で評価した燃料費（無負荷費を除く）[円/h]。

    安いセグメントから順に埋めた値であり、線形計画の最適解が実際に取る
    値と一致する（凸性から、埋める順序は最適解でも安い順になる）。
    元の 2 次曲線に対して **必ず過大評価側**である。
    """
    remaining = float(p_mw) - unit.p_min_mw
    if remaining < -_TOL_MW:
        raise ValueError(
            f"号機 {unit.name} の出力 {p_mw:.6g} MW が最低出力 "
            f"{unit.p_min_mw:.6g} MW を下回っている。"
        )
    cost = _variable_cost(unit, unit.p_min_mw)
    remaining = max(remaining, 0.0)
    for width, slope in _segments(unit, n_segments):
        take = min(remaining, width)
        cost += slope * take
        remaining -= take
        if remaining <= 0.0:
            break
    return cost


# ======================================================================
# 入力の整理と実行可能性の下調べ
# ======================================================================
def _units_for(case: Case, units) -> list[Unit]:
    """号機の指定（``None`` / 名前の列 / :class:`Unit` の列）を正規化する。"""
    if units is None:
        chosen = list(case.units)
    else:
        by_name = {unit.name: unit for unit in case.units}
        chosen = []
        for item in units:
            if isinstance(item, Unit):
                chosen.append(item)
            elif item in by_name:
                chosen.append(by_name[item])
            else:
                raise ValueError(
                    f"号機 '{item}' はケース '{case.name}' に無い。"
                    f"使えるのは {sorted(by_name)}。"
                )
    usable = [unit for unit in chosen if unit.p_max_mw > 0.0]
    if not usable:
        raise ValueError(
            f"ケース '{case.name}' に出力上限が正の号機が 1 つも無い。"
            "units 層を読み込めているか（Case.require('units')）を確認すること。"
        )
    return usable


def _reserve_rate(case: Case, reserve_rate: float | None) -> float:
    """予備力率を決める。``None`` ならケースの ``commitment.reserve_rate``。"""
    if reserve_rate is None:
        rate = float(case.commitment.get("reserve_rate", 0.0) or 0.0)
    else:
        rate = float(reserve_rate)
    if rate < 0.0:
        raise ValueError(f"予備力率が負である: {rate}")
    return rate


def _inputs(case: Case, demand_mw, *, units, reserve_rate, vre_mw) -> dict:
    """需要・純需要・予備力要求・号機をまとめて用意する。"""
    case.require("units")
    unit_list = _units_for(case, units)

    gross = np.asarray(demand_mw, dtype=float).ravel()
    if gross.size == 0:
        raise ValueError("需要の系列が空である。demand_profile(case) などで作ること。")
    if not np.all(np.isfinite(gross)):
        raise ValueError("需要に有限でない値（NaN / inf）が含まれている。")
    if np.any(gross < -_TOL_MW):
        raise ValueError("需要に負の値がある。VRE は vre_mw で渡すこと。")

    vre = _vre_output(case, gross.size, vre_mw)
    net = gross - vre
    rate = _reserve_rate(case, reserve_rate)
    # 予備力要求は純需要に対して課す。純需要が負（VRE が需要を上回る）の
    # 時刻に負の要求を課しても意味が無いので 0 で切る。
    reserve = np.maximum(rate * net, 0.0)
    return {
        "units": unit_list,
        "gross": gross,
        "vre": vre,
        "demand": net,
        "reserve": reserve,
        "reserve_rate": rate,
    }


def _forced_states(unit: Unit, horizon: int) -> tuple[int, int]:
    """期間の頭で強制される運転／停止の時間数（時刻数）を返す。"""
    return (
        min(horizon, unit.remaining_min_up()),
        min(horizon, unit.remaining_min_down()),
    )


def _feasibility_check(units, demand, reserve, *, allow_shortfall: bool) -> None:
    """解く前に **必要条件**だけを調べ、破れていれば日本語で止める。

    ここで見るのは「どうやっても足りない」という量の話だけである。
    必要条件なので、通ったからといって実行可能とは限らない（ランプ率や
    最低運転時間の組み合わせで詰むことはある）。その場合は
    :func:`gridops.solvers.solve` が Infeasible の診断を出す。

    Raises
    ------
    ValueError
        供給力または予備力が原理的に足りない時刻があるとき。
    """
    horizon = len(demand)
    problems: list[str] = []
    for t in range(horizon):
        available = [
            unit for unit in units if t >= min(horizon, unit.remaining_min_down())
        ]
        capacity = sum(unit.p_max_mw for unit in available)
        headroom = sum(unit.p_max_mw - unit.p_min_mw for unit in available)
        if not allow_shortfall and capacity < demand[t] - _TOL_MW:
            problems.append(
                f"  時刻 {t}: 需要 {demand[t]:.1f} MW に対し、起動できる号機の容量は "
                f"{capacity:.1f} MW（{demand[t] - capacity:.1f} MW 不足）"
            )
        if headroom < reserve[t] - _TOL_MW:
            problems.append(
                f"  時刻 {t}: 予備力要求 {reserve[t]:.1f} MW に対し、"
                f"未負荷容量の上限は {headroom:.1f} MW"
                f"（Σ(Pmax - Pmin) が {reserve[t] - headroom:.1f} MW 不足）"
            )
    if not problems:
        return

    shown = problems[:6]
    tail = [] if len(problems) <= 6 else [f"  （ほか {len(problems) - 6} 件）"]
    hint = (
        "  allow_shortfall=True にすると「何時に何 MW 足りないか」が "
        "shortfall_mw に返り、授業を止めずに済む。"
        if not allow_shortfall
        else "  予備力率を下げるか、最低出力の低い号機を足すこと。"
    )
    raise ValueError(
        "起動停止計画が実行不可能である（解く前の必要条件で判定した）。\n"
        + "\n".join(shown + tail)
        + "\n"
        + hint
    )


def _identical_groups(units: Sequence[Unit]) -> list[list[int]]:
    """費用も諸元も完全に同じ号機の添字をまとめる。

    対称性除去 :math:`u_{it} \\ge u_{i+1,t}` を入れてよいのは、この意味で
    **完全に同じ**号機の間だけである。燃料費が 1 円でも違えば、安い方を
    後ろの番号に置いた解が最適である可能性があり、順序を強制すると
    最適解を切り落とす。
    """
    signatures: dict[tuple, list[int]] = {}
    for i, unit in enumerate(units):
        key = (
            unit.p_max_mw, unit.p_min_mw, unit.var_cost, unit.quadratic,
            unit.noload_cost, unit.startup_cost, unit.min_up, unit.min_down,
            unit.ramp_up, unit.ramp_down, unit.su_ramp, unit.sd_ramp,
            unit.u0, unit.hours_in_state,
        )
        signatures.setdefault(key, []).append(i)
    return [group for group in signatures.values() if len(group) > 1]


# ======================================================================
# 混合整数計画（と、入切を固定した線形計画）
# ======================================================================
def _build_problem(
    units: Sequence[Unit],
    demand: np.ndarray,
    reserve: np.ndarray,
    *,
    n_segments: int,
    voll: float,
    spill_price: float,
    allow_shortfall: bool,
    symmetry_breaking: bool,
    schedule: Sequence[Sequence[float]] | None = None,
    name: str = "unit_commitment",
):
    """問題を組み立てる。``schedule`` を与えると入切を固定した線形計画になる。

    1 つの関数で 2 つの問題（混合整数計画と、入切を固定した線形計画）を
    組むのは、**両者が同じ制約を持つことをコードで保証する**ためである。
    別々に書くと、限界費用を取るための線形計画がいつのまにか元の問題と
    違う制約を持つ、という事故が起きる。

    入切を固定した側では :math:`u, v, w` は変数ではなく **数**になる。
    整数変数が 1 つも無いので :meth:`pulp.LpProblem.isMIP` が偽になり、
    :attr:`gridops.solvers.Solution.duals` に双対が入る。
    """
    horizon = len(demand)
    fixed = schedule is not None
    problem = solvers.problem(name)

    segments = [_segments(unit, n_segments) for unit in units]
    p: dict[tuple[int, int], object] = {}
    d: dict[tuple[int, int, int], object] = {}
    u: dict[tuple[int, int], object] = {}
    v: dict[tuple[int, int], object] = {}
    w: dict[tuple[int, int], object] = {}

    for i, unit in enumerate(units):
        for t in range(horizon):
            p[i, t] = solvers.variable(f"p_{unit.name}_{t}", 0.0, unit.p_max_mw)
            for k, (width, _slope) in enumerate(segments[i]):
                d[i, t, k] = solvers.variable(f"d_{unit.name}_{t}_{k}", 0.0, width)
        if fixed:
            row = np.asarray(schedule[i], dtype=float)
            previous = np.concatenate(([float(unit.u0)], row[:-1]))
            for t in range(horizon):
                u[i, t] = float(row[t])
                v[i, t] = max(0.0, float(row[t] - previous[t]))
                w[i, t] = max(0.0, float(previous[t] - row[t]))
        else:
            for t in range(horizon):
                u[i, t] = solvers.variable(f"u_{unit.name}_{t}", 0, 1, cat="Binary")
                # v だけを 0-1 にすれば w は遷移式と v + w <= 1 から一意に
                # 決まる。v を連続にすると、運転を続けている時刻に
                # v = w = 0.5 と置いて起動ランプ SU の半分を「盗む」解が
                # 出てしまう（起動費を半分払う代わりにランプ制約を緩める）。
                v[i, t] = solvers.variable(f"v_{unit.name}_{t}", 0, 1, cat="Binary")
                w[i, t] = solvers.variable(f"w_{unit.name}_{t}", 0.0, 1.0)

    shed = {
        t: solvers.variable(
            f"shed_{t}", 0.0, max(float(demand[t]), 0.0) if allow_shortfall else 0.0
        )
        for t in range(horizon)
    }
    # 抑制の上限は「全機が全出力で、しかも純需要が負」の場合まで見込む。
    # 純需要は VRE が需要を上回れば負になり、そのぶん抑制量は増える。
    total_capacity = float(sum(unit.p_max_mw for unit in units))
    spill = {
        t: solvers.variable(
            f"spill_{t}", 0.0, total_capacity + max(0.0, -float(demand[t]))
        )
        for t in range(horizon)
    }

    # --- 号機ごとの制約 ------------------------------------------------
    for i, unit in enumerate(units):
        must_run, must_off = _forced_states(unit, horizon)
        ramp_up = min(unit.ramp_up, unit.p_max_mw)      # Pmax 以上は非拘束
        ramp_down = min(unit.ramp_down, unit.p_max_mw)

        for t in range(horizon):
            # 出力の上下限。下限 p >= Pmin u は区分線形の等式から従う。
            problem += p[i, t] <= unit.p_max_mw * u[i, t], f"cap_{unit.name}_{t}"
            problem += (
                p[i, t]
                == unit.p_min_mw * u[i, t]
                + solvers.lp_sum([d[i, t, k] for k in range(len(segments[i]))]),
                f"pw_{unit.name}_{t}",
            )

            previous_u = u[i, t - 1] if t > 0 else float(unit.u0)

            # ランプ制約。1 時刻目は「期間直前の出力」が要るが、それは
            # ケースデータに無い。停止していた号機 (u0 = 0) だけは
            # p = 0 と分かるので、その場合に限って課す。運転していた
            # 号機の 1 時刻目に勝手な初期出力を仮定すると、その仮定が
            # そのまま 1 時刻目の意思決定を支配してしまう。
            if t > 0 or unit.u0 == 0:
                previous_p = p[i, t - 1] if t > 0 else 0.0
                problem += (
                    p[i, t] - previous_p
                    <= ramp_up * previous_u + unit.su_ramp * v[i, t],
                    f"rampup_{unit.name}_{t}",
                )
                problem += (
                    previous_p - p[i, t]
                    <= ramp_down * u[i, t] + unit.sd_ramp * w[i, t],
                    f"rampdn_{unit.name}_{t}",
                )

            if fixed:
                continue

            # 状態遷移と、起動・停止が同時に立たないこと。
            problem += (
                u[i, t] - previous_u == v[i, t] - w[i, t], f"link_{unit.name}_{t}"
            )
            problem += v[i, t] + w[i, t] <= 1, f"once_{unit.name}_{t}"

            # 最低運転／停止時間。窓は期間の頭で切り詰める（期間の外の
            # 起動は存在しないので、切り詰めても拘束は失われない）。
            # 期間の頭から持ち越した拘束は下の must_run / must_off が持つ。
            problem += (
                solvers.lp_sum([v[i, s] for s in range(max(0, t - unit.min_up + 1), t + 1)])
                <= u[i, t],
                f"minup_{unit.name}_{t}",
            )
            problem += (
                solvers.lp_sum(
                    [w[i, s] for s in range(max(0, t - unit.min_down + 1), t + 1)]
                )
                <= 1 - u[i, t],
                f"mindown_{unit.name}_{t}",
            )

        if not fixed:
            # 初期条件。remaining_min_up / remaining_min_down の時間だけ、
            # 入切を強制的に固定する。窓和の切り詰めでは代用できない。
            for t in range(must_run):
                problem += u[i, t] == 1, f"mustrun_{unit.name}_{t}"
            for t in range(must_off):
                problem += u[i, t] == 0, f"mustoff_{unit.name}_{t}"

    # --- 系統全体の制約 ------------------------------------------------
    for t in range(horizon):
        # 需給バランス。**右辺に需要を正の符号で置く向き**を守ること。
        # この向きのときだけ双対がそのまま限界費用 [円/MWh] になる
        # （gridops.solvers のモジュール docstring の規約）。
        problem += (
            solvers.lp_sum([p[i, t] for i in range(len(units))]) + shed[t] - spill[t]
            == float(demand[t]),
            f"balance_{t}",
        )
        # 運転予備力は「同期並列している未負荷容量」。
        problem += (
            solvers.lp_sum(
                [units[i].p_max_mw * u[i, t] - p[i, t] for i in range(len(units))]
            )
            >= float(reserve[t]),
            f"reserve_{t}",
        )

    if symmetry_breaking and not fixed:
        for group in _identical_groups(units):
            for first, second in zip(group, group[1:]):
                for t in range(horizon):
                    problem += u[first, t] >= u[second, t], f"sym_{first}_{second}_{t}"

    # --- 目的関数 ------------------------------------------------------
    terms = []
    for i, unit in enumerate(units):
        fixed_hourly = unit.noload_cost + _variable_cost(unit, unit.p_min_mw)
        for t in range(horizon):
            terms.append(fixed_hourly * u[i, t])
            terms.append(unit.startup_cost * v[i, t])
            for k, (_width, slope) in enumerate(segments[i]):
                terms.append(slope * d[i, t, k])
    for t in range(horizon):
        terms.append(voll * shed[t])
        terms.append(spill_price * spill[t])
    problem += solvers.lp_sum(terms)

    return problem


def _read_schedule(units, horizon, solution) -> list[np.ndarray]:
    """解から入切表を取り出す。0-1 の読み取りは :func:`gridops.solvers.binary`。"""
    rows = []
    for unit in units:
        values = [
            solvers.binary(solution.values[f"u_{unit.name}_{t}"])
            for t in range(horizon)
        ]
        rows.append(np.array(values, dtype=float))
    return rows


def _read_dispatch(units, horizon, solution, schedule) -> list[np.ndarray]:
    """解から出力を取り出す。停止中の時刻は厳密に 0 にする。"""
    rows = []
    for i, unit in enumerate(units):
        values = np.array(
            [solution.values[f"p_{unit.name}_{t}"] for t in range(horizon)], dtype=float
        )
        values = np.where(np.asarray(schedule[i]) > 0.5, values, 0.0)
        rows.append(np.where(np.abs(values) < _TOL_MW, 0.0, values))
    return rows


def _cost_breakdown(
    units, schedule, dispatch, shortfall, spill, *, n_segments, voll, spill_price
) -> dict[str, float]:
    """入切と出力から費用の内訳を **数え直す** [円]。

    ソルバの目的関数の値をそのまま分解するのではなく、得られた
    :math:`u` と :math:`p` から独立に計算する。合計が目的関数の値と
    一致することが、モデルの読み方が正しいことの検算になる。
    """
    fuel = noload = startup = 0.0
    for i, unit in enumerate(units):
        u = np.asarray(schedule[i], dtype=float)
        p = np.asarray(dispatch[i], dtype=float)
        previous = np.concatenate(([float(unit.u0)], u[:-1]))
        noload += unit.noload_cost * float(u.sum())
        startup += unit.startup_cost * float(np.maximum(u - previous, 0.0).sum())
        for t in range(u.size):
            if u[t] > 0.5:
                fuel += _piecewise_cost(unit, float(p[t]), n_segments)
    penalty = voll * float(np.sum(shortfall)) + spill_price * float(np.sum(spill))
    return {"fuel": fuel, "noload": noload, "startup": startup, "penalty": penalty}


def unit_commitment(
    case: Case,
    demand_mw,
    *,
    reserve_rate: float | None = None,
    vre_mw=None,
    n_segments: int = DEFAULT_SEGMENTS,
    allow_shortfall: bool = True,
    voll: float = DEFAULT_VOLL,
    time_limit: float = 60.0,
    gap: float = 1e-4,
    symmetry_breaking: bool = True,
) -> CommitmentResult:
    """混合整数計画で起動停止計画を解く（PuLP + CBC）。

    制約と設計の意図はモジュール docstring に書いた。ここでは引数と、
    引数によって何が変わるかだけを述べる。

    Parameters
    ----------
    case:
        ``units`` 層を持つケース。
    demand_mw:
        ``(T,)`` の需要 [MW]。:func:`demand_profile` で作る。
    reserve_rate:
        予備力率。``None`` ならケースの ``commitment.reserve_rate``。
        予備力要求は :math:`R_t = r D_t`（純需要に対して）である。
    vre_mw:
        ``None`` なら **VRE 無し**。スカラーならその設備容量 [MW] で
        ケースの出力形状を使う。配列なら VRE 出力 [MW] そのもの。
        （:func:`net_demand` の ``vre_mw=None`` とは意味が違う。そちらは
        ケースの VRE 層を使う。）
    n_segments:
        区分線形近似の分割数。1 でも解は出るが、費用は過大評価になる。
    allow_shortfall:
        供給不足 ``shed`` を許すか。既定は ``True``。``False`` にすると
        供給力が足りない時刻がある場合に日本語の :class:`ValueError` になる。
    voll:
        供給支障費用 [円/MWh]。``shed`` に掛かる。
    time_limit, gap:
        ソルバの打ち切り時間 [s] と相対ギャップ。既定のギャップ 1e-4 は
        「最適値から 0.01% 以内」の意味で、**総費用を厳密に比較したい
        テストでは 1e-9 程度まで絞ること**。
    symmetry_breaking:
        費用も諸元も完全に同じ号機に :math:`u_{it} \\ge u_{i+1,t}` を
        入れるか。同梱ケース wscc9 では同一プラント内でも燃料費をずらして
        あるので、**既定でも 1 本も入らない**（そういう号機の組が無い）。

    Returns
    -------
    CommitmentResult

    Raises
    ------
    ValueError
        供給力または予備力が原理的に足りないとき（解く前に判定する）。
        実行不可能・非有界のときは :func:`gridops.solvers.solve` が投げる。

    Notes
    -----
    **1 時刻目のランプ制約**: ランプ制約には計画期間の直前の出力
    :math:`p_{i,-1}` が要るが、ケースデータにあるのは運転状態 :math:`u^0`
    だけである。停止していた号機なら :math:`p_{i,-1} = 0` と分かるので、
    1 時刻目のランプ制約は **その号機にだけ**課す（起動した時刻の上限
    :math:`SU` はこれで効く）。運転していた号機については、1 時刻目の
    ランプ制約を課さない。:math:`P^{min}` などの値を仮に置くと、その仮定が
    そのまま 1 時刻目の入切を支配してしまうためである（wscc9 では
    「1 時刻目に全機を運転し続けなければ需要を賄えない」という、データに
    無い結論が出る）。知らないことは知らないままにしておくほうがよい。

    **spill は既定で常に入る。** 純需要が起動中の号機の
    :math:`\\sum P^{min}` を下回る時刻（太陽光の多い正午前後）で、
    spill を入れないと問題は実行不可能になる。「解けない」ではなく
    「抑制した」と答えるのが正しい。

    Examples
    --------
    >>> case = load_case("wscc9")                      # doctest: +SKIP
    >>> demand = demand_profile(case)                  # doctest: +SKIP
    >>> result = unit_commitment(case, demand)         # doctest: +SKIP
    >>> print(result.to_table())                       # doctest: +SKIP
    """
    data = _inputs(
        case, demand_mw, units=None, reserve_rate=reserve_rate, vre_mw=vre_mw
    )
    units = data["units"]
    demand, reserve = data["demand"], data["reserve"]
    spill_price = voll * SPILL_PRICE_RATIO
    _feasibility_check(units, demand, reserve, allow_shortfall=allow_shortfall)

    problem = _build_problem(
        units,
        demand,
        reserve,
        n_segments=n_segments,
        voll=voll,
        spill_price=spill_price,
        allow_shortfall=allow_shortfall,
        symmetry_breaking=symmetry_breaking,
        name="unit_commitment",
    )
    solution = solvers.solve(
        problem,
        context=f"起動停止計画 (T={len(demand)}, 号機 {len(units)} 台, "
        f"予備力率 {data['reserve_rate']:.0%})",
        time_limit=time_limit,
        gap=gap,
    )

    horizon = len(demand)
    schedule = _read_schedule(units, horizon, solution)
    dispatch = _read_dispatch(units, horizon, solution, schedule)
    shortfall = np.array([solution.values[f"shed_{t}"] for t in range(horizon)])
    spill = np.array([solution.values[f"spill_{t}"] for t in range(horizon)])
    shortfall = np.where(np.abs(shortfall) < _TOL_MW, 0.0, shortfall)
    spill = np.where(np.abs(spill) < _TOL_MW, 0.0, spill)

    return _result(
        case,
        data,
        units,
        schedule,
        dispatch,
        shortfall,
        spill,
        status=solution.status,
        total_cost=float(solution.objective),
        seconds=solution.seconds,
        method="milp",
        n_segments=n_segments,
        voll=voll,
        spill_price=spill_price,
        allow_shortfall=allow_shortfall,
        symmetry_breaking=symmetry_breaking,
        vre_mw=vre_mw,
    )


def _result(
    case,
    data,
    units,
    schedule,
    dispatch,
    shortfall,
    spill,
    *,
    status,
    total_cost,
    seconds,
    method,
    n_segments,
    voll,
    spill_price,
    allow_shortfall,
    symmetry_breaking,
    vre_mw,
) -> CommitmentResult:
    """共通の後処理。費用の内訳を数え直して :class:`CommitmentResult` を組む。"""
    breakdown = _cost_breakdown(
        units,
        schedule,
        dispatch,
        shortfall,
        spill,
        n_segments=n_segments,
        voll=voll,
        spill_price=spill_price,
    )
    return CommitmentResult(
        case=case,
        demand_mw=data["demand"],
        status=status,
        schedule={unit.name: schedule[i] for i, unit in enumerate(units)},
        dispatch={unit.name: dispatch[i] for i, unit in enumerate(units)},
        shortfall_mw=np.asarray(shortfall, dtype=float),
        spill_mw=np.asarray(spill, dtype=float),
        total_cost=float(total_cost),
        cost_breakdown=breakdown,
        seconds=float(seconds),
        method=method,
        options={
            "reserve_rate": data["reserve_rate"],
            "reserve_mw": data["reserve"],
            "gross_demand_mw": data["gross"],
            "vre_mw": data["vre"],
            "vre_spec": vre_mw,
            "n_segments": int(n_segments),
            "voll": float(voll),
            "spill_price": float(spill_price),
            "allow_shortfall": bool(allow_shortfall),
            "symmetry_breaking": bool(symmetry_breaking),
            "unit_names": tuple(unit.name for unit in units),
        },
    )


# ======================================================================
# 優先順位法
# ======================================================================
def _priority_schedule(units, demand, reserve) -> list[np.ndarray]:
    """全負荷平均費用の安い順に、需要 + 予備力を満たすまで起動する。

    最低運転停止時間は「状態機械」で事後的に均す。すなわち、止めたい／
    起動したいと思っても、最低運転（停止）時間を満たしていなければその
    まま続ける。この均し方は初期条件とも自動的に整合する
    （:meth:`~gridops.case.Unit.remaining_min_up` の時間だけ、止めたくても
    止まらない）。**厳密でないのはここではなく順位づけの側**である。
    全負荷平均費用は起動費も最低運転時間も見ていないので、
    「いま安い」ことと「1 日を通して安い」ことの区別がつかない。
    """
    horizon = len(demand)
    count = len(units)
    order = sorted(range(count), key=lambda i: (units[i].full_load_average_cost(), i))
    state = [int(unit.u0) for unit in units]
    timer = [int(unit.hours_in_state) for unit in units]
    rows = [np.zeros(horizon) for _ in units]

    for t in range(horizon):
        can_stop = [state[i] == 0 or timer[i] >= units[i].min_up for i in range(count)]
        can_start = [state[i] == 1 or timer[i] >= units[i].min_down for i in range(count)]
        # 止めたくても止められない号機は、無条件に運転が続く。
        committed = {i for i in range(count) if state[i] == 1 and not can_stop[i]}
        capacity = sum(units[i].p_max_mw for i in committed)
        need = demand[t] + reserve[t]
        for i in order:
            if capacity >= need - _TOL_MW:
                break
            if i in committed or not can_start[i]:
                continue
            committed.add(i)
            capacity += units[i].p_max_mw
        for i in range(count):
            new_state = 1 if i in committed else 0
            if new_state != state[i]:
                state[i] = new_state
                timer[i] = 1
            else:
                timer[i] += 1
            rows[i][t] = float(state[i])
    return rows


def priority_list(
    case: Case, demand_mw, *, reserve_rate: float | None = None, vre_mw=None
) -> CommitmentResult:
    """優先順位法（ヒューリスティック）で起動停止計画を作る。

    全負荷平均費用 :math:`C(P^{max})/P^{max}` の安い順に、需要と予備力を
    満たすまで号機を起動する。最低運転停止時間は事後的に均す
    （:func:`_priority_schedule` を参照）。

    **この方法は厳密ではない。** 全負荷平均費用は「1 時間だけ見た安さ」
    であり、次の 3 つを見ていない。

    1. 起動費（何度も入り切りすると、燃料費の差はすぐに食い潰される）
    2. 最低運転停止時間（安い機を止めると、必要になっても戻せない）
    3. ランプ率（形を追えない）

    それでも 1970 年代まで実務で使われ、いまも初期解として使われる。
    混合整数計画との **費用の差**（その大半は起動費に出る）が、
    0-1 の意思決定を最適化することの価値そのものである。

    出力は入切を固定した線形計画で決めるので、ランプ率と予備力は
    厳密に満たされる。費用は :func:`unit_commitment` と同じ区分線形近似
    （``n_segments`` = :data:`DEFAULT_SEGMENTS`）で評価するので、
    総費用をそのまま比較してよい。

    Returns
    -------
    CommitmentResult
        ``method="priority"``。供給不足は常に許す（ヒューリスティックが
        作った入切表では足りない時刻がありうる。それを隠さずに返す）。
    """
    data = _inputs(
        case, demand_mw, units=None, reserve_rate=reserve_rate, vre_mw=vre_mw
    )
    units = data["units"]
    demand, reserve = data["demand"], data["reserve"]
    _feasibility_check(units, demand, reserve, allow_shortfall=True)

    start = time.perf_counter()
    schedule = _priority_schedule(units, demand, reserve)
    voll = DEFAULT_VOLL
    spill_price = voll * SPILL_PRICE_RATIO
    problem = _build_problem(
        units,
        demand,
        reserve,
        n_segments=DEFAULT_SEGMENTS,
        voll=voll,
        spill_price=spill_price,
        allow_shortfall=True,
        symmetry_breaking=False,
        schedule=schedule,
        name="priority_list",
    )
    solution = solvers.solve(
        problem, context=f"優先順位法の出力配分 (T={len(demand)})"
    )
    seconds = time.perf_counter() - start

    horizon = len(demand)
    dispatch = _read_dispatch(units, horizon, solution, schedule)
    shortfall = np.array([solution.values[f"shed_{t}"] for t in range(horizon)])
    spill = np.array([solution.values[f"spill_{t}"] for t in range(horizon)])
    shortfall = np.where(np.abs(shortfall) < _TOL_MW, 0.0, shortfall)
    spill = np.where(np.abs(spill) < _TOL_MW, 0.0, spill)

    return _result(
        case, data, units, schedule, dispatch, shortfall, spill,
        status=solution.status,
        total_cost=float(solution.objective),
        seconds=seconds,
        method="priority",
        n_segments=DEFAULT_SEGMENTS,
        voll=voll,
        spill_price=spill_price,
        allow_shortfall=True,
        symmetry_breaking=False,
        vre_mw=vre_mw,
    )


# ======================================================================
# 全列挙（テスト専用の基準）
# ======================================================================
def _feasible_schedules(unit: Unit, horizon: int) -> list[tuple[int, ...]]:
    """最低運転停止時間と初期条件を満たす入切表をすべて列挙する。

    期間の終わりで窓が入りきらない場合（残り時間 < 最低運転時間）は、
    そこで打ち切ることを許す。混合整数計画側の窓和も同じ扱いなので、
    両者の実行可能領域は一致する。
    """
    results: list[tuple[int, ...]] = []
    path: list[int] = []

    def walk(t: int, state: int, timer: int) -> None:
        if t == horizon:
            results.append(tuple(path))
            return
        for nxt in (0, 1):
            if nxt != state:
                if state == 1 and timer < unit.min_up:
                    continue
                if state == 0 and timer < unit.min_down:
                    continue
                path.append(nxt)
                walk(t + 1, nxt, 1)
            else:
                path.append(nxt)
                walk(t + 1, state, timer + 1)
            path.pop()

    walk(0, int(unit.u0), int(unit.hours_in_state))
    return results


def _hour_dispatch(units, segments, on, previous, following, demand, reserve,
                   *, voll, spill_price):
    """1 時刻の最小費用配分を貪欲法で解く。

    入切が決まっていて **ランプ率が拘束しない**なら、各時刻は独立に解ける。
    可変費用は凸な区分線形なので、最適解は「安いセグメントから順に埋める」
    で得られる（線形計画を呼ぶ必要がない）。全列挙をソルバから独立させる
    ための要点である。

    ``on`` / ``previous`` / ``following`` は 0-1 のビットマスク。
    ``following`` が ``None`` なら期間の最終時刻。

    Returns
    -------
    tuple | None
        ``(費用, 出力配列, 供給不足, 出力抑制)``。この入切では実行不可能な
        ときは ``None``（予備力が確保できない、など）。
    """
    count = len(units)
    low = np.zeros(count)
    high = np.zeros(count)
    cost = 0.0
    capacity = 0.0
    for i, unit in enumerate(units):
        if not (on >> i) & 1:
            continue
        capacity += unit.p_max_mw
        cost += unit.noload_cost + _variable_cost(unit, unit.p_min_mw)
        upper = unit.p_max_mw
        if not (previous >> i) & 1:
            cost += unit.startup_cost
            upper = min(upper, unit.su_ramp)          # 起動した時刻の上限
        if following is not None and not (following >> i) & 1:
            upper = min(upper, unit.sd_ramp)          # 次の時刻に止まる
        if upper < unit.p_min_mw - _TOL_MW:
            return None
        low[i] = unit.p_min_mw
        high[i] = max(upper, unit.p_min_mw)

    total_low = float(low.sum())
    total_high = min(float(high.sum()), capacity - float(reserve))
    if total_high < total_low - _TOL_MW:
        return None                                    # 予備力が確保できない

    target = min(max(float(demand), total_low), total_high)
    shed = max(0.0, float(demand) - total_high)
    spill = max(0.0, total_low - float(demand))

    blocks: list[tuple[float, int, float]] = []
    for i, unit in enumerate(units):
        if not (on >> i) & 1:
            continue
        room = high[i] - low[i]
        for width, slope in segments[i]:
            if room <= 0.0:
                break
            take = min(width, room)
            blocks.append((slope, i, take))
            room -= take
    blocks.sort(key=lambda block: block[0])

    output = low.copy()
    remaining = target - total_low
    for slope, i, width in blocks:
        if remaining <= 0.0:
            break
        take = min(width, remaining)
        output[i] += take
        cost += slope * take
        remaining -= take

    cost += voll * shed + spill_price * spill
    return cost, output, shed, spill


def enumerate_commitment(
    case: Case,
    demand_mw,
    *,
    reserve_rate: float | None = None,
    units=None,
    vre_mw=None,
) -> CommitmentResult:
    """全列挙で厳密な最適解を求める（**テストの基準専用**）。

    分枝限定法とはまったく別の道筋、すなわち「すべての入切表を作って
    数える」で同じ答えに到達することを確かめるための関数である。
    :math:`2^{nT}` 通りなので、号機数 x 時間数が小さいときにしか使えない。

    出力の配分にも線形計画を使わない。入切が決まっていてランプ率が
    拘束しないなら各時刻は独立に解け、凸な区分線形費用は
    「安いセグメントから順に埋める」で最適になる。ソルバから完全に
    独立した基準にするための設計である。

    Parameters
    ----------
    units:
        使う号機（:class:`~gridops.case.Unit` か号機名の列）。
        ``None`` ならケースの全号機。

    Returns
    -------
    CommitmentResult
        ``method="enumeration"``、``status="Optimal"``。区分線形の分割数と
        VOLL は :data:`DEFAULT_SEGMENTS` / :data:`DEFAULT_VOLL`、すなわち
        :func:`unit_commitment` の既定と同じ値を使う（総費用をそのまま
        比較できるようにするため）。

    Raises
    ------
    ValueError
        候補数が :data:`ENUMERATION_LIMIT` を超えるとき。
        ランプ率が拘束しうる号機が含まれているとき（時刻を独立に解けなく
        なるため。``dataclasses.replace(unit, ramp_up=math.inf)`` で外すか、
        :func:`unit_commitment` を使うこと）。
    """
    data = _inputs(
        case, demand_mw, units=units, reserve_rate=reserve_rate, vre_mw=vre_mw
    )
    unit_list = data["units"]
    demand, reserve = data["demand"], data["reserve"]
    horizon = len(demand)
    count = len(unit_list)

    if count * horizon > 20:
        raise ValueError(
            f"全列挙の規模が大きすぎる: 号機 {count} 台 x {horizon} 時刻 = "
            f"2^{count * horizon} 通り。"
            "units= で号機を絞るか、需要を短く切ること"
            "（この関数はテストの基準専用である。実用には unit_commitment を使う）。"
        )

    binding = [
        unit.name
        for unit in unit_list
        if min(unit.ramp_up, unit.ramp_down) < unit.p_max_mw - unit.p_min_mw - _TOL_MW
    ]
    if binding:
        raise ValueError(
            f"ランプ率が拘束しうる号機がある: {binding}。"
            "全列挙は各時刻を独立に解くので、時間方向に結合するランプ制約を"
            "扱えない（扱うと候補ごとに線形計画が要り、全列挙が終わらない）。\n"
            "  dataclasses.replace(unit, ramp_up=math.inf, ramp_down=math.inf) で"
            "外した号機で比べるか、unit_commitment を使うこと。\n"
            f"  拘束しない条件は ramp >= Pmax - Pmin である。"
        )

    per_unit = [_feasible_schedules(unit, horizon) for unit in unit_list]
    candidates = 1
    for schedules in per_unit:
        candidates *= len(schedules)
    if candidates > ENUMERATION_LIMIT:
        raise ValueError(
            f"最低運転停止時間で絞ったあとでも候補が {candidates:,} 通りあり、"
            f"上限 {ENUMERATION_LIMIT:,} を超える。号機か時間数を減らすこと。"
        )

    segments = [_segments(unit, DEFAULT_SEGMENTS) for unit in unit_list]
    voll = DEFAULT_VOLL
    spill_price = voll * SPILL_PRICE_RATIO
    initial_mask = sum(1 << i for i, unit in enumerate(unit_list) if unit.u0)
    cache: dict[tuple[int, int, int, int], float | None] = {}

    def hour_cost(t: int, previous: int, on: int, following: int) -> float | None:
        key = (t, previous, on, following)
        if key not in cache:
            outcome = _hour_dispatch(
                unit_list, segments, on, previous,
                None if following < 0 else following,
                demand[t], reserve[t], voll=voll, spill_price=spill_price,
            )
            cache[key] = None if outcome is None else outcome[0]
        return cache[key]

    start = time.perf_counter()
    best_cost = math.inf
    best_masks: tuple[int, ...] | None = None

    def walk(index: int, masks: list[int]) -> None:
        nonlocal best_cost, best_masks
        if index == count:
            total = 0.0
            for t in range(horizon):
                cost = hour_cost(
                    t,
                    masks[t - 1] if t > 0 else initial_mask,
                    masks[t],
                    masks[t + 1] if t + 1 < horizon else -1,
                )
                if cost is None:
                    return
                total += cost
                if total >= best_cost:
                    return
            best_cost = total
            best_masks = tuple(masks)
            return
        bit = 1 << index
        for schedule in per_unit[index]:
            walk(index + 1, [m + (bit if s else 0) for m, s in zip(masks, schedule)])

    walk(0, [0] * horizon)
    seconds = time.perf_counter() - start

    if best_masks is None:
        raise ValueError(
            "全列挙で実行可能な入切表が 1 つも見つからなかった。"
            "予備力率が高すぎないか、最低運転停止時間の初期条件と需要が"
            "両立しているかを確認すること。"
        )

    schedule = [
        np.array([float((best_masks[t] >> i) & 1) for t in range(horizon)])
        for i in range(count)
    ]
    dispatch = [np.zeros(horizon) for _ in range(count)]
    shortfall = np.zeros(horizon)
    spill = np.zeros(horizon)
    for t in range(horizon):
        outcome = _hour_dispatch(
            unit_list, segments, best_masks[t],
            best_masks[t - 1] if t > 0 else initial_mask,
            best_masks[t + 1] if t + 1 < horizon else None,
            demand[t], reserve[t], voll=voll, spill_price=spill_price,
        )
        _cost, output, shed_t, spill_t = outcome
        for i in range(count):
            dispatch[i][t] = output[i]
        shortfall[t] = shed_t
        spill[t] = spill_t

    return _result(
        case, data, unit_list, schedule, dispatch, shortfall, spill,
        status="Optimal",
        total_cost=best_cost,
        seconds=seconds,
        method="enumeration",
        n_segments=DEFAULT_SEGMENTS,
        voll=voll,
        spill_price=spill_price,
        allow_shortfall=True,
        symmetry_breaking=False,
        vre_mw=vre_mw,
    )


# ======================================================================
# 限界費用（入切を固定した線形計画の双対）
# ======================================================================
def marginal_prices(case: Case, result: CommitmentResult) -> np.ndarray:
    """時間別の限界費用 [円/MWh] を求める。

    **混合整数計画は双対を返さない。** 整数変数を含む問題の最適値は右辺
    について階段状に変化するので、微分（＝双対）がそもそも存在しない。
    CBC は分枝限定の最後の緩和問題の双対を返してくるが、その値は探索の
    経路に依存し、限界費用としての意味を持たない
    （:mod:`gridops.solvers` はこれを空の辞書にして堰き止めている）。

    そこで 2 段階にする。

    1. 混合整数計画で入切 :math:`u` を決める（:func:`unit_commitment`）
    2. その :math:`u` を **定数として固定**し、同じ制約の線形計画に
       落とし直して需給バランス制約の双対を取る

    得られる価格は「その時刻に 1 MWh 追加で必要になったとき、いま動かせる
    号機を動かして賄う費用」である。起動費や無負荷費は、入切が固定されて
    いる以上この価格には現れない。**限界費用の合計は総費用に一致しない**
    のはそのためで、これは誤りではなく起動停止問題の性質である
    （固定費を回収できない、という卸電力市場の古典的な論点そのもの）。

    価格が :math:`P^{max}` に張り付いた時刻では、価格は最後に動かせた
    号機の増分費用に跳ね上がる。供給不足が立っている時刻では VOLL に
    なる（需要を 1 MWh 増やすと、その分がそのまま供給不足になるため）。

    Parameters
    ----------
    case:
        ``result`` を作ったときと同じケース。
    result:
        :func:`unit_commitment` などの結果。``options`` に入っている
        設定（予備力・区分線形の分割数・VOLL）をそのまま使う。

    Returns
    -------
    numpy.ndarray
        ``(T,)`` の限界費用 [円/MWh]。
    """
    options = dict(result.options or {})
    names = options.get("unit_names") or tuple(result.schedule)
    units = _units_for(case, names)
    demand = np.asarray(result.demand_mw, dtype=float)
    horizon = demand.size
    schedule = [np.asarray(result.schedule[unit.name], dtype=float) for unit in units]
    reserve = np.asarray(
        options.get("reserve_mw", np.zeros(horizon)), dtype=float
    )
    n_segments = int(options.get("n_segments", DEFAULT_SEGMENTS))
    voll = float(options.get("voll", DEFAULT_VOLL))
    spill_price = float(options.get("spill_price", voll * SPILL_PRICE_RATIO))

    problem = _build_problem(
        units,
        demand,
        reserve,
        n_segments=n_segments,
        voll=voll,
        spill_price=spill_price,
        allow_shortfall=bool(options.get("allow_shortfall", True)),
        symmetry_breaking=False,
        schedule=schedule,
        name="marginal_prices",
    )
    solution = solvers.solve(
        problem, context=f"限界費用（入切を固定した線形計画, T={horizon}）"
    )
    if not solution.duals:
        raise ValueError(
            "入切を固定した線形計画から双対が取れなかった。"
            "整数変数が残っていないか（u が固定されているか）を確認すること。"
        )
    return np.array([solution.duals[f"balance_{t}"] for t in range(horizon)])
