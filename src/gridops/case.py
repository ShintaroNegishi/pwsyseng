"""科目全体で共有する系統データ。

安定度の教材 genstab の ``cases/wscc9.yaml`` は、母線の ``(v, angle_deg)``
と発電機の ``(p, q)`` を **入力として** 持っている。潮流計算を対象外に
していたのでそれで正しかったが、潮流を学生自身に解かせるには、入力と解を
層に分けなければならない。答えを読み込んで答えを出す教材になってしまう
ためである。そこで本パッケージのケースは次の層に分かれている。

===============  ====================================================
層               内容
===============  ====================================================
``network``      母線種別・設定電圧・需要・線路（熱容量つき）
``units``        号機の諸元（容量・費用・信頼度・慣性）
``commitment``   時系列需要と予備力率
``reliability``  年間需要の生成パラメータ
``stability``    事故スケジュール（genstab へ渡す）
``solution``     潮流解。**答え合わせと初期値にだけ使い、入力には使わない**
===============  ====================================================

テーマごとに別スキーマを作ることはしない。1 つのファイルに全テーマ分の
層を置き、その回で使う層だけを読む。学生が回ごとに系統を覚え直さずに
済むことを優先した。層が欠けているケースを使うと :meth:`Case.require` が
「その層はどの回で必要になるか」を添えて ``ValueError`` を投げる。

単位の層
--------
本パッケージは 2 つの単位系を意図的に混在させ、**識別子の接尾辞で区別する**。

===========================================  ==========  ==============
量                                           単位        接尾辞
===========================================  ==========  ==============
ネットワーク量（電圧・潮流・インピーダンス）  p.u.        なし
費用量（容量・出力・ランプ率・燃料費）        MW, 円      ``_mw``
===========================================  ==========  ==============

両者の変換は :meth:`Case.to_mw` と :meth:`Case.to_pu` の 2 つだけを通す。
研究室の既存コード ``OPF/OPF.py`` は p.u. 化した出力を $/(MWh)^2 の係数に
そのまま代入しており、二次項が 10^4 倍小さくなっていた。同じ事故を
規約で構造的に防ぐのが目的である。

符号の規約
----------
母線への **注入** を正とする。すなわち発電は正、負荷は負である。
注入を組み立てる箇所は :meth:`Case.bus_injection` 1 箇所に閉じている。
負荷を「負の発電」として足し込む符号ミスは最も頻出するバグなので、
:class:`Bus` が ``pd`` / ``qd`` を持ち、合成をこのメソッドだけが行う。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Iterable, Mapping, Sequence

import numpy as np

#: 層の名前と、それを最初に使う notebook の番号。
#: :meth:`Case.require` が欠落を報告するときの案内に使う。
BLOCK_LESSONS = {
    "network": "01（Ybus）",
    "units": "05（経済負荷配分）",
    "commitment": "07（起動停止計画）",
    "reliability": "10（アデカシー）",
    "stability": "genstab の 01（動揺方程式）",
    "solution": "02（潮流計算の答え合わせ）",
    "contingencies": "09（セキュリティ）",
}


#: 枝の許容容量として指定できる属性名。これ以外（``x`` や ``tap`` など）を
#: 容量として使うと、もっともらしいが無意味な潮流制約になる。
RATING_ATTRIBUTES = ("rate_a", "rate_b")


def validate_rating_attribute(limit: str) -> str:
    """``limit`` が枝の許容容量の属性名であることを確かめて返す。

    ``getattr(branch, limit)`` で読む実装は、放っておくと ``x`` や ``tap``
    のような **容量でない属性**も受け取れてしまう。その場合エラーには
    ならず、リアクタンスを送電容量と解釈した「もっともらしい誤答」が
    返る（外部レビューの指摘 #2）。入口で名前を検査して防ぐ。
    """
    if limit not in RATING_ATTRIBUTES:
        raise ValueError(
            f"limit={limit!r} は枝の許容容量ではない。"
            f"使えるのは {RATING_ATTRIBUTES}（常時 / 緊急時）。"
        )
    return limit


class BusType(Enum):
    """母線の種別。未知数と方程式の対応を決める。

    潮流計算では母線ごとに 4 つの量 :math:`(P, Q, |V|, \\theta)` のうち
    2 つを与え、残り 2 つを求める。どの 2 つを与えるかがこの種別である。
    """

    SLACK = "slack"   #: :math:`|V|, \\theta` を与え :math:`P, Q` を求める（ちょうど 1 つ）
    PV = "pv"         #: :math:`P, |V|` を与え :math:`Q, \\theta` を求める
    PQ = "pq"         #: :math:`P, Q` を与え :math:`|V|, \\theta` を求める


@dataclass(frozen=True)
class Bus:
    """母線。潮流計算の *入力* として必要な量だけを持つ。

    電圧の解は :class:`ReferenceSolution` 側にあり、こちらには入らない。

    Parameters
    ----------
    id:
        母線番号。ケースファイルの中で一意であること。
    type:
        母線種別。
    v_set:
        PV / slack 母線の設定電圧 [p.u.]。PQ 母線では初期値としてのみ使う。
    pd, qd:
        負荷の有効・無効電力 [p.u.]。**正の値が負荷**であり、注入としては
        符号が反転する（:meth:`Case.bus_injection` が処理する）。
    gs, bs:
        母線に直接つながるシャントのコンダクタンス・サセプタンス [p.u.]。
        調相設備を表すのに使う。
    v_min, v_max:
        運用上の電圧の下限・上限 [p.u.]。N-1 の判定に使う。
    """

    id: int
    type: BusType = BusType.PQ
    v_set: float = 1.0
    pd: float = 0.0
    qd: float = 0.0
    gs: float = 0.0
    bs: float = 0.0
    v_min: float = 0.95
    v_max: float = 1.05
    name: str = ""

    @property
    def label(self) -> str:
        """作図と表示に使う名前。空なら ``bus 5`` の形にする。"""
        return self.name or f"bus {self.id}"


@dataclass(frozen=True)
class Branch:
    """線路または変圧器（π 型等価回路 + 理想変圧器）。

    タップ比と位相調整角をまとめて :math:`\\bar a = \\tau e^{j\\phi}` と置くと、
    枝の 2x2 アドミタンス行列は

    .. math::

        Y_{ff} = \\frac{y_s + jb/2}{\\tau^2}, \\qquad
        Y_{ft} = -\\frac{y_s}{\\bar a^{*}}, \\qquad
        Y_{tf} = -\\frac{y_s}{\\bar a}, \\qquad
        Y_{tt} = y_s + jb/2

    となる（タップを from 側に置く MATPOWER の規約）。``tap=1``,
    ``shift_deg=0`` のとき素の π 型に縮退し、``genstab.multimachine.Branch``
    と厳密に一致する。

    Notes
    -----
    ``rate_a`` / ``rate_b`` は **皮相電力** :math:`|S|` の制限 [p.u.] である。
    直流潮流の有効電力 P と比べてはいけない。WSCC 9 母線の枝 4-5 では
    交流の :math:`|S|` が直流の P より 47.6% 大きい（0.5614 対 0.3803）。

    ``shift_deg != 0`` があると :math:`Y_{ft} \\ne Y_{tf}` となり **Ybus は
    非対称になる**。対称性を仮定したコードやテストを書かないこと。
    """

    from_bus: int
    to_bus: int
    r: float = 0.0            #: 直列抵抗 [p.u.]
    x: float = 0.1            #: 直列リアクタンス [p.u.]
    b: float = 0.0            #: 全充電サセプタンス [p.u.]（両端に半分ずつ配分）
    tap: float = 1.0          #: タップ比 :math:`\\tau`
    shift_deg: float = 0.0    #: 位相調整角 :math:`\\phi` [deg]
    rate_a: float = math.inf  #: 常時許容容量 :math:`|S|` [p.u.]
    rate_b: float = math.inf  #: 緊急時許容容量 :math:`|S|` [p.u.]（N-1 判定に使う）
    name: str = ""

    def key(self) -> tuple[int, int]:
        """母線番号の順序によらない枝の識別子。"""
        return (min(self.from_bus, self.to_bus), max(self.from_bus, self.to_bus))

    @property
    def label(self) -> str:
        return self.name or f"{self.from_bus}-{self.to_bus}"

    def series_admittance(self) -> complex:
        """直列アドミタンス :math:`y_s = 1/(r + jx)`。"""
        impedance = complex(self.r, self.x)
        if impedance == 0:
            raise ValueError(
                f"枝 {self.label} のインピーダンスがゼロ。"
                "ゼロインピーダンス枝は Ybus を特異にする。"
                "母線を 1 つに統合するか、小さな x を与えること。"
            )
        return 1.0 / impedance

    def primitive(self) -> np.ndarray:
        """2x2 の枝アドミタンス行列 ``[[Y_ff, Y_ft], [Y_tf, Y_tt]]``。

        Ybus の組み立ては、この行列を母線番号の位置に足し込むだけになる。
        タップと位相調整の扱いをここ 1 箇所に閉じるのが狙いである。
        """
        y_s = self.series_admittance()
        y_shunt = 1j * self.b / 2.0
        a = self.tap * np.exp(1j * math.radians(self.shift_deg))
        if self.tap <= 0:
            raise ValueError(f"枝 {self.label} のタップ比が非正: tap={self.tap}")
        return np.array(
            [
                [(y_s + y_shunt) / (self.tap**2), -y_s / np.conj(a)],
                [-y_s / a, y_s + y_shunt],
            ],
            dtype=complex,
        )


@dataclass(frozen=True)
class Unit:
    """号機。発電所は「同一母線に並ぶ号機の集合」として表す。

    テーマごとに使うフィールドが違う。潮流計算だけなら ``bus`` と無効電力
    制限があれば足り、使わないものは既定値のままでよい。

    出力と費用は必ず MW と円である（``_mw`` 接尾辞が目印）。無効電力の
    制限だけは電圧の話なので p.u. で持つ。

    Parameters
    ----------
    plant:
        発電所名。同じ ``plant`` を持つ号機は、安定度の計算に渡すときに
        1 台へ集約される（慣性定数は加算、過渡リアクタンスは並列合成）。
    u0, hours_in_state:
        計画期間が始まる直前の運転状態と、その状態が続いている時間 [h]。
        最低運転時間・最低停止時間の初期条件になる。
    """

    name: str
    bus: int
    plant: str = ""

    # --- 出力と無効電力 ------------------------------------------------
    p_max_mw: float = 0.0
    p_min_mw: float = 0.0
    q_min: float = -math.inf        #: [p.u.]
    q_max: float = math.inf         #: [p.u.]

    # --- 経済負荷配分・起動停止計画 ------------------------------------
    var_cost: float = 0.0           #: 燃料費の 1 次係数 [円/MWh]
    quadratic: float = 0.0          #: 燃料費の 2 次係数 [円/(MW^2 h)]
    noload_cost: float = 0.0        #: 無負荷費 [円/h]
    startup_cost: float = 0.0       #: 起動費 [円/回]
    min_up: int = 1                 #: 最低運転時間 [h]
    min_down: int = 1               #: 最低停止時間 [h]
    ramp_up: float = math.inf       #: 増出力率 [MW/h]
    ramp_down: float = math.inf     #: 減出力率 [MW/h]
    startup_ramp: float | None = None    #: 起動した時刻に到達できる出力 [MW]
    shutdown_ramp: float | None = None   #: 停止する直前に許される出力 [MW]
    u0: int = 0                     #: 期間直前の運転状態（1: 運転, 0: 停止）
    hours_in_state: int = 24        #: その状態が続いている時間 [h]

    # --- アデカシー ----------------------------------------------------
    forced_outage_rate: float = 0.0  #: 強制停止率 FOR [-]
    mttf: float | None = None        #: 平均無故障時間 [h]（逐次モンテカルロ用）
    mttr: float | None = None        #: 平均修復時間 [h]（同上）

    # --- 安定度（genstab へ渡す）---------------------------------------
    h: float | None = None          #: 慣性の寄与分 [s]（系統共通基準に換算済み。単純加算できる）
    xd_prime: float | None = None   #: 過渡リアクタンス [p.u.]
    d: float = 0.0                  #: 制動の寄与分 [p.u.]（同上。機器容量基準の値を入れないこと）

    # ------------------------------------------------------------------
    @property
    def outage_rate(self) -> float:
        """``forced_outage_rate`` の別名。

        文献では FOR と書かれるが、``for`` は Python の予約語なので
        フィールド名にできない。式を写すときの読み替えを減らすための別名。
        """
        return self.forced_outage_rate

    @property
    def su_ramp(self) -> float:
        """起動時に到達できる出力 [MW]。既定は最低出力。"""
        return self.p_min_mw if self.startup_ramp is None else self.startup_ramp

    @property
    def sd_ramp(self) -> float:
        """停止直前に許される出力 [MW]。既定は最低出力。"""
        return self.p_min_mw if self.shutdown_ramp is None else self.shutdown_ramp

    def fuel_cost(self, p_mw: float | np.ndarray) -> float | np.ndarray:
        """出力 ``p_mw`` [MW] における燃料費 [円/h]（無負荷費を含む）。

        .. math:: C(P) = c_2 P^2 + c_1 P + c_0
        """
        return self.quadratic * np.square(p_mw) + self.var_cost * p_mw + self.noload_cost

    def incremental_cost(self, p_mw: float | np.ndarray) -> float | np.ndarray:
        """増分燃料費 :math:`dC/dP` [円/MWh]。等 λ 法の主役。"""
        return 2.0 * self.quadratic * p_mw + self.var_cost

    def full_load_average_cost(self) -> float:
        """全負荷平均費用 [円/MWh]。優先順位法の順位づけに使う。

        .. math:: \\frac{C(P_{max})}{P_{max}}
        """
        if self.p_max_mw <= 0:
            return math.inf
        return float(self.fuel_cost(self.p_max_mw) / self.p_max_mw)

    def remaining_min_up(self) -> int:
        """期間開始時点で残っている最低運転時間 [h]。

        ``u0 = 1`` で ``hours_in_state`` が ``min_up`` に満たなければ、
        その差の時間だけは停止できない。ここを「窓和の切り詰め」で
        済ませると、初期状態から持ち越した拘束が課されず、
        8 時間の最低運転時間を持つ石炭機を 1 時間目に止められてしまう。
        """
        if self.u0 != 1:
            return 0
        return max(0, self.min_up - self.hours_in_state)

    def remaining_min_down(self) -> int:
        """期間開始時点で残っている最低停止時間 [h]。"""
        if self.u0 != 0:
            return 0
        return max(0, self.min_down - self.hours_in_state)


@dataclass(frozen=True)
class ReferenceSolution:
    """教科書に載っている潮流解。**入力ではなく答え合わせに使う。**

    Parameters
    ----------
    v, angle_deg:
        母線電圧の大きさ [p.u.] と位相 [deg]。``Case.buses`` と同じ順。
    generation:
        母線単位の発電 ``{母線番号: (P, Q)}`` [p.u.]。
    checks:
        独立に確かめられる量（総損失・slack 出力・内部起電力など）。
        テストの期待値をケースファイル側に持たせるための入れ物。
    digits:
        出典に記載されている桁数。許容差の根拠として記録しておく。
    source:
        出典の書誌。
    """

    v: np.ndarray
    angle_deg: np.ndarray
    generation: Mapping[int, tuple[float, float]] = field(default_factory=dict)
    checks: Mapping[str, object] = field(default_factory=dict)
    digits: int = 4
    source: str = ""

    @property
    def voltage(self) -> np.ndarray:
        """複素電圧 :math:`\\bar V = |V| e^{j\\theta}`。"""
        return self.v * np.exp(1j * np.radians(self.angle_deg))


# ======================================================================
# ケース本体
# ======================================================================
@dataclass
class Case:
    """5 つのテーマが共有する系統ケース。

    ``frozen`` にしていないのは、:func:`dataclasses.replace` による掃引
    （負荷倍率を変えて P-V 曲線を描くなど）を素直に書けるようにするため
    である。要素の :class:`Bus` / :class:`Branch` / :class:`Unit` は
    ``frozen`` なので、うっかり書き換わる事故は起きない。
    """

    name: str
    base_mva: float = 100.0
    frequency_hz: float = 60.0
    buses: Sequence[Bus] = ()
    branches: Sequence[Branch] = ()
    units: Sequence[Unit] = ()
    commitment: Mapping[str, object] = field(default_factory=dict)
    reliability: Mapping[str, object] = field(default_factory=dict)
    stability: Mapping[str, object] = field(default_factory=dict)
    contingencies: Sequence[tuple[int, int]] = ()
    reference: ReferenceSolution | None = None
    source: str = ""
    modifications: str = ""     #: 原典から改変した点

    def __post_init__(self) -> None:
        self._index = {bus.id: i for i, bus in enumerate(self.buses)}

    # ------------------------------------------------------------------
    # 層の検査
    # ------------------------------------------------------------------
    @property
    def blocks(self) -> frozenset[str]:
        """このケースが持っている層の集合。"""
        present = {"network"} if self.buses else set()
        if self.units:
            present.add("units")
        for name in ("commitment", "reliability", "stability"):
            if getattr(self, name):
                present.add(name)
        if self.contingencies:
            present.add("contingencies")
        if self.reference is not None:
            present.add("solution")
        return frozenset(present)

    def require(self, *blocks: str) -> None:
        """必要な層が欠けていれば、それを使う回の番号を添えて例外を投げる。

        授業の依存構造をそのまま実行可能な形にするための仕掛けである。

        Raises
        ------
        ValueError
            指定した層がこのケースにないとき。
        """
        missing = [b for b in blocks if b not in self.blocks]
        if not missing:
            return
        lines = [f"ケース '{self.name}' には次の層がない: {', '.join(missing)}"]
        for name in missing:
            lesson = BLOCK_LESSONS.get(name, "（対応する回は未登録）")
            lines.append(f"  - {name}: 第 {lesson} 以降で必要になる")
        lines.append("  全テーマの層を備えたケースとして 'wscc9' を用意している。")
        raise ValueError("\n".join(lines))

    # ------------------------------------------------------------------
    def check(self) -> list[str]:
        """データの不整合を列挙する（空リストなら問題なし）。

        **ソルバの設定を疑う前に必ずここを通すこと。** 数値的に解けない
        原因の大半は、ソルバではなくデータとトポロジーの矛盾にある。

        Returns
        -------
        list[str]
            見つかった問題の説明。1 件も無ければ空リスト。
        """
        problems: list[str] = []

        # --- 母線 ------------------------------------------------------
        ids = [bus.id for bus in self.buses]
        duplicated = {i for i in ids if ids.count(i) > 1}
        if duplicated:
            problems.append(f"母線番号が重複している: {sorted(duplicated)}")

        slack = [bus.id for bus in self.buses if bus.type is BusType.SLACK]
        if len(slack) != 1:
            problems.append(
                f"slack 母線がちょうど 1 つでない（{len(slack)} 個: {slack}）。"
                "位相の基準と損失の受け皿がなくなるので潮流計算が定義できない。"
            )

        for bus in self.buses:
            if bus.v_min > bus.v_max:
                problems.append(f"母線 {bus.id}: v_min > v_max")
            if bus.type in (BusType.PV, BusType.SLACK) and bus.v_set <= 0:
                problems.append(f"母線 {bus.id}: 設定電圧が非正")

        # --- 枝 --------------------------------------------------------
        known = set(ids)
        for branch in self.branches:
            if branch.from_bus not in known or branch.to_bus not in known:
                problems.append(f"枝 {branch.label}: 存在しない母線を参照している")
            if branch.from_bus == branch.to_bus:
                problems.append(f"枝 {branch.label}: 自己ループ")
            if branch.r == 0.0 and branch.x == 0.0:
                problems.append(f"枝 {branch.label}: ゼロインピーダンス")
            if branch.tap <= 0:
                problems.append(f"枝 {branch.label}: タップ比が非正")
            if branch.rate_b < branch.rate_a:
                problems.append(
                    f"枝 {branch.label}: 緊急時容量が常時容量より小さい"
                )

        problems.extend(self._connectivity_problems())

        # --- 号機 ------------------------------------------------------
        for unit in self.units:
            if unit.bus not in known:
                problems.append(f"号機 {unit.name}: 存在しない母線 {unit.bus}")
            if unit.p_min_mw > unit.p_max_mw:
                problems.append(f"号機 {unit.name}: p_min_mw > p_max_mw")
            if unit.q_min > unit.q_max:
                problems.append(f"号機 {unit.name}: q_min > q_max")
            if unit.quadratic < 0.0:
                problems.append(
                    f"号機 {unit.name}: 燃料費の 2 次係数が負 "
                    f"(quadratic={unit.quadratic})。費用が凹になり、"
                    "等 λ 法の単調性と区分線形化の前提（凸性）が崩れる。"
                )
            if not 0.0 <= unit.forced_outage_rate <= 1.0:
                problems.append(f"号機 {unit.name}: FOR が [0, 1] の外")
            if unit.min_up < 1 or unit.min_down < 1:
                problems.append(f"号機 {unit.name}: 最低運転／停止時間が 1 未満")
            # 単位の取り違えの検出。容量が p.u. のオーダーに見える場合。
            if 0.0 < unit.p_max_mw < 10.0:
                problems.append(
                    f"号機 {unit.name}: p_max_mw={unit.p_max_mw} は MW にしては"
                    "小さすぎる。p.u. の値を入れていないか確認すること。"
                )

        if self.units:
            slack_units = [u for u in self.units if u.bus in slack]
            if slack and not slack_units:
                problems.append(
                    f"slack 母線 {slack[0]} に号機がない。"
                    "損失と需給の差を引き受ける発電機が必要である。"
                )
            total = sum(u.p_max_mw for u in self.units)
            peak = float(self.commitment.get("peak_mw", 0.0) or 0.0)
            if peak and total < peak:
                problems.append(
                    f"号機容量の合計 {total:.1f} MW が最大需要 {peak:.1f} MW を"
                    "下回っている。起動停止計画が必ず実行不可能になる。"
                )

        return problems

    def _connectivity_problems(self) -> list[str]:
        """非連結な島と孤立母線を検出する。"""
        if not self.buses:
            return []
        adjacency: dict[int, set[int]] = {bus.id: set() for bus in self.buses}
        for branch in self.branches:
            if branch.from_bus in adjacency and branch.to_bus in adjacency:
                adjacency[branch.from_bus].add(branch.to_bus)
                adjacency[branch.to_bus].add(branch.from_bus)

        seen: set[int] = set()
        islands: list[list[int]] = []
        for start in adjacency:
            if start in seen:
                continue
            stack, island = [start], []
            seen.add(start)
            while stack:
                node = stack.pop()
                island.append(node)
                for neighbour in adjacency[node] - seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
            islands.append(sorted(island))

        if len(islands) <= 1:
            return []
        return [
            f"系統が {len(islands)} 個の島に分かれている: "
            + " / ".join(str(island) for island in islands)
        ]

    # ------------------------------------------------------------------
    # 添字と諸量
    # ------------------------------------------------------------------
    def index_of(self, bus_id: int) -> int:
        """母線番号から行列の添字を引く。"""
        try:
            return self._index[bus_id]
        except KeyError:
            raise KeyError(
                f"母線 {bus_id} はケース '{self.name}' にない。"
                f"存在するのは {sorted(self._index)}"
            ) from None

    @property
    def n_bus(self) -> int:
        return len(self.buses)

    @property
    def n_branch(self) -> int:
        return len(self.branches)

    @property
    def bus_ids(self) -> list[int]:
        return [bus.id for bus in self.buses]

    def type_indices(
        self, dispatch: Mapping[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(slack, pv, pq)`` の添字配列を返す。未知数の数合わせの根拠。"""
        types = self.effective_bus_types(dispatch)
        slack, pv, pq = [], [], []
        for i, bus in enumerate(self.buses):
            kind = types[bus.id]
            (slack if kind is BusType.SLACK else pv if kind is BusType.PV else pq).append(i)
        return np.array(slack, int), np.array(pv, int), np.array(pq, int)

    def n_unknowns(self, dispatch: Mapping[str, float] | None = None) -> int:
        """未知数の数 :math:`2 n_{PQ} + n_{PV}`。方程式の数と一致するはず。"""
        _, pv, pq = self.type_indices(dispatch)
        return 2 * len(pq) + len(pv)

    def units_at(self, bus_id: int) -> list[Unit]:
        """指定した母線につながる号機。"""
        return [u for u in self.units if u.bus == bus_id]

    def plants(self) -> dict[str, list[Unit]]:
        """発電所名で号機をまとめる。安定度へ渡すときの集約に使う。"""
        grouped: dict[str, list[Unit]] = {}
        for unit in self.units:
            grouped.setdefault(unit.plant or unit.name, []).append(unit)
        return grouped

    # ------------------------------------------------------------------
    def effective_bus_types(
        self, dispatch: Mapping[str, float] | None = None
    ) -> dict[int, BusType]:
        """運転状態を反映した母線種別を返す。

        起動停止計画である母線の号機が全台停止すると、その母線は電圧を
        支持できなくなるので、PV 母線ではなく **注入ゼロの PQ 母線**に
        なる。この再判定を忘れると、停止した発電機が電圧を支え続ける
        物理的にありえない解が出る。

        Parameters
        ----------
        dispatch:
            号機名から出力 [MW] への対応。``None`` なら全機が運転中と
            みなし、ケースの母線種別をそのまま返す。

        Raises
        ------
        ValueError
            slack 母線の号機が全台停止しているとき。位相の基準と損失の
            受け皿がなくなるので、潮流計算そのものが定義できない。
        """
        types = {bus.id: bus.type for bus in self.buses}
        if dispatch is None:
            return types

        for bus in self.buses:
            if bus.type is BusType.PQ:
                continue
            running = [u for u in self.units_at(bus.id) if dispatch.get(u.name, 0.0) > 0.0]
            if running:
                continue
            if bus.type is BusType.SLACK:
                raise ValueError(
                    f"slack 母線 {bus.id} の号機がすべて停止している。"
                    "位相の基準と損失を引き受ける発電機がなくなるため、"
                    "潮流計算が定義できない。起動停止計画の結果を確認すること。"
                )
            types[bus.id] = BusType.PQ
        return types

    def bus_injection(
        self, dispatch: Mapping[str, float] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """指定注入 :math:`(P^{sp}, Q^{sp})` [p.u.] を組む。

        **注入を組み立てる箇所は本メソッドだけである。** 負荷を負の発電と
        して足し込む符号ミスを構造的に防ぐため、他のモジュールは自分で
        注入を作らずここを呼ぶ。

        Parameters
        ----------
        dispatch:
            号機名から出力 [MW] への対応。``None`` なら参照解の母線単位の
            発電を使う（参照解がなければ発電はゼロ）。

        Notes
        -----
        無効電力の注入は PQ 母線でしか意味を持たない。PV / slack 母線の
        :math:`Q^{sp}` は潮流計算では使われないが、Q 制限の判定で
        参照するので値は入れてある。
        """
        p = np.zeros(self.n_bus)
        q = np.zeros(self.n_bus)

        if dispatch is None:
            if self.reference is not None:
                for bus_id, (pg, qg) in self.reference.generation.items():
                    i = self.index_of(bus_id)
                    p[i] += pg
                    q[i] += qg
        else:
            for unit in self.units:
                i = self.index_of(unit.bus)
                p[i] += self.to_pu(dispatch.get(unit.name, 0.0))

        for i, bus in enumerate(self.buses):
            p[i] -= bus.pd
            q[i] -= bus.qd
        return p, q

    # ------------------------------------------------------------------
    # 単位の変換
    # ------------------------------------------------------------------
    def to_mw(self, pu: float | np.ndarray) -> float | np.ndarray:
        """p.u. を MW に変換する。"""
        return np.asarray(pu) * self.base_mva if np.ndim(pu) else pu * self.base_mva

    def to_pu(self, mw: float | np.ndarray) -> float | np.ndarray:
        """MW を p.u. に変換する。"""
        return np.asarray(mw) / self.base_mva if np.ndim(mw) else mw / self.base_mva

    # ------------------------------------------------------------------
    # 変形
    # ------------------------------------------------------------------
    def _carried_reference(self, keep_generation: bool) -> "ReferenceSolution | None":
        """変形後のケースに載せる参照解を決める。

        既定 (``keep_generation=False``) では ``None`` を返す。系統や負荷を
        変えたケースに元の潮流解を残すと、それが答えであるかのように読めて
        しまうためである。

        ``keep_generation=True`` のときは **発電だけを引き継ぐ**ために参照解を
        残すが、``checks``（総損失・slack 出力など）は変形で必ず変わるので
        空にし、``source`` に引き継ぎである旨を書き足す。残った ``v`` /
        ``angle_deg`` は変形前の値であって答え合わせには使えない。
        """
        if not keep_generation or self.reference is None:
            return None
        return replace(
            self.reference,
            checks={},
            source=(
                f"{self.reference.source} [変形前のケースから発電のみ引き継いだ。"
                "電圧・位相は変形前の値であり答え合わせには使えない]"
            ),
        )

    def scaled(
        self,
        factor: float,
        *,
        buses: Iterable[int] | None = None,
        keep_generation: bool = False,
    ) -> "Case":
        """負荷を ``factor`` 倍したケースを返す。P-V 曲線の追跡に使う。

        Parameters
        ----------
        factor:
            負荷倍率。
        buses:
            倍率をかける母線。``None`` なら全母線。
        keep_generation:
            ``True`` なら参照解の**発電**を引き継ぐ。既定の ``False`` では
            参照解を落とすので、返ったケースを ``dispatch`` なしで潮流計算に
            渡すと :meth:`bus_injection` が**発電ゼロ**を返し、「slack 母線
            1 台で全負荷を賄う」まったく別の系統を解くことになる（WSCC 9 母線
            では収束しない）。負荷余裕を見る P-V 曲線のように「発電を据え置いた
            まま負荷だけを増やす」用途では ``True`` にすること。

        Notes
        -----
        ``keep_generation=True`` で残る参照解は **発電だけが意味を持つ**。
        電圧・位相は変形前の値なので、それを答え合わせに使ってはならない
        （その旨は ``reference.source`` に書き足してある）。
        """
        target = set(self.bus_ids if buses is None else buses)
        scaled_buses = [
            replace(bus, pd=bus.pd * factor, qd=bus.qd * factor)
            if bus.id in target
            else bus
            for bus in self.buses
        ]
        return replace(
            self,
            buses=scaled_buses,
            reference=self._carried_reference(keep_generation),
        )

    def without_branch(
        self, key: tuple[int, int], *, keep_generation: bool = False
    ) -> "Case":
        """枝を 1 本開放したケースを返す。N-1 の評価に使う。

        Parameters
        ----------
        key:
            開放する枝。``Branch.key()`` と同じく順序は問わない。多重回線
            （同じ母線対の 2 回線）は 1 本だけ開放できず、両方が外れる。
        keep_generation:
            ``True`` なら参照解の**発電**を引き継ぐ。N-1 は「事故前の発電を
            据え置いたまま枝を 1 本外す」評価なので、事故後潮流を解くなら
            ``True`` にするか ``dispatch`` を明示的に渡すこと。既定の
            ``False`` では :meth:`bus_injection` が発電ゼロを返す（:meth:`scaled`
            の ``keep_generation`` と同じ注意が当てはまる）。

        Raises
        ------
        ValueError
            指定した枝がケースにないとき。
        """
        target = (min(key), max(key))
        remaining = [b for b in self.branches if b.key() != target]
        if len(remaining) == len(self.branches):
            raise ValueError(f"枝 {target} はケース '{self.name}' にない。")
        return replace(
            self,
            branches=remaining,
            reference=self._carried_reference(keep_generation),
        )

    # ------------------------------------------------------------------
    def describe(self) -> str:
        """構成の要約を返す。"""
        slack, pv, pq = self.type_indices()
        lines = [
            f"Case '{self.name}'",
            f"  基準       : {self.base_mva} MVA, {self.frequency_hz} Hz",
            f"  母線       : {self.n_bus} (slack {len(slack)}, PV {len(pv)}, PQ {len(pq)})",
            f"  枝         : {self.n_branch}",
            f"  号機       : {len(self.units)}"
            + (f" / 設備容量 {sum(u.p_max_mw for u in self.units):.0f} MW" if self.units else ""),
            f"  未知数     : {self.n_unknowns()}",
            f"  層         : {', '.join(sorted(self.blocks))}",
        ]
        total_load = sum(bus.pd for bus in self.buses)
        lines.append(f"  基準負荷   : {self.to_mw(total_load):.1f} MW")
        return "\n".join(lines)
