"""静的セキュリティ解析（N-1 スクリーニングとセキュリティ制約付き経済配分）。

第 09 回。系統は「いま健全であること」ではなく「**想定した事故が 1 つ
起きても健全であり続けること**」を要求される。これが N-1 基準であり、
本モジュールはその判定（:func:`screen_n1`）と、判定に通る運転点を
費用最小で探すこと（:func:`sced`）を扱う。

なぜ 2 段構えにするのか
-----------------------
事故 1 件ごとに交流潮流を解けば正確だが、実系統では候補が数千件あり、
運用の周期（数分）に間に合わない。直流近似は線形なので、線路開放分布
係数 :math:`\\mathrm{LODF}` を一度だけ作れば事故後潮流が

.. math:: f' = f + \\mathrm{LODF}[:, k]\\, f_k

という行列とベクトルの積 1 回で出る。そこで **直流で絞り込み、交流で
判定する**。この住み分けが実務の流れそのものであり、本モジュールの
``method="lodf"`` + ``check_voltage=True`` がそれを再現する。

**スクリーニングは絞り込みであって判断ではない。** この一文が第 09 回の
主題である。直流スクリーニングが取りこぼすものが 2 つある。

1. **電圧を原理的に見ない。** 直流潮流には :math:`|V|` も無効電力も
   無い。WSCC 9 母線で枝 ``4-6`` を開放すると、熱容量は最悪でも
   ``rate_b`` の **75.7%** で完全に健全なのに、母線 6 の電圧が
   **0.9418 p.u.** まで落ちて下限 0.95 を割る。**熱容量だけを見る
   N-1 スクリーニングは、この事故を「健全」と誤判定する。**
   ``check_voltage=False`` にするとこの誤判定が実際に起きることを、
   ``tests/test_security.py`` で固定してある（誤判定そのものをテストに
   書いてあるのは、それがこの回の教材だからである）。
2. **負荷率を外す。しかも安全側とは限らない。** 定格は皮相電力
   :math:`|S|` の制限なのに、直流が持っているのは有効電力 :math:`P`
   だけである。同じ WSCC 9 母線で枝 ``4-5`` を開放すると、直流の最悪
   負荷率は 86.2% で健全に見えるが、交流では枝 ``5-7`` が **101.5%**
   で逸脱している。スクリーニングの閾値を 100% に置く（= 直流が越えた
   ものだけ交流で見る）と、この事故を見落とす。

   誤差の向きは 6 候補中 5 件が過小評価だが、枝 ``6-9`` の開放だけは
   直流 102.1% に対し交流 101.5% で **過大評価**である（直流は
   :math:`P` そのものも近似しているため）。したがって「直流の値に
   一律の余裕を足せば安全側になる」とは言えない。既定の
   ``screen_threshold=0.0``（全候補を交流で解き直す）はこの事実から
   来ている。

WSCC 9 母線で分かること
-----------------------
交流で 6 候補すべてを解き直すと、枝ごとの N-1 最大負荷率（``rate_b``
基準）は次のようになる。

===== ======  ===== ======  ===== ======
枝     最大    枝     最大    枝     最大
===== ======  ===== ======  ===== ======
4-5    91.1%  5-7   112.4%  7-8   112.5%
4-6    90.5%  6-9    88.2%  8-9    88.3%
===== ======  ===== ======  ===== ======

**拘束するのは 5-7 と 7-8 のちょうど 2 本**であり、非拘束枝の最大は
91.1% で、100% までに 9 ポイントの余裕がある。9 本の枝のうち 2 本が
運転点を縛り、残りは縛らない — 「どの設備が効いているか」を数えられる
ことが、N-1 解析が単なる合否判定でない理由である。

なお ``5-7`` は、ケースの ``stability`` 層が持つ標準事故（母線 7 の
三相地絡を線路 5-7 の開放で除去する）と **同じ枝**である。静的な
過負荷と過渡安定度という別々の物理が、同じ 1 本の枝で出会う。

橋は候補にしない
----------------
変圧器 3 本 ``(1,4), (2,7), (3,9)`` は **橋**（開放すると系統が 2 つの
島に分かれる枝）である。連結系統を前提とする LODF を適用できないので候補から外すが、**外した事実は捨てずに** :attr:`SecurityReport.skipped`
に理由つきで残す。「候補になかった」と「検査して健全だった」を
取り違えると、解析の穴がそのまま見えなくなるからである。同じ 3 本を
:func:`gridops.ybus.bridges` が独立に検出し、:func:`gridops.dc.lodf` は
分母 :math:`1-\\Psi[k,k]` がゼロになることで拒否する。

性能指標（PI）について
----------------------
:func:`performance_index` は事故の順位づけに使う 1 つの数だが、
**PI ランキングは順位を誤るものである**。詳細はその docstring を参照。

セキュリティ制約付き経済配分
----------------------------
:func:`sced` は「全事故に耐える運転点のうち最も安いもの」を線形計画で
求める。事故の数だけ制約を最初から並べると問題が巨大になるので、
**制約生成**（違反した事故だけを足していき、違反が無くなるまで繰り返す）
を使う。予防的 (``mode="preventive"``) と是正的 (``mode="corrective"``)
の費用差が「セキュリティの値段」である。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import solvers
from .case import Case, Unit, validate_rating_attribute
from .dc import dc_powerflow, lodf, ptdf
from .powerflow import PowerFlowSolution
from .powerflow import solve as solve_powerflow
from .ybus import bridges

__all__ = [
    "LIMIT_NAMES",
    "SCREEN_METHODS",
    "SCED_MODES",
    "LOADING_TOLERANCE",
    "VOLTAGE_TOLERANCE",
    "FLOW_TOLERANCE",
    "ContingencyResult",
    "SecurityReport",
    "SCEDResult",
    "performance_index",
    "screen_n1",
    "sced",
]

#: 熱容量の属性名。``rate_a`` が常時、``rate_b`` が緊急時（事故後）許容容量。
LIMIT_NAMES = ("rate_a", "rate_b")

#: :func:`screen_n1` の ``method``。
SCREEN_METHODS = ("lodf", "ac")

#: :func:`sced` の ``mode``。
SCED_MODES = ("preventive", "corrective")

#: 負荷率が 1.0 を超えたと判定する幅。潮流解の残差（既定 1e-10 p.u.）より
#: 十分に大きく、教材で問題にする 0.1 ポイントよりはるかに小さい。
LOADING_TOLERANCE = 1e-9

#: 電圧が下限・上限を外れたと判定する幅。単位は p.u.。
VOLTAGE_TOLERANCE = 1e-9

#: :func:`sced` が「制約に違反した」と判定する枝潮流の幅 [p.u.]。
#: CBC が解ファイルに書き出す桁数（9 桁前後）に起因する誤差より大きく取る。
FLOW_TOLERANCE = 1e-6


# ======================================================================
# 内部ヘルパ
# ======================================================================
def _normalize_key(key: Sequence[int]) -> tuple[int, int]:
    """``(母線, 母線)`` を :meth:`Branch.key` と同じ ``(小, 大)`` に正規化する。"""
    try:
        a, b = key
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"枝の指定 {key!r} が (母線, 母線) の組になっていない。"
            "N-1 の候補は [(4, 5), (4, 6)] のように **組のリスト**で渡すこと"
            "（(4, 5) と書くと母線番号 2 つに分解される）。"
        ) from exc
    return (min(int(a), int(b)), max(int(a), int(b)))


def _branch_keys(case: Case) -> list[tuple[int, int]]:
    """枝の識別子を :attr:`Case.branches` の順で返す。"""
    return [branch.key() for branch in case.branches]


def _limit_array(case: Case, limit: str) -> np.ndarray:
    """枝の熱容量を枝順の配列で返す。

    Raises
    ------
    ValueError
        ``limit`` が :data:`LIMIT_NAMES` にないとき、容量が非正の枝が
        あるとき。
    """
    if limit not in LIMIT_NAMES:
        raise ValueError(
            f"limit={limit!r} は使えない。{LIMIT_NAMES} のいずれかを指定すること"
            "（rate_a が常時許容容量、rate_b が緊急時許容容量。"
            "N-1 の事故後は rate_b で見るのが規約である）。"
        )
    validate_rating_attribute(limit)
    rates = np.array([float(getattr(branch, limit)) for branch in case.branches])
    if np.any(rates <= 0.0):
        bad = [
            branch.label
            for branch, rate in zip(case.branches, rates)
            if rate <= 0.0
        ]
        raise ValueError(
            f"枝 {bad} の {limit} が非正で、負荷率が定義できない。"
            "熱容量を与えない枝は inf のままにすること。"
        )
    return rates


def _candidates(
    case: Case, contingencies: Sequence[Sequence[int]] | None
) -> list[tuple[int, int]]:
    """N-1 の候補を正規化して返す。

    ``contingencies`` が ``None`` ならケースの ``contingencies`` 層を使い、
    それも空なら全枝を候補にする。

    Raises
    ------
    ValueError
        ケースにない枝を指定したとき。**黙って無視しない**のは、
        「開放したつもりで開放されていない」N-1 結果が最も気づきにくい
        誤りだからである。
    """
    known = set(_branch_keys(case))
    if contingencies is None:
        source: Sequence[Sequence[int]] = case.contingencies or _branch_keys(case)
    else:
        source = contingencies

    result: list[tuple[int, int]] = []
    for key in source:
        normalized = _normalize_key(key)
        if normalized not in known:
            raise ValueError(
                f"枝 {normalized} はケース '{case.name}' にない。"
                f"存在する枝は {sorted(known)}。"
            )
        if normalized not in result:
            result.append(normalized)
    return result


def _outaged_case(case: Case, key: tuple[int, int]) -> Case:
    """枝を 1 本開放したケースを返す（**事故前の発電を据え置く**）。

    :meth:`Case.without_branch` は既定で ``reference=None`` にする。参照解が
    消えると :meth:`Case.bus_injection` が発電ゼロを返し、「slack 母線 1 台で
    全負荷を賄う」まったく別の系統を解くことになる（WSCC 9 母線では収束
    しない）。N-1 は **事故前の発電を据え置いて**評価するものなので、
    ``keep_generation=True`` を指定して発電だけを引き継ぐ。
    """
    return case.without_branch(key, keep_generation=True)


def _voltage_violations(case: Case, v: np.ndarray) -> tuple[str, ...]:
    """母線ごとの電圧逸脱を日本語で列挙する（空なら健全）。"""
    messages: list[str] = []
    for i, bus in enumerate(case.buses):
        if v[i] < bus.v_min - VOLTAGE_TOLERANCE:
            messages.append(
                f"母線 {bus.id}: 電圧 {v[i]:.4f} p.u. が下限 "
                f"{bus.v_min:.2f} を下回っている"
            )
        elif v[i] > bus.v_max + VOLTAGE_TOLERANCE:
            messages.append(
                f"母線 {bus.id}: 電圧 {v[i]:.4f} p.u. が上限 "
                f"{bus.v_max:.2f} を上回っている"
            )
    return tuple(messages)


# ======================================================================
# 性能指標
# ======================================================================
def performance_index(
    flows: Sequence[float] | np.ndarray,
    limits: Sequence[float] | np.ndarray,
    *,
    n: int = 1,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """事故の重大さを 1 つの数にまとめる性能指標（PI）。

    .. math::

        \\mathrm{PI} = \\frac{1}{2n} \\sum_{\\ell} w_{\\ell}
        \\left(\\frac{f_{\\ell}}{f_{\\ell}^{\\max}}\\right)^{2n}

    偶数乗なので向きに依らず、定格を超えた枝が指数関数的に効く。
    事故を 1 件ずつ交流で解く余裕がないとき、まず PI で並べて上位だけ
    詳しく見る、という使い方をする。

    Parameters
    ----------
    flows:
        枝潮流。交流なら :math:`|S|`、直流なら :math:`|P|` [p.u.]。
    limits:
        枝の熱容量 [p.u.]。``inf`` の枝は比がゼロになり寄与しない。
    n:
        指数の半分。``n=1`` なら 2 乗、``n=4`` なら 8 乗。
    weights:
        枝ごとの重み。``None`` なら全部 1。

    Returns
    -------
    float
        PI の値（無次元、非負）。

    Raises
    ------
    ValueError
        配列の長さが揃わないとき、``n`` が 1 未満のとき、容量が
        非正の枝があるとき。

    Notes
    -----
    **PI ランキングは順位を誤るものである。** これは実装の粗さではなく
    指標の性質であり、masking（隠蔽）と呼ばれる。:math:`n=1`（2 乗）の
    とき、軽く載った枝が多数ある事故の PI が、1 本だけ深刻に過負荷して
    いる事故の PI を **上回ってしまう**。

    ``tests/test_security.py`` に入れてある最小の例（容量はすべて 1.0）:

    ============== ============================ ================ =======
    事故            枝潮流                        最悪負荷率        PI (n=1)
    ============== ============================ ================ =======
    A（軽い多数）    0.95 が 10 本                 95%（健全）       4.5125
    B（重い 1 本）   2.0 が 1 本 + 0.1 が 9 本      **200%**         2.0450
    ============== ============================ ================ =======

    PI は A を上位に置くが、**実際に危険なのは B** である。A は 1 本も
    定格を超えていない。:math:`n=4`（8 乗）にすると A が 0.83、B が 32.0
    となって順位が入れ替わる。指数を上げれば masking は弱まるが、今度は
    「わずかに定格を超えた枝が 1 本」と「大きく超えた枝が 1 本」の差が
    潰れる（8 乗は 1.05 と 1.10 をほとんど区別しない）。

    結論として、**PI は候補を並べるための道具であって、判定ではない**。
    順位の上位から順に交流で解き直し、判定は交流の結果で行うこと。
    :meth:`SecurityReport.ranked` が ``by="worst_loading"`` も
    ``by="v_min"`` も受け付けるのは、単一の指標に判断を預けないためである。

    電圧はこの指標に **入っていない**。PI を電圧まで含めて定義する流儀も
    あるが（電圧 PI を別に作って足す）、熱容量と電圧は単位も逸脱の意味も
    違うので、本モジュールでは分けたまま扱う。
    """
    f = np.asarray(flows, dtype=float).ravel()
    cap = np.asarray(limits, dtype=float).ravel()
    if f.size != cap.size:
        raise ValueError(
            f"flows の長さ {f.size} と limits の長さ {cap.size} が違う。"
            "どちらも枝の並び順（Case.branches の順）の配列で渡すこと。"
        )
    if n < 1:
        raise ValueError(
            f"n={n} は使えない。PI の指数は 2n で、n は 1 以上の整数である"
            "（n=1 で 2 乗。masking を弱めたいなら n を上げる）。"
        )
    if np.any(cap <= 0.0):
        raise ValueError(
            "熱容量が非正の枝がある。負荷率が定義できないので PI も作れない。"
            "熱容量を与えない枝は inf にすること（比がゼロになり寄与しない）。"
        )
    if weights is None:
        w = np.ones_like(f)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != f.size:
            raise ValueError(
                f"weights の長さ {w.size} が flows の長さ {f.size} と違う。"
            )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(np.isfinite(cap), np.abs(f) / cap, 0.0)
    return float(np.sum(w * ratio ** (2 * n)) / (2 * n))


# ======================================================================
# 事故 1 件の結果
# ======================================================================
@dataclass(frozen=True)
class ContingencyResult:
    """事故 1 件の評価結果。

    Parameters
    ----------
    outage:
        開放した枝の :meth:`Branch.key`。
    flows:
        事故後の枝潮流の **大きさ** [p.u.]。並びは（事故前の）
        :attr:`Case.branches` の順で、開放した枝の要素はゼロである。
        枝の番号が事故ごとにずれないようにするための規約であり、
        :func:`gridops.dc.lodf` の行番号ともそのまま対応する。
        交流で評価したなら両端の :math:`|S|` の大きい方、直流の
        スクリーニングで評価したなら :math:`|P|` が入る（``method``
        フィールドを見ること）。
    loading:
        ``limit`` に対する負荷率（1.0 で定格）。``flows`` と同じ並び。
    v_min:
        最も低い母線電圧 [p.u.]。電圧を見ていない結果では ``nan``。
    v_min_bus:
        その母線 **番号**。電圧を見ていない結果では ``-1``。
    performance_index:
        :func:`performance_index` の値。
    worst_branch:
        負荷率が最大の枝の :meth:`Branch.key`。
    worst_loading:
        その負荷率。
    islanding:
        開放によって系統が島に分かれるか。**橋は候補の段階で
        :attr:`SecurityReport.skipped` に回るので、通常この値は
        ``False`` である。** フィールドを残してあるのは、候補を外から
        与えて自前で島を扱う拡張のためである。
    converged:
        評価が成立したか。交流潮流が収束しなかったときだけ ``False``
        になる（直流は反復しないので常に ``True``）。
    limit:
        負荷率の分母に使った熱容量の名前（契約に無い追加）。
    v_limit:
        ``v_min_bus`` の電圧下限 [p.u.]（契約に無い追加）。
    v_violations:
        電圧逸脱の日本語の一覧（契約に無い追加）。空なら電圧は健全。
        :attr:`voltage_secure` はこれを見る。母線ごとの
        :attr:`Bus.v_min` / :attr:`Bus.v_max` を使うので、母線ごとに
        違う限度を置いたケースでも正しく判定できる。
    voltage_checked:
        電圧を実際に見たか（契約に無い追加）。``False`` のときは
        :attr:`voltage_secure` が **無条件に** ``True`` を返す。
        「見ていない」と「見て健全だった」を取り違えないための印である。
    method:
        この結果を得た方法。``"ac"`` か ``"lodf"``（契約に無い追加）。
    screening_loading:
        直流スクリーニングが出した最悪負荷率（契約に無い追加）。交流の
        ``worst_loading`` と並べると、直流がどれだけ過小評価するかが
        そのまま読める。スクリーニングをしていないときは ``nan``。
    branch_keys:
        ``flows`` / ``loading`` の添字に対応する枝の識別子（契約に無い
        追加）。配列だけを持ち回すと「何番目がどの枝か」が呼び出し側の
        暗黙の了解になるので、結果に同梱してある。

    Notes
    -----
    :attr:`thermal_secure` と :attr:`voltage_secure` を **別々に**
    持っているのがこのクラスの要点である。WSCC 9 母線で枝 ``4-6`` を
    開放すると

    .. code-block:: text

        thermal_secure = True    （最悪 75.7%、rate_b に対し健全）
        voltage_secure = False   （母線 6 が 0.9418 で下限 0.95 を割る）
        is_secure      = False

    となる。熱容量だけを見ていれば「健全」で通ってしまう事故であり、
    第 09 回の主教材である。
    """

    outage: tuple[int, int]
    flows: np.ndarray
    loading: np.ndarray
    v_min: float
    v_min_bus: int
    performance_index: float
    worst_branch: tuple[int, int]
    worst_loading: float
    islanding: bool
    converged: bool
    # --- 契約に無い追加フィールド（すべて既定値つき）-------------------
    limit: str = "rate_b"
    v_limit: float = 0.95
    v_violations: tuple[str, ...] = ()
    voltage_checked: bool = True
    method: str = "ac"
    screening_loading: float = float("nan")
    branch_keys: tuple[tuple[int, int], ...] = ()

    # ------------------------------------------------------------------
    @property
    def thermal_secure(self) -> bool:
        """熱容量の面で健全か。

        収束しなかった事故と島に分かれる事故は、判定そのものが成立
        しないので ``False`` とする（「分からない」を「健全」に丸めない）。
        """
        if self.islanding or not self.converged:
            return False
        return self.worst_loading <= 1.0 + LOADING_TOLERANCE

    @property
    def voltage_secure(self) -> bool:
        """電圧の面で健全か。

        Notes
        -----
        **``voltage_checked`` が ``False`` のときは無条件に ``True`` を
        返す。** これは「電圧を見なければ何でも健全に見える」という
        誤判定そのものであり、意図的にそう作ってある。
        :func:`screen_n1` に ``check_voltage=False`` を渡すと、WSCC 9 母線
        の枝 ``4-6`` の開放が :attr:`is_secure` で ``True`` になる —
        母線 6 の電圧が 0.9418 p.u. まで落ちているにもかかわらず、である。

        直流の LODF は電圧を **原理的に**持っていない。だから直流の
        スクリーニングを「判定」と呼んではいけない。
        """
        if self.islanding or not self.converged:
            return False
        if not self.voltage_checked:
            return True
        return not self.v_violations

    @property
    def is_secure(self) -> bool:
        """熱容量と電圧の **両方**で健全か。"""
        return self.thermal_secure and self.voltage_secure

    # ------------------------------------------------------------------
    def violations(self) -> list[str]:
        """逸脱の一覧（日本語）。空リストなら健全である。"""
        messages: list[str] = []
        if self.islanding:
            messages.append(
                f"枝 {self.outage[0]}-{self.outage[1]} は橋であり、開放すると"
                "系統が島に分かれる（事故後潮流という概念が成り立たない）。"
            )
            return messages
        if not self.converged:
            messages.append(
                "事故後の潮流が収束していない。以下の判定は意味を持たない"
                "（収束しないこと自体が最も重い異常である）。"
            )
        if self.worst_loading > 1.0 + LOADING_TOLERANCE:
            for key, ratio in zip(self._keys(), self.loading):
                if ratio > 1.0 + LOADING_TOLERANCE:
                    messages.append(
                        f"枝 {key[0]}-{key[1]}: 負荷率 {ratio * 100:.1f}% "
                        f"（{self.limit} 基準）"
                    )
        messages.extend(self.v_violations)
        if not self.voltage_checked:
            messages.append(
                "※ 電圧は見ていない（check_voltage=False）。"
                "熱容量が健全でも電圧が下限を割っている可能性がある。"
            )
        return messages

    def _keys(self) -> list[tuple[int, int]]:
        """``loading`` の添字に対応する枝の識別子。"""
        if self.branch_keys:
            return list(self.branch_keys)
        return [(i, i) for i in range(len(self.loading))]

    def summary(self) -> str:
        """この事故 1 件の要約（日本語）。"""
        head = (
            f"事故 {self.outage[0]}-{self.outage[1]} — "
            + ("健全" if self.is_secure else "**逸脱あり**")
            + f"（評価: {self.method}）"
        )
        lines = [
            head,
            f"  最悪の枝   : {self.worst_branch[0]}-{self.worst_branch[1]} "
            f"負荷率 {self.worst_loading * 100:.1f}% ({self.limit})",
        ]
        if math.isfinite(self.screening_loading):
            lines.append(
                f"  直流の予測 : 最悪負荷率 {self.screening_loading * 100:.1f}%"
                f"（交流との差 {(self.worst_loading - self.screening_loading) * 100:+.1f} ポイント）"
            )
        if math.isfinite(self.v_min):
            note = "" if self.voltage_checked else "  ← **判定に使っていない**"
            lines.append(
                f"  最低電圧   : 母線 {self.v_min_bus} で {self.v_min:.4f} p.u."
                f"（下限 {self.v_limit:.2f}）{note}"
            )
        else:
            lines.append("  最低電圧   : 未評価（直流には電圧が無い）")
        lines.append(f"  PI         : {self.performance_index:.4f}")
        problems = self.violations()
        if problems:
            lines.append(f"  逸脱       : {len(problems)} 件")
            lines.extend(f"    - {message}" for message in problems)
        else:
            lines.append("  逸脱       : なし")
        return "\n".join(lines)


# ======================================================================
# 報告
# ======================================================================
@dataclass
class SecurityReport:
    """N-1 スクリーニングの結果一式。

    Parameters
    ----------
    base:
        事故前の :class:`~gridops.powerflow.PowerFlowSolution`。
    results:
        評価した事故の一覧。**候補から外した枝はここに入らない。**
    skipped:
        評価しなかった枝と理由の組。橋がここに入る。
        「候補になかった」と「検査して健全だった」を取り違えないための
        フィールドであり、:meth:`summary` は必ずこれを表示する。
    case:
        解析したケース（契約に無い追加）。枝の並びを復元するのに使う。
    limit:
        熱容量の判定に使った名前（契約に無い追加）。
    method:
        ``"lodf"`` か ``"ac"``（契約に無い追加）。
    check_voltage:
        電圧を見たか（契約に無い追加）。
    seconds:
        解析に要した時間 [s]（契約に無い追加）。
    """

    base: object
    results: list[ContingencyResult]
    skipped: list[tuple[tuple[int, int], str]] = field(default_factory=list)
    case: Case | None = None
    limit: str = "rate_b"
    method: str = "lodf"
    check_voltage: bool = True
    seconds: float = 0.0

    # ------------------------------------------------------------------
    def _case(self) -> Case:
        """ケースを取り出す（``case`` が無ければ ``base`` のものを使う）。"""
        if self.case is not None:
            return self.case
        return getattr(self.base, "case")

    def _keys(self) -> list[tuple[int, int]]:
        return _branch_keys(self._case())

    # ------------------------------------------------------------------
    @property
    def has_unassessed(self) -> bool:
        """未評価の想定事故（橋など）が残っているか。

        ``is_secure`` は **評価した事故だけ**の判定である。除外された事故が
        あるまま「N-1 健全」と結論しないよう、まずここを確かめること。
        """
        return bool(self.skipped)

    @property
    def is_secure(self) -> bool:
        """評価したすべての事故で健全か。

        Notes
        -----
        **``skipped`` が空でないなら、この値は「候補にした事故の範囲で」
        という但し書きつきである。** 橋を除いたことを忘れて
        「N-1 に耐える」と言ってはいけない。橋の開放は起きれば必ず
        供給支障になるのであって、健全なのではない。
        """
        return all(result.is_secure for result in self.results)

    def insecure(self) -> list[ContingencyResult]:
        """健全でない事故だけを、負荷率の高い順に返す。"""
        return sorted(
            (result for result in self.results if not result.is_secure),
            key=lambda r: r.worst_loading,
            reverse=True,
        )

    def ranked(self, by: str = "performance_index") -> list[ContingencyResult]:
        """事故を重大さの順に並べる。

        Parameters
        ----------
        by:
            ``"performance_index"`` / ``"worst_loading"``（どちらも降順）、
            ``"v_min"``（**昇順**。電圧は低いほど重大）。

        Returns
        -------
        list of ContingencyResult

        Raises
        ------
        ValueError
            ``by`` が上の 3 つ以外のとき。

        Notes
        -----
        **3 つの順位は一致しない。** PI は masking で順位を誤る
        (:func:`performance_index` の Notes)。``worst_loading`` は
        熱容量しか見ない。``v_min`` は電圧しか見ない。WSCC 9 母線では
        ``worst_loading`` の 1 位が枝 ``5-7`` の開放（7-8 が 112.5%）
        なのに対し、``v_min`` の 1 位は枝 ``4-5`` の開放（母線 5 が
        0.8388 p.u.）である。**どの順位で並べたかを言わずに「最悪の
        事故」と呼んではいけない。**
        """
        if by == "performance_index":
            return sorted(self.results, key=lambda r: r.performance_index, reverse=True)
        if by == "worst_loading":
            return sorted(self.results, key=lambda r: r.worst_loading, reverse=True)
        if by == "v_min":
            return sorted(
                self.results,
                key=lambda r: (r.v_min if math.isfinite(r.v_min) else math.inf),
            )
        raise ValueError(
            f"by={by!r} は使えない。'performance_index'（PI の降順）、"
            "'worst_loading'（負荷率の降順）、'v_min'（電圧の昇順）"
            "のいずれかを指定すること。"
        )

    # ------------------------------------------------------------------
    def worst_loading_by_branch(self) -> dict[tuple[int, int], float]:
        """枝ごとの **N-1 最大負荷率**（全事故にわたる最大）。

        Notes
        -----
        「どの事故が重いか」ではなく「**どの設備が運転点を縛るか**」を
        見る量である。WSCC 9 母線（``rate_b`` 基準、参照解の発電）では

        .. code-block:: text

            4-5  91.1%   5-7  112.4%   7-8  112.5%
            4-6  90.5%   6-9   88.2%   8-9   88.3%

        となり、100% を超えるのは 5-7 と 7-8 のちょうど 2 本である。
        """
        keys = self._keys()
        worst = {key: 0.0 for key in keys}
        for result in self.results:
            for key, ratio in zip(keys, result.loading):
                if ratio > worst[key]:
                    worst[key] = float(ratio)
        return worst

    def binding_branches(self) -> list[tuple[int, int]]:
        """N-1 で拘束する枝（最大負荷率が 100% を超える枝）を返す。

        Returns
        -------
        list of tuple
            枝の識別子。母線番号の順に並べる。

        Notes
        -----
        WSCC 9 母線では ``[(5, 7), (7, 8)]`` のちょうど 2 本になる。
        非拘束枝の最大は 91.1%（枝 4-5）で 9 ポイントの余裕があるので、
        この 2 本は「たまたま境界に近い」のではなく **系統の設計上の
        隘路**である。

        枝 ``5-7`` は、ケースの ``stability`` 層の標準事故（母線 7 の
        三相地絡を線路 5-7 の開放で除去する）と同じ枝でもある。静的な
        過負荷と過渡安定度が同じ 1 本で出会うことを、演習で確かめさせる
        とよい。
        """
        worst = self.worst_loading_by_branch()
        return sorted(
            key for key, ratio in worst.items() if ratio > 1.0 + LOADING_TOLERANCE
        )

    # ------------------------------------------------------------------
    def to_table(self) -> str:
        """事故の一覧を ASCII の表で返す（列見出しは英語）。"""
        header = (
            f"{'outage':>8}  {'worst branch':>13}  {'loading':>8}  "
            f"{'v_min bus':>10}  {'v_min':>7}  {'PI':>9}  {'verdict':>9}"
        )
        lines = [
            f"N-1 screening ({self.method}, limit = {self.limit}, "
            f"voltage = {'on' if self.check_voltage else 'OFF'})",
            header,
            "-" * len(header),
        ]
        for result in self.ranked(by="worst_loading"):
            outage = f"{result.outage[0]}-{result.outage[1]}"
            worst = f"{result.worst_branch[0]}-{result.worst_branch[1]}"
            if math.isfinite(result.v_min):
                bus = f"{result.v_min_bus}"
                # 電圧を判定に使っていないときは括弧に入れて区別する。
                v = (
                    f"{result.v_min:.4f}"
                    if result.voltage_checked
                    else f"({result.v_min:.4f})"
                )
            else:
                bus = "-"
                v = "-"
            verdict = "secure" if result.is_secure else "INSECURE"
            lines.append(
                f"{outage:>8}  {worst:>13}  {result.worst_loading * 100:7.1f}%  "
                f"{bus:>10}  {v:>7}  {result.performance_index:9.4f}  {verdict:>9}"
            )
        for key, _reason in self.skipped:
            lines.append(
                f"{key[0]}-{key[1]:<6}  {'(skipped)':>13}  {'-':>8}  "
                f"{'-':>10}  {'-':>7}  {'-':>9}  {'skipped':>9}"
            )
        return "\n".join(lines)

    def summary(self) -> str:
        """要約を返す（日本語）。"""
        case = self._case()
        lines = [
            f"N-1 スクリーニング '{case.name}' — "
            f"{'健全' if self.is_secure else '**逸脱あり**'}"
            f"（{self.method}, {self.limit}, "
            f"電圧判定 {'あり' if self.check_voltage else '**なし**'}）",
            f"  評価した事故 : {len(self.results)} 件 / "
            f"除外 {len(self.skipped)} 件 / {self.seconds:.3f} s",
        ]
        bad = self.insecure()
        if bad:
            lines.append(f"  逸脱した事故 : {len(bad)} 件")
            for result in bad:
                for message in result.violations():
                    lines.append(
                        f"    - 事故 {result.outage[0]}-{result.outage[1]}: {message}"
                    )
        else:
            lines.append("  逸脱した事故 : なし")
        binding = self.binding_branches()
        if binding:
            worst = self.worst_loading_by_branch()
            lines.append(
                "  拘束する枝   : "
                + ", ".join(
                    f"{k[0]}-{k[1]} ({worst[k] * 100:.1f}%)" for k in binding
                )
            )
        if self.skipped:
            lines.append("  除外した枝   :")
            for key, reason in self.skipped:
                lines.append(f"    - {key[0]}-{key[1]}: {reason}")
            lines.append(
                "    ※ 除外は「健全」ではない。候補に入れなかっただけである。"
            )
        if not self.check_voltage:
            lines.append(
                "  ※ 電圧を見ていない。熱容量だけの判定は、電圧が下限を割る事故を"
                "「健全」と誤判定する（WSCC 9 母線の枝 4-6 がその例）。"
            )
        return "\n".join(lines)


# ======================================================================
# N-1 スクリーニング
# ======================================================================
def screen_n1(
    case: Case,
    base: PowerFlowSolution | None = None,
    *,
    method: str = "lodf",
    limit: str = "rate_b",
    check_voltage: bool = True,
    contingencies: Sequence[Sequence[int]] | None = None,
    pi_n: int = 1,
    screen_threshold: float = 0.0,
) -> SecurityReport:
    """N-1 スクリーニングを行う。

    Parameters
    ----------
    case:
        系統ケース。
    base:
        事故前の **交流**潮流解。``None`` なら :func:`gridops.powerflow.solve`
        で解く（``method`` や ``check_voltage`` に依らず基準は交流で持つ。
        事故前の電圧と :math:`|S|` を報告に載せるためで、費用は反復 4 回
        ぶんしかない）。与えた解の ``dispatch`` が事故後の計算にも
        使われるので、経済配分の結果に対して N-1 を掛けたいときは
        ここに渡すこと。
    method:
        ``"lodf"``（直流で絞り込む）または ``"ac"``（最初から全件を
        交流で解き直す）。
    limit:
        熱容量の判定に使う名前。事故後は ``"rate_b"``（緊急時許容容量）。
    check_voltage:
        交流潮流を解き直して電圧を見るか。``False`` にすると **直流の
        熱容量だけ**で判定する（下の Notes を必ず読むこと）。
    contingencies:
        候補の枝。``None`` ならケースの ``contingencies`` 層、それも
        空なら全枝。
    pi_n:
        :func:`performance_index` の ``n``（契約に無い追加）。
    screen_threshold:
        直流スクリーニングの最悪負荷率が **この値以上**の候補だけを
        交流で解き直す（契約に無い追加）。既定の ``0.0`` は「全候補を
        交流で解き直す」という安全側であり、このとき ``method="lodf"``
        の値打ちは「並べ替えと事前の見積り」に限られる。
        ``method="ac"`` または ``check_voltage=False`` のときは使われない。

    Returns
    -------
    SecurityReport

    Raises
    ------
    ValueError
        ``method`` / ``limit`` が不正なとき。候補にケースにない枝を
        指定したとき。

    Notes
    -----
    **2 段構えの意味。** 直流で絞り込み、交流で判定する。この順番は
    速さのためだが、**絞り込みの網を細かくしすぎると判定に届かない**。
    WSCC 9 母線で ``screen_threshold`` を 1.0 に上げる（= 直流が定格を
    超えた候補だけ交流で見る）と、次の 2 件を取りこぼす。

    - 枝 ``4-5`` の開放: 直流の最悪は 86.2% で健全に見えるが、交流では
      枝 5-7 が **101.5%** で逸脱している。定格は皮相電力 :math:`|S|` の
      制限なのに直流は有効電力 :math:`P` しか持たないためである。ただし
      **誤差は安全側に偏っていない**。枝 6-9 の開放では直流 102.1% に
      対し交流 101.5% で、直流のほうが大きく出る。余裕の取り方では
      判定の代わりにならない。
    - 枝 ``4-6`` の開放: 直流でも交流でも熱容量は健全（72.4% / 75.7%）
      だが、母線 6 の電圧が **0.9418 p.u.** で下限 0.95 を割る。

    後者は閾値をいくら下げても直流では捕まらない。**直流の LODF は
    電圧を原理的に持っていない**からである。だから既定は
    ``screen_threshold=0.0``（全件を交流で解く）にしてあり、閾値を
    上げたときに何を失うかを演習で確かめさせる作りにしてある。

    **``check_voltage=False`` は誤判定を作る。** このとき枝 ``4-6`` の
    開放は ``thermal_secure=True`` かつ ``voltage_secure=True``
    （見ていないので）となり、``is_secure`` が ``True`` になる。母線 6 の
    電圧が 0.9418 p.u. まで落ちているにもかかわらず「健全」と報告される。
    これはバグではなく、**熱容量だけを見る N-1 スクリーニングが実際に
    犯す誤り**である。テストでもこの誤判定そのものを固定してある。

    **橋は候補から外す。** 開放すると系統が島に分かれる枝
    （WSCC 9 母線では変圧器 3 本 ``(1,4), (2,7), (3,9)``）は、事故後
    潮流という概念自体が成り立たない。:func:`gridops.ybus.bridges` が
    検出したものを :attr:`SecurityReport.skipped` に理由つきで移す。
    ``results`` には入らないので、``is_secure`` の意味が「候補にした
    事故の範囲で」であることを忘れないこと。

    **発電は事故前の値に据え置く。** 事故後のケースは枝だけを差し替えて
    作る（:meth:`Case.without_branch` は参照解を落としてしまうので使わ
    ない）。事故直後に発電機が動かない前提であり、動かしてよいとする
    のが :func:`sced` の ``mode="corrective"`` である。

    Examples
    --------
    >>> from gridops import load_case
    >>> from gridops.security import screen_n1
    >>> report = screen_n1(load_case("wscc9"))
    >>> report.binding_branches()
    [(5, 7), (7, 8)]
    """
    case.require("network")
    if method not in SCREEN_METHODS:
        raise ValueError(
            f"method={method!r} は使えない。{SCREEN_METHODS} のいずれかを"
            "指定すること（'lodf' が直流の絞り込み、'ac' が全件の交流計算）。"
        )
    rates = _limit_array(case, limit)
    started = time.perf_counter()

    if base is None:
        base = solve_powerflow(case)
    dispatch = getattr(base, "dispatch", None)

    keys = _branch_keys(case)
    candidates = _candidates(case, contingencies)
    bridge_set = set(bridges(case))

    # ケース側で橋を候補から除いていても、既定解析では『未評価』として記録する。
    # 除外した事実を残さないと、『候補に無かった』と『評価して健全だった』を区別できない。
    skipped: list[tuple[tuple[int, int], str]] = [
        (
            key,
            "橋（唯一の連絡路）なので開放後は系統が島に分かれる。"
            "連結系統を前提とする LODF は適用できず、本教材は島ごとの"
            "基準母線・需給再配分・周波数変動・負荷遮断を扱わないため未評価とする。",
        )
        for key in sorted(bridge_set)
        if contingencies is None and key not in candidates
    ]
    kept: list[tuple[int, int]] = []
    for key in candidates:
        if key in bridge_set:
            skipped.append(
                (
                    key,
                    "橋（この枝が唯一の連絡路）なので、開放すると系統が島に"
                    "分かれる。連結系統を前提とする LODF は適用できず、"
                    "本教材は島ごとの需給再配分を扱わない。gridops.ybus.bridges() "
                    "が独立に同じ枝を返す。除外は健全という意味ではない。",
                )
            )
        else:
            kept.append(key)

    if not kept:
        details = "\n".join(f"  - {k}: {reason[:40]}…" for k, reason in skipped)
        raise ValueError(
            "評価できる想定事故が 1 件もない。候補が空か、すべて橋である。\n"
            + (details or "  （候補そのものが空）")
            + "\n  contingencies に評価したい枝を指定すること。"
        )

    # --- 第 1 段: 直流スクリーニング（method="lodf" のときだけ）--------
    screening: dict[tuple[int, int], np.ndarray] = {}
    if method == "lodf" and kept:
        base_dc = dc_powerflow(case, dispatch=dispatch)
        L = lodf(case, outages=kept)
        for key in kept:
            k = keys.index(key)
            post = base_dc.flows + L[:, k] * base_dc.flows[k]
            post[k] = 0.0          # 開放した枝の潮流は厳密にゼロ
            screening[key] = post

    # --- 第 2 段: 判定 --------------------------------------------------
    results: list[ContingencyResult] = []
    for key in kept:
        screen_loading = (
            float(np.max(np.abs(screening[key]) / rates))
            if key in screening
            else float("nan")
        )
        use_ac = method == "ac" or (
            check_voltage
            and (math.isnan(screen_loading) or screen_loading >= screen_threshold)
        )
        if use_ac:
            result = _evaluate_ac(
                case, key, keys, rates, limit, dispatch, pi_n, screen_loading,
                check_voltage,
            )
        else:
            result = _evaluate_lodf(
                key, keys, rates, limit, screening[key], pi_n
            )
        results.append(result)

    return SecurityReport(
        base=base,
        results=results,
        skipped=skipped,
        case=case,
        limit=limit,
        method=method,
        check_voltage=check_voltage,
        seconds=time.perf_counter() - started,
    )


def _evaluate_lodf(
    key: tuple[int, int],
    keys: list[tuple[int, int]],
    rates: np.ndarray,
    limit: str,
    post_flows: np.ndarray,
    pi_n: int,
) -> ContingencyResult:
    """直流スクリーニングの結果だけで :class:`ContingencyResult` を作る。

    電圧は見ていないので ``voltage_checked=False``、``v_min`` は ``nan``
    である。``flows`` に入るのは有効電力 :math:`|P|` であって皮相電力
    :math:`|S|` ではない（直流には無効電力が無い）。負荷率はふつう
    過小評価になるが、直流は :math:`P` 自体も近似しているので上に
    外れることもある（安全側の近似ではない）。
    """
    flows = np.abs(np.asarray(post_flows, dtype=float))
    loading = flows / rates
    worst = int(np.argmax(loading))
    return ContingencyResult(
        outage=key,
        flows=flows,
        loading=np.asarray(loading, dtype=float),
        v_min=float("nan"),
        v_min_bus=-1,
        performance_index=performance_index(flows, rates, n=pi_n),
        worst_branch=keys[worst],
        worst_loading=float(loading[worst]),
        islanding=False,
        converged=True,
        limit=limit,
        v_limit=float("nan"),
        v_violations=(),
        voltage_checked=False,
        method="lodf",
        screening_loading=float(loading[worst]),
        branch_keys=tuple(keys),
    )


def _evaluate_ac(
    case: Case,
    key: tuple[int, int],
    keys: list[tuple[int, int]],
    rates: np.ndarray,
    limit: str,
    dispatch: Mapping[str, float] | None,
    pi_n: int,
    screen_loading: float,
    check_voltage: bool,
) -> ContingencyResult:
    """事故後の交流潮流を解いて :class:`ContingencyResult` を作る。

    収束しなかった場合は ``converged=False`` の結果を返す（例外にしない）。
    N-1 の掃引が 1 件の未収束で止まってしまうと、他の事故の情報まで
    失われるためである。未収束は :attr:`ContingencyResult.thermal_secure`
    を ``False`` にするので、健全に丸められることはない。

    ``check_voltage=False`` のときも交流は解く（``method="ac"`` なら
    そう指示されているため）が、電圧は **判定に使わない**。最低電圧の
    値そのものは結果に残るので、「0.9418 という数字が目の前にあるのに
    判定は健全」という誤判定の姿がそのまま見える。
    """
    outaged = _outaged_case(case, key)
    try:
        solution = solve_powerflow(outaged, dispatch=dispatch)
    except RuntimeError:
        # 未収束は「この定式化・初期値・ソルバでは事故後の定常状態を
        # 確認できない」ことを意味する。解の不存在の証明ではないが、
        # 確認できない以上、安全側に不合格として返す。
        zeros = np.zeros(len(keys))
        return ContingencyResult(
            outage=key,
            flows=zeros,
            loading=zeros,
            v_min=float("nan"),
            v_min_bus=-1,
            performance_index=float("nan"),
            worst_branch=key,
            worst_loading=float("nan"),
            islanding=False,
            converged=False,
            limit=limit,
            v_limit=float("nan"),
            v_violations=("事故後の潮流が収束しなかった。",),
            voltage_checked=check_voltage,
            method="ac",
            screening_loading=screen_loading,
            branch_keys=tuple(keys),
        )

    magnitudes = solution.apparent_flows()
    ratios = solution.loading(limit)
    flows = np.array([magnitudes.get(k, 0.0) for k in keys])
    loading = np.array([ratios.get(k, 0.0) for k in keys])
    worst = int(np.argmax(loading))
    v_min_bus, v_min = solution.min_voltage()
    return ContingencyResult(
        outage=key,
        flows=flows,
        loading=loading,
        v_min=float(v_min),
        v_min_bus=int(v_min_bus),
        performance_index=performance_index(flows, rates, n=pi_n),
        worst_branch=keys[worst],
        worst_loading=float(loading[worst]),
        islanding=False,
        converged=bool(solution.converged),
        limit=limit,
        v_limit=float(case.buses[case.index_of(v_min_bus)].v_min),
        v_violations=_voltage_violations(outaged, solution.v),
        voltage_checked=check_voltage,
        method="ac",
        screening_loading=screen_loading,
        branch_keys=tuple(keys),
    )


# ======================================================================
# セキュリティ制約付き経済配分（SCED）
# ======================================================================
def _bus_loads_pu(case: Case, demand_mw: float | None) -> np.ndarray:
    """母線ごとの負荷 [p.u.] を母線の並び順で返す。

    ``demand_mw`` を与えたときは、母線ごとの負荷の **比を保ったまま**
    合計がその値になるよう一律にスケールする。``None`` ならケースの
    負荷そのもの（WSCC 9 母線では合計 3.15 p.u. = 315 MW）。

    Raises
    ------
    ValueError
        総負荷がゼロのケースに ``demand_mw`` を与えたとき（比を保った
        配分ができない）。
    """
    load = np.array([bus.pd for bus in case.buses], dtype=float)
    if demand_mw is None:
        return load
    total = float(load.sum())
    if total <= 0.0:
        raise ValueError(
            f"ケース '{case.name}' の総負荷がゼロなので、demand_mw="
            f"{demand_mw} を母線に配分できない。母線ごとの負荷 Bus.pd を"
            "与えるか、demand_mw を省くこと。"
        )
    return load * (case.to_pu(float(demand_mw)) / total)


def _sensitivity(
    case: Case, units: Sequence[Unit], load_pu: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """枝潮流を号機の出力 [MW] の 1 次式で表す係数を返す。

    .. math::

        f_{\\ell}(p) = \\sum_j \\frac{\\mathrm{PTDF}[\\ell, b(j)]}{S_{base}}
        \\, p_j - \\left(\\mathrm{PTDF}\\, d\\right)_{\\ell}

    Returns
    -------
    tuple
        ``(G, f0)``。``G`` は ``(n_branch, n_unit)`` で単位は
        [p.u./MW]、``f0`` は ``(n_branch,)`` で単位は [p.u.]。

    Notes
    -----
    PTDF は slack の取り方に依存するが、注入の総和がゼロ
    （:math:`\\sum_j p_j = \\sum_b d_b`、直流は無損失）である限り
    :math:`f` は slack に依存しない。:func:`sced` が需給バランスを
    等式で課しているので、この条件は常に満たされる。
    """
    H = ptdf(case)
    G = np.zeros((case.n_branch, len(units)))
    for j, unit in enumerate(units):
        G[:, j] = H[:, case.index_of(unit.bus)] / case.base_mva
    return G, -(H @ load_pu)


@dataclass
class SCEDResult:
    """セキュリティ制約付き経済配分の結果。

    Parameters
    ----------
    dispatch:
        号機名から **事故前**の出力 [MW] への対応。実際に運転する点は
        これである（是正的でも、事故が起きるまではこの点で運転する）。
    total_cost:
        目的関数の値 :math:`\\sum_i c_i p_i` [円/h]。1 次費用のみ
        （PuLP は 2 次費用を扱えない）。
    binding:
        ``(事故, 拘束した枝)`` の組の一覧。最適解で等号が成立した
        制約だけを並べる。
    iterations:
        制約生成の周回数。最後の 1 周は「違反が 1 件も見つからなかった」
        ことを確かめる周回である（ほかに基準用の線形計画を 1 回解く）。
    case:
        解いたケース（契約に無い追加）。
    mode:
        ``"preventive"`` / ``"corrective"``（契約に無い追加）。
    limit:
        事故後の熱容量の名前（契約に無い追加）。
    base_limit:
        事故前の熱容量の名前（契約に無い追加）。
    unconstrained_cost:
        送電制約を **1 本も**課さない経済配分の費用 [円/h]（契約に無い
        追加）。:meth:`cost_of_security` の基準にそのまま使える。
    base_binding:
        事故前（N-0）の段階で拘束した枝（契約に無い追加）。
    corrective_dispatch:
        是正的のとき、事故ごとの **事故後**の出力 [MW]（契約に無い追加）。
        予防的では空。テストで実行可能性を独立に検算するのに要る。
    status:
        ソルバの状態（契約に無い追加）。
    seconds:
        求解に要した時間 [s]（契約に無い追加）。

    Notes
    -----
    費用は **事故前の出力**で測る。是正的でも、事故は「起きるかもしれ
    ない」ものであって毎時間起きるわけではないからである。是正的が安く
    なるのは事故後に動く余地を織り込めるからであって、事故後の運転費が
    ただになるからではない。
    """

    dispatch: dict[str, float]
    total_cost: float
    binding: list[tuple[tuple[int, int], tuple[int, int]]]
    iterations: int
    case: Case | None = None
    mode: str = "preventive"
    limit: str = "rate_b"
    base_limit: str = "rate_a"
    unconstrained_cost: float = float("nan")
    base_binding: list[tuple[int, int]] = field(default_factory=list)
    corrective_dispatch: dict[tuple[int, int], dict[str, float]] = field(
        default_factory=dict
    )
    status: str = ""
    seconds: float = 0.0

    # ------------------------------------------------------------------
    def cost_of_security(self, base_cost: float) -> float:
        """セキュリティの値段 [円/h]。

        Parameters
        ----------
        base_cost:
            比較の基準にする費用。ふつうは送電制約を課さない経済配分の
            費用（:attr:`unconstrained_cost` にも入っている）。

        Returns
        -------
        float
            ``total_cost - base_cost``。制約を足したぶん必ず非負になる
            （同じ実行可能領域の部分集合を最適化しているため）。

        Notes
        -----
        WSCC 9 母線（315 MW、1 次費用、``rate_a`` / ``rate_b``）では

        .. code-block:: text

            制約なし  3,343,500 円/h
            是正的    3,383,029 円/h   （+   39,529）
            予防的    3,896,400 円/h   （+  552,900）

        となり、予防的の値段は是正的の 14 倍である。**同じ系統・同じ
        需要・同じ事故の集合でも、事故後に再給電できるかどうかで
        セキュリティの値段が桁で変わる。** 系統に投資するか運用で
        しのぐかの判断は、この差を測ることから始まる。
        """
        return float(self.total_cost - base_cost)

    def summary(self) -> str:
        """要約を返す（日本語）。"""
        name = self.case.name if self.case is not None else "?"
        lines = [
            f"SCED '{name}' — {self.mode}"
            f"（事故前 {self.base_limit} / 事故後 {self.limit}）",
            f"  状態         : {self.status} / 制約生成 {self.iterations} 周 /"
            f" {self.seconds:.3f} s",
            f"  総費用       : {self.total_cost:,.0f} 円/h",
        ]
        if math.isfinite(self.unconstrained_cost):
            lines.append(
                f"  制約なし     : {self.unconstrained_cost:,.0f} 円/h"
                f"（セキュリティの値段 "
                f"{self.cost_of_security(self.unconstrained_cost):+,.0f} 円/h）"
            )
        lines.append(
            "  出力         : "
            + ", ".join(
                f"{name_}={value:.1f}" for name_, value in sorted(self.dispatch.items())
            )
            + " MW"
        )
        if self.base_binding:
            lines.append(
                "  N-0 で拘束   : "
                + ", ".join(f"{k[0]}-{k[1]}" for k in self.base_binding)
            )
        if self.binding:
            lines.append(f"  N-1 で拘束   : {len(self.binding)} 件")
            for outage, branch in self.binding:
                lines.append(
                    f"    - 事故 {outage[0]}-{outage[1]} のとき枝 "
                    f"{branch[0]}-{branch[1]}"
                )
        else:
            lines.append("  N-1 で拘束   : なし（事故は運転点を縛っていない）")
        return "\n".join(lines)


def sced(
    case: Case,
    *,
    demand_mw: float | None = None,
    contingencies: Sequence[Sequence[int]] | None = None,
    mode: str = "preventive",
    limit: str = "rate_b",
    max_iter: int = 20,
    base_limit: str = "rate_a",
    corrective_ramp_fraction: float = 1.0,
) -> SCEDResult:
    """セキュリティ制約付き経済配分を制約生成で解く。

    「全事故に耐える運転点のうち最も安いもの」を直流最適潮流として解く。
    事故の数だけ制約を最初から並べると問題が大きくなるので、
    **違反した事故だけを制約として足し、違反が無くなるまで繰り返す**
    （制約生成 / lazy constraints）。実系統の SCED はこの形で実装される。

    Parameters
    ----------
    case:
        系統ケース（``network`` と ``units`` の層が要る）。
    demand_mw:
        総需要 [MW]。``None`` ならケースの母線負荷そのもの
        （WSCC 9 母線では 315 MW）。値を与えると母線ごとの比を保った
        まま一律にスケールする。
    contingencies:
        想定事故。``None`` ならケースの ``contingencies`` 層。橋は
        自動的に除かれる。
    mode:
        ``"preventive"`` か ``"corrective"``（下の Notes を参照）。
    limit:
        **事故後**の熱容量。既定は ``"rate_b"``（緊急時許容容量）。
    max_iter:
        制約生成の上限周回数。
    base_limit:
        **事故前**の熱容量（契約に無い追加）。既定は ``"rate_a"``
        （常時許容容量）。事故前と事故後で許される負荷率が違う、という
        運用の規約をそのまま引数にしてある。
    corrective_ramp_fraction:
        是正的のとき、事故後に動かせる幅を :attr:`Unit.ramp_up` /
        :attr:`Unit.ramp_down` の何倍にするか（契約に無い追加）。
        ``1.0`` は「1 時間ぶんのランプを丸ごと使える」、``0.0`` は
        「まったく動かせない」で、後者は予防的と厳密に同じ問題になる。

    Returns
    -------
    SCEDResult

    Raises
    ------
    ValueError
        ``mode`` / ``limit`` が不正なとき。実行可能な運転点が存在
        しないとき（:func:`gridops.solvers.solve` の日本語診断が出る）。
        ``max_iter`` 周しても違反が消えないとき。

    Notes
    -----
    **予防的と是正的の違い。** 予防的 (preventive) は「事故が起きても
    発電機を動かさずに耐える」。したがって事故前後で出力は同じ 1 組
    :math:`p` であり、すべての事故の潮流制約を **同じ** :math:`p` が
    満たさねばならない（:func:`screen_n1` が事故前の発電を据え置いて
    評価するのと同じ前提である）。是正的 (corrective) は「事故後に再給電できる」
    ことを前提にし、事故 :math:`c` ごとに別の出力 :math:`p^c` を許す。
    ただし発電機が動ける幅はランプ率で決まるので

    .. math:: |p^c_i - p_i| \\le \\alpha\\, R_i, \\qquad \\sum_i p^c_i = D

    を課す（:math:`\\alpha` が ``corrective_ramp_fraction``）。**是正的の
    実行可能領域は予防的を含む**（:math:`p^c = p` と置けばよい）ので、
    費用は必ず

    .. code-block:: text

        予防的の費用 >= 是正的の費用 >= 制約なしの経済配分の費用

    となる。WSCC 9 母線では 3,896,400 >= 3,383,029 >= 3,343,500 円/h で、
    3 つとも真に異なる。``corrective_ramp_fraction=0.0`` にすると
    是正的の費用は予防的にぴったり一致する（動けないなら是正できない）。

    **直流で解く理由。** 費用最小化を線形計画にするには潮流が出力の
    1 次式でなければならない。交流潮流は非線形なのでそのままでは載らない
    （実務では逐次線形化するが、教材の範囲を超える）。したがって
    :func:`sced` の答えは **直流の意味で** N-1 に耐える運転点である。
    交流の :math:`|S|` は直流の :math:`P` より大きく出るのがふつう
    （常にではない。:func:`screen_n1` の Notes を参照）なので、得られた
    運転点を :func:`screen_n1` に掛けて交流で確かめる手順まで含めて
    1 つの作業とすること。

    **なぜ制約生成で最適解が出るのか。** 各周回で解いているのは、
    まだ足していない事故の制約を落とした **緩和問題**である。緩和問題の
    最適値は元の問題の最適値以下（下界）になる。一方、打ち切ったときの
    解は「全事故を検査して違反ゼロ」を確認しているので元の問題に対して
    **実行可能**であり、その費用は最適値以上（上界）である。下界と上界が
    同じ 1 つの解で達成されるので、それが最適解である。**足していない
    制約は「無視した」のではなく「効かないことを確かめた」のである。**
    WSCC 9 母線では 6 事故 x 9 枝 = 54 本の潮流制約のうち、実際に足されるの
    はごく一部で済む。

    **費用は 1 次費用のみ。** :attr:`Unit.var_cost` だけを使い、2 次項
    :attr:`Unit.quadratic` と無負荷費 :attr:`Unit.noload_cost` は入って
    いない（PuLP は線形計画しか扱えない）。同じ理由で
    :attr:`gridops.dispatch.DCOPFResult.total_cost` とは直接比較できるが、
    :attr:`gridops.dispatch.DispatchResult.total_cost` とは比較できない。

    **下限は :attr:`Unit.p_min_mw`。** 起動停止は扱わないので、すべての
    号機が運転している前提である。入切まで決めたいなら
    :func:`gridops.commitment.unit_commitment` の担当である。

    Examples
    --------
    >>> from gridops import load_case
    >>> from gridops.security import sced
    >>> case = load_case("wscc9")
    >>> preventive = sced(case)
    >>> corrective = sced(case, mode="corrective")
    >>> preventive.total_cost >= corrective.total_cost >= preventive.unconstrained_cost
    True
    """
    case.require("network", "units")
    if mode not in SCED_MODES:
        raise ValueError(
            f"mode={mode!r} は使えない。{SCED_MODES} のいずれかを指定すること"
            "（'preventive' が事前に耐える、'corrective' が事故後に再給電する）。"
        )
    if max_iter < 1:
        raise ValueError(f"max_iter={max_iter} は 1 以上でなければならない。")
    if corrective_ramp_fraction < 0.0:
        raise ValueError(
            f"corrective_ramp_fraction={corrective_ramp_fraction} が負。"
            "0.0（動かせない = 予防的と同じ）以上に取ること。"
        )
    post_rates = _limit_array(case, limit)
    base_rates = _limit_array(case, base_limit)
    started = time.perf_counter()

    units = list(case.units)
    if not units:
        raise ValueError(
            f"ケース '{case.name}' に号機がない。SCED は出力を動かす対象が"
            "無ければ成り立たない。"
        )
    keys = _branch_keys(case)
    load_pu = _bus_loads_pu(case, demand_mw)
    demand = float(case.to_mw(load_pu.sum()))
    G, f0 = _sensitivity(case, units, load_pu)

    candidates = _candidates(case, contingencies)
    bridge_set = set(bridges(case))
    kept = [key for key in candidates if key not in bridge_set]
    L = lodf(case, outages=kept) if kept else np.zeros((case.n_branch, case.n_branch))

    # 事故 c ごとの「出力 -> 事故後潮流」の 1 次式を先に作っておく。
    post: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, int]] = {}
    for key in kept:
        k = keys.index(key)
        post[key] = (G + np.outer(L[:, k], G[k]), f0 + L[:, k] * f0[k], k)

    ramp_up = np.array(
        [corrective_ramp_fraction * unit.ramp_up for unit in units], dtype=float
    )
    ramp_down = np.array(
        [corrective_ramp_fraction * unit.ramp_down for unit in units], dtype=float
    )

    # --- 基準: 送電制約を 1 本も課さない経済配分 -----------------------
    unconstrained = _solve_sced_lp(
        case, units, demand, G, f0, base_rates, post_rates, post,
        active={}, corrective=False, ramp_up=ramp_up, ramp_down=ramp_down,
        network=False,
    )

    # --- 制約生成 -------------------------------------------------------
    corrective = mode == "corrective"
    active: dict[tuple[int, int], set[int]] = {}
    iterations = 0
    solution = None
    for iterations in range(1, max_iter + 1):
        solution = _solve_sced_lp(
            case, units, demand, G, f0, base_rates, post_rates, post,
            active=active, corrective=corrective,
            ramp_up=ramp_up, ramp_down=ramp_down, network=True,
        )
        p = np.array([solution["dispatch"][unit.name] for unit in units])
        added = 0
        for key in kept:
            Gc, fc0, k = post[key]
            used = solution["corrective"].get(key)
            p_used = (
                np.array([used[unit.name] for unit in units]) if used is not None else p
            )
            flow = Gc @ p_used + fc0
            for ell in range(case.n_branch):
                if ell == k:
                    continue
                if abs(flow[ell]) > post_rates[ell] + FLOW_TOLERANCE:
                    known = active.setdefault(key, set())
                    if ell not in known:
                        known.add(ell)
                        added += 1
        if added == 0:
            break
    else:
        raise ValueError(
            f"SCED の制約生成が {max_iter} 周しても収束しなかった"
            f"（ケース '{case.name}', mode={mode}）。max_iter を増やす前に、"
            "熱容量 rate_b が需要に対して足りているか（screen_n1 の結果）と、"
            "橋を候補に入れていないかを確かめること。"
        )

    # max_iter >= 1 を上で検査しているので、ここで solution は必ず存在する。
    assert solution is not None
    p = np.array([solution["dispatch"][unit.name] for unit in units])

    # --- 拘束した制約を読み取る（等号が成立したものだけ）---------------
    base_flow = G @ p + f0
    base_binding = sorted(
        keys[ell]
        for ell in range(case.n_branch)
        if math.isfinite(base_rates[ell])
        and abs(base_flow[ell]) >= base_rates[ell] - FLOW_TOLERANCE
    )
    binding: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for key in sorted(active):
        Gc, fc0, k = post[key]
        used = solution["corrective"].get(key)
        p_used = (
            np.array([used[unit.name] for unit in units]) if used is not None else p
        )
        flow = Gc @ p_used + fc0
        for ell in sorted(active[key]):
            if abs(flow[ell]) >= post_rates[ell] - FLOW_TOLERANCE:
                binding.append((key, keys[ell]))

    return SCEDResult(
        dispatch=dict(solution["dispatch"]),
        total_cost=float(solution["cost"]),
        binding=binding,
        iterations=iterations,
        case=case,
        mode=mode,
        limit=limit,
        base_limit=base_limit,
        unconstrained_cost=float(unconstrained["cost"]),
        base_binding=base_binding,
        corrective_dispatch={
            key: dict(value) for key, value in solution["corrective"].items()
        },
        status=str(solution["status"]),
        seconds=time.perf_counter() - started,
    )


def _solve_sced_lp(
    case: Case,
    units: Sequence[Unit],
    demand: float,
    G: np.ndarray,
    f0: np.ndarray,
    base_rates: np.ndarray,
    post_rates: np.ndarray,
    post: Mapping[tuple[int, int], tuple[np.ndarray, np.ndarray, int]],
    *,
    active: Mapping[tuple[int, int], set[int]],
    corrective: bool,
    ramp_up: np.ndarray,
    ramp_down: np.ndarray,
    network: bool,
) -> dict[str, object]:
    """SCED の線形計画を 1 回組んで解く（内部）。

    ``active`` に入っている事故の、入っている枝の制約だけを課す。
    これが制約生成の「足していく」側の実装である。

    需給バランスは :func:`gridops.solvers.solve` の規約どおり
    ``lpSum(p) == demand`` の向きで書く。この向きなら双対がそのまま
    限界費用 [円/MWh] になる。
    """
    problem = solvers.problem("sced")
    p = {
        unit.name: solvers.variable(f"p_{unit.name}", unit.p_min_mw, unit.p_max_mw)
        for unit in units
    }
    problem += solvers.lp_sum(unit.var_cost * p[unit.name] for unit in units), "fuel_cost"
    problem += solvers.lp_sum(p.values()) == demand, "balance"

    if network:
        for ell, rate in enumerate(base_rates):
            if not math.isfinite(rate):
                continue
            flow = solvers.lp_sum(
                G[ell, j] * p[unit.name] for j, unit in enumerate(units)
            )
            problem += (flow + f0[ell] <= rate, f"base-pos-{ell}")
            problem += (-(flow + f0[ell]) <= rate, f"base-neg-{ell}")

    corrective_vars: dict[tuple[int, int], dict[str, object]] = {}
    for key, branches in active.items():
        Gc, fc0, _ = post[key]
        tag = f"{key[0]}_{key[1]}"
        if corrective:
            q = {
                unit.name: solvers.variable(
                    f"p_{tag}_{unit.name}", unit.p_min_mw, unit.p_max_mw
                )
                for unit in units
            }
            corrective_vars[key] = q
            problem += solvers.lp_sum(q.values()) == demand, f"balance-{tag}"
            for j, unit in enumerate(units):
                problem += (
                    q[unit.name] - p[unit.name] <= ramp_up[j],
                    f"rampup-{tag}-{unit.name}",
                )
                problem += (
                    p[unit.name] - q[unit.name] <= ramp_down[j],
                    f"rampdn-{tag}-{unit.name}",
                )
        else:
            q = p
        for ell in sorted(branches):
            if not math.isfinite(post_rates[ell]):
                continue
            flow = solvers.lp_sum(
                Gc[ell, j] * q[unit.name] for j, unit in enumerate(units)
            )
            problem += (flow + fc0[ell] <= post_rates[ell], f"post-{tag}-pos-{ell}")
            problem += (
                -(flow + fc0[ell]) <= post_rates[ell],
                f"post-{tag}-neg-{ell}",
            )

    solution = solvers.solve(
        problem,
        context=(
            f"セキュリティ制約付き経済配分 (ケース '{case.name}', "
            f"需要 {demand:.1f} MW, 事故 {len(active)} 件)"
        ),
    )
    dispatch = {
        unit.name: float(solution.values[f"p_{unit.name}"]) for unit in units
    }
    corrective_dispatch = {
        key: {
            unit.name: float(solution.values[f"p_{key[0]}_{key[1]}_{unit.name}"])
            for unit in units
        }
        for key in corrective_vars
    }
    return {
        "dispatch": dispatch,
        "corrective": corrective_dispatch,
        "cost": float(
            sum(unit.var_cost * dispatch[unit.name] for unit in units)
        ),
        "status": solution.status,
    }
