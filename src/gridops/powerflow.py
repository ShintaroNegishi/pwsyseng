"""交流潮流計算 — Newton-Raphson / Gauss-Seidel / Fast Decoupled。

潮流計算は電力系統の全テーマの土台である。経済負荷配分の損失も、
N-1 の過負荷判定も、安定度の内部起電力も、すべてここで得た
:math:`(|V|, \\theta)` の上に乗る。したがって本モジュールの設計目標は
「速く解く」ことではなく、**なぜ解けるのか・なぜ解けないのかが
学生に見えること**である。

なぜ解析ヤコビアンなのか
------------------------
安定度の教材 ``genstab`` は状態行列 :math:`A` を **数値微分（中心差分）**
で作っている。発電機モデルや制御器を差し替えてもコードを書き換えずに
線形化できるようにするためであり、あちらではそれが正しい判断である。

本モジュールは逆に **解析式**でヤコビアンを書く。潮流方程式

.. math::

    P_i = \\sum_j |V_i||V_j|
          \\bigl(G_{ij}\\cos\\theta_{ij} + B_{ij}\\sin\\theta_{ij}\\bigr),
    \\qquad
    Q_i = \\sum_j |V_i||V_j|
          \\bigl(G_{ij}\\sin\\theta_{ij} - B_{ij}\\cos\\theta_{ij}\\bigr)

は **固定であってモデルが差し替わらない**ので、微分を一度手で書いて
しまえば以後変わらない。解析式なら 1 反復あたりの計算量が
:math:`O(n^2)` で済み（数値微分だと未知数 1 つにつき 1 回ずつ電力を
計算し直すので :math:`O(n^3)`）、丸め誤差も入らないので Newton 法が
本来の二次収束を示す。刻み幅 :math:`h` の選び方という別の問題も
持ち込まずに済む。

その代わり、解析式には **写し間違いを実行時に教えてくれる仕組みが
ない**。そこで正しさは中心差分と突き合わせて確認する
（``tests/test_powerflow.py`` の ``test_jacobian_matches_central_difference``）。
「解析式で速く解き、数値微分で正しさを担保する」という役割分担であって、
どちらか一方が優れているという話ではない。

修正量の形
----------
Newton 法の修正量は :math:`[\\Delta\\theta;\\ \\Delta|V|/|V|]` の形にとる。
ヤコビアンの N・L ブロックを :math:`\\partial P/\\partial |V| \\cdot |V|`,
:math:`\\partial Q/\\partial |V| \\cdot |V|` と定義してあるのはこのためで、
両者は釣り合っている。この形にすると 4 つのブロックがすべて電力の
次元をもち、成分の大きさが揃うのでヤコビアンの条件数が良くなる。
高電圧系統で :math:`|N|, |M| \\ll |H|, |L|` が見えるのもこの形のときで、
Fast Decoupled 法の根拠がそのまま数値として現れる。

収束判定
--------
**判定はミスマッチ（電力の不釣り合い）の無限大ノルムで行い、修正量
:math:`\\Delta x` では判定しない。** 修正量で判定すると、ヤコビアンが
悪条件（電圧安定限界の近傍など）のときに「動きが小さいだけ」を収束と
誤判定する。ミスマッチは物理量そのもの（p.u. の電力）なので、
``tol=1e-10`` は「どの母線でも 1e-8 MW 以内で釣り合っている」と読める。

符号の規約
----------
母線への **注入**を正とする。指定注入 :math:`(P^{sp}, Q^{sp})` を
組み立てるのは :meth:`gridops.case.Case.bus_injection` だけであり、
本モジュールは自分で発電と負荷を合成しない。唯一の例外は
``enforce_q_limits`` で PV 母線を PQ に切り替えるときの
:math:`Q^{sp}` の差し替えで、これは「発電機の無効電力が上下限に
張り付いた」という物理そのものである。

3 つの解法の位置づけ
--------------------
=================  ======================================================
``newton``         標準。二次収束（ミスマッチが 1e-1 → 1e-3 → 1e-7 →
                   1e-14 と桁が倍々に増える）。毎反復ヤコビアンを組む
``gauss_seidel``   一次収束。反復回数は桁違いに多いが 1 回が軽く、
                   ヤコビアンを持たないので記憶容量が要らない
``fast_decoupled`` 定数行列 B', B''（**XB 版**）を 1 度だけ LU 分解して
                   使い回す。:math:`|N|, |M| \\ll |H|, |L|` という近似の上に立つ
``dc``             :mod:`gridops.dc` に委譲する線形近似（反復なし）
=================  ======================================================

3 つの交流解法は **同じ方程式を解いている**ので、収束すれば同じ解に
行き着く。違うのは「どう近づくか」だけである。この事実を確かめるのが
第 02 回の主題であり、テストでも 1e-6 以内の一致として固定してある。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from .case import BusType, Case
from .ybus import build_ybus, islands

__all__ = [
    "DEFAULT_MAX_ITER",
    "METHODS",
    "PowerFlowSolution",
    "mismatch",
    "jacobian_blocks",
    "jacobian",
    "solve",
]

#: 解法ごとの最大反復回数の既定値。
#:
#: Gauss-Seidel だけ桁が違うのは一次収束だからであって、実装が悪い
#: からではない。ここを 20 に揃えてしまうと「Gauss-Seidel は収束しない
#: 手法だ」という誤った印象を与える。
DEFAULT_MAX_ITER: dict[str, int] = {
    "newton": 20,
    "fast_decoupled": 50,
    "gauss_seidel": 1000,
    "dc": 1,
}

#: 使える解法の名前。
METHODS = tuple(DEFAULT_MAX_ITER)


# ======================================================================
# 内部ヘルパ — 添字・電力・ヤコビアンの素
# ======================================================================
def _sets_from_types(
    case: Case, types: Mapping[int, BusType]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """母線種別の対応から ``(slack, pv, pq, non_slack)`` の添字を作る。

    ``non_slack`` は PV と PQ を **母線の並び順**に混ぜた配列である。
    P 方程式の並びをこの順にしておくと、母線番号と方程式の対応が
    ``case.buses`` の順そのままになり、ミスマッチの最大成分がどの母線の
    ものかを人が読める（収束しなかったときの診断でこれが効く）。
    """
    slack, pv, pq = [], [], []
    for i, bus in enumerate(case.buses):
        kind = types[bus.id]
        (slack if kind is BusType.SLACK else pv if kind is BusType.PV else pq).append(i)
    non_slack = sorted(pv + pq)
    return (
        np.array(slack, dtype=int),
        np.array(pv, dtype=int),
        np.array(pq, dtype=int),
        np.array(non_slack, dtype=int),
    )


def _injected_power(Y: np.ndarray, voltage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """複素電圧から母線注入 :math:`\\bar S = \\bar V (Y \\bar V)^{*}` を求める。

    ヤコビアンの三角関数表現とは別の（複素数のままの）道筋で同じ量を
    出しているので、両者の一致は組み立ての検算になる。
    """
    s = voltage * np.conj(Y @ voltage)
    return s.real.copy(), s.imag.copy()


def _mismatch_vector(
    Y: np.ndarray,
    v_mag: np.ndarray,
    theta: np.ndarray,
    p_sp: np.ndarray,
    q_sp: np.ndarray,
    non_slack: np.ndarray,
    pq: np.ndarray,
) -> np.ndarray:
    """``[ΔP(slack 以外); ΔQ(PQ のみ)]`` を組む（内部用）。"""
    p_calc, q_calc = _injected_power(Y, v_mag * np.exp(1j * theta))
    return np.concatenate([(p_sp - p_calc)[non_slack], (q_sp - q_calc)[pq]])


def _full_blocks(
    Y: np.ndarray, v_mag: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """母線全体（縮約前）の 4 ブロックを解析式で作る。

    :math:`\\theta_{ij} = \\theta_i - \\theta_j` と置き、

    .. math::

        a_{ij} = |V_i||V_j|(G_{ij}\\cos\\theta_{ij} + B_{ij}\\sin\\theta_{ij}),
        \\qquad
        b_{ij} = |V_i||V_j|(G_{ij}\\sin\\theta_{ij} - B_{ij}\\cos\\theta_{ij})

    とすると :math:`P_i = \\sum_j a_{ij}`, :math:`Q_i = \\sum_j b_{ij}` で
    あり、非対角成分は

    .. math::

        H_{ij} = b_{ij}, \\quad N_{ij} = a_{ij}, \\quad
        M_{ij} = -a_{ij}, \\quad L_{ij} = b_{ij}
        \\qquad (i \\ne j)

    対角成分は

    .. math::

        H_{ii} = -Q_i - B_{ii}|V_i|^2, \\quad N_{ii} = P_i + G_{ii}|V_i|^2,
        \\quad M_{ii} = P_i - G_{ii}|V_i|^2, \\quad L_{ii} = Q_i - B_{ii}|V_i|^2

    となる。:math:`a_{ii} = G_{ii}|V_i|^2`, :math:`b_{ii} = -B_{ii}|V_i|^2`
    に注意すると、4 ブロックは ``a`` と ``b`` の対角に :math:`\\pm P`,
    :math:`\\pm Q` を足すだけで書ける。

    Notes
    -----
    この導出は :math:`Y` の対称性を **一切使っていない**。位相調整器
    (``shift_deg != 0``) があると Ybus は非対称になるが、上の式は
    そのまま成り立つ。
    """
    G, B = Y.real, Y.imag
    difference = theta[:, None] - theta[None, :]
    products = np.outer(v_mag, v_mag)
    a = products * (G * np.cos(difference) + B * np.sin(difference))
    b = products * (G * np.sin(difference) - B * np.cos(difference))
    p = a.sum(axis=1)
    q = b.sum(axis=1)
    return b - np.diag(q), a + np.diag(p), -a + np.diag(p), b + np.diag(q)


def _blocks(
    Y: np.ndarray,
    v_mag: np.ndarray,
    theta: np.ndarray,
    non_slack: np.ndarray,
    pq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """縮約した 4 ブロックを返す（内部用）。"""
    H, N, M, L = _full_blocks(Y, v_mag, theta)
    return (
        H[np.ix_(non_slack, non_slack)],
        N[np.ix_(non_slack, pq)],
        M[np.ix_(pq, non_slack)],
        L[np.ix_(pq, pq)],
    )


def _assemble(H: np.ndarray, N: np.ndarray, M: np.ndarray, L: np.ndarray) -> np.ndarray:
    """4 ブロックを ``[[H, N], [M, L]]`` に並べる。"""
    return np.block([[H, N], [M, L]])


def _q_capability(
    case: Case, bus_id: int, dispatch: Mapping[str, float] | None
) -> tuple[float, float]:
    """母線につながる号機の無効電力の可能範囲 ``(q_min, q_max)`` [p.u.]。

    号機が 1 台もなければ制限なし（``(-inf, +inf)``）とする。``dispatch``
    を与えた場合は **運転中の号機だけ**を数える。停止した発電機が
    無効電力を出し続ける解を防ぐためで、
    :meth:`Case.effective_bus_types` の判定と揃えてある。
    """
    units = case.units_at(bus_id)
    if dispatch is not None:
        units = [u for u in units if dispatch.get(u.name, 0.0) > 0.0]
    if not units:
        return (-math.inf, math.inf)
    return (
        float(sum(u.q_min for u in units)),
        float(sum(u.q_max for u in units)),
    )


# ======================================================================
# 解
# ======================================================================
@dataclass
class PowerFlowSolution:
    """潮流解と、そこから読み取れる量。

    電圧の解そのもの :math:`(|V|, \\theta)` だけを状態として持ち、
    枝潮流も損失も slack 出力もすべてここから導く。導いた量を
    フィールドに持たせないのは、片方だけ更新して不整合になる事故
    （``genstab`` の内部起電力で実際に起こりやすい種類の事故）を
    構造的に防ぐためである。

    Parameters
    ----------
    case:
        解いた系統ケース。N-1 の解では **開放後**のケースが入る。
    v, theta:
        電圧の大きさ [p.u.] と位相 [rad]。並びは :attr:`Case.buses` の順。
    converged:
        収束したか。``False`` の解を握って先へ進んではいけない。
    iterations:
        反復回数。
    mismatch_history:
        反復ごとのミスマッチの無限大ノルム [p.u.]。先頭は **初期値での**
        値であり、長さは ``iterations + 1`` になる。対数軸で描くと
        Newton の二次収束（傾きが倍々に急になる）と Gauss-Seidel の
        一次収束（直線）の違いがそのまま見える。
    method:
        使った解法の名前。
    dispatch:
        号機名から出力 [MW] への対応。``None`` なら参照解の発電を使った。
    q_limited:
        無効電力の上下限に張り付いて PQ 母線に切り替えた母線の番号。
        ``enforce_q_limits=False`` なら常に空。

    Notes
    -----
    ``method="dc"`` の解は :math:`|V| = 1`, 有効電力のみの線形近似で
    あり、:meth:`branch_flows` に交流の式を通すと直流潮流の値とは
    一致しない（充電容量と損失の分だけずれる）。直流の枝潮流は
    :class:`gridops.dc.DCSolution` の ``flows`` を見ること。
    """

    case: Case
    v: np.ndarray
    theta: np.ndarray
    converged: bool
    iterations: int
    mismatch_history: list[float] = field(default_factory=list)
    method: str = "newton"
    dispatch: Mapping[str, float] | None = None
    q_limited: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.v = np.asarray(self.v, dtype=float)
        self.theta = np.asarray(self.theta, dtype=float)
        self._ybus: np.ndarray | None = None

    # ------------------------------------------------------------------
    @property
    def voltage(self) -> np.ndarray:
        """複素電圧 :math:`\\bar V = |V| e^{j\\theta}`。"""
        return self.v * np.exp(1j * self.theta)

    @property
    def angle_deg(self) -> np.ndarray:
        """位相 [deg]。教科書の表と見比べるための便宜。"""
        return np.degrees(self.theta)

    def ybus(self) -> np.ndarray:
        """このケースの母線アドミタンス行列（初回だけ組んで使い回す）。"""
        if self._ybus is None:
            self._ybus = build_ybus(self.case)
        return self._ybus

    # ------------------------------------------------------------------
    def injections(self) -> tuple[np.ndarray, np.ndarray]:
        """実際の注入 :math:`(P, Q)` [p.u.]（母線の並び順）。

        指定した注入ではなく **解から計算した**注入である。PQ 母線では
        指定値に一致し（それが収束したということ）、slack 母線では
        発電が、PV 母線では無効電力が、解いた結果として出てくる。
        両者の差がそのままミスマッチである。
        """
        return _injected_power(self.ybus(), self.voltage)

    def _branch_powers(self) -> list[tuple[complex, complex]]:
        """枝ごとの ``(S_from->to, S_to->from)`` を :attr:`Case.branches` の順に返す。"""
        voltage = self.voltage
        flows: list[tuple[complex, complex]] = []
        for branch in self.case.branches:
            f = self.case.index_of(branch.from_bus)
            t = self.case.index_of(branch.to_bus)
            primitive = branch.primitive()
            current = primitive @ np.array([voltage[f], voltage[t]])
            terminal = np.array([voltage[f], voltage[t]]) * np.conj(current)
            flows.append((complex(terminal[0]), complex(terminal[1])))
        return flows

    def branch_flows(self) -> dict[tuple[int, int], tuple[complex, complex]]:
        """枝ごとの両端の複素潮流 ``{key: (S_from->to, S_to->from)}`` [p.u.]。

        枝の 2x2 行列 :meth:`Branch.primitive` から端子電流を作り、
        :math:`\\bar S = \\bar V \\bar I^{*}` で電力にする。**2 つの値の
        和が枝の損失**であり、符号は「その端子から枝に流れ込む向きが正」
        である。したがって両方が正なら枝は電力を消費しており（普通の
        線路）、片方が負ならそちらへ流れ出している。

        Notes
        -----
        キーは :meth:`Branch.key` なので、**同じ母線対を結ぶ多重回線は
        1 つのキーに潰れる**（後の枝が前の枝を上書きする）。回線ごとに
        区別したい場合は :meth:`Case.branches` の順に並んだ内部表現を
        使うこと。WSCC 9 母線に多重回線はない。
        """
        return {
            branch.key(): flow
            for branch, flow in zip(self.case.branches, self._branch_powers())
        }

    def apparent_flows(self) -> dict[tuple[int, int], float]:
        """枝ごとの皮相電力 :math:`|S|` [p.u.]。**両端の大きい方**を採る。

        熱容量の判定は両端で見なければならない。線路の損失と充電容量の
        ぶんだけ両端の :math:`|S|` は異なり、どちらが大きいかは潮流の
        向きと負荷の重さで変わるからである。
        """
        return {
            branch.key(): float(max(abs(s_ft), abs(s_tf)))
            for branch, (s_ft, s_tf) in zip(self.case.branches, self._branch_powers())
        }

    def loading(self, limit: str = "rate_a") -> dict[tuple[int, int], float]:
        """熱容量に対する負荷率（``1.0`` で定格）。

        Parameters
        ----------
        limit:
            ``"rate_a"``（常時）または ``"rate_b"``（緊急時）。N-1 の
            事故後は ``"rate_b"`` で見るのが規約である。

        Notes
        -----
        分母は **皮相電力**の制限である。直流潮流の有効電力 :math:`P` と
        比べてはいけない。WSCC 9 母線の枝 4-5 では交流の :math:`|S|` が
        直流の :math:`P` より 47.6% 大きい。

        制限が ``inf``（未設定）の枝は負荷率 0 とする。制限が 0 以下の
        枝は ``inf`` を返す（データの誤りとして目立たせる）。
        """
        if limit not in ("rate_a", "rate_b"):
            raise ValueError(
                f"limit='{limit}' は使えない。'rate_a'（常時）か "
                "'rate_b'（緊急時）を指定すること。"
            )
        result: dict[tuple[int, int], float] = {}
        for branch, (s_ft, s_tf) in zip(self.case.branches, self._branch_powers()):
            magnitude = float(max(abs(s_ft), abs(s_tf)))
            rating = float(getattr(branch, limit))
            if not math.isfinite(rating):
                result[branch.key()] = 0.0
            elif rating <= 0.0:
                result[branch.key()] = math.inf
            else:
                result[branch.key()] = magnitude / rating
        return result

    # ------------------------------------------------------------------
    @property
    def losses(self) -> float:
        """枝の有効電力損失の合計 [p.u.]。

        各枝の両端の有効電力の和 :math:`\\mathrm{Re}(S_{ft} + S_{tf})` を
        足したものである。母線シャント ``gs`` がゼロなら、これは注入の
        総和 :math:`\\sum_i P_i` に厳密に一致する（発電の合計 - 負荷の
        合計 = 損失）。``gs`` がある系統では、その消費のぶんだけ注入の
        総和のほうが大きくなる。
        """
        return float(sum((s_ft + s_tf).real for s_ft, s_tf in self._branch_powers()))

    @property
    def slack_power(self) -> complex:
        """slack 母線の **発電** :math:`P + jQ` [p.u.]。

        注入に母線の負荷を足し戻した値である。slack 母線に負荷がない
        WSCC 9 母線では注入と一致する。この値は解いてはじめて分かる量
        （需給の差と損失を引き受けた結果）であって、入力ではない。
        """
        slack, _, _, _ = _sets_from_types(self.case, {b.id: b.type for b in self.case.buses})
        if slack.size == 0:
            raise ValueError(
                f"ケース '{self.case.name}' に slack 母線がない。"
                "Case.check() を先に通すこと。"
            )
        index = int(slack[0])
        p, q = self.injections()
        bus = self.case.buses[index]
        return complex(p[index] + bus.pd, q[index] + bus.qd)

    def min_voltage(self) -> tuple[int, float]:
        """最も電圧の低い母線 ``(母線番号, |V|)``。"""
        index = int(np.argmin(self.v))
        return self.case.buses[index].id, float(self.v[index])

    # ------------------------------------------------------------------
    def violations(self, limit: str = "rate_a") -> list[str]:
        """過負荷と電圧逸脱の一覧（日本語）。空リストなら健全である。

        Parameters
        ----------
        limit:
            熱容量の判定に使う制限。事故後の評価では ``"rate_b"``。

        Notes
        -----
        **熱容量と電圧は独立に見なければならない。** WSCC 9 母線で枝
        4-6 を開放すると、最悪の枝でも 75.7% で熱容量は健全なのに、
        母線 6 の電圧が 0.9418 p.u. まで落ちて下限 0.95 を割る。
        熱容量だけを見る N-1 スクリーニングはこの事故を「健全」と
        誤判定する。この関数が両方を返すのはそのためである。
        """
        messages: list[str] = []
        if not self.converged:
            messages.append(
                "潮流が収束していない。以下の判定は意味を持たない"
                "（収束しないこと自体が最も重い異常である）。"
            )
        magnitudes = self.apparent_flows()
        for branch, ratio in self.loading(limit).items():
            if ratio > 1.0:
                rating = getattr(self.case.branches[self._branch_index(branch)], limit)
                messages.append(
                    f"枝 {branch[0]}-{branch[1]}: 負荷率 {ratio * 100:.1f}% "
                    f"（{limit} = {rating:.2f} p.u. に対し "
                    f"{magnitudes[branch]:.4f} p.u.）"
                )
        for i, bus in enumerate(self.case.buses):
            if self.v[i] < bus.v_min:
                messages.append(
                    f"母線 {bus.id}: 電圧 {self.v[i]:.4f} p.u. が下限 "
                    f"{bus.v_min:.2f} を下回っている"
                )
            elif self.v[i] > bus.v_max:
                messages.append(
                    f"母線 {bus.id}: 電圧 {self.v[i]:.4f} p.u. が上限 "
                    f"{bus.v_max:.2f} を上回っている"
                )
        return messages

    def _branch_index(self, key: tuple[int, int]) -> int:
        """枝の識別子から :attr:`Case.branches` の添字を引く。"""
        for i, branch in enumerate(self.case.branches):
            if branch.key() == key:
                return i
        raise KeyError(f"枝 {key} はケース '{self.case.name}' にない。")

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """解の要約を返す。"""
        bus_id, v_min = self.min_voltage()
        slack = self.slack_power
        residual = self.mismatch_history[-1] if self.mismatch_history else float("nan")
        lines = [
            f"潮流解 '{self.case.name}' — {self.method}"
            + ("（収束）" if self.converged else "（**未収束**）"),
            f"  反復回数     : {self.iterations}"
            + (f" / 最終ミスマッチ {residual:.3e} p.u." if self.mismatch_history else ""),
            f"  slack 出力   : {slack.real:.6f} + {slack.imag:.6f}j p.u."
            f" ({self.case.to_mw(slack.real):.1f} MW)",
            f"  総損失       : {self.losses:.6f} p.u."
            f" ({self.case.to_mw(self.losses):.2f} MW)",
            f"  最低電圧     : 母線 {bus_id} で {v_min:.4f} p.u.",
        ]
        if self.q_limited:
            lines.append(
                "  Q 制限       : 母線 "
                + ", ".join(str(b) for b in self.q_limited)
                + " が無効電力の上下限に張り付き PQ 母線に切り替わった"
            )
        problems = self.violations()
        if problems:
            lines.append(f"  逸脱         : {len(problems)} 件")
            lines.extend(f"    - {message}" for message in problems)
        else:
            lines.append("  逸脱         : なし（rate_a と電圧の両方）")
        return "\n".join(lines)


# ======================================================================
# 方程式とヤコビアン
# ======================================================================
def mismatch(
    case: Case,
    Y: np.ndarray,
    v_mag: np.ndarray,
    theta: np.ndarray,
    *,
    dispatch: Mapping[str, float] | None = None,
) -> np.ndarray:
    """電力ミスマッチ ``[ΔP(slack 以外); ΔQ(PQ のみ)]`` [p.u.] を返す。

    .. math::

        \\Delta P_i = P_i^{sp} - P_i(|V|, \\theta), \\qquad
        \\Delta Q_i = Q_i^{sp} - Q_i(|V|, \\theta)

    Parameters
    ----------
    case:
        系統ケース。
    Y:
        母線アドミタンス行列（:func:`gridops.ybus.build_ybus`）。
    v_mag, theta:
        電圧の大きさ [p.u.] と位相 [rad]。:attr:`Case.buses` の並び順。
    dispatch:
        号機名から出力 [MW] への対応。``None`` なら参照解の発電を使う。

    Returns
    -------
    numpy.ndarray
        長さ :math:`n_{PV} + 2 n_{PQ}` のベクトル。前半が P の残差
        （slack を除く全母線、**母線の並び順**）、後半が Q の残差
        （PQ 母線のみ）。

    Notes
    -----
    指定注入 :math:`(P^{sp}, Q^{sp})` は :meth:`Case.bus_injection` から
    取る。**注入を組み立てるのはあの 1 箇所だけ**という規約であり、
    負荷を負の発電として足し込む符号ミスを構造的に防いでいる。

    slack 母線の P 方程式と、PV・slack 母線の Q 方程式が式の側から
    落ちているのは、それらが「未知数 :math:`P, Q` を後から決める式」
    だからである。方程式の数 :math:`n_{PV} + 2 n_{PQ}` は
    :meth:`Case.n_unknowns` と必ず一致する。
    """
    p_sp, q_sp = case.bus_injection(dispatch)
    _, _, pq, non_slack = _sets_from_types(case, case.effective_bus_types(dispatch))
    return _mismatch_vector(
        Y, np.asarray(v_mag, float), np.asarray(theta, float), p_sp, q_sp, non_slack, pq
    )


def jacobian_blocks(
    case: Case, Y: np.ndarray, v_mag: np.ndarray, theta: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ヤコビアンの 4 ブロック :math:`(H, N, M, L)` を **解析式**で返す。

    .. math::

        H = \\frac{\\partial P}{\\partial \\theta}, \\qquad
        N = \\frac{\\partial P}{\\partial |V|}|V|, \\qquad
        M = \\frac{\\partial Q}{\\partial \\theta}, \\qquad
        L = \\frac{\\partial Q}{\\partial |V|}|V|

    Returns
    -------
    tuple of numpy.ndarray
        形はそれぞれ ``(n_ns, n_ns)``, ``(n_ns, n_pq)``,
        ``(n_pq, n_ns)``, ``(n_pq, n_pq)``。``n_ns`` は slack 以外の
        母線数である。行と列の並びは :func:`mismatch` と揃えてある。

    Notes
    -----
    **4 ブロックを個別に取り出せるようにしてあるのは、高電圧系統で
    :math:`|N|, |M| \\ll |H|, |L|` であることを学生に測らせるため**で
    ある。送電線は :math:`x \\gg r` なので、有効電力は主に位相差で、
    無効電力は主に電圧差で決まる。この非対角ブロックの小ささが
    Fast Decoupled 法の唯一の根拠であり、WSCC 9 母線では
    :math:`\\max|N| / \\max|H|` は 0.084、:math:`\\max|M| / \\max|L|` は
    0.102 に収まる（実測）。配電系統のように
    :math:`r \\sim x` の系統ではこの前提が崩れ、Fast Decoupled は
    収束しなくなる（Newton は収束する）。

    ``N`` と ``L`` に :math:`|V|` を掛けてあるのは、修正量を
    :math:`\\Delta |V| / |V|` の形（無次元）で扱うためである。こうすると
    4 ブロックがすべて電力の次元になり、成分の大きさが揃う。

    :func:`jacobian_blocks` は ``dispatch`` を取らない。母線種別は
    ケースのものをそのまま使うので、起動停止の結果を反映した種別で
    ヤコビアンを組みたい場合は :func:`jacobian` に ``dispatch`` を
    渡すこと。
    """
    _, _, pq, non_slack = _sets_from_types(case, {b.id: b.type for b in case.buses})
    return _blocks(Y, np.asarray(v_mag, float), np.asarray(theta, float), non_slack, pq)


def jacobian(
    case: Case,
    Y: np.ndarray,
    v_mag: np.ndarray,
    theta: np.ndarray,
    *,
    dispatch: Mapping[str, float] | None = None,
) -> np.ndarray:
    """潮流ヤコビアン :math:`J = [[H, N], [M, L]]`。

    未知数の並びは :math:`[\\Delta\\theta;\\ \\Delta|V|/|V|]`、方程式の
    並びは :func:`mismatch` と同じ ``[ΔP; ΔQ]`` である。すなわち
    Newton の 1 歩は :math:`J \\Delta x = \\Delta S` を解いて

    .. math::

        \\theta \\leftarrow \\theta + \\Delta\\theta, \\qquad
        |V| \\leftarrow |V| \\left(1 + \\frac{\\Delta |V|}{|V|}\\right)

    と更新することになる。**2 番目の式が掛け算であること**に注意する
    （L ブロックに :math:`|V|` を掛けてあるぶんを戻している）。

    Returns
    -------
    numpy.ndarray
        ``(n_unknowns, n_unknowns)`` の正方行列。次元は
        :meth:`Case.n_unknowns` に一致する。

    Notes
    -----
    この行列が特異になる点が **電圧安定の限界（ノーズ点）**である。
    そこでは「解が無い」のであって「ソルバが悪い」のではない。
    最小特異値を余裕の指標に使う話は :mod:`gridops.voltage` で扱う。
    """
    _, _, pq, non_slack = _sets_from_types(case, case.effective_bus_types(dispatch))
    return _assemble(
        *_blocks(Y, np.asarray(v_mag, float), np.asarray(theta, float), non_slack, pq)
    )


# ======================================================================
# 解法の実体
# ======================================================================
def _norm(vector: np.ndarray) -> float:
    """ミスマッチの無限大ノルム。空ベクトルは 0 とする。"""
    return float(np.max(np.abs(vector))) if vector.size else 0.0


def _newton_kernel(
    Y: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    p_sp: np.ndarray,
    q_sp: np.ndarray,
    non_slack: np.ndarray,
    pq: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[bool, int, list[float], str | None]:
    """Newton-Raphson 法。``v`` と ``theta`` をその場で更新する。

    修正量は :math:`[\\Delta\\theta;\\ \\Delta|V|/|V|]` なので、電圧の
    更新は **掛け算** :math:`|V| \\leftarrow |V|(1 + \\Delta|V|/|V|)` に
    なる。足し算で書くと N・L ブロックに掛けた :math:`|V|` のぶんだけ
    修正が過大になり、収束が遅くなる（多くの場合は収束するので
    気づきにくい種類の誤りである）。
    """
    n_theta = int(non_slack.size)
    residual = _mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq)
    history = [_norm(residual)]
    iterations = 0

    while history[-1] > tol and iterations < max_iter:
        matrix = _assemble(*_blocks(Y, v, theta, non_slack, pq))
        try:
            step = np.linalg.solve(matrix, residual)
        except np.linalg.LinAlgError:
            return (
                False,
                iterations,
                history,
                "ヤコビアンが特異になった（数値的に逆行列が作れない）。"
                "電圧安定の限界（ノーズ点）を越えたか、系統が島に分かれている。",
            )
        theta[non_slack] += step[:n_theta]
        v[pq] *= 1.0 + step[n_theta:]
        iterations += 1
        residual = _mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq)
        history.append(_norm(residual))
        if not np.isfinite(history[-1]):
            return (False, iterations, history, "ミスマッチが発散した（inf / nan）。")

    return history[-1] <= tol, iterations, history, None


def _gauss_seidel_kernel(
    Y: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    p_sp: np.ndarray,
    q_sp: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    non_slack: np.ndarray,
    v_set: np.ndarray,
    tol: float,
    max_iter: int,
    acceleration: float,
) -> tuple[bool, int, list[float], str | None]:
    """加速係数つき Gauss-Seidel 法。

    母線 :math:`i` の電圧を、他の母線の **最新の**値を使って

    .. math::

        \\bar V_i \\leftarrow \\frac{1}{Y_{ii}}
        \\left(\\frac{P_i^{sp} - jQ_i^{sp}}{\\bar V_i^{*}}
              - \\sum_{j \\ne i} Y_{ij}\\bar V_j\\right)

    で置き換えていく。得られた値をそのまま採らず

    .. math::

        \\bar V_i^{new} = \\bar V_i + \\alpha(\\bar V_i^{calc} - \\bar V_i)

    と **行き過ぎさせる**のが加速係数 :math:`\\alpha` である。
    :math:`\\alpha = 1` が素の Gauss-Seidel で、1.4〜1.7 で反復回数が
    半分以下になる。大きくしすぎると発散する（:math:`\\alpha \\ge 2` は
    ほぼ必ず発散する）ので、教材では 1.6 を既定にしてある。

    PV 母線の扱いが要点である。PV 母線では :math:`Q^{sp}` が未知なので、

    1. 現在の電圧から :math:`Q_i = \\mathrm{Im}
       \\bigl(\\bar V_i (Y\\bar V)_i^{*}\\bigr)` を **計算し**、
    2. それを使って上の式で電圧を更新し、
    3. 得られた電圧の **大きさを設定値 :math:`V_i^{set}` に戻す**
       （位相だけを採用する）

    という 3 段階を踏む。3 を忘れると PV 母線が実質 PQ 母線になり、
    「収束したが電圧が設定値からずれている」解が出る。

    Notes
    -----
    一次収束なので、ミスマッチは反復ごとにほぼ一定の比で減る（対数軸で
    直線になる）。Newton の二次収束（対数軸で下に折れ曲がる）と並べて
    描くと違いが一目で分かる。反復 1 回は Newton よりずっと軽く、
    ヤコビアンを持たないので記憶容量も要らない。大規模系統で
    Newton が使われるのは反復回数が母線数にほとんど依存しないからで、
    「1 回の軽さ」ではなく「回数の少なさ」が効いている。
    """
    voltage = v * np.exp(1j * theta)
    pv_set = set(int(i) for i in pv)
    residual = _mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq)
    history = [_norm(residual)]
    iterations = 0

    while history[-1] > tol and iterations < max_iter:
        for index in non_slack:
            i = int(index)
            if i in pv_set:
                # PV 母線: 無効電力を「計算して」使う。
                q_calc = float(np.imag(voltage[i] * np.conj(Y[i] @ voltage)))
                power = complex(p_sp[i], q_calc)
            else:
                power = complex(p_sp[i], q_sp[i])
            others = Y[i] @ voltage - Y[i, i] * voltage[i]
            updated = (np.conj(power) / np.conj(voltage[i]) - others) / Y[i, i]
            updated = voltage[i] + acceleration * (updated - voltage[i])
            if i in pv_set:
                # 大きさは設定値に戻し、位相だけを採用する。
                updated = v_set[i] * updated / abs(updated)
            voltage[i] = updated

        iterations += 1
        v[:] = np.abs(voltage)
        theta[:] = np.angle(voltage)
        residual = _mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq)
        history.append(_norm(residual))
        if not np.isfinite(history[-1]):
            return (
                False,
                iterations,
                history,
                f"ミスマッチが発散した（inf / nan）。加速係数 "
                f"acceleration={acceleration} が大きすぎる可能性がある。",
            )

    return history[-1] <= tol, iterations, history, None


def _fdlf_matrices(
    case: Case, non_slack: np.ndarray, pq: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fast Decoupled 法の定数行列 :math:`B'`, :math:`B''` を作る（**XB 版**）。

    版がいくつもあるので、どれを採ったかを必ず書いておく必要がある。
    本実装は **XB 版**（Stott & Alsac の原型）である。

    ==========  ==========================================================
    :math:`B'`  枝の直列 **抵抗を無視**（``r = 0``）し、線路の充電容量
                ``b``・母線シャント・タップ比・位相調整角も落とした
                行列の :math:`-\\mathrm{Im}`。実質 :math:`1/x` だけの
                行列になる
    :math:`B''` 実際の Ybus の :math:`-\\mathrm{Im}`（抵抗も充電容量も
                母線シャントも含む）。位相調整角だけ落とす
    ==========  ==========================================================

    BX 版は :math:`B'` と :math:`B''` で抵抗を落とす側が逆になる。
    :math:`r/x` が大きい系統では BX 版のほうが収束が良いことが知られて
    いるが、教材では原型の XB 版で「近似の出どころ」を追いやすくした。

    Notes
    -----
    :math:`B'` は :math:`\\Delta P/|V|` から :math:`\\Delta\\theta` を、
    :math:`B''` は :math:`\\Delta Q/|V|` から :math:`\\Delta|V|` を出す。
    どちらも **反復を通じて変わらない**ので、LU 分解は最初の 1 回だけで
    済む。これが「Fast」の中身であって、収束が速いわけではない
    （反復回数は Newton より多い）。
    """
    for branch in case.branches:
        if branch.x == 0.0:
            raise ValueError(
                f"枝 {branch.label} のリアクタンスがゼロ。Fast Decoupled 法は "
                "B' を 1/x から組むので x = 0 の枝があると作れない。"
                "Newton 法（method='newton'）を使うこと。"
            )

    prime = [
        replace(branch, r=0.0, b=0.0, tap=1.0, shift_deg=0.0)
        for branch in case.branches
    ]
    double = [replace(branch, shift_deg=0.0) for branch in case.branches]

    b_prime = -np.imag(
        build_ybus(replace(case, branches=prime), include_shunts=False)
    )
    b_double = -np.imag(build_ybus(replace(case, branches=double)))
    return (
        b_prime[np.ix_(non_slack, non_slack)],
        b_double[np.ix_(pq, pq)],
    )


def _fast_decoupled_kernel(
    case: Case,
    Y: np.ndarray,
    v: np.ndarray,
    theta: np.ndarray,
    p_sp: np.ndarray,
    q_sp: np.ndarray,
    non_slack: np.ndarray,
    pq: np.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[bool, int, list[float], str | None]:
    """Fast Decoupled 法（XB 版）。

    .. math::

        B' \\Delta\\theta = \\frac{\\Delta P}{|V|}, \\qquad
        B'' \\Delta |V| = \\frac{\\Delta Q}{|V|}

    を交互に解く。1 反復は「P-θ の半歩 → 電力を計算し直す → Q-V の
    半歩」であり、**半歩ごとにミスマッチを作り直す**のが要点である
    （まとめて 1 歩にすると収束が目に見えて悪くなる）。

    :math:`B'`, :math:`B''` は定数なので LU 分解を 1 度だけ行い、以後は
    前進後退代入だけで済ませる。ヤコビアンを毎回組み直す Newton 法との
    差はここにある。
    """
    from scipy.linalg import lu_factor, lu_solve   # 分解は 1 度だけ

    b_prime, b_double = _fdlf_matrices(case, non_slack, pq)
    try:
        prime_lu = lu_factor(b_prime)
        double_lu = lu_factor(b_double)
    except (np.linalg.LinAlgError, ValueError):
        return (
            False,
            0,
            [_norm(_mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq))],
            "B' または B'' が特異で LU 分解できない。系統が島に分かれている可能性がある。",
        )

    residual = _mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq)
    history = [_norm(residual)]
    iterations = 0

    while history[-1] > tol and iterations < max_iter:
        # --- P-θ の半歩 ---------------------------------------------
        p_calc, _ = _injected_power(Y, v * np.exp(1j * theta))
        theta[non_slack] += lu_solve(
            prime_lu, (p_sp - p_calc)[non_slack] / v[non_slack]
        )
        # --- Q-V の半歩（電力を作り直してから）-----------------------
        _, q_calc = _injected_power(Y, v * np.exp(1j * theta))
        v[pq] += lu_solve(double_lu, (q_sp - q_calc)[pq] / v[pq])

        iterations += 1
        residual = _mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq)
        history.append(_norm(residual))
        if not np.isfinite(history[-1]):
            return (False, iterations, history, "ミスマッチが発散した（inf / nan）。")

    return history[-1] <= tol, iterations, history, None


# ======================================================================
# 収束しなかったときの診断
# ======================================================================
def _worst_mismatch(
    case: Case,
    residual: np.ndarray,
    non_slack: np.ndarray,
    pq: np.ndarray,
) -> str:
    """ミスマッチの最大成分が「どの母線の何の式か」を日本語で返す。"""
    if residual.size == 0:
        return "ミスマッチの成分がない（未知数がゼロ）。"
    k = int(np.argmax(np.abs(residual)))
    if k < non_slack.size:
        index, kind = int(non_slack[k]), "有効電力 ΔP"
    else:
        index, kind = int(pq[k - non_slack.size]), "無効電力 ΔQ"
    bus = case.buses[index]
    value = float(residual[k])
    return (
        f"母線 {bus.id}（{bus.label}）の{kind} = {value:+.4e} p.u. "
        f"= {case.to_mw(value):+.2f} MW"
    )


def _q_report(
    case: Case,
    q_calc: np.ndarray,
    types: Mapping[int, BusType],
    dispatch: Mapping[str, float] | None,
) -> list[str]:
    """PV 母線の無効電力と、その上下限との関係を並べる。"""
    lines: list[str] = []
    for i, bus in enumerate(case.buses):
        if types[bus.id] is BusType.PQ:
            continue
        q_min, q_max = _q_capability(case, bus.id, dispatch)
        q_gen = float(q_calc[i]) + bus.qd
        if not math.isfinite(q_min) and not math.isfinite(q_max):
            lines.append(f"母線 {bus.id}: Q_gen = {q_gen:+.4f} p.u.（制限なし）")
            continue
        mark = ""
        if q_gen > q_max:
            mark = f" ← 上限 {q_max:+.2f} を超過"
        elif q_gen < q_min:
            mark = f" ← 下限 {q_min:+.2f} を下回る"
        lines.append(
            f"母線 {bus.id}: Q_gen = {q_gen:+.4f} p.u. "
            f"（可能範囲 [{q_min:+.2f}, {q_max:+.2f}]）{mark}"
        )
    return lines


def _divergence_error(
    case: Case,
    method: str,
    v: np.ndarray,
    theta: np.ndarray,
    p_sp: np.ndarray,
    q_sp: np.ndarray,
    types: Mapping[int, BusType],
    non_slack: np.ndarray,
    pq: np.ndarray,
    dispatch: Mapping[str, float] | None,
    tol: float,
    iterations: int,
    history: Sequence[float],
    note: str | None,
) -> RuntimeError:
    """収束しなかったときの :class:`RuntimeError` を組み立てる。

    「解が無い」のか「初期値が悪い」のかを切り分けられるよう、
    (a) 最大ミスマッチとその母線、(b) 発電機の無効電力と上下限、
    (c) 島の有無 の 3 つを必ず並べる。ソルバの設定をいじる前に
    データとトポロジーを疑うのが順序である。
    """
    Y = build_ybus(case)
    residual = _mismatch_vector(Y, v, theta, p_sp, q_sp, non_slack, pq)
    _, q_calc = _injected_power(Y, v * np.exp(1j * theta))
    component = islands(case)

    lines = [
        f"{method} 法が {iterations} 回の反復で収束しなかった"
        f"（ミスマッチの無限大ノルム {history[-1]:.3e} > tol {tol:.1e}）。",
    ]
    if note:
        lines.append(f"  直接の原因: {note}")
    lines.append(f"  (a) 最大ミスマッチ: {_worst_mismatch(case, residual, non_slack, pq)}")
    lines.append("  (b) 発電機の無効電力:")
    lines.extend(f"      {line}" for line in _q_report(case, q_calc, types, dispatch))
    if len(component) == 1:
        lines.append("  (c) トポロジー: 系統は連結（島は 1 個）。")
    else:
        lines.append(
            f"  (c) トポロジー: 系統が {len(component)} 個の島に分かれている: "
            + " / ".join(str(island) for island in component)
            + "。島ごとに slack 母線がなければ潮流計算は定義できない。"
        )
    trend = ", ".join(f"{value:.2e}" for value in list(history)[:4])
    if len(history) > 5:
        trend += ", ..., " + ", ".join(f"{value:.2e}" for value in list(history)[-2:])
    lines.append(f"  ミスマッチの推移: {trend}")
    lines.append(
        "  切り分け: ミスマッチが増え続けているなら初期値ではなく "
        "**その運転点に解が存在しない**ことを疑う（負荷が重すぎる／"
        "電圧安定の限界を越えている）。Case.check() を先に通し、"
        "gridops.voltage.pv_curve で負荷倍率を下げながらノーズ点を探すこと。"
        "減ってはいるが遅い場合は max_iter を増やすか method='newton' を試すこと。"
    )
    return RuntimeError("\n".join(lines))


# ======================================================================
# 入口
# ======================================================================
def solve(
    case: Case,
    method: str = "newton",
    *,
    tol: float = 1e-10,
    max_iter: int | None = None,
    dispatch: Mapping[str, float] | None = None,
    v0: Sequence[float] | np.ndarray | None = None,
    theta0: Sequence[float] | np.ndarray | None = None,
    enforce_q_limits: bool = False,
    acceleration: float = 1.6,
) -> PowerFlowSolution:
    """交流潮流を解く。

    Parameters
    ----------
    case:
        系統ケース。N-1 の評価では枝を開放したケースを渡すが、そのとき
        発電の据え置きに注意すること（下の Notes を参照）。
    method:
        ``"newton"`` / ``"gauss_seidel"`` / ``"fast_decoupled"`` / ``"dc"``。
    tol:
        収束判定に使う **ミスマッチの無限大ノルム** [p.u.]。既定の
        1e-10 は 100 MVA 基準で 1e-8 MW にあたる。
    max_iter:
        最大反復回数。``None`` なら :data:`DEFAULT_MAX_ITER`
        （newton 20 / fast_decoupled 50 / gauss_seidel 1000）。
    dispatch:
        号機名から出力 [MW] への対応。``None`` なら参照解の母線単位の
        発電を使う。運転している号機が 1 台もない PV 母線は
        :meth:`Case.effective_bus_types` により PQ 母線に落ちる。
    v0, theta0:
        初期値。``None`` なら :attr:`Bus.v_set` と位相ゼロの
        **フラットスタート**。PV・slack 母線の :math:`|V|` は、``v0`` を
        与えた場合でも設定値で上書きする（設定電圧を保つことが PV 母線の
        定義だからである）。
    enforce_q_limits:
        発電機の無効電力の上下限を反映するか（下記参照）。
    acceleration:
        Gauss-Seidel の加速係数 :math:`\\alpha`。他の解法では使わない。
        1.4〜1.7 が実用域で、2 以上はほぼ必ず発散する。

    Returns
    -------
    PowerFlowSolution
        収束した解。

    Raises
    ------
    ValueError
        解法の名前が不正なとき、slack 母線がちょうど 1 つでないとき、
        初期値の長さが母線数と合わないとき、加速係数が非正のとき。
    RuntimeError
        収束しなかったとき。メッセージには (a) 最大ミスマッチとその母線、
        (b) 発電機の無効電力と上下限、(c) 島の有無 が並ぶので、
        「解が無い」のか「初期値が悪い」のかを切り分けられる。

    Notes
    -----
    **収束判定はミスマッチの無限大ノルムで行う。** 修正量
    :math:`\\Delta x` では判定しない。修正量で判定すると、ヤコビアンが
    悪条件のときに「動きが小さいだけ」を収束と誤判定するからである。
    電圧安定の限界の近くではこの誤判定が実際に起こる。

    ``method="fast_decoupled"`` は **XB 版**（Stott & Alsac の原型）である。
    :math:`B'` は抵抗・充電容量・タップ・位相調整角を落とした
    :math:`1/x` だけの行列、:math:`B''` は実際の Ybus の虚部（位相調整角
    のみ落とす）から作る。BX 版は抵抗を落とす側が逆で、:math:`r/x` の
    大きい系統ではそちらのほうが収束が良い。

    **N-1 の評価で注意すること。** :meth:`Case.without_branch` は既定で
    ``reference=None`` にするので、``dispatch`` を与えずにその結果を
    渡すと :meth:`Case.bus_injection` が **発電ゼロ**を返し、
    「slack 母線 1 台で全負荷を賄う」まったく別の系統を解くことになる
    （WSCC 9 母線では収束しない）。事故前の出力を据え置いて解きたい
    場合は ``case.without_branch(key, keep_generation=True)`` を使うか、
    ``dispatch`` を明示すること（:func:`gridops.security.screen_n1` は
    前者を使っている）。負荷倍率をかける :meth:`Case.scaled` にも同じ
    ``keep_generation`` がある。

    ``method="dc"`` は :func:`gridops.dc.dc_powerflow` に委譲する
    （import は関数の中で行い、``gridops.dc`` と ``gridops.powerflow``
    の循環参照を避けている）。返る解は :math:`|V| = 1`, 損失ゼロの
    線形近似であり、:meth:`PowerFlowSolution.branch_flows` に交流の式を
    通しても直流の枝潮流とは一致しない。直流の枝潮流は
    :class:`gridops.dc.DCSolution` の ``flows`` を見ること。

    無効電力の上下限（``enforce_q_limits=True``）
    ---------------------------------------------
    PV 母線の発電機が無効電力の上下限に達したら、その母線は電圧を
    支えられなくなる。本実装は **反復の外で** 次の手順を踏む。

    1. まず全 PV 母線を PV のまま解く
    2. 各 PV 母線の :math:`Q_{gen} = Q_i + q_d` を上下限と比べる
    3. 外れた母線を **PQ 母線に切り替え**、:math:`Q^{sp}` を超えた側の
       制限値に固定して解き直す（前回の解を初期値にする）
    4. 切り替えが起きなくなるまで繰り返す

    **これは厳密な扱いではない。** 正しくは反復の途中で判定し、電圧が
    設定値まで戻せる場合には PQ から PV へ **戻す**必要がある。そこまで
    実装すると、ある母線を PQ にすると別の母線が制限内に戻り、戻すと
    また外れる、という **切替の振動が起きて収束しないことがある**
    （実務のソルバはこの振動を抑えるための工夫を持っている）。本実装は
    PV → PQ の一方向にしか切り替えないので反復は必ず止まるが、その
    代わり得られる解は厳密でない。どちらの母線が制限に達したかは
    :attr:`PowerFlowSolution.q_limited` に残るので、結果を読むときは
    必ず確認すること。

    無効電力の可能範囲は :attr:`Unit.q_min` / :attr:`Unit.q_max` の和で
    ある。号機が 1 台もない PV 母線には制限がない（例えば調相設備だけの
    母線を PV として置いた場合）。

    Examples
    --------
    >>> from gridops import load_case
    >>> from gridops.powerflow import solve
    >>> solution = solve(load_case("wscc9"))
    >>> round(solution.losses, 6)
    0.04641
    """
    if method not in DEFAULT_MAX_ITER:
        raise ValueError(
            f"method='{method}' は使えない。指定できるのは "
            f"{', '.join(repr(name) for name in METHODS)}。"
        )
    if acceleration <= 0.0:
        raise ValueError(
            f"acceleration={acceleration} が非正。Gauss-Seidel の加速係数は "
            "1.0（素の Gauss-Seidel）以上 2.0 未満に取ること。"
        )

    if method == "dc":
        # 循環 import を避けるため関数の中で読み込む。
        from .dc import dc_powerflow

        linear = dc_powerflow(case, dispatch=dispatch)
        return PowerFlowSolution(
            case=case,
            v=np.ones(case.n_bus),
            theta=np.asarray(linear.theta, dtype=float),
            converged=True,
            iterations=0,
            mismatch_history=[],
            method="dc",
            dispatch=dispatch,
        )

    limit = DEFAULT_MAX_ITER[method] if max_iter is None else int(max_iter)
    Y = build_ybus(case)
    types = dict(case.effective_bus_types(dispatch))
    p_sp, q_sp = case.bus_injection(dispatch)

    slack, pv_initial, _, _ = _sets_from_types(case, types)
    if slack.size != 1:
        raise ValueError(
            f"ケース '{case.name}' の slack 母線が {slack.size} 個ある"
            "（ちょうど 1 つでなければならない）。位相の基準と損失の"
            "受け皿がなくなるので潮流計算が定義できない。"
            "Case.check() を先に通すこと。"
        )

    # --- 初期値 -------------------------------------------------------
    v = (
        np.array([bus.v_set for bus in case.buses], dtype=float)
        if v0 is None
        else np.array(v0, dtype=float).copy()
    )
    theta = (
        np.zeros(case.n_bus)
        if theta0 is None
        else np.array(theta0, dtype=float).copy()
    )
    if v.shape != (case.n_bus,) or theta.shape != (case.n_bus,):
        raise ValueError(
            f"初期値の長さが母線数 {case.n_bus} と合わない"
            f"（v0: {v.shape}, theta0: {theta.shape}）。"
            "並びは Case.buses の順（母線番号の順ではない）である。"
        )

    q_limited: list[int] = []
    v_set = np.array([bus.v_set for bus in case.buses], dtype=float)

    # --- Q 制限の切り替えを含む外側のループ ---------------------------
    for _round in range(int(pv_initial.size) + 1):
        _, pv, pq, non_slack = _sets_from_types(case, types)
        # PV・slack 母線の大きさは設定値に固定する。
        for index in (*slack, *pv):
            v[index] = v_set[index]

        if method == "newton":
            converged, iterations, history, note = _newton_kernel(
                Y, v, theta, p_sp, q_sp, non_slack, pq, tol, limit
            )
        elif method == "gauss_seidel":
            converged, iterations, history, note = _gauss_seidel_kernel(
                Y, v, theta, p_sp, q_sp, pv, pq, non_slack,
                v_set, tol, limit, acceleration,
            )
        else:
            converged, iterations, history, note = _fast_decoupled_kernel(
                case, Y, v, theta, p_sp, q_sp, non_slack, pq, tol, limit
            )

        if not converged:
            raise _divergence_error(
                case, method, v, theta, p_sp, q_sp, types, non_slack, pq,
                dispatch, tol, iterations, history, note,
            )

        if not enforce_q_limits:
            break

        # --- Q が上下限を外れた PV 母線を PQ に落とす ------------------
        _, q_calc = _injected_power(Y, v * np.exp(1j * theta))
        switched = False
        for index in pv:
            i = int(index)
            bus = case.buses[i]
            q_min, q_max = _q_capability(case, bus.id, dispatch)
            q_gen = float(q_calc[i]) + bus.qd
            if q_gen > q_max:
                bound = q_max
            elif q_gen < q_min:
                bound = q_min
            else:
                continue
            types[bus.id] = BusType.PQ
            q_sp[i] = bound - bus.qd     # 制限値に張り付いた発電機
            q_limited.append(bus.id)
            switched = True
        if not switched:
            break

    return PowerFlowSolution(
        case=case,
        v=v,
        theta=theta,
        converged=True,
        iterations=iterations,
        mismatch_history=[float(value) for value in history],
        method=method,
        dispatch=dispatch,
        q_limited=tuple(sorted(q_limited)),
    )
