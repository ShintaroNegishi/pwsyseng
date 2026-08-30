"""直流潮流と感度係数（PTDF / LODF）。

交流潮流は非線形で、解くたびに反復が要る。N-1 のスクリーニングでは
「事故 1 件につき 1 回」の潮流計算を何百回と繰り返すので、この非線形性が
そのまま計算時間になる。直流近似の値打ちは精度ではなく **線形性**に
ある。線形であれば重ね合わせが効き、感度係数（PTDF）を一度だけ作って
おけば、以後の事故後潮流は行列とベクトルの積 1 回で出る。

近似の中身
----------
次の 4 つを同時に仮定する。

1. すべての母線で :math:`|V| \\simeq 1` p.u.
2. 枝の抵抗を無視する（:math:`r \\ll x`）
3. 位相差が小さい（:math:`\\sin\\theta_{ij} \\simeq \\theta_{ij}`,
   :math:`\\cos\\theta_{ij} \\simeq 1`）
4. 線路の充電容量と母線シャントを無視する

このとき枝潮流と母線注入は

.. math::

    f_{\\ell} = \\frac{\\theta_f - \\theta_t}{\\tau_{\\ell} x_{\\ell}},
    \\qquad
    P = B' \\theta, \\qquad
    B' = A^{T} \\operatorname{diag}(1/\\tau x) A

と、位相 :math:`\\theta` について完全に線形になる。:math:`A` は
:func:`gridops.ybus.incidence_matrix` の枝-母線接続行列である。

**捨てているもの**
------------------
- **無効電力と電圧**。直流潮流は電圧を原理的に見ない。第 09 回で
  「熱容量では健全だが電圧が下限を割る」事故（枝 4-6 の開放）を
  直流のスクリーニングが見逃すのは、この仮定の当然の帰結である。
- **損失**。:math:`r=0` としたので枝の両端の有効電力は符号だけが逆に
  なり、損失は恒等的にゼロである。slack 母線の出力は「総需要 - 他機の
  出力」ちょうどになる。
- **位相調整角**。位相調整器 :math:`\\phi \\ne 0` があると

  .. math:: f_{\\ell} = \\frac{\\theta_f - \\theta_t - \\phi_{\\ell}}{\\tau_{\\ell} x_{\\ell}}

  となり、注入側に :math:`-A^{T}\\operatorname{diag}(1/\\tau x)\\phi` という
  **定数項**が加わる。本モジュールは :attr:`Branch.shift_deg` を
  **無視している**（同梱の WSCC 9 母線には位相調整器がないため）。
  位相調整器のあるケースを扱うときは、注入に上の定数項を足してから
  この関数群を使うこと。B' と PTDF / LODF は定数項の影響を受けないので、
  変更が要るのは :func:`dc_powerflow` の右辺だけである。

熱容量の判定に使ってはいけない理由
----------------------------------
:attr:`Branch.rate_a` / :attr:`rate_b` は **皮相電力** :math:`|S|` の
制限であって、有効電力 :math:`P` の制限ではない。WSCC 9 母線の枝 4-5
では交流の :math:`|S| = 0.5614` に対し直流の :math:`P` は 0.38 前後で、
**47% ほど小さく出る**。直流潮流の値をそのまま定格と比べると、
過負荷を「健全」と誤判定する。直流は候補を絞り込むために使い、
判定は交流で行う。この住み分けが :func:`gridops.security.screen_n1` の
2 段構えの根拠である。

slack の取り方について
----------------------
:func:`ptdf` は slack 母線を位相の基準に選んで解く。PTDF の **1 列**は
「その母線に 1 p.u. 注入し、slack から 1 p.u. 引き抜いたときの枝潮流」
なので、slack を変えれば値が変わる。一方、

- **列の差** :math:`\\mathrm{PTDF}[:, i] - \\mathrm{PTDF}[:, j]`
  （母線 i から母線 j への 1 p.u. の送電）
- **LODF**

は slack の取り方に依存しない。前者は slack の寄与が引き算で消え、
後者は枝の両端の列の差だけで書けるからである。テストではこの
「変わること」と「変わらないこと」の両方を固定してある。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .case import Case
from .ybus import incidence_matrix, islands

__all__ = ["DCSolution", "susceptance_matrix", "dc_powerflow", "ptdf", "lodf"]

#: :meth:`DCSolution.loading` が受け付ける熱容量の属性名。
LIMIT_NAMES = ("rate_a", "rate_b")


# ======================================================================
# 内部ヘルパ
# ======================================================================
def _susceptances(case: Case) -> np.ndarray:
    """枝ごとの :math:`1/(\\tau x)` を枝の並び順で返す。

    直流近似では :math:`r` を捨てるので、枝の「アドミタンス」は
    リアクタンスの逆数だけになる。タップ比が :math:`\\tau^2` ではなく
    1 次で入るのは、交流の枝行列の **非対角**成分が
    :math:`-y_s / (\\tau e^{\\mp j\\phi})` だからである。:math:`|V|=1`,
    :math:`r=0` とすると枝潮流は

    .. math:: P_{ft} = \\frac{\\theta_f - \\theta_t - \\phi}{\\tau x}

    になり、:math:`\\tau^2` の付いた対角成分（純サセプタンス）は有効電力に
    寄与しない。``tap=1`` なら素の :math:`1/x` に戻る。

    Raises
    ------
    ValueError
        :math:`x = 0` の枝、またはタップ比が非正の枝があるとき。
    """
    values = np.empty(case.n_branch)
    for i, branch in enumerate(case.branches):
        if branch.tap <= 0:
            raise ValueError(
                f"枝 {branch.label} のタップ比が非正: tap={branch.tap}。"
                "直流潮流の B' が組めない。ケースファイルの tap を確認すること。"
            )
        if branch.x == 0.0:
            raise ValueError(
                f"枝 {branch.label} のリアクタンスがゼロ。直流潮流は 1/x を"
                "使うので発散する。母線を 1 つに統合するか、小さな x を"
                "与えること（Case.check() でも検出できる）。"
            )
        values[i] = 1.0 / (branch.tap * branch.x)
    return values


def _slack_index(case: Case, slack: int | None) -> int:
    """位相の基準にする母線の **添字**を決める。

    Parameters
    ----------
    slack:
        母線 **番号**（添字ではない）。``None`` ならケースの slack 母線。

    Raises
    ------
    ValueError
        ``slack=None`` でケースに slack 母線がないとき。
    KeyError
        指定した母線番号がケースにないとき（:meth:`Case.index_of`）。
    """
    if slack is None:
        slack_idx, _, _ = case.type_indices()
        if len(slack_idx) == 0:
            raise ValueError(
                f"ケース '{case.name}' に slack 母線がない。直流潮流は位相の"
                "基準を 1 つ決めないと解けない（B' は行和がゼロで特異）。"
                "母線種別を確認するか、slack=母線番号 を明示すること。"
            )
        return int(slack_idx[0])
    return case.index_of(int(slack))


def _require_connected(
    case: Case, removed_branches: Sequence[tuple[int, int]], *, context: str
) -> None:
    """系統が連結でなければ、島の中身を添えて日本語で止める。

    直流潮流が解けない原因の大半はソルバではなくトポロジーである。
    「特異行列」という numpy のメッセージのままにしておくと、学生は
    ソルバの設定を疑いに行ってしまう。
    """
    groups = islands(case, removed_branches=removed_branches)
    if len(groups) <= 1:
        return
    listing = " / ".join(str(group) for group in groups)
    raise ValueError(
        f"{context}: 枝 {sorted(map(tuple, removed_branches))} を開放すると"
        f"系統が {len(groups)} 個の島に分かれる: {listing}。"
        "島ごとに需給が独立するので、位相の基準が 1 つの直流潮流では"
        "解が定義できない。これは数値の破綻ではなくトポロジーの事実である。"
        "開放した枝が橋かどうかは gridops.ybus.bridges() で確かめられる。"
    )


def _outage_columns(
    case: Case, outages: Sequence[tuple[int, int]] | None
) -> np.ndarray:
    """:func:`lodf` で列を計算する枝の **通し番号**を返す。

    ``None`` なら全枝。指定があれば :meth:`Branch.key` で照合する。
    存在しない枝を黙って無視すると、NaN のままの列を「計算した」と
    思い込む事故になるので、ここで止める。

    Raises
    ------
    ValueError
        指定した枝がケースにないとき。
    """
    if outages is None:
        return np.arange(case.n_branch)

    wanted = _normalized_keys(outages)
    positions = {branch.key(): i for i, branch in enumerate(case.branches)}
    missing = [key for key in wanted if key not in positions]
    if missing:
        raise ValueError(
            f"outages に指定した枝 {missing} はケース '{case.name}' にない。"
            f"存在する枝は {sorted(positions)}。母線番号の順序は問わない。"
        )
    return np.array(sorted({positions[key] for key in wanted}), dtype=int)


def _normalized_keys(
    removed_branches: Sequence[tuple[int, int]]
) -> tuple[tuple[int, int], ...]:
    """開放指定を :meth:`Branch.key` と同じ正規形（昇順の組）に直す。

    存在検査は :mod:`gridops.ybus` 側で済んでいる前提で呼ぶこと。
    """
    return tuple(
        sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in removed_branches)
    )


# ======================================================================
# 直流潮流の解
# ======================================================================
@dataclass
class DCSolution:
    """直流潮流の解。

    Parameters
    ----------
    case:
        解いたケース。
    theta:
        母線位相 [rad]。並びは :attr:`Case.buses` の順で、slack 母線が
        ちょうどゼロになる。
    flows:
        枝の有効電力 [p.u.]。並びは :attr:`Case.branches` の順で、
        符号は **from_bus から to_bus へ流れる向きを正**とする。
        開放した枝の要素はゼロである。
    slack:
        位相の基準にした母線 **番号**（添字ではない）。
    removed_branches:
        開放した枝の :meth:`Branch.key`。解を見ただけでどの事故の結果か
        分かるように残してある（契約に無い追加フィールド）。

    Notes
    -----
    直流潮流に損失はない。枝の両端の有効電力は符号だけが逆になるので、
    ``flows`` は枝あたり 1 つの数で足りる（交流の
    :meth:`PowerFlowSolution.branch_flows` が両端の値を返すのと対照的
    である）。総発電は総需要にちょうど一致し、slack 母線の出力は
    「総需要 - 他機の出力」になる。
    """

    case: Case
    theta: np.ndarray
    flows: np.ndarray
    slack: int
    removed_branches: tuple[tuple[int, int], ...] = ()

    # ------------------------------------------------------------------
    @property
    def angle_deg(self) -> np.ndarray:
        """母線位相 [deg]。交流の解や参照解と並べるとき用（追加 API）。"""
        return np.degrees(self.theta)

    def flow_of(self, key: tuple[int, int]) -> float:
        """枝 1 本の有効電力 [p.u.] を母線番号の組で引く。

        Parameters
        ----------
        key:
            ``(母線, 母線)`` の組。順序は問わない（:meth:`Branch.key` と
            同じく識別にだけ使う）。

        Returns
        -------
        float
            **枝の from_bus -> to_bus の向きを正**とする有効電力。
            引数の順序を入れ替えても符号は変わらない。``flows`` の
            該当要素そのものであり、向きの規約を 1 か所に保つためである。

        Raises
        ------
        ValueError
            その枝がケースにないとき。
        """
        a, b = int(key[0]), int(key[1])
        target = (min(a, b), max(a, b))
        for i, branch in enumerate(self.case.branches):
            if branch.key() == target:
                return float(self.flows[i])
        raise ValueError(
            f"枝 {target} はケース '{self.case.name}' にない。"
            f"存在する枝は {sorted({b.key() for b in self.case.branches})}。"
        )

    def loading(self, limit: str = "rate_a") -> np.ndarray:
        """熱容量に対する負荷率（1.0 で定格）を枝順で返す。

        Parameters
        ----------
        limit:
            ``"rate_a"``（常時）または ``"rate_b"``（緊急時）。

        Returns
        -------
        numpy.ndarray
            :math:`|P_{\\ell}| / \\mathrm{rate}`。容量が ``inf`` の枝は
            ゼロになる。

        Raises
        ------
        ValueError
            ``limit`` が ``rate_a`` / ``rate_b`` 以外のとき。容量が
            非正の枝があるとき。

        Notes
        -----
        **この値を過負荷の判定に使ってはいけない。** 定格は皮相電力
        :math:`|S|` の制限であるのに対し、直流潮流が持っているのは
        有効電力 :math:`P` だけで、無効電力の分がまるごと抜けている。
        WSCC 9 母線の枝 4-5 では交流の :math:`|S|=0.5614` に対し直流の
        :math:`P` は 0.38 前後（負荷率で 56% 対 38%）であり、**直流は
        負荷率を系統的に過小評価する**。直流で絞り込み、交流で判定する
        こと（:func:`gridops.security.screen_n1`）。
        """
        if limit not in LIMIT_NAMES:
            raise ValueError(
                f"limit={limit!r} は使えない。{LIMIT_NAMES} のいずれかを指定すること"
                "（rate_a が常時許容容量、rate_b が緊急時許容容量）。"
            )
        from .case import validate_rating_attribute

        validate_rating_attribute(limit)
        rates = np.array([getattr(branch, limit) for branch in self.case.branches])
        if np.any(rates <= 0.0):
            bad = [
                branch.label
                for branch, rate in zip(self.case.branches, rates)
                if rate <= 0.0
            ]
            raise ValueError(
                f"枝 {bad} の {limit} が非正。負荷率が定義できない。"
                "熱容量を与えないときは inf のままにすること。"
            )
        return np.abs(self.flows) / rates


# ======================================================================
# B' と直流潮流
# ======================================================================
def susceptance_matrix(
    case: Case, *, removed_branches: Sequence[tuple[int, int]] = ()
) -> np.ndarray:
    """直流潮流のサセプタンス行列 :math:`B'` を組む。

    .. math::

        B' = A^{T} \\operatorname{diag}\\!\\left(\\frac{1}{\\tau_{\\ell}
             x_{\\ell}}\\right) A

    :math:`A` は :func:`gridops.ybus.incidence_matrix`。この形で書くと、
    「対角は自分につながる枝の :math:`1/x` の和、非対角は :math:`-1/x`」
    という手計算の規則が行列積 1 つに収まる。

    Parameters
    ----------
    case:
        系統ケース。
    removed_branches:
        開放する枝の ``(母線, 母線)`` の組。接続行列の該当行がゼロに
        なるだけなので、行列の大きさは変わらない。

    Returns
    -------
    numpy.ndarray
        ``(n_bus, n_bus)`` の実対称行列。

    Raises
    ------
    ValueError
        :math:`x=0` の枝やタップ比が非正の枝があるとき。開放指定の枝が
        ケースにないとき（:mod:`gridops.ybus` が投げる）。

    Notes
    -----
    :math:`B'` は **必ず対称**である。交流の Ybus は位相調整器があると
    非対称になるが、直流近似では位相調整角が右辺の定数項に移り、行列
    そのものには入らない（モジュール docstring 参照）。

    行和はゼロである（全母線を同じ位相にすれば潮流が流れない）。
    したがって :math:`B'` は特異であり、そのままでは解けない。slack の
    行と列を落として初めて正則になる。これは「位相の絶対値には意味が
    なく、差にだけ意味がある」という物理の行列としての現れである。

    抵抗と充電容量を落としたケースでは :math:`B' = -\\operatorname{Im}
    (Y_{bus})` に一致する（``tap=1`` のとき）。この一致は
    :func:`gridops.ybus.build_ybus` と本関数のどちらかが壊れたことを
    検出する独立な手掛かりになる。
    """
    A = incidence_matrix(case, removed_branches=removed_branches)
    b = _susceptances(case)
    return A.T @ (b[:, None] * A)


def dc_powerflow(
    case: Case,
    *,
    dispatch: Mapping[str, float] | None = None,
    removed_branches: Sequence[tuple[int, int]] = (),
    slack: int | None = None,
) -> DCSolution:
    """直流潮流を解く。

    slack の行と列を落とした :math:`B'` の連立 1 次方程式

    .. math:: B'_{rr}\\,\\theta_r = P_r, \\qquad \\theta_{slack} = 0

    を 1 回解くだけである。反復も収束判定もない。

    Parameters
    ----------
    case:
        系統ケース。
    dispatch:
        号機名から出力 [MW] への対応。``None`` なら参照解の母線単位の
        発電を使う。注入の組み立ては :meth:`Case.bus_injection` に
        任せる（符号の規約を 1 か所に閉じるため）。
    removed_branches:
        開放する枝。N-1 の事故後潮流を直接解くのに使う。
    slack:
        位相の基準にする母線 **番号**。``None`` ならケースの slack 母線。

    Returns
    -------
    DCSolution

    Raises
    ------
    ValueError
        開放によって系統が島に分かれるとき（枝が橋のとき）。ケースに
        slack 母線がないとき。:math:`B'_{rr}` が特異なとき。

    Notes
    -----
    **slack 母線に指定した注入は使われない。** slack の行を落とすので、
    その母線の :math:`P` は方程式に入らず、残り全部の帳尻として決まる。
    直流潮流には損失がないので、その値は「総需要 - 他機の出力」に
    ちょうど一致する。参照解の発電（総和が総需要より損失の分だけ多い）を
    そのまま渡しても、余った 0.046 p.u. は slack の出力から差し引かれる
    だけで、枝潮流には現れない。

    位相調整角 :attr:`Branch.shift_deg` は **無視している**（モジュール
    docstring の「捨てているもの」を参照）。
    """
    slack_index = _slack_index(case, slack)
    _require_connected(case, removed_branches, context="dc_powerflow")

    B = susceptance_matrix(case, removed_branches=removed_branches)
    p_injection, _ = case.bus_injection(dispatch)

    keep = np.delete(np.arange(case.n_bus), slack_index)
    theta = np.zeros(case.n_bus)
    try:
        theta[keep] = np.linalg.solve(B[np.ix_(keep, keep)], p_injection[keep])
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"直流潮流の B' が特異で解けない（slack = 母線 "
            f"{case.buses[slack_index].id}）。ソルバではなくデータを疑うこと: "
            "孤立した母線がないか（gridops.ybus.islands）、リアクタンスが"
            "ゼロの枝がないか（Case.check）を先に確かめる。"
        ) from exc

    A = incidence_matrix(case, removed_branches=removed_branches)
    flows = _susceptances(case) * (A @ theta)
    return DCSolution(
        case=case,
        theta=theta,
        flows=flows,
        slack=case.buses[slack_index].id,
        removed_branches=_normalized_keys(removed_branches),
    )


# ======================================================================
# 感度係数
# ======================================================================
def ptdf(
    case: Case,
    *,
    slack: int | None = None,
    removed_branches: Sequence[tuple[int, int]] = (),
) -> np.ndarray:
    """発電移動分布係数（PTDF）を作る。

    :math:`\\mathrm{PTDF}[\\ell, i]` は「母線 :math:`i` に 1 p.u. 注入し、
    slack 母線から 1 p.u. 引き抜いたときの枝 :math:`\\ell` の潮流」で
    ある。直流潮流が線形なので、これは注入に対する偏微分そのもの
    :math:`\\partial f_{\\ell} / \\partial P_i` でもある。

    .. math::

        \\mathrm{PTDF} = \\operatorname{diag}\\!\\left(\\frac{1}{\\tau x}
        \\right) A\\, X, \\qquad
        X = \\begin{pmatrix} 0 & 0 \\\\ 0 & (B'_{rr})^{-1}\\end{pmatrix}

    :math:`X` は slack の行と列にゼロを詰め戻した行列である。この
    「落として解いて詰め戻す」が直流潮流のすべてで、PTDF は
    :func:`dc_powerflow` を単位ベクトルの注入で :math:`n_{bus}` 回
    解いたものと厳密に同じになる（テストで数値微分と突き合わせてある）。

    Parameters
    ----------
    case:
        系統ケース。
    slack:
        位相の基準にする母線 **番号**。``None`` ならケースの slack 母線。
    removed_branches:
        開放した状態の PTDF が欲しいときに指定する。開放した枝の行は
        ゼロになる。

    Returns
    -------
    numpy.ndarray
        ``(n_branch, n_bus)`` の実行列。行は :attr:`Case.branches` の順、
        列は :attr:`Case.buses` の順。

    Raises
    ------
    ValueError
        系統が島に分かれているとき。:math:`B'_{rr}` が特異なとき。

    Notes
    -----
    **単一の列は slack の取り方に依存する。** slack 母線の列は恒等的に
    ゼロであり（自分に注入して自分から引き抜けば何も動かない）、他の
    列の値も slack を変えれば変わる。「母線 5 の PTDF」という言い方は
    slack を言わなければ意味を持たない。

    一方、**列の差** :math:`\\mathrm{PTDF}[:, i] - \\mathrm{PTDF}[:, j]`
    は slack に依存しない。これは「母線 :math:`i` から母線 :math:`j` へ
    1 p.u. 送る」という slack を含まない取引に対応するからである。
    :func:`lodf` が slack に依存しないのも、枝の両端の列の差だけで
    書けるからである。

    行の和 :math:`\\sum_i \\mathrm{PTDF}[\\ell, i]` には意味がない。
    意味があるのは列の差である。
    """
    slack_index = _slack_index(case, slack)
    _require_connected(case, removed_branches, context="ptdf")

    B = susceptance_matrix(case, removed_branches=removed_branches)
    A = incidence_matrix(case, removed_branches=removed_branches)
    b = _susceptances(case)

    keep = np.delete(np.arange(case.n_bus), slack_index)
    X = np.zeros((case.n_bus, case.n_bus))
    try:
        X[np.ix_(keep, keep)] = np.linalg.inv(B[np.ix_(keep, keep)])
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"PTDF の B' が特異で反転できない（slack = 母線 "
            f"{case.buses[slack_index].id}）。孤立した母線か、リアクタンスが"
            "ゼロの枝がないかを先に確かめること"
            "（gridops.ybus.islands / Case.check）。"
        ) from exc

    return (b[:, None] * A) @ X


def lodf(
    case: Case,
    *,
    tolerance: float = 1e-8,
    outages: Sequence[tuple[int, int]] | None = None,
) -> np.ndarray:
    """線路開放分布係数（LODF）を作る。

    :math:`\\mathrm{LODF}[\\ell, k]` は「枝 :math:`k` の開放による枝
    :math:`\\ell` の潮流変化 / 枝 :math:`k` の事故前潮流」である。
    事故後潮流はこれで

    .. math:: f' = f + \\mathrm{LODF}[:, k]\\, f_k

    と、行列とベクトルの積 1 回で出る。潮流を解き直さずに N-1 を
    掃けるのが直流近似の最大の実利である。

    補償定理
    --------
    枝 :math:`k`（母線 :math:`m`-:math:`n`）の開放を、**枝を残したまま**
    両端に置いた等価注入対 :math:`+\\Delta` / :math:`-\\Delta` で
    置き換える。枝 :math:`k` の潮流がちょうどゼロになる :math:`\\Delta` は

    .. math::

        \\Delta = \\frac{f_k}{1 - \\Psi[k, k]}, \\qquad
        \\Psi[\\ell, k] = \\mathrm{PTDF}[\\ell, m] - \\mathrm{PTDF}[\\ell, n]

    で、他の枝の変化は :math:`\\Psi[\\ell, k]\\,\\Delta` である。よって

    .. math::

        \\mathrm{LODF}[\\ell, k] = \\frac{\\Psi[\\ell, k]}{1 - \\Psi[k, k]}
        \\quad (\\ell \\ne k), \\qquad \\mathrm{LODF}[k, k] = -1

    対角が :math:`-1` なのは、開放した枝の潮流が事故前の値をちょうど
    打ち消してゼロになるからである。

    Parameters
    ----------
    case:
        系統ケース。
    tolerance:
        分母 :math:`1 - \\Psi[k,k]` をゼロとみなす閾値。
    outages:
        **列を計算する枝**を ``(母線, 母線)`` の組で限定する（契約に無い
        追加引数）。``None`` なら全枝を計算し、橋が 1 本でもあれば
        例外になる。橋を含む系統（WSCC 9 母線がまさにそれである）で
        N-1 のスクリーニングをするには、橋を除いた候補だけを渡すこと:

        .. code-block:: python

            L = lodf(case, outages=case.contingencies)   # 橋は入っていない
            post = flows + L[:, k] * flows[k]

        計算しなかった列は ``numpy.nan`` で埋める。**ゼロで埋めない**の
        は、うっかり使ったときに「事故の影響がゼロ」という*もっともらしい*
        誤った答えが出るのを防ぐためである。NaN なら結果全体が NaN に
        なって直ちに気づける。行と列の並びは常に :attr:`Case.branches`
        と一致し、枝の番号がずれることはない。

    Returns
    -------
    numpy.ndarray
        ``(n_branch, n_branch)`` の実行列。行が観測する枝、列が開放する枝。

    Raises
    ------
    ValueError
        計算対象の枝の分母が ``tolerance`` を下回るとき。``outages`` に
        ケースにない枝を指定したとき。

    Notes
    -----
    **分母がゼロになるのは数値の破綻ではない。** :math:`\\Psi[k,k] = 1`
    は「枝 :math:`k` の両端の間で送る電力が全部その枝を通る」、すなわち
    **その枝が唯一の連絡路（橋）である**ことを意味する。橋を開放すれば
    系統は 2 つの島に分かれ、事故後潮流という概念自体が成り立たない。
    どんなに丁寧に数値計算をしてもこの値は出ないので、tolerance を
    緩めて先へ進んではいけない。WSCC 9 母線では変圧器 3 本
    ``(1,4), (2,7), (3,9)`` がこれに当たり、ケースの N-1 候補から
    外してある理由もこれである（:func:`gridops.ybus.bridges` が同じ
    3 本を返す）。

    LODF は **slack の取り方に依存しない**。枝の両端に対する PTDF の
    列の差だけで書けるからである。:func:`ptdf` の単一列が slack で
    変わるのと対照的で、この違いはテストで両方向とも固定してある。

    事故前潮流 :math:`f_k` が LODF の定義に入っているように見えるが、
    **LODF 自体は潮流に依存しない**（トポロジーとリアクタンスだけで
    決まる）。だからこそ需要が変わっても作り直さずに使い回せる。
    """
    A = incidence_matrix(case)
    psi = ptdf(case) @ A.T
    denominator = 1.0 - np.diag(psi)

    columns = _outage_columns(case, outages)
    singular = np.abs(denominator[columns]) < tolerance
    if np.any(singular):
        labels = ", ".join(
            str(case.branches[k].key()) for k in np.asarray(columns)[singular]
        )
        raise ValueError(
            f"枝 {labels} の LODF の分母 1 - PTDF[k,(m,n)] が "
            f"{tolerance} を下回る。これは数値の破綻ではなく、**その枝が橋で"
            "あり、開放すると系統が 2 つの島に分かれる**という位相の事実で"
            "ある（分母がゼロとは、両端の間で送る電力が 100% その枝を通る"
            "ということ）。tolerance を緩めても意味のある値は出ない。"
            "どの枝が橋かは gridops.ybus.bridges() で確かめられる。"
            "N-1 の候補からその枝を外すか（lodf(case, outages=...) で列を"
            "限定できる）、島ごとに分けて解析すること。"
        )

    matrix = np.full((case.n_branch, case.n_branch), np.nan)
    matrix[:, columns] = psi[:, columns] / denominator[np.newaxis, columns]
    matrix[columns, columns] = -1.0
    return matrix
