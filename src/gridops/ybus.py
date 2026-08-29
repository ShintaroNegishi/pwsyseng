"""母線アドミタンス行列と系統トポロジー。

Ybus の組み立てを **足し込みのループ 1 つ**に切り詰めるのが本モジュールの
設計意図である。タップ比と位相調整角の扱いは
:meth:`gridops.case.Branch.primitive` に閉じてあるので、ここでやることは
2x2 の枝行列を母線番号の位置に足し、シャントを対角に足すだけになる。
変圧器の式をあちこちに散らかさないことが、符号やタップの置き場所を
めぐるバグを構造的に潰す唯一の方法である。

符号と向きの規約
----------------
:math:`\\bar I = Y \\bar V` の向きで定義する。すなわち :math:`\\bar I` は
**母線への注入電流**であり、注入が正である（:class:`gridops.case.Case` の
規約と同じ）。母線の複素電力は

.. math::

    \\bar S_i = \\bar V_i \\left(\\sum_j Y_{ij} \\bar V_j\\right)^{*}

で得られ、発電機母線では発電量、負荷母線では負荷の符号を反転した値に
なる。この検算は :func:`gridops.ybus.build_ybus` の正しさを外から確かめる
一番安い手段である。

対称性についての注意
--------------------
``shift_deg != 0`` の枝があると :math:`Y_{ft} \\ne Y_{tf}` となり、
**Ybus は非対称になる**。タップ比だけなら対称のままである。位相調整器は
潮流を能動的に押し込む装置であり、その非相反性が行列の非対称として
現れる。``Y == Y.T`` を仮定したコードやテストを書いてはならない。

トポロジーを扱う理由
--------------------
:func:`bridges` と :func:`islands` を Ybus と同じ場所に置いてあるのは、
両者が同じ「枝の接続関係」から出るからであり、かつ第 09 回の
セキュリティ解析で必ず必要になるからである。橋（開放すると系統が
2 つの島に分かれる枝）を開放したケースの Ybus は特異になり、直流でも
LODF の分母 :math:`1 - \\mathrm{PTDF}[k,(m,n)]` が機械精度でゼロになる。
これは数値の破綻ではなく **位相の事実**であって、WSCC 9 母線では
変圧器 3 本 ``(1,4), (2,7), (3,9)`` がそれに当たる。N-1 の候補から
この 3 本が外れている理由を、データではなくアルゴリズムで説明できる
ようにしておくのが狙いである。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .case import Case

__all__ = ["build_ybus", "incidence_matrix", "bridges", "islands"]


# ======================================================================
# 内部ヘルパ
# ======================================================================
def _removed_keys(
    case: Case, removed_branches: Iterable[tuple[int, int]]
) -> set[tuple[int, int]]:
    """開放指定を :meth:`Branch.key` と同じ正規形に直し、存在を検査する。

    存在しない枝の開放を黙って無視すると「開放したつもりで開放されて
    いない」結果が出る。N-1 の解析では最も気づきにくい種類の誤りなので、
    ここで止める。

    Raises
    ------
    ValueError
        指定した枝がケースにないとき。要素が ``(母線, 母線)`` の組として
        読めないとき。
    """
    present = {branch.key() for branch in case.branches}
    keys: set[tuple[int, int]] = set()
    for item in removed_branches:
        try:
            pair = tuple(item)
            a, b = int(pair[0]), int(pair[1])
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                f"removed_branches の要素 {item!r} を (母線, 母線) の組として"
                "読めない。枝 1 本だけを開放するときも [(4, 5)] のように"
                "**組のリスト**で渡すこと。(4, 5) と書くと母線番号 2 つに"
                "分解されてしまう。"
            ) from None
        if len(pair) != 2:
            raise ValueError(
                f"removed_branches の要素 {item!r} の長さが 2 でない。"
                "枝は (母線, 母線) の組で指定すること。"
            )
        key = (min(a, b), max(a, b))
        if key not in present:
            raise ValueError(
                f"開放しようとした枝 {key} はケース '{case.name}' にない。"
                f"存在する枝は {sorted(present)}。"
                "母線番号の順序は問わない（(4, 5) と (5, 4) は同じ枝）ので、"
                "番号そのものか、既に開放済みでないかを確認すること。"
            )
        keys.add(key)
    return keys


def _adjacency(
    case: Case, removed: set[tuple[int, int]] | None = None
) -> list[list[tuple[int, int]]]:
    """隣接リスト ``[(隣の母線の添字, 枝の通し番号), ...]`` を作る。

    隣の母線ではなく **枝の通し番号**を持たせるのが要点である。橋の
    判定で「入ってきた枝そのもの」を除くとき、母線番号で除くと多重枝
    （同じ母線対を結ぶ 2 回線）のもう 1 本まで除いてしまい、多重枝を
    誤って橋と判定する。
    """
    removed = removed or set()
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(case.n_bus)]
    for edge, branch in enumerate(case.branches):
        if branch.key() in removed:
            continue
        i = case.index_of(branch.from_bus)
        j = case.index_of(branch.to_bus)
        if i == j:
            continue
        adjacency[i].append((j, edge))
        adjacency[j].append((i, edge))
    return adjacency


# ======================================================================
# 母線アドミタンス行列
# ======================================================================
def build_ybus(
    case: Case,
    *,
    removed_branches: Sequence[tuple[int, int]] = (),
    include_shunts: bool = True,
) -> np.ndarray:
    """母線アドミタンス行列 :math:`Y` を組む。

    枝ごとの 2x2 行列 :meth:`gridops.case.Branch.primitive` を母線番号の
    位置に足し込むだけである。タップ比と位相調整角の式は
    :class:`~gridops.case.Branch` 側にあり、ここには現れない。

    .. math::

        Y[f, f] \\mathrel{+}= Y_{ff}, \\quad
        Y[f, t] \\mathrel{+}= Y_{ft}, \\quad
        Y[t, f] \\mathrel{+}= Y_{tf}, \\quad
        Y[t, t] \\mathrel{+}= Y_{tt}

    Parameters
    ----------
    case:
        系統ケース。
    removed_branches:
        開放する枝を ``(母線, 母線)`` の組で指定する。順序は問わない。
        N-1 の評価で線路を切り離すのに使う。識別は
        :meth:`Branch.key` 単位なので、**多重回線は 1 本だけを開放
        できず、同じ母線対の枝がまとめて外れる**
        （:meth:`Case.without_branch` と同じ規約）。
    include_shunts:
        母線シャント :attr:`Bus.gs` / :attr:`Bus.bs` を対角に足すか。
        ``False`` にすると **枝だけの行列**になり、潮流解との整合性の
        検算（:math:`\\bar S = \\bar V (Y \\bar V)^{*}`）や、
        ``genstab`` の ``MultiMachineNetwork.ybus(include_loads=False)``
        との突き合わせに使える。

    Returns
    -------
    numpy.ndarray
        ``(n_bus, n_bus)`` の複素行列。行と列の順は :attr:`Case.buses`
        の順であり、母線番号の順ではない（:meth:`Case.index_of` を通す）。

    Raises
    ------
    ValueError
        自己ループの枝があるとき、開放指定の枝がケースにないとき、
        タップ比が非正のとき（:meth:`Branch.primitive` が投げる）。

    Notes
    -----
    ``tap=1``, ``shift_deg=0`` のときは素の :math:`\\pi` 型に縮退し、
    ``genstab.multimachine.MultiMachineNetwork.ybus()`` と 1e-14 以内で
    一致する。2 つのパッケージが同じ系統を別々に組み立てているので、
    この一致はどちらかが壊れたことを検出する砦になっている。

    ``shift_deg != 0`` の枝があると :math:`Y_{ft} \\ne Y_{tf}` となり、
    **戻り値は非対称になる**。対称性を仮定してはならない。

    行和は、シャントと線路の充電容量を除けばゼロになる。すなわち
    ``build_ybus(case, include_shunts=False).sum(axis=1)`` は各母線に
    つながる枝の :math:`jb/2` の合計に等しい（``tap=1`` の場合）。
    全母線の電圧が等しいとき電流が流れないという当たり前の事実の
    行列表現であり、組み立ての誤りを見つけるのに有効である。
    """
    removed = _removed_keys(case, removed_branches)
    n = case.n_bus
    Y = np.zeros((n, n), dtype=complex)

    for branch in case.branches:
        if branch.key() in removed:
            continue
        f = case.index_of(branch.from_bus)
        t = case.index_of(branch.to_bus)
        if f == t:
            raise ValueError(
                f"枝 {branch.label} が自己ループになっている。"
                "同じ母線を両端に持つ枝は Ybus に足し込めない。"
                "Case.check() で検出できるので先に通すこと。"
            )
        index = np.array([f, t])
        Y[np.ix_(index, index)] += branch.primitive()

    if include_shunts:
        for i, bus in enumerate(case.buses):
            Y[i, i] += complex(bus.gs, bus.bs)
    return Y


def incidence_matrix(
    case: Case, *, removed_branches: Sequence[tuple[int, int]] = ()
) -> np.ndarray:
    """枝-母線接続行列 :math:`A` を作る。

    行が枝、列が母線で、from 側に ``+1``、to 側に ``-1`` を置く。枝の
    両端の位相差は :math:`A \\theta`、母線の注入は :math:`A^{T} f` で
    書けるようになり、直流潮流の :math:`B' = A^{T} \\mathrm{diag}(1/x) A`
    が 1 行で出る。

    Parameters
    ----------
    case:
        系統ケース。
    removed_branches:
        開放する枝。

    Returns
    -------
    numpy.ndarray
        ``(n_branch, n_bus)`` の実行列。

    Notes
    -----
    **開放した枝の行は削除せず、ゼロ行として残す。** 枝の並びが
    :attr:`Case.branches` と常に一致するので、潮流ベクトルや LODF の
    行番号が事故の有無でずれない。行を削る実装にすると、N-1 の結果を
    枝ごとに並べるところで添字がずれる事故が起きる。

    ``+1`` / ``-1`` の向きは :attr:`Branch.from_bus` / :attr:`Branch.to_bus`
    の並びで決まるだけで、潮流の向きの物理的な意味はない。潮流の符号は
    「from から to へ流れる向きを正とする」という規約に過ぎない。
    """
    removed = _removed_keys(case, removed_branches)
    A = np.zeros((case.n_branch, case.n_bus))
    for row, branch in enumerate(case.branches):
        if branch.key() in removed:
            continue
        A[row, case.index_of(branch.from_bus)] += 1.0
        A[row, case.index_of(branch.to_bus)] -= 1.0
    return A


# ======================================================================
# トポロジー
# ======================================================================
def bridges(case: Case) -> list[tuple[int, int]]:
    """開放すると系統が分離する枝（橋）を返す。

    深さ優先探索の low-link で求める（Tarjan の橋検出）。``scipy`` を
    使わないのは、20 行ほどで書けるものをブラックボックスにしないため
    である。探索木の枝 :math:`(u, v)` について

    .. math:: \\mathrm{low}[v] > \\mathrm{disc}[u]

    が成り立つとき、:math:`v` 以下の部分木から :math:`u` を飛び越えて
    戻る道が 1 本もない、すなわちその枝が唯一の連絡路である。

    Returns
    -------
    list[tuple[int, int]]
        橋の :meth:`Branch.key` を母線番号の昇順に並べたもの。

    Notes
    -----
    WSCC 9 母線では変圧器 3 本 ``[(1, 4), (2, 7), (3, 9)]`` がちょうど
    返る。**この 3 本は N-1 の候補にしてはならない。** 開放すると
    発電機母線が島になり、交流潮流は解を持たず、直流でも LODF の分母
    :math:`1 - \\mathrm{PTDF}[k,(m,n)]` がゼロになるからである
    （:func:`gridops.dc.lodf` はここを検出して日本語で案内する）。

    多重枝（同じ母線対を結ぶ 2 回線）は橋ではない。1 本開放しても
    もう 1 本が残るからである。判定で「入ってきた枝」を母線番号ではなく
    **枝の通し番号**で除いているのはこのためである。ここで
    :func:`islands` の ``removed_branches`` とは粒度が違うことに注意する。
    橋の判定は **回線 1 本ごと**だが、開放の指定は :meth:`Branch.key`
    単位なので、多重回線に ``(1, 4)`` を指定すると 2 本ともまとめて
    外れて島ができる。多重回線を個別に扱いたい場合は、その枝だけを
    除いたケースを :func:`dataclasses.replace` で作ること。

    再帰ではなく明示スタックで書いてあるのは、母線数が Python の再帰
    上限（既定 1000）を超える系統でも動くようにするためである。
    """
    adjacency = _adjacency(case)
    edge_key = [branch.key() for branch in case.branches]

    n = case.n_bus
    disc = [-1] * n          # 発見時刻。-1 は未訪問
    low = [0] * n            # 部分木から戻れる最も古い発見時刻
    timer = 0
    found: list[tuple[int, int]] = []

    for root in range(n):
        if disc[root] >= 0:
            continue
        disc[root] = low[root] = timer
        timer += 1
        # スタックの要素は (母線, 入ってきた枝, 未処理の隣接の反復子)
        stack: list[tuple[int, int, object]] = [
            (root, -1, iter(adjacency[root]))
        ]
        while stack:
            node, in_edge, neighbours = stack[-1]
            descended = False
            for neighbour, edge in neighbours:  # type: ignore[misc]
                if edge == in_edge:
                    continue                     # 来た枝は戻り道に数えない
                if disc[neighbour] < 0:
                    disc[neighbour] = low[neighbour] = timer
                    timer += 1
                    stack.append((neighbour, edge, iter(adjacency[neighbour])))
                    descended = True
                    break
                low[node] = min(low[node], disc[neighbour])  # 後退枝
            if descended:
                continue
            stack.pop()
            if stack:
                parent = stack[-1][0]
                low[parent] = min(low[parent], low[node])
                if low[node] > disc[parent]:
                    found.append(edge_key[in_edge])
    return sorted(found)


def islands(
    case: Case, *, removed_branches: Sequence[tuple[int, int]] = ()
) -> list[list[int]]:
    """連結成分ごとの母線番号を返す。要素が 1 個なら系統は連結である。

    Parameters
    ----------
    case:
        系統ケース。
    removed_branches:
        開放する枝。橋を 1 本開放すると戻り値は 2 個になる。

    Returns
    -------
    list[list[int]]
        島ごとの母線番号（各島は昇順、島どうしは最小の母線番号の順）。

    Notes
    -----
    潮流計算の前にこれを見る癖をつけること。島に分かれた系統の Ybus は
    特異であり、Newton 法は「収束しない」という形でしか異常を教えて
    くれない。ソルバの設定を疑う前にトポロジーを疑うのが順序である。
    島ごとに slack 母線がなければ、その島には位相の基準も損失の受け皿も
    存在しない。
    """
    removed = _removed_keys(case, removed_branches)
    adjacency = _adjacency(case, removed)
    bus_ids = case.bus_ids

    seen = [False] * case.n_bus
    components: list[list[int]] = []
    for start in range(case.n_bus):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(bus_ids[node])
            for neighbour, _edge in adjacency[node]:
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        components.append(sorted(component))
    return sorted(components, key=lambda island: island[0])
