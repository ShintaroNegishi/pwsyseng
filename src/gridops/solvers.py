"""線形計画・混合整数計画のソルバ層（PuLP + CBC）。

**本パッケージで ``pulp`` を import してよいのはこのモジュールだけである。**
経済負荷配分・直流最適潮流・起動停止計画はいずれも数理計画を解くが、
それぞれが :mod:`pulp` を直接叩くと、次の 3 つが各モジュールに散らばる。

1. ソルバをどう見つけるか（環境によって使える CBC が違う）
2. 実行不可能だったときに何を疑うか（診断の文言）
3. 双対をどの符号で受け取るか（限界費用の符号）

どれも「電力系統の話」ではなく「PuLP の話」である。授業で読ませたいのは
前者なので、後者をこの 1 枚に閉じ込めた。下流のモジュールは
:func:`solve` の返す :class:`Solution` だけを見ればよい。

商用ソルバ（Gurobi など）には**一切触れない**。``gurobipy`` は import した
だけでライセンス確認が走ることがあり、学生の環境で「先生の画面では動くのに」
が起きる。再現性を優先し、無償の CBC のみを対象にする。

双対の符号の規約
----------------
本パッケージでは、需給バランス制約を **右辺に需要を正の符号で置く**向き、
すなわち

.. math::

    \\sum_i p_i = D

の形（PuLP では ``problem += lpSum(p) == demand, "balance"``）で書く。
このとき :attr:`Solution.duals` の値は

.. math::

    \\pi = \\frac{\\partial (\\text{目的関数})}{\\partial D}

そのもの、すなわち **限界費用 [円/MWh] そのもの**になる（CBC で実測して
確認済み）。同じ問題を ``demand - lpSum(p) == 0`` と書くと同じ最適値に
対して :math:`\\pi` の**符号だけが反転する**。制約式を等価変形しただけで
価格の符号が変わるのだから、これは「ソルバのバグ」ではなく「規約が要る」
ということである。

:class:`Solution` は CBC が返した値を**そのまま**格納する。符号を揃える
責任は呼び出し側にあり、その約束が

    バランス制約は ``lpSum(...) == 需要`` の向きで書く

である。不等式制約の双対の符号も同様に測って決まる。最小化問題では

======================  ==========  ==================================
制約の向き              双対の符号  意味
======================  ==========  ==================================
``lpSum(p) == D``       正          需要が 1 増えたときの費用増（限界費用）
``lpSum(p) >= D``       正          同上
``f <= f_max``          **負**      容量が 1 増えたときの費用減
======================  ==========  ==================================

送電容量制約の双対（混雑料金）を「正の量」として扱いたい場所では、
**受け取ってから符号を反転する**こと。:func:`gridops.dispatch.dc_opf` が
そうしている。ここで反転してしまうと、上の表が壊れて規約が二重になる。

変数名の落とし穴
----------------
PuLP は変数名の ``-+[] >/`` を、制約名の ``-+[] `` を、それぞれ黙って
``_`` に置き換える。号機名 ``G1-1`` で作った変数は ``G1_1`` になり、
呼び出し側が組み立てた ``f"p_{unit.name}_{t}"`` では引けなくなる。
この事故を防ぐため、

* :func:`variable` で作った変数は**元の名前**で :attr:`Solution.values`
  に入る（元の名前を変数オブジェクトに覚えさせている）
* :attr:`Solution.values` と :attr:`Solution.duals` は、置換後の名前でも
  置換前の名前でも引ける辞書である（:func:`safe_name` で正規化して再検索
  する）

の二重の手当てをしてある。素の :class:`pulp.LpVariable` を使っても
``solution.values["p_G1-1_3"]`` が引ける、ということである。
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Mapping

import pulp

# ======================================================================
# 既定値
# ======================================================================
#: 求解の打ち切り時間 [s]。授業中に返ってこない計算を作らないための上限。
#: 起動停止計画（7 号機 x 24 時間）は CBC で 0.5 秒程度なので十分に広い。
SOLVER_TIME_LIMIT = 60.0

#: 混合整数計画の相対ギャップ。``1e-4`` は「最適値から 0.01% 以内」の意味。
#: 厳密最適を求めると分枝限定が長引くだけで、教材としての結論は変わらない。
SOLVER_GAP = 1e-4

#: 元の変数名を :class:`pulp.LpVariable` に覚えさせるための属性名。
_ORIGINAL_NAME = "_gridops_original_name"

#: :func:`safe_name` が受け付ける名前の種別。
_TRANSLATIONS = {
    "variable": pulp.LpElement.trans,          # "-+[] >/" を "_" にする
    "constraint": pulp.LpAffineExpression.trans,  # "-+[] " を "_" にする
    # 問題名で PuLP が咎めるのは空白だけだが、制約と同じ表を使っておけば
    # LP ファイルに書き出したときの読みやすさも揃う。
    "problem": pulp.LpAffineExpression.trans,
}


# ======================================================================
# 名前の正規化
# ======================================================================
def safe_name(name: str, *, kind: str = "variable") -> str:
    """PuLP が実際に使う名前へ正規化する。

    PuLP 自身が持つ変換表を引いているので、PuLP の版が変わって禁止文字が
    増えても追随する。自前で ``str.replace("-", "_")`` と書かないのは、
    変数と制約で禁止文字の集合が違う（変数は ``>`` と ``/`` も置換される
    が、制約は置換されない）ためである。

    Parameters
    ----------
    name:
        呼び出し側が付けた名前。
    kind:
        ``"variable"`` か ``"constraint"``。

    Returns
    -------
    str
        PuLP 内部での名前。

    Raises
    ------
    ValueError
        ``kind`` が未知のとき。

    Examples
    --------
    >>> safe_name("p_G1-1_3")
    'p_G1_1_3'
    """
    try:
        table = _TRANSLATIONS[kind]
    except KeyError:
        raise ValueError(
            f"kind は {sorted(_TRANSLATIONS)} のいずれかであること（渡された値: {kind!r}）"
        ) from None
    return str(name).translate(table)


class _NameLookup(dict):
    """置換前の名前でも置換後の名前でも引ける辞書。

    :attr:`Solution.values` と :attr:`Solution.duals` の型である。
    ``dict`` の部分型なので、``len`` も ``items()`` も普通の辞書として
    振る舞う（キーは 1 つずつしか持たない）。違うのは検索だけで、
    見つからなかったときに :func:`safe_name` で正規化して**もう一度だけ**
    探す。号機名 ``G1-1`` を含むキーで引けるようにするための仕掛けである。
    """

    __slots__ = ("_kind",)

    def __init__(self, mapping: Mapping[str, float] | None = None, *, kind: str = "variable"):
        super().__init__(mapping or {})
        self._kind = kind

    def _resolve(self, key):
        """正規化した名前が入っていればそれを返す。無ければ ``None``。"""
        if not isinstance(key, str):
            return None
        alternative = safe_name(key, kind=self._kind)
        if alternative != key and dict.__contains__(self, alternative):
            return alternative
        return None

    def __missing__(self, key):
        alternative = self._resolve(key)
        if alternative is not None:
            return dict.__getitem__(self, alternative)
        kind = "変数" if self._kind == "variable" else "制約"
        raise KeyError(
            f"{kind} '{key}' は解に含まれていない。"
            f"PuLP は名前の '-+[] ' を '_' に置き換えるので、"
            f"正規化した '{safe_name(key, kind=self._kind)}' でも探したが見つからなかった。"
            f"存在するのは {sorted(self)[:12]}"
        )

    def __contains__(self, key) -> bool:
        return dict.__contains__(self, key) or self._resolve(key) is not None

    def get(self, key, default=None):
        if dict.__contains__(self, key):
            return dict.__getitem__(self, key)
        alternative = self._resolve(key)
        return default if alternative is None else dict.__getitem__(self, alternative)


def variable(
    name: str,
    low: float | None = None,
    up: float | None = None,
    *,
    cat: str = "Continuous",
) -> pulp.LpVariable:
    """元の名前を保ったまま :class:`pulp.LpVariable` を作る。

    PuLP は名前の禁止文字を置換したうえ、名前が禁止文字で**始まる**ときだけ
    警告を出す（途中のハイフンは黙って置換される）。ここでは先に
    :func:`safe_name` を通してから渡すので警告は出ず、置換前の名前は
    変数オブジェクトに覚えさせておく。:func:`solve` はそれを読んで
    :attr:`Solution.values` のキーにする。

    Parameters
    ----------
    name:
        呼び出し側が付けたい名前（``"p_G1-1_3"`` のようにハイフンを含んでよい）。
    low, up:
        下限・上限。``None`` は無限大。
    cat:
        ``"Continuous"`` / ``"Integer"`` / ``"Binary"``。

    Returns
    -------
    pulp.LpVariable
    """
    var = pulp.LpVariable(safe_name(name), lowBound=low, upBound=up, cat=cat)
    setattr(var, _ORIGINAL_NAME, str(name))
    return var


_SENSES = {"min": pulp.LpMinimize, "max": pulp.LpMaximize}


def problem(name: str = "problem", *, sense: str = "min") -> pulp.LpProblem:
    """空の :class:`pulp.LpProblem` を作る。

    **このモジュールが ``pulp`` を import してよい唯一の場所である**という
    規約を、下流のモジュール（:mod:`gridops.dispatch` /
    :mod:`gridops.commitment` / :mod:`gridops.security`）が字面どおり守れる
    ようにするためのヘルパである。:func:`solve` は完成した ``LpProblem`` しか
    受け取らないので、これが無いと下流は ``pulp`` を直接 import するか
    ``solvers.pulp`` を借りるかの二択になってしまう。

    Parameters
    ----------
    name:
        問題名。:func:`safe_name` を通すので空白やハイフンを含んでよい。
    sense:
        ``"min"``（最小化、既定）または ``"max"``（最大化）。

    Returns
    -------
    pulp.LpProblem

    Raises
    ------
    ValueError
        ``sense`` が ``"min"`` / ``"max"`` のどちらでもないとき。

    Examples
    --------
    >>> prob = problem("toy")
    >>> x = variable("x", 0.0, 10.0)
    >>> prob += 3.0 * x, "cost"
    >>> prob += x >= 4.0, "floor"
    >>> result = solve(prob, context="toy")
    >>> round(result.objective, 6), round(result.values["x"], 6)
    (12.0, 4.0)
    """
    try:
        lp_sense = _SENSES[str(sense).lower()]
    except KeyError:
        raise ValueError(
            f"sense は 'min' か 'max' のどちらかである（受け取った値: {sense!r}）。"
        ) from None
    return pulp.LpProblem(safe_name(name, kind="problem"), lp_sense)


def lp_sum(terms):
    """線形式の総和を取る（:func:`pulp.lpSum` の薄い包み）。

    素の :func:`sum` を使うと ``0 + 式`` の連鎖で式が深くなり、変数が
    数千個ある起動停止計画では目に見えて遅くなる。:func:`problem` と
    同じ理由で、下流が ``pulp`` を直接触らずに済ませるために公開している。

    Parameters
    ----------
    terms:
        線形式（またはその要素）の反復可能オブジェクト。空でもよい。

    Examples
    --------
    >>> xs = [variable(f"x{i}", 0.0, 1.0) for i in range(3)]
    >>> str(lp_sum(xs))
    'x0 + x1 + x2'
    """
    return pulp.lpSum(terms)


# ======================================================================
# 解
# ======================================================================
@dataclass(frozen=True)
class Solution:
    """数理計画の解。PuLP の :class:`~pulp.LpProblem` から値だけを抜いた器。

    下流のモジュールに :mod:`pulp` のオブジェクトを渡さないための境界で
    ある。``LpProblem`` を持ち回すと、解いたあとに制約を足すといった
    「解と問題がずれる」書き方ができてしまう。

    Parameters
    ----------
    status:
        ``"Optimal"`` / ``"Not Solved"`` など、PuLP の状態名。
    objective:
        最適値。目的関数を置かなかった問題（実行可能性だけを問う問題）では
        ``0.0``。
    values:
        変数名 -> 値。置換前・置換後どちらの名前でも引ける
        （:class:`_NameLookup` を参照）。
    duals:
        制約名 -> 双対 [目的関数の単位 / 制約右辺の単位]。符号はモジュール
        docstring の規約どおり、CBC が返した値そのままである。
        **混合整数計画では空の辞書になる。**
    seconds:
        求解に要した時間 [s]。:func:`time.perf_counter` で測る。
    """

    status: str
    objective: float
    values: dict[str, float] = field(default_factory=_NameLookup)
    duals: dict[str, float] = field(
        default_factory=lambda: _NameLookup(kind="constraint")
    )
    seconds: float = 0.0

    @property
    def is_optimal(self) -> bool:
        """最適解が得られたか。

        時間切れ（``"Not Solved"``）で実行可能解だけが返ったときは ``False``
        になる。起動停止計画で ``time_limit`` を短く切った場合に相当する。
        """
        return self.status == "Optimal"


# ======================================================================
# ソルバの発見
# ======================================================================
def _no_solver_error() -> RuntimeError:
    """CBC が 1 つも見つからないときの案内。"""
    return RuntimeError(
        "CBC ソルバが見つからない。PuLP 同梱の PULP_CBC_CMD も外部の COIN_CMD も"
        "使えなかった。次の順に確認すること。\n"
        "  1. conda activate pwsyseng を忘れていないか。"
        "（環境を有効にしないと cbc が PATH に載らない）\n"
        "  2. cbc が入っているか。`which cbc` で何も出なければ\n"
        "     conda install -c conda-forge coin-or-cbc\n"
        "  3. pip 版の PuLP なら CBC が同梱されている: pip install pulp\n"
        "  4. 使えるソルバの一覧は pulp.listSolvers(onlyAvailable=True) で見られる。\n"
        "  なお本パッケージは商用ソルバ（Gurobi 等）を参照しない。"
        "ライセンスの有無で授業の再現性が崩れるのを避けるためである。"
    )


def _time_limit_value(time_limit: float | None) -> float | None:
    """``timeLimit`` に渡す値を決める。非正または無限大なら制限なし。"""
    if time_limit is None:
        time_limit = SOLVER_TIME_LIMIT
    if time_limit <= 0 or math.isinf(time_limit):
        return None
    return float(time_limit)


def available_solver(
    *,
    msg: bool = False,
    time_limit: float | None = None,
    gap: float | None = None,
):
    """使える CBC を返す。``PULP_CBC_CMD`` -> ``COIN_CMD`` の順に探す。

    2 つを順に試すのは、CBC の入り方が環境で違うためである。pip 版の PuLP は
    CBC のバイナリを同梱しており ``PULP_CBC_CMD`` が使える。conda-forge 版の
    PuLP は同梱しないので、別途入れた ``coin-or-cbc`` を ``COIN_CMD`` が
    PATH から探す。研究室の ``genstab`` 環境は後者である。どちらでも同じ
    コードが動くようにしておかないと、「先生の画面では動くのに」が起きる。

    Parameters
    ----------
    msg:
        CBC のログを標準出力に流すか。既定は ``False``（notebook が
        ソルバのログで埋まるのを避ける）。分枝限定の様子を見せたい回では
        ``True`` にする。
    time_limit:
        打ち切り時間 [s]。``None`` なら :data:`SOLVER_TIME_LIMIT`。
        ``0`` 以下または ``math.inf`` なら無制限。
    gap:
        混合整数計画の相対ギャップ。``None`` なら :data:`SOLVER_GAP`。
        線形計画には影響しない。

    Returns
    -------
    pulp.LpSolver
        :meth:`~pulp.LpSolver.available` が真を返したソルバ。

    Raises
    ------
    RuntimeError
        どちらも使えないとき。conda 環境の有効化と
        ``conda install -c conda-forge coin-or-cbc`` を案内する。
    """
    options = {
        "msg": bool(msg),
        "timeLimit": _time_limit_value(time_limit),
        "gapRel": SOLVER_GAP if gap is None else float(gap),
    }

    for factory in (pulp.PULP_CBC_CMD, pulp.COIN_CMD):
        try:
            solver = factory(**options)
        except Exception:
            # 未インストールのソルバはコンストラクタで例外を投げることがある。
            continue
        try:
            found = bool(solver.available())
        except Exception:
            found = False
        if found:
            return solver

    raise _no_solver_error()


# ======================================================================
# 求解
# ======================================================================
#: 解が存在しないとみなす PuLP の状態名と、その日本語の診断。
_FATAL_STATUS = {
    "Infeasible": (
        "実行不可能 (Infeasible): 制約をすべて満たす点が存在しない。\n"
        "  ソルバの設定を触る前に、データとトポロジーの整合を先に疑うこと。\n"
        "  1. Case.check() を通す（母線番号の重複・島・容量と需要の矛盾を拾う）\n"
        "  2. 需給バランス: 需要が Σ p_max を超えていないか、Σ p_min を下回っていないか\n"
        "  3. 予備力率・最低運転停止時間・ランプ率が同時には満たせない組み合わせに"
        "なっていないか\n"
        "  4. 単位の取り違え（MW と p.u.、円/MWh と円/kWh）\n"
        "  授業を止めないためには、供給不足を許す緩和変数（allow_shortfall=True）を"
        "入れて\n"
        "  「何時に何 MW 足りないか」が返る形にするのが有効である。"
    ),
    "Unbounded": (
        "非有界 (Unbounded): 目的関数をいくらでも良くできてしまう。\n"
        "  1. 変数の上下限を付け忘れていないか\n"
        "  2. 費用係数の符号を取り違えていないか（最小化に負の費用を入れていないか）\n"
        "  3. 最大化と最小化を取り違えていないか"
    ),
    "Undefined": (
        "求解できなかった (Undefined): ソルバが状態を判定できなかった。\n"
        "  問題が空（変数も制約もない）でないか、係数に NaN や inf が"
        "紛れ込んでいないかを確認すること。"
    ),
    "Not Solved": (
        "求解できなかった (Not Solved): 実行可能解が 1 つも見つからないまま"
        "打ち切られた。\n"
        "  time_limit を伸ばすか、gap を緩めること。整数変数が多すぎる場合は"
        "問題の規模を落とすこと。"
    ),
}


def solve(
    problem: pulp.LpProblem,
    *,
    context: str = "",
    time_limit: float | None = None,
    gap: float | None = None,
    msg: bool = False,
) -> Solution:
    """:class:`pulp.LpProblem` を解いて、値と双対を取り出す。

    **状態を確かめてから値を読む。** ``pulp.value(problem.objective)`` は
    実行不可能な問題に対しても平然と数値を返す（CBC が探索の途中で持って
    いた値がそのまま残るため）。状態を見ずに読むと「解けているが間違って
    いる」結果が下流に流れる。ここで必ず堰き止める。

    Parameters
    ----------
    problem:
        解く問題。バランス制約は ``lpSum(...) == 需要`` の向きで書くこと
        （モジュール docstring の「双対の符号の規約」を参照）。
    context:
        例外メッセージに添える文脈。``"起動停止計画 (T=24, 予備力 10%)"``
        のように、どの計算で失敗したかが分かる文字列を渡す。
    time_limit, gap, msg:
        :func:`available_solver` に渡す。

    Returns
    -------
    Solution

    Raises
    ------
    ValueError
        実行不可能・非有界・判定不能のとき、および実行可能解を 1 つも
        得られずに打ち切られたとき。制約が厳しすぎるのか、データが壊れて
        いるのかを切り分けるための手順を日本語で並べる。
    TypeError
        ``problem`` が :class:`pulp.LpProblem` でないとき。

    Notes
    -----
    **混合整数計画では :attr:`Solution.duals` は空の辞書になる。**
    これは実装の手抜きではない。整数変数を含む問題の最適値は右辺について
    区分的に一定な階段関数であり、微分（＝双対）がそもそも存在しない。
    CBC は分枝限定の最後の緩和問題の双対を返してくるが、その値は探索の
    経路に依存し、限界費用としての意味を持たない。意味のある値が欲しい
    ときは、得られた整数解を固定して線形計画に落とし直してから双対を取る
    （:func:`gridops.commitment.marginal_prices` がそうしている）。
    """
    if not isinstance(problem, pulp.LpProblem):
        raise TypeError(
            f"solve() は pulp.LpProblem を受け取る（渡された型: {type(problem).__name__}）。"
        )

    solver = available_solver(msg=msg, time_limit=time_limit, gap=gap)

    start = time.perf_counter()
    try:
        problem.solve(solver)
    except pulp.PulpSolverError as exc:
        raise RuntimeError(
            f"CBC の呼び出しに失敗した{_suffix(context)}: {exc}\n"
            "  一時ファイルを書けるディレクトリにいるか、cbc が実行可能かを確認すること。"
        ) from exc
    seconds = time.perf_counter() - start

    status = pulp.LpStatus[problem.status]
    _check_status(problem, status, context)

    # 目的関数を置かない問題（実行可能性だけを問う問題）では objective が
    # None になり、pulp.value(None) は AttributeError を投げる。
    objective = None if problem.objective is None else pulp.value(problem.objective)
    solution = Solution(
        status=status,
        objective=0.0 if objective is None else float(objective),
        values=_collect_values(problem),
        duals=_collect_duals(problem),
        seconds=seconds,
    )
    return solution


def _suffix(context: str) -> str:
    """文脈があれば ``（...）`` の形に整える。"""
    return f"（{context}）" if context else ""


def _check_status(problem: pulp.LpProblem, status: str, context: str) -> None:
    """状態を見て、値を読んではいけない場合に例外を投げる。

    ``"Not Solved"`` だけは 2 通りある。時間切れでも実行可能解が手元に
    あるなら、それは「粗いが使える答え」であって捨てる理由がない。
    起動停止計画で ``time_limit`` を短く切ったときに相当するので、警告を
    出したうえで返す。解が 1 つも無いときだけ例外にする。
    """
    if status == "Optimal":
        return

    if status == "Not Solved" and _has_feasible_solution(problem):
        warnings.warn(
            f"time_limit で打ち切られた{_suffix(context)}。"
            "実行可能解は得られているが最適性は保証されない。"
            "Solution.is_optimal が False になる。",
            UserWarning,
            stacklevel=3,
        )
        return

    diagnosis = _FATAL_STATUS.get(status, f"想定していない状態 (status={status})。")
    raise ValueError(f"{diagnosis}\n  文脈: {context or '（指定なし）'}")


def _has_feasible_solution(problem: pulp.LpProblem) -> bool:
    """打ち切られた問題に実行可能解が残っているか。"""
    sol_status = getattr(problem, "sol_status", None)
    return sol_status in (
        getattr(pulp.constants, "LpSolutionOptimal", 1),
        getattr(pulp.constants, "LpSolutionIntegerFeasible", 2),
    )


def _collect_values(problem: pulp.LpProblem) -> _NameLookup:
    """変数の値を集める。キーは呼び出し側が付けた名前。

    :func:`variable` で作った変数は元の名前を覚えているのでそれを使い、
    素の :class:`pulp.LpVariable` は PuLP が置換したあとの名前になる。
    どちらでも :class:`_NameLookup` が正規化して引けるようにする。
    """
    values: dict[str, float] = {}
    for var in problem.variables():
        name = getattr(var, _ORIGINAL_NAME, var.name)
        value = var.value()
        # 目的関数にも制約にも係数 0 でしか現れない変数は、ソルバの解ファイル
        # に載らず None になることがある。その場合は 0 とみなす。
        values[name] = 0.0 if value is None else float(value)
    return _NameLookup(values, kind="variable")


def _collect_duals(problem: pulp.LpProblem) -> _NameLookup:
    """制約の双対を集める。混合整数計画では空になる。

    符号は CBC が返したままである（モジュール docstring の規約を参照）。
    ``pi`` が ``None`` の制約は**入れない**。0 で埋めると「双対が取れて
    いない」ことが「双対が 0」に化けて、混雑していないという誤読を生む。
    """
    if problem.isMIP():
        return _NameLookup(kind="constraint")

    duals = {
        name: float(constraint.pi)
        for name, constraint in problem.constraints.items()
        if getattr(constraint, "pi", None) is not None
    }
    return _NameLookup(duals, kind="constraint")


# ======================================================================
# 0-1 変数の読み取り
# ======================================================================
#: CBC の整数許容誤差。PuLP の ``roundSolution`` も同じ値を使う。
BINARY_TOLERANCE = 1e-5


def binary(value: float, *, tolerance: float = BINARY_TOLERANCE) -> int:
    """ソルバが返した 0-1 変数の値を ``0`` か ``1`` に丸める。

    分枝限定は整数性を厳密には満たさない。``0.9999999998`` や ``-3e-11``
    のような値が普通に返ってくるので、``int(value)`` と書くと
    ``0.9999999998`` が ``0`` になる（切り捨てのため）。丸めるのが正しい。

    ただし**黙って丸めない**。``0.5`` のような値が来たら、それは整数変数
    として宣言し忘れているか、連続変数を読んでいるかである。そのまま 0 か 1
    にすると、起動していない号機が出力を持つ矛盾した計画が出来上がる。

    Parameters
    ----------
    value:
        ソルバが返した値。
    tolerance:
        0 または 1 からのずれをどこまで許すか。既定は CBC の整数許容誤差
        :data:`BINARY_TOLERANCE`。

    Returns
    -------
    int
        ``0`` または ``1``。

    Raises
    ------
    ValueError
        ``value`` が ``None`` のとき、または 0 からも 1 からも ``tolerance``
        より離れているとき。

    Examples
    --------
    >>> binary(0.9999999998), binary(1e-9)
    (1, 0)
    """
    if value is None:
        raise ValueError(
            "0-1 変数の値が None である。求解前の変数を読んでいないか、"
            "目的関数にも制約にも現れない変数でないかを確認すること。"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"0-1 変数の値が有限でない: {value!r}")

    for target in (0, 1):
        if abs(number - target) <= tolerance:
            return target

    raise ValueError(
        f"0-1 変数のはずの値が {number!r} である（0 からも 1 からも "
        f"{tolerance:g} より離れている）。\n"
        "  1. 変数を cat='Binary' で宣言し忘れていないか\n"
        "  2. 連続変数（出力 p など）を誤って読んでいないか\n"
        "  3. 混合整数計画を線形緩和のまま解いていないか"
    )


__all__ = [
    "BINARY_TOLERANCE",
    "SOLVER_GAP",
    "SOLVER_TIME_LIMIT",
    "Solution",
    "available_solver",
    "binary",
    "lp_sum",
    "problem",
    "safe_name",
    "solve",
    "variable",
]
