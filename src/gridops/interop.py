"""gridops で解いた潮流解を、安定度の教材 genstab へ渡すための橋。

2 つのパッケージは対象とする時間スケールが違う。gridops は準定常
（潮流・経済負荷配分・起動停止計画）を扱い、genstab は事故直後の
1 秒未満を扱う。両者をつなぐ唯一の面が **事故前潮流解** である。

======================  ==============================================
gridops が渡すもの      genstab がそれを何に使うか
======================  ==============================================
母線の複素電圧 V        負荷の定インピーダンス変換 ``Y = conj(S)/|V|^2``
発電機の P, Q           内部起電力 ``E = V + j x'd I``
線路・変圧器の r, x, b  事故前・事故中・事故後の Ybus
号機の H, x'd           集約して古典モデルの 1 台にする
``stability`` 層        事故スケジュールと開放する枝
======================  ==============================================

genstab のソースには一切手を入れない。公開クラス
(:class:`~genstab.multimachine.MultiMachineNetwork`,
:class:`~genstab.multimachine.GeneratorData`,
:class:`~genstab.multimachine.Branch`,
:class:`~genstab.multimachine.Load`,
:class:`~genstab.multimachine.MultiMachineSystem`) をこちら側から
組み立てるだけである。``import genstab`` は関数の中で行うので、
genstab が入っていない環境でも gridops の第 01〜10 回は動く。

なぜ号機を集約するのか
----------------------
原典 (Anderson & Fouad) の WSCC 9 母線系統は 3 機だが、それでは
起動停止計画の意思決定がほぼ自明になってしまうので、gridops の
ケースは 7 号機に分けてある。安定度へ渡すときは、同じ発電所の
号機が同じ母線で同じ角度で振れる（同一プラント内の機器間動揺は
無視する）とみなして 1 台に戻す。

.. math::

    H = \\sum_i H_i, \\qquad
    \\frac{1}{x'_d} = \\sum_i \\frac{1}{x'_{d,i}}, \\qquad
    D = \\sum_i D_i, \\qquad
    P = \\sum_i P_i

一般には慣性定数を各機の容量基準のまま足してはならない。本ケースの
``Unit.h`` は 100 MVA 共通基準へ換算済みの慣性寄与分として定義しているため、
その寄与分を加算できる。過渡リアクタンスは並列につながる枝なので逆数の和になる。制動係数を **加算** するのは
H と揃えるためで、こうすると減衰比 :math:`D / (2\\sqrt{2HK})` が
1 台のときと変わらない。D だけ加算しないと、号機を分けただけで
減衰比が号機数分の 1 になってしまう。

この集約は可逆で、gridops の wscc9 は原典の 3 機に厳密に戻る。

===  ===============================  =================
機   gridops の号機                   集約後
===  ===============================  =================
G1   3 台 x (H=7.880, x'd=0.1824)     (23.64, 0.0608)
G2   2 台 x (H=3.200, x'd=0.2396)     ( 6.40, 0.1198)
G3   2 台 x (H=1.505, x'd=0.3626)     ( 3.01, 0.1813)
===  ===============================  =================

最も踏みやすい落とし穴
----------------------
**潮流解と発電機の P, Q は必ずセットで渡すこと。** 負荷の定インピー
ダンス変換は |V| に依存し、内部起電力は P, Q に依存する。片方だけ
更新すると事故前の運転点が平衡点でなくなり、事故が無くても角度が
動き出す。「解けているが間違っている」状態になり、波形を見ても
気づきにくい。本モジュールは P, Q を **潮流解そのものから**
（:math:`\\bar S = \\bar V (\\bar Y \\bar V)^{*}` に負荷を足し戻して）
求めることで、この食い違いを構造的に起こせないようにしている。
``dispatch`` を渡した場合は、その出力が潮流解と整合しているかを
:data:`DISPATCH_TOLERANCE` の幅で照合し、食い違えば例外にする。

制動係数の注意（実測）
----------------------
D = 0 の運転点では固有値が虚軸上に乗り、数値線形化の誤差で実部の
符号が定まらない。``genstab/eac.py`` は実部が 1e-6 を超えると
「定態不安定」と警告するので、運転点をわずかに動かすだけで警告が
出たり出なかったりする。gridops の同梱ケースが D = 2.0（原典は
D = 0）にしてあるのはこのためである。genstab の ``cases/wscc9.yaml``
と数値を突き合わせるときは ``damping=0.0`` を指定して条件を揃えること。
"""

from __future__ import annotations

import collections.abc
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .case import BusType, Case, Unit
from .ybus import build_ybus

if TYPE_CHECKING:  # 実行時には import しない（genstab が無くても動くため）
    from genstab.multimachine import MultiMachineSystem

__all__ = [
    "TAP_TOLERANCE",
    "SHIFT_TOLERANCE",
    "DISPATCH_TOLERANCE",
    "aggregate_plants",
    "to_genstab",
    "check_against_reference",
]

#: タップ比を 1 とみなす幅。genstab の Branch はタップを持たない。
TAP_TOLERANCE = 1e-9

#: 位相調整角を 0 とみなす幅 [deg]。同上。
SHIFT_TOLERANCE = 1e-9

#: ``dispatch`` と潮流解の発電が一致しているとみなす幅 [p.u.]。
#: 潮流計算の既定の収束判定 1e-10 p.u. より 4 桁緩い。ここに引っかかる
#: ようなら「解いたあとに出力だけ書き換えた」ことを疑うべきである。
DISPATCH_TOLERANCE = 1e-6

_INSTALL_HINT = (
    "安定度の教材 genstab が import できない。多機系統への変換にはこの"
    "パッケージが要る。\n"
    "  gridops と同じ親ディレクトリに genstab を置いたうえで\n"
    "      pip install -e .\n"
    "  をリポジトリ直下で実行すること（genstab も一緒に入る）。\n"
    "  genstab を使わない回（潮流・経済負荷配分・起動停止計画・"
    "アデカシー）は gridops だけで動く。"
)


# ======================================================================
# genstab の遅延 import
# ======================================================================
def _genstab() -> dict[str, Any]:
    """genstab の公開クラスをまとめて返す（関数の中で import する）。

    モジュールの先頭で import してしまうと、genstab が入っていない環境で
    ``import gridops.interop`` そのものが失敗し、潮流計算まで巻き添えに
    なる。教材としては「安定度の回だけが動かない」で止めたい。

    Raises
    ------
    ImportError
        genstab が入っていないとき。``pip install -e .`` のやり直しを案内する。
    """
    try:
        import genstab  # noqa: F401  最初にここで落として案内を出す

        from genstab.events import FaultSchedule
        from genstab.multimachine import (
            Branch,
            GeneratorData,
            Load,
            MultiMachineNetwork,
            MultiMachineSystem,
        )
        from genstab.units import SystemBase
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(f"{_INSTALL_HINT}\n  元の例外: {exc}") from exc

    return {
        "Branch": Branch,
        "FaultSchedule": FaultSchedule,
        "GeneratorData": GeneratorData,
        "Load": Load,
        "MultiMachineNetwork": MultiMachineNetwork,
        "MultiMachineSystem": MultiMachineSystem,
        "SystemBase": SystemBase,
    }


# ======================================================================
# 潮流解の読み取り
# ======================================================================
def _voltage_of(case: Case, solution: Any) -> np.ndarray:
    """いろいろな形の「潮流解」から複素電圧の配列を取り出す。

    受け付ける形は次の 4 つ。並びはすべて :attr:`Case.buses` の順である
    （母線番号の順ではない）。

    * :class:`gridops.powerflow.PowerFlowSolution`（``voltage`` を持つ）
    * :class:`gridops.case.ReferenceSolution`（同上）
    * 複素数の配列
    * ``{母線番号: 複素電圧}`` の写像
    * ``None``（ケースの参照解を使う）

    Raises
    ------
    ValueError
        長さが母線数と合わないとき、収束していない解を渡したとき、
        ``None`` を渡したのにケースが参照解を持たないとき。
    """
    if solution is None:
        case.require("solution")
        solution = case.reference

    if getattr(solution, "converged", True) is False:
        raise ValueError(
            "収束していない潮流解から安定度の系を組もうとしている。"
            "収束していない V から作った定インピーダンス負荷と内部起電力は"
            "物理的な意味を持たない。まず gridops.powerflow.solve が返す"
            "RuntimeError の本文（最大ミスマッチの母線・Q の可能範囲・島の有無）"
            "を読むこと。"
        )

    voltage = getattr(solution, "voltage", None)
    if voltage is None:
        if isinstance(solution, collections.abc.Mapping):
            try:
                voltage = np.array(
                    [complex(solution[bus.id]) for bus in case.buses], dtype=complex
                )
            except KeyError as exc:
                raise ValueError(
                    f"潮流解の写像に母線 {exc.args[0]} の電圧がない。"
                    f"必要なのは {case.bus_ids} のすべて。"
                ) from None
        else:
            voltage = np.asarray(solution, dtype=complex)

    voltage = np.asarray(voltage, dtype=complex)
    if voltage.shape != (case.n_bus,):
        raise ValueError(
            f"潮流解の長さ {voltage.shape} がケース '{case.name}' の母線数 "
            f"{case.n_bus} と合わない。並びは Case.buses の順（母線番号の順では"
            "ない）であることにも注意すること。"
        )
    return voltage


def _bus_generation(case: Case, voltage: np.ndarray) -> np.ndarray:
    """各母線の **発電** を複素電力 [p.u.] で返す（母線の並び順）。

    .. math::

        \\bar S^{gen}_i = \\bar V_i \\Bigl(\\sum_j Y_{ij} \\bar V_j\\Bigr)^{*}
                          + (P^d_i + j Q^d_i)

    第 1 項は母線への正味の注入、第 2 項で負荷を足し戻す。号機ごとの
    出力表 (``dispatch``) ではなく **潮流解そのもの** から求めるのが要点で、
    こうすると slack 母線の発電が損失を引き受けた後の値になり、内部起電力
    :math:`E = V + j x'_d I` が事故前の平衡点と厳密に整合する。
    """
    ybus = build_ybus(case)
    injection = voltage * np.conj(ybus @ voltage)
    load = np.array([complex(bus.pd, bus.qd) for bus in case.buses], dtype=complex)
    return injection + load


def _branch_losses(case: Case, voltage: np.ndarray) -> float:
    """枝損失の合計 :math:`\\sum \\mathrm{Re}(S_{ft} + S_{tf})` [p.u.]。"""
    total = 0.0
    for branch in case.branches:
        f = case.index_of(branch.from_bus)
        t = case.index_of(branch.to_bus)
        terminal = np.array([voltage[f], voltage[t]])
        power = terminal * np.conj(branch.primitive() @ terminal)
        total += float(np.real(power.sum()))
    return total


# ======================================================================
# 号機の集約
# ======================================================================
def _online(unit: Unit, dispatch: Mapping[str, float] | None) -> bool:
    """その号機が運転中か。``dispatch`` が無ければ全機運転中とみなす。

    判定を ``dispatch.get(name, 0.0) > 0`` に揃えてあるのは
    :meth:`Case.effective_bus_types` と同じ規約にするためである。
    停止した号機は慣性も過渡リアクタンスも系に寄与しない。
    """
    if dispatch is None:
        return True
    return float(dispatch.get(unit.name, 0.0)) > 0.0


def _plant_groups(
    case: Case, dispatch: Mapping[str, float] | None
) -> list[tuple[str, int, list[Unit]]]:
    """``(発電所名, 母線番号, 運転中の号機)`` を号機の登場順に返す。

    Raises
    ------
    ValueError
        同じ発電所の号機が別々の母線に置かれているとき。集約すると
        どちらの母線につなぐか決まらないので、データの誤りとして止める。
    """
    groups: list[tuple[str, int, list[Unit]]] = []
    for name, units in case.plants().items():
        buses = {unit.bus for unit in units}
        if len(buses) > 1:
            raise ValueError(
                f"発電所 '{name}' の号機が複数の母線に分かれている: {sorted(buses)}。"
                "集約すると 1 台をどの母線につなぐか決まらない。"
                "別の発電所として plant 名を分けること。"
            )
        running = [unit for unit in units if _online(unit, dispatch)]
        if running:
            groups.append((name, units[0].bus, running))
    return groups


def _require_stability_data(units: Sequence[Unit]) -> None:
    """慣性定数と過渡リアクタンスが揃っているか確かめる。"""
    for unit in units:
        if unit.h is None:
            raise ValueError(
                f"号機 {unit.name} に慣性定数 h が設定されていないため、"
                "安定度の系を組み立てられない。ケースファイルの units 層に "
                "h [s] と xd_prime [p.u.] を書くこと。"
                "潮流計算や起動停止計画だけならこの 2 つは不要である。"
            )
        if unit.xd_prime is None:
            raise ValueError(
                f"号機 {unit.name} に過渡リアクタンス xd_prime が設定されて"
                "いないため、安定度の系を組み立てられない。ケースファイルの "
                "units 層に h [s] と xd_prime [p.u.] を書くこと。"
                "潮流計算や起動停止計画だけならこの 2 つは不要である。"
            )
        if unit.xd_prime <= 0.0:
            raise ValueError(
                f"号機 {unit.name} の過渡リアクタンス xd_prime={unit.xd_prime} が"
                "非正である。並列合成の逆数和が発散するので集約できない。"
            )


def _check_dispatch(
    case: Case,
    generation: np.ndarray,
    dispatch: Mapping[str, float],
) -> None:
    """``dispatch`` と潮流解の発電が同じものを指しているか照合する。

    slack 母線は損失を引き受けるので出力が指定値と一致しない。そこだけ
    除いて、母線ごとに :math:`\\sum P^{dispatch}` と潮流解の発電を比べる。
    ここで止まる典型例は「潮流を解いたあとに出力表だけ書き換えた」で、
    そのまま進むと内部起電力が運転点と食い違ったまま計算が通ってしまう。
    """
    types = case.effective_bus_types(dispatch)
    for bus in case.buses:
        units = case.units_at(bus.id)
        if not units or types[bus.id] is BusType.SLACK:
            continue
        scheduled = case.to_pu(sum(float(dispatch.get(u.name, 0.0)) for u in units))
        solved = float(generation[case.index_of(bus.id)].real)
        if abs(scheduled - solved) > DISPATCH_TOLERANCE:
            raise ValueError(
                f"母線 {bus.id} で dispatch の合計 {case.to_mw(scheduled):.4f} MW と"
                f" 潮流解の発電 {case.to_mw(solved):.4f} MW が食い違っている"
                f"（差 {case.to_mw(abs(scheduled - solved)):.6f} MW）。\n"
                "潮流解と発電機の P, Q はセットで渡すこと。片方だけ更新すると"
                "内部起電力が運転点と整合せず、事故が無くても角度が動き出す"
                "「解けているが間違っている」状態になる。\n"
                "  対処: gridops.powerflow.solve(case, dispatch=dispatch) で"
                "解き直してからその解を渡すこと。"
            )


def aggregate_plants(
    case: Case,
    solution: Any,
    *,
    dispatch: Mapping[str, float] | None = None,
) -> list[dict]:
    """同一 ``plant`` の号機を 1 台に集約する。

    共通系統基準へ換算済みの慣性寄与 H は加算、過渡リアクタンス x'd は並列合成、P と Q は合計。
    制動係数 D も H と同じく加算する（減衰比を保つため。モジュールの
    docstring を参照）。

    Parameters
    ----------
    case:
        ``units`` 層を持つケース。
    solution:
        事故前潮流解。:class:`gridops.powerflow.PowerFlowSolution`、
        :class:`gridops.case.ReferenceSolution`、複素電圧の配列、
        ``{母線番号: 複素電圧}``、``None``（ケースの参照解）のいずれか。
    dispatch:
        号機名から出力 [MW] への対応。与えると **停止中の号機を集約から
        外す**（停止した発電機は慣性を供給しない）。潮流解と整合しない
        出力表を渡すと :class:`ValueError` になる。

    Returns
    -------
    list of dict
        発電所ごとに次のキーを持つ辞書。並びは号機の登場順。

        ============  ==================================================
        キー          内容
        ============  ==================================================
        ``name``      発電所名
        ``bus``       母線番号
        ``H``         100 MVA 共通基準上の慣性寄与 [s]（加算）
        ``xd_prime``  過渡リアクタンス [p.u.]（並列合成）
        ``D``         制動係数 [p.u.]（加算）
        ``P``         有効電力 [p.u.]
        ``Q``         無効電力 [p.u.]
        ``p_mw``      有効電力 [MW]
        ``q_mvar``    無効電力 [Mvar]
        ``units``     集約した号機名の組
        ``n_units``   集約した号機数
        ============  ==================================================

    Raises
    ------
    ValueError
        ``h`` または ``xd_prime`` を持たない号機があるとき。同じ発電所の
        号機が別の母線に置かれているとき。``dispatch`` が潮流解と
        食い違っているとき。

    Notes
    -----
    P と Q は **潮流解から** 求める（:func:`_bus_generation` を参照）。
    ``dispatch`` の値をそのまま使わないのは、slack 母線の出力が損失を
    引き受けた後の値でなければ運転点が平衡点にならないからである。
    1 つの母線に複数の発電所がある場合は、母線の発電を各発電所へ
    比例配分する（``dispatch`` があればその出力比、無ければ ``p_max_mw``
    の比）。同梱ケース wscc9 は 1 母線 1 発電所なので配分は効かない。

    Examples
    --------
    >>> from gridops import load_case                      # doctest: +SKIP
    >>> from gridops.powerflow import solve                # doctest: +SKIP
    >>> case = load_case("wscc9")                          # doctest: +SKIP
    >>> plants = aggregate_plants(case, solve(case))       # doctest: +SKIP
    >>> [(p["name"], round(p["H"], 3)) for p in plants]    # doctest: +SKIP
    [('G1', 23.64), ('G2', 6.4), ('G3', 3.01)]
    """
    case.require("network", "units")
    voltage = _voltage_of(case, solution)
    generation = _bus_generation(case, voltage)
    groups = _plant_groups(case, dispatch)
    if dispatch is not None:
        _check_dispatch(case, generation, dispatch)

    # 母線ごとの発電所の重み（1 母線 1 発電所なら効かない）。
    per_bus: dict[int, list[int]] = {}
    for k, (_, bus, _) in enumerate(groups):
        per_bus.setdefault(bus, []).append(k)

    weights = np.ones(len(groups))
    for bus, members in per_bus.items():
        if len(members) == 1:
            continue
        if dispatch is None:
            share = [sum(u.p_max_mw for u in groups[k][2]) for k in members]
        else:
            share = [
                sum(float(dispatch.get(u.name, 0.0)) for u in groups[k][2])
                for k in members
            ]
        total = float(sum(share))
        for k, value in zip(members, share):
            weights[k] = (value / total) if total > 0 else 1.0 / len(members)

    plants: list[dict] = []
    for k, (name, bus, units) in enumerate(groups):
        _require_stability_data(units)
        power = generation[case.index_of(bus)] * weights[k]
        plants.append(
            {
                "name": name,
                "bus": bus,
                "H": float(sum(u.h for u in units)),
                "xd_prime": float(1.0 / sum(1.0 / u.xd_prime for u in units)),
                "D": float(sum(u.d for u in units)),
                "P": float(power.real),
                "Q": float(power.imag),
                "p_mw": float(case.to_mw(power.real)),
                "q_mvar": float(case.to_mw(power.imag)),
                "units": tuple(u.name for u in units),
                "n_units": len(units),
            }
        )
    return plants


# ======================================================================
# 変換本体
# ======================================================================
def _check_branches(case: Case) -> None:
    """genstab で表せない枝がないか確かめる。"""
    offenders = [
        branch
        for branch in case.branches
        if abs(branch.tap - 1.0) > TAP_TOLERANCE
        or abs(branch.shift_deg) > SHIFT_TOLERANCE
    ]
    if not offenders:
        return
    detail = ", ".join(
        f"{b.label} (tap={b.tap:g}, shift_deg={b.shift_deg:g})" for b in offenders
    )
    raise ValueError(
        f"genstab.multimachine.Branch はタップ比と位相調整角を持たない"
        f"（素の π 型等価回路だけ）ため、次の枝を変換できない: {detail}。\n"
        "  対処 1: タップ比を線路のリアクタンスに織り込んだ等価回路に直す。\n"
        "  対処 2: dataclasses.replace(branch, tap=1.0, shift_deg=0.0) で"
        "落として構わないかを、まず潮流解が変わらないかで確かめる。\n"
        "変圧器のタップは事故前潮流には効くが、古典モデルの動揺には"
        "「そのリアクタンスで送れる電力」としてしか効かない点に注意すること。"
    )


def _shunt_loads(case: Case, voltage: np.ndarray) -> list[tuple[int, float, float]]:
    """母線シャントを定インピーダンス負荷に読み替えた ``(母線, P, Q)`` を返す。

    genstab には母線シャントの入れ物が無いが、シャントはもともと定
    インピーダンスなので、負荷として厳密に等価な形に書き直せる。

    .. math::

        \\frac{(P + jQ)^{*}}{|V|^{2}} = g_s + j b_s
        \\;\\Longleftrightarrow\\;
        P = g_s |V|^{2}, \\quad Q = -b_s |V|^{2}

    調相用のコンデンサ (:math:`b_s > 0`) は **負の Q を持つ負荷** として
    現れる。無効電力を供給する設備なのだから符号はこれで正しい。
    """
    extra: list[tuple[int, float, float]] = []
    for i, bus in enumerate(case.buses):
        if bus.gs == 0.0 and bus.bs == 0.0:
            continue
        magnitude_squared = float(abs(voltage[i]) ** 2)
        extra.append(
            (bus.id, bus.gs * magnitude_squared, -bus.bs * magnitude_squared)
        )
    return extra


def _fault_data(case: Case) -> dict[str, Any]:
    """``stability`` 層の事故スケジュールを取り出す（無ければ事故なし）。"""
    fault = (case.stability or {}).get("fault", {}) or {}
    return {
        "t_fault": float(fault.get("t_fault", math.inf)),
        "t_clear": float(fault.get("t_clear", math.inf)),
        "buses": [int(b) for b in fault.get("buses", [])],
        "tripped": [
            (int(item["from"]), int(item["to"]))
            for item in fault.get("tripped_branches", [])
        ],
    }


def _damping_for(plant: dict, damping: float | Mapping[str, float] | None) -> float:
    """集約後の機の制動係数を決める。"""
    if damping is None:
        return float(plant["D"])
    if isinstance(damping, collections.abc.Mapping):
        if plant["name"] not in damping:
            raise ValueError(
                f"damping に発電所 '{plant['name']}' の値がない。"
                f"与えられているのは {sorted(damping)}。"
                "全機を同じ値にするなら数値をそのまま渡すこと。"
            )
        return float(damping[plant["name"]])
    return float(damping)


def to_genstab(
    case: Case,
    solution: Any,
    *,
    dispatch: Mapping[str, float] | None = None,
    damping: float | Mapping[str, float] | None = None,
) -> "MultiMachineSystem":
    """潮流解から genstab の :class:`MultiMachineSystem` を組み立てる。

    **genstab のソースには手を入れない。** 公開クラス
    (:class:`MultiMachineNetwork`, :class:`GeneratorData`, :class:`Branch`,
    :class:`Load`, :class:`MultiMachineSystem`) をこちらから組み立てる
    だけである。genstab の import は関数の中で行うので、genstab が入って
    いない環境でも gridops の他の機能は動く。

    Parameters
    ----------
    case:
        ``network`` と ``units`` の層を持つケース。``stability`` 層が
        あれば事故スケジュールも写す（無ければ事故なしの系になる）。
    solution:
        事故前潮流解（:func:`aggregate_plants` と同じ形を受ける）。
    dispatch:
        号機名から出力 [MW] への対応。停止中の号機を集約から外す。
    damping:
        制動係数 [p.u.] の上書き。数値なら **全機** をその値にし、
        ``{発電所名: 値}`` の写像なら発電所ごとに指定する。``None``
        （既定）ならケースの号機の ``d`` を加算した値を使う。

    Returns
    -------
    genstab.multimachine.MultiMachineSystem

    Raises
    ------
    ImportError
        genstab が入っていないとき。``pip install -e .`` のやり直しを案内する。
    ValueError
        ``tap != 1`` または ``shift_deg != 0`` の枝があるとき（genstab の
        :class:`Branch` はタップを持たない）。``h`` または ``xd_prime`` を
        持たない号機があるとき。``dispatch`` が潮流解と食い違うとき。

    Notes
    -----
    genstab の負荷の定インピーダンス変換 :math:`Y = \\mathrm{conj}(S)/|V|^2`
    は潮流解の :math:`|V|` に依存し、内部起電力 :math:`E = V + j x'_d I` は
    P, Q に依存する。**潮流解と発電機の P, Q は必ずセットで渡すこと。**
    片方だけ更新すると内部起電力が不整合になり「解けているが間違っている」
    状態になる。本関数は P, Q を潮流解そのものから求め、``dispatch`` を
    渡した場合はその照合まで行うことで、この事故を構造的に防いでいる。

    母線シャント ``gs`` / ``bs`` は :math:`P = g_s|V|^2`,
    :math:`Q = -b_s|V|^2` の定インピーダンス負荷に読み替えて渡す
    （genstab に母線シャントの入れ物が無いため。読み替えは厳密で、
    Ybus は 1 ビットも変わらない）。同梱ケース wscc9 はシャントを
    持たないのでこの経路は効かない。

    ``stability`` 層の ``protection_time`` は :class:`MultiMachineSystem`
    に対応する入れ物が無いので渡していない。事故除去時間と CCT を
    比べるときに使う値なので、必要なら ``case.stability`` から直接読むこと。

    Examples
    --------
    >>> from gridops import load_case                            # doctest: +SKIP
    >>> from gridops.powerflow import solve                      # doctest: +SKIP
    >>> from genstab import eac                                  # doctest: +SKIP
    >>> case = load_case("wscc9")                                # doctest: +SKIP
    >>> system = to_genstab(case, solve(case), damping=0.0)      # doctest: +SKIP
    >>> eac.critical_clearing_time(system, t_end=5.0,
    ...                            tolerance=1e-4, upper_bound=1.0)  # doctest: +SKIP
    0.161133
    """
    api = _genstab()
    case.require("network", "units")

    voltage = _voltage_of(case, solution)
    _check_branches(case)
    plants = aggregate_plants(case, solution, dispatch=dispatch)
    if not plants:
        raise ValueError(
            f"ケース '{case.name}' に運転中の発電機が 1 台もない。"
            "dispatch がすべてゼロになっていないか確認すること。"
        )

    voltages = {bus.id: complex(voltage[i]) for i, bus in enumerate(case.buses)}

    loads = []
    shunt_equivalent = dict(
        (bus_id, (p, q)) for bus_id, p, q in _shunt_loads(case, voltage)
    )
    for bus in case.buses:
        p_shunt, q_shunt = shunt_equivalent.get(bus.id, (0.0, 0.0))
        p_total = bus.pd + p_shunt
        q_total = bus.qd + q_shunt
        if p_total == 0.0 and q_total == 0.0:
            continue
        loads.append(api["Load"](bus=bus.id, P=p_total, Q=q_total))

    network = api["MultiMachineNetwork"](
        buses=[bus.id for bus in case.buses],
        branches=[
            api["Branch"](
                from_bus=branch.from_bus,
                to_bus=branch.to_bus,
                r=branch.r,
                x=branch.x,
                b=branch.b,
            )
            for branch in case.branches
        ],
        loads=loads,
        voltages=voltages,
    )

    generators = [
        api["GeneratorData"](
            bus=plant["bus"],
            H=plant["H"],
            xd_prime=plant["xd_prime"],
            P=plant["P"],
            Q=plant["Q"],
            D=_damping_for(plant, damping),
            name=plant["name"],
        )
        for plant in plants
    ]

    fault = _fault_data(case)
    return api["MultiMachineSystem"](
        network=network,
        generators=generators,
        fault=api["FaultSchedule"](
            t_fault=fault["t_fault"], t_clear=fault["t_clear"]
        ),
        faulted_buses=fault["buses"],
        tripped_branches=fault["tripped"],
        base=api["SystemBase"](
            frequency_hz=case.frequency_hz, s_base_mva=case.base_mva
        ),
    )


# ======================================================================
# 答え合わせの表
# ======================================================================
def _row(values: Sequence[str], widths: Sequence[int]) -> str:
    """1 行分の欄を並べる。先頭の欄だけ左寄せ（名前の欄なので）。"""
    cells = [str(values[0]).ljust(widths[0])]
    cells += [str(text).rjust(width) for text, width in zip(values[1:], widths[1:])]
    return "  ".join(cells).rstrip()


def check_against_reference(case: Case, solution: Any) -> str:
    """自力で解いた解と参照解を並べた表（英語ヘッダ）を返す。

    見出しと列名を英語にしてあるのは、そのまま notebook の出力や図の
    キャプションに貼れるようにするためである（本パッケージの作図の
    軸ラベルが英語なのと同じ理由）。

    Parameters
    ----------
    case:
        ``solution`` 層（参照解）を持つケース。
    solution:
        突き合わせる解（:func:`aggregate_plants` と同じ形を受ける）。

    Returns
    -------
    str
        母線電圧・内部起電力・スカラー量（slack 出力と総損失）を
        並べた表。差の欄は指数表記なので、桁で読める。

    Raises
    ------
    ValueError
        ケースが参照解を持たないとき。

    Notes
    -----
    差が残るのは実装の誤りとは限らない。参照解は出典の掲載桁数
    （``ReferenceSolution.digits``、wscc9 では 4 桁）で丸められて
    いるので、|V| で 5e-5、位相で 5e-5 deg の丸めが必ず乗る。
    **どこまでの差なら丸めで説明できるかを先に見積もってから**
    表を読むこと。

    内部起電力の位相は特に敏感である。wscc9 の G3 は Q が小さく
    x'd が大きいので、出典が Q を 3 桁で丸めた影響が 0.009 deg 程度の
    差になって現れる。同じ表の G1 / G2 が 1e-4 deg で合っているのに
    G3 だけ 2 桁大きい、という読み方ができれば、原因が実装ではなく
    データの丸めであることが特定できる。
    """
    case.require("solution")
    reference = case.reference
    voltage = _voltage_of(case, solution)
    checks: Mapping[str, Any] = reference.checks or {}

    lines: list[str] = [
        f"Comparison against the reference solution -- {case.name}",
        f"  Reference source : {reference.source or '(not recorded)'}",
        f"  Published digits : {reference.digits}"
        f"   (rounding alone gives up to {0.5 * 10.0 ** -reference.digits:.1e})",
        "",
    ]

    # --- 母線電圧 -----------------------------------------------------
    widths = (5, 11, 11, 10, 13, 13, 10)
    header = ("Bus", "|V| solved", "|V| ref", "d|V|", "angle solved", "angle ref",
              "d angle")
    lines.append("Bus voltages   (magnitude in p.u., angle in deg)")
    lines.append(_row(header, widths))
    lines.append("-" * len(_row(header, widths)))

    solved_magnitude = np.abs(voltage)
    solved_angle = np.degrees(np.angle(voltage))
    for i, bus in enumerate(case.buses):
        lines.append(
            _row(
                (
                    str(bus.id),
                    f"{solved_magnitude[i]:.6f}",
                    f"{reference.v[i]:.6f}",
                    f"{solved_magnitude[i] - reference.v[i]:+.2e}",
                    f"{solved_angle[i]:.6f}",
                    f"{reference.angle_deg[i]:.6f}",
                    f"{solved_angle[i] - reference.angle_deg[i]:+.2e}",
                ),
                widths,
            )
        )
    lines.append(
        _row(
            (
                "max",
                "",
                "",
                f"{np.abs(solved_magnitude - reference.v).max():.2e}",
                "",
                "",
                f"{np.abs(solved_angle - reference.angle_deg).max():.2e}",
            ),
            widths,
        )
    )
    lines.append("")

    # --- 内部起電力 ---------------------------------------------------
    emf_reference = checks.get("internal_emf")
    angle_reference = checks.get("internal_angle_deg")
    has_stability_data = bool(case.units) and all(
        unit.h is not None and unit.xd_prime is not None for unit in case.units
    )
    if emf_reference is not None and angle_reference is not None and has_stability_data:
        plants = aggregate_plants(case, voltage)
        emf = np.array(
            [
                voltage[case.index_of(p["bus"])]
                + 1j
                * p["xd_prime"]
                * np.conj(
                    complex(p["P"], p["Q"]) / voltage[case.index_of(p["bus"])]
                )
                for p in plants
            ]
        )
        widths = (9, 11, 11, 10, 13, 13, 10)
        header = ("Machine", "|E| solved", "|E| ref", "d|E|", "delta solved",
                  "delta ref", "d delta")
        lines.append("Internal EMF   (magnitude in p.u., angle in deg)")
        lines.append(_row(header, widths))
        lines.append("-" * len(_row(header, widths)))
        for k, plant in enumerate(plants):
            magnitude = float(abs(emf[k]))
            angle = float(np.degrees(np.angle(emf[k])))
            lines.append(
                _row(
                    (
                        plant["name"],
                        f"{magnitude:.6f}",
                        f"{float(emf_reference[k]):.6f}",
                        f"{magnitude - float(emf_reference[k]):+.2e}",
                        f"{angle:.6f}",
                        f"{float(angle_reference[k]):.6f}",
                        f"{angle - float(angle_reference[k]):+.2e}",
                    ),
                    widths,
                )
            )
        lines.append("")

    # --- スカラー量 ---------------------------------------------------
    generation = _bus_generation(case, voltage)
    slack_index = [
        i for i, bus in enumerate(case.buses) if bus.type is BusType.SLACK
    ]
    scalars: list[tuple[str, float, float | None]] = []
    if slack_index:
        i = slack_index[0]
        scalars.append(("Slack P [p.u.]", float(generation[i].real),
                        _as_float(checks.get("slack_p"))))
        scalars.append(("Slack Q [p.u.]", float(generation[i].imag),
                        _as_float(checks.get("slack_q"))))
    scalars.append(("Total losses [p.u.]", _branch_losses(case, voltage),
                    _as_float(checks.get("losses_pu"))))

    widths = (21, 12, 12, 11)
    header = ("Quantity", "Solved", "Reference", "Difference")
    lines.append("Scalar checks")
    lines.append(_row(header, widths))
    lines.append("-" * len(_row(header, widths)))
    for label, solved, expected in scalars:
        lines.append(
            _row(
                (
                    label,
                    f"{solved:.6f}",
                    "n/a" if expected is None else f"{expected:.6f}",
                    "n/a" if expected is None else f"{solved - expected:+.2e}",
                ),
                widths,
            )
        )
    return "\n".join(lines)


def _as_float(value: Any) -> float | None:
    """``checks`` の値を float にする（無ければ ``None``）。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - データの誤り
        return None
