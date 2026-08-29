"""Ybus とトポロジー（第 01 回）の検証。

このモジュールの正しさは、**gridops の外にある基準**と突き合わせて
確かめる。具体的には次の 4 つである。

1. 安定度の教材 ``genstab`` が独立に組んでいる Ybus との一致
2. 教科書（Anderson & Fouad）の潮流解から計算した注入電力が、
   ケースファイルの発電量・負荷と一致すること
3. 直流潮流の別定式化 :math:`Y = A^{T} \\mathrm{diag}(y) A` との一致
4. 「橋かどうか」を連結成分の総当たりで求めた結果との一致

実装の出力を実装で再計算するだけのテストは書かない。1 と 3 は式の
出所が違い、2 と 4 はアルゴリズムそのものが違う。
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from gridops import Branch, Bus, BusType, Case, load_case
from gridops.ybus import bridges, build_ybus, incidence_matrix, islands

#: WSCC 9 母線の橋。変圧器 3 本で、開放すると発電機母線が島になる。
EXPECTED_BRIDGES = [(1, 4), (2, 7), (3, 9)]

#: 事故前潮流解における各母線の複素注入電力 [p.u.]（発電 - 負荷）。
#: ケースファイルの generation と buses の pd/qd から手で組んだ値であり、
#: gridops の計算結果ではない。
EXPECTED_INJECTIONS = {
    1: 0.716 + 0.270j,
    2: 1.630 + 0.067j,
    3: 0.850 - 0.109j,
    4: 0.0 + 0.0j,
    5: -(1.25 + 0.50j),
    6: -(0.90 + 0.30j),
    7: 0.0 + 0.0j,
    8: -(1.00 + 0.35j),
    9: 0.0 + 0.0j,
}

#: API 契約に載っている基準潮流の枝潮流。
#: ``枝: (両端の |S| の大きい方, from 側の有効電力, 基準負荷率 [%])``
EXPECTED_FLOWS = {
    (1, 4): (0.7658, 0.7164, 38.3),
    (2, 7): (1.6326, 1.6300, 81.6),
    (3, 9): (0.8631, 0.8500, 43.2),
    (4, 5): (0.5614, 0.4094, 56.1),
    (4, 6): (0.3473, 0.3070, 43.4),
    (5, 7): (0.8702, -0.8432, 79.1),
    (6, 9): (0.6345, -0.5946, 57.7),
    (7, 8): (0.7666, 0.7638, 69.7),
    (8, 9): (0.3422, -0.2410, 42.8),
}


@pytest.fixture(scope="module")
def case() -> Case:
    return load_case("wscc9")


def _two_bus_case(**branch_kwargs) -> Case:
    """タップと位相調整を確かめるための最小ケース。"""
    return Case(
        name="two-bus",
        buses=[Bus(id=1, type=BusType.SLACK), Bus(id=2)],
        branches=[Branch(from_bus=1, to_bus=2, **branch_kwargs)],
    )


# ----------------------------------------------------------------------
# 独立な実装との一致（ドリフト検出の砦）
# ----------------------------------------------------------------------
def test_matches_genstab_ybus(case):
    """genstab が独立に組んだ Ybus と一致すること。

    genstab 側は負荷を定インピーダンスとして対角に足すので、
    ``include_loads=False`` で枝だけの行列にしてから比べる。両者は
    別々のソースコードで同じ物理を組み立てているので、片方が壊れれば
    この一致が真っ先に崩れる。許容差 1e-14 は倍精度の丸めの水準。
    """
    genstab = pytest.importorskip("genstab.multimachine")

    branches = [
        genstab.Branch(from_bus=b.from_bus, to_bus=b.to_bus, r=b.r, x=b.x, b=b.b)
        for b in case.branches
    ]
    loads = [
        genstab.Load(bus=bus.id, P=bus.pd, Q=bus.qd)
        for bus in case.buses
        if bus.pd or bus.qd
    ]
    voltages = {
        bus.id: complex(v) for bus, v in zip(case.buses, case.reference.voltage)
    }
    network = genstab.MultiMachineNetwork(
        buses=case.bus_ids, branches=branches, loads=loads, voltages=voltages
    )

    assert all(b.tap == 1.0 and b.shift_deg == 0.0 for b in case.branches)
    expected = network.ybus(include_loads=False)
    actual = build_ybus(case, include_shunts=False)
    assert np.abs(actual - expected).max() < 1e-14


def test_matches_genstab_ybus_with_a_branch_removed(case):
    """枝を開放した状態でも genstab と一致すること。"""
    genstab = pytest.importorskip("genstab.multimachine")

    branches = [
        genstab.Branch(from_bus=b.from_bus, to_bus=b.to_bus, r=b.r, x=b.x, b=b.b)
        for b in case.branches
    ]
    voltages = {
        bus.id: complex(v) for bus, v in zip(case.buses, case.reference.voltage)
    }
    network = genstab.MultiMachineNetwork(
        buses=case.bus_ids, branches=branches, loads=[], voltages=voltages
    )
    expected = network.ybus([(5, 7)], include_loads=False)
    actual = build_ybus(case, removed_branches=[(7, 5)], include_shunts=False)
    assert np.abs(actual - expected).max() < 1e-14


def test_matches_incidence_form_for_a_lossless_network(case):
    """抵抗と充電容量を落とすと :math:`Y = A^{T}\\,\\mathrm{diag}(y)\\,A` に一致すること。

    足し込みのループとは無関係な行列積の定式化と突き合わせる。
    :math:`\\pi` 型の充電容量とタップがなければ、Ybus は接続行列と
    枝アドミタンスだけで書けるという事実の確認である。
    """
    lossless = replace(
        case,
        branches=[replace(b, r=0.0, b=0.0) for b in case.branches],
        reference=None,
    )
    A = incidence_matrix(lossless)
    y = np.array([b.series_admittance() for b in lossless.branches])
    expected = A.T @ np.diag(y) @ A
    actual = build_ybus(lossless, include_shunts=False)
    assert np.abs(actual - expected).max() < 1e-13


# ----------------------------------------------------------------------
# 潮流解との整合（教科書の値が基準）
# ----------------------------------------------------------------------
def test_injections_from_reference_solution_match_generation_and_load(case):
    """:math:`\\bar S = \\bar V (Y \\bar V)^{*}` が発電量と負荷に一致すること。

    Ybus が正しく、かつ参照解が正しければ、電圧だけから注入電力が
    再現できる。許容差 2e-3 の根拠: 参照解は 4 桁で丸めてあるので
    :math:`|\\Delta V| \\le 5\\times10^{-5}`、Ybus の行の絶対値和は
    最大 79 p.u. なので、注入の誤差は数 1e-3 のオーダーになりうる。
    実測の最大誤差は 1.1e-3（母線 9）である。
    """
    Y = build_ybus(case, include_shunts=False)
    v = case.reference.voltage
    s = v * np.conj(Y @ v)
    for bus_id, expected in EXPECTED_INJECTIONS.items():
        actual = s[case.index_of(bus_id)]
        assert abs(actual - expected) < 2e-3, f"母線 {bus_id} が不整合"


def test_total_injection_equals_losses(case):
    """注入の総和が総損失（0.046410 p.u.）に一致すること。

    期待値はケースファイルの ``checks.losses_pu``（独立に検算済みの値）。
    """
    Y = build_ybus(case, include_shunts=False)
    v = case.reference.voltage
    losses = float(np.sum(v * np.conj(Y @ v)).real)
    assert losses == pytest.approx(case.reference.checks["losses_pu"], abs=2e-3)


def test_branch_flows_match_the_contract_table(case):
    """枝潮流と負荷率が API 契約の表と一致すること。

    Ybus と同じ 2x2 の枝行列から
    :math:`\\bar S_{ft} = \\bar V_f (Y_{ff}\\bar V_f + Y_{ft}\\bar V_t)^{*}`
    を計算する。許容差 1e-3 は参照解の丸め（実測の最大差は 3.5e-4）。
    """
    v = case.reference.voltage
    for branch in case.branches:
        f = case.index_of(branch.from_bus)
        t = case.index_of(branch.to_bus)
        primitive = branch.primitive()
        s_ft = v[f] * np.conj(primitive[0, 0] * v[f] + primitive[0, 1] * v[t])
        s_tf = v[t] * np.conj(primitive[1, 0] * v[f] + primitive[1, 1] * v[t])

        expected_s, expected_p, expected_loading = EXPECTED_FLOWS[branch.key()]
        assert max(abs(s_ft), abs(s_tf)) == pytest.approx(expected_s, abs=1e-3)
        assert s_ft.real == pytest.approx(expected_p, abs=1e-3)
        loading = 100.0 * max(abs(s_ft), abs(s_tf)) / branch.rate_a
        assert loading == pytest.approx(expected_loading, abs=0.15)


# ----------------------------------------------------------------------
# 組み立ての性質
# ----------------------------------------------------------------------
def test_row_sums_are_the_charging_susceptance_only(case):
    """行和がシャントと充電容量だけになること。

    全母線の電圧が等しければ枝には電流が流れないので、行和には
    :math:`\\pi` 型の充電容量 :math:`jb/2` しか残らない。期待値は
    ケースファイルの ``b`` から直接組むので、Ybus の組み立てとは独立。
    """
    charging = np.zeros(case.n_bus, dtype=complex)
    for branch in case.branches:
        charging[case.index_of(branch.from_bus)] += 1j * branch.b / 2.0
        charging[case.index_of(branch.to_bus)] += 1j * branch.b / 2.0

    Y = build_ybus(case, include_shunts=False)
    assert np.abs(Y.sum(axis=1) - charging).max() < 1e-12


def test_shunts_go_to_the_diagonal_only(case):
    """``Bus.gs`` / ``Bus.bs`` が対角にだけ入り、除けること。"""
    shunted = replace(
        case,
        buses=[
            replace(bus, gs=0.02, bs=0.19) if bus.id == 6 else bus
            for bus in case.buses
        ],
        reference=None,
    )
    without = build_ybus(shunted, include_shunts=False)
    with_shunt = build_ybus(shunted, include_shunts=True)

    difference = with_shunt - without
    expected = np.zeros_like(difference)
    expected[shunted.index_of(6), shunted.index_of(6)] = complex(0.02, 0.19)
    assert np.abs(difference - expected).max() < 1e-15
    # wscc9 にはシャントがないので、元のケースでは両者が一致する。
    assert np.array_equal(
        build_ybus(case, include_shunts=True), build_ybus(case, include_shunts=False)
    )


def test_removing_a_branch_removes_exactly_its_contribution(case):
    """開放した行列との差が、その枝 1 本の 2x2 行列に一致すること。"""
    target = next(b for b in case.branches if b.key() == (4, 5))
    full = build_ybus(case, include_shunts=False)
    opened = build_ybus(case, removed_branches=[(4, 5)], include_shunts=False)

    difference = full - opened
    expected = np.zeros_like(difference)
    index = np.array([case.index_of(4), case.index_of(5)])
    expected[np.ix_(index, index)] = target.primitive()
    assert np.abs(difference - expected).max() < 1e-15


def test_unknown_removed_branch_is_rejected(case):
    """存在しない枝の開放を黙って無視しないこと。"""
    with pytest.raises(ValueError, match="開放しようとした枝"):
        build_ybus(case, removed_branches=[(4, 9)])


def test_self_loop_branch_is_rejected():
    """自己ループの枝を Ybus に足し込ませないこと。"""
    looped = Case(
        name="self-loop",
        buses=[Bus(id=1, type=BusType.SLACK), Bus(id=2)],
        branches=[Branch(from_bus=2, to_bus=2, x=0.1)],
    )
    with pytest.raises(ValueError, match="自己ループ"):
        build_ybus(looped)


# ----------------------------------------------------------------------
# タップと位相調整（対称性）
# ----------------------------------------------------------------------
def test_tap_keeps_the_matrix_symmetric(case):
    """タップ比だけなら Ybus は対称のままであること。"""
    tapped = replace(
        case,
        branches=[
            replace(b, tap=1.05) if b.key() == (1, 4) else b for b in case.branches
        ],
        reference=None,
    )
    Y = build_ybus(tapped, include_shunts=False)
    assert np.abs(Y - Y.T).max() < 1e-15
    # タップは行列を確かに変える（変えないなら実装が読み飛ばしている）。
    assert not np.allclose(Y, build_ybus(case, include_shunts=False))


def test_phase_shift_makes_the_matrix_asymmetric(case):
    """位相調整角を入れると Ybus が非対称になること。

    :math:`Y_{ft} = -y_s/\\bar a^{*}`、:math:`Y_{tf} = -y_s/\\bar a` なので、
    大きさは等しいまま位相だけが逆向きにずれる。位相調整器は潮流を
    能動的に押し込む非相反な装置であり、それが行列の非対称として現れる。
    """
    shifted = replace(
        case,
        branches=[
            replace(b, shift_deg=5.0) if b.key() == (1, 4) else b
            for b in case.branches
        ],
        reference=None,
    )
    Y = build_ybus(shifted, include_shunts=False)
    assert not np.allclose(Y, Y.T)

    f, t = shifted.index_of(1), shifted.index_of(4)
    assert Y[f, t] != Y[t, f]
    assert abs(Y[f, t]) == pytest.approx(abs(Y[t, f]), rel=1e-15)
    # 非対称なのはその枝の 2 要素だけである。
    asymmetry = np.abs(Y - Y.T)
    asymmetry[f, t] = asymmetry[t, f] = 0.0
    assert asymmetry.max() < 1e-15


def test_lossless_phase_shifter_consumes_no_active_power():
    """無損失の位相調整変圧器で有効電力の収支がゼロになること。

    理想変圧器と純リアクタンスだけの枝は有効電力を消費しない。
    どんな電圧を与えても :math:`\\Re(\\bar S_1 + \\bar S_2) = 0` になる
    はずで、これはタップと位相の置き方が正しいことの物理的な検算に
    なる（式を写し直す自己参照テストではない）。
    """
    shifter = _two_bus_case(r=0.0, x=0.12, b=0.0, tap=1.04, shift_deg=-8.0)
    Y = build_ybus(shifter)

    rng = np.random.default_rng(0)
    for _ in range(5):
        v = (0.9 + 0.2 * rng.random(2)) * np.exp(
            1j * math.radians(30.0) * (rng.random(2) - 0.5)
        )
        s = v * np.conj(Y @ v)
        assert abs(s.sum().real) < 1e-14
        # 無効電力はリアクタンスが消費するので、ゼロにはならない。
        assert s.sum().imag > 0.0


# ----------------------------------------------------------------------
# 接続行列
# ----------------------------------------------------------------------
def test_incidence_matrix_shape_and_rows(case):
    """形と各行の非ゼロが ``+1`` / ``-1`` の 2 つだけであること。"""
    A = incidence_matrix(case)
    assert A.shape == (case.n_branch, case.n_bus)
    assert np.array_equal(A.sum(axis=1), np.zeros(case.n_branch))
    for row, branch in enumerate(case.branches):
        assert A[row, case.index_of(branch.from_bus)] == 1.0
        assert A[row, case.index_of(branch.to_bus)] == -1.0
        assert np.count_nonzero(A[row]) == 2


def test_incidence_matrix_keeps_a_zero_row_for_removed_branches(case):
    """開放した枝の行は削除せずゼロ行として残すこと。

    枝の並びが :attr:`Case.branches` と常に一致していないと、N-1 の
    結果を枝ごとに並べるところで添字がずれる。
    """
    A = incidence_matrix(case, removed_branches=[(4, 5)])
    assert A.shape == (case.n_branch, case.n_bus)
    row = [b.key() for b in case.branches].index((4, 5))
    assert np.count_nonzero(A[row]) == 0
    assert np.count_nonzero(A) == 2 * (case.n_branch - 1)


def test_incidence_matrix_rank_shows_the_number_of_islands(case):
    """接続行列の階数が ``n_bus - 島の数`` になること。

    連結なら階数は :math:`n-1`（位相の基準が 1 つ自由）。橋を開放して
    2 島になれば :math:`n-2` に落ちる。連結成分の探索とは無関係な
    線形代数の側から島の数を数えている。
    """
    A = incidence_matrix(case)
    assert np.linalg.matrix_rank(A) == case.n_bus - len(islands(case))

    key = EXPECTED_BRIDGES[0]
    opened = incidence_matrix(case, removed_branches=[key])
    assert np.linalg.matrix_rank(opened) == case.n_bus - len(
        islands(case, removed_branches=[key])
    )
    assert len(islands(case, removed_branches=[key])) == 2


# ----------------------------------------------------------------------
# トポロジー
# ----------------------------------------------------------------------
def test_bridges_of_wscc9_are_the_three_transformers(case):
    """WSCC 9 母線の橋がちょうど変圧器 3 本であること。"""
    assert bridges(case) == EXPECTED_BRIDGES


def test_bridges_agree_with_brute_force_islanding(case):
    """橋の判定が「1 本ずつ開放して島を数える」総当たりと一致すること。

    low-link の実装とはまったく別のアルゴリズム（連結成分の数え上げ）で
    同じ答えが出ることを確かめる。
    """
    found = bridges(case)
    for branch in case.branches:
        splits = len(islands(case, removed_branches=[branch.key()])) > 1
        assert (branch.key() in found) == splits, f"枝 {branch.label} の判定が不一致"


def test_bridges_are_excluded_from_the_contingency_list(case):
    """ケースの N-1 候補に橋が 1 本も入っていないこと。

    第 09 回で「なぜ変圧器 3 本が候補から外れているか」を学生に
    答えさせる。データ側の記述とアルゴリズムの答えが一致していないと
    その問いが成立しない。
    """
    assert set(case.contingencies).isdisjoint(bridges(case))
    assert len(case.contingencies) == case.n_branch - len(bridges(case))


def test_parallel_branches_are_not_bridges(case):
    """多重回線は橋でないこと。

    変圧器 1-4 を 2 回線にすると、1 本開放してももう 1 本が残るので
    橋ではなくなる。「入ってきた枝」を母線番号で除く実装だと、ここで
    誤って橋と判定される。
    """
    transformer = next(b for b in case.branches if b.key() == (1, 4))
    doubled = replace(
        case,
        branches=[*case.branches, replace(transformer, name="T1b")],
        reference=None,
    )
    assert bridges(doubled) == [(2, 7), (3, 9)]

    # 1 本だけ開放したケースを作れば、母線 1 は孤立しない。
    one_circuit = replace(doubled, branches=doubled.branches[1:])
    assert len(islands(one_circuit)) == 1


def test_removal_by_key_opens_every_parallel_circuit(case):
    """開放の指定は :meth:`Branch.key` 単位であること。

    多重回線に ``(1, 4)`` を指定すると 2 本ともまとめて外れる
    （:meth:`Case.without_branch` と同じ規約）。橋の判定が回線 1 本
    ごとなのに対し、開放の指定は母線対ごとという **粒度の違い**が
    あるので、その差をここで固定しておく。
    """
    transformer = next(b for b in case.branches if b.key() == (1, 4))
    doubled = replace(
        case,
        branches=[*case.branches, replace(transformer, name="T1b")],
        reference=None,
    )
    assert (1, 4) not in bridges(doubled)
    assert islands(doubled, removed_branches=[(1, 4)]) == [
        [1],
        [2, 3, 4, 5, 6, 7, 8, 9],
    ]


def test_islands_of_the_intact_network(case):
    """無事故時は 1 つの島に全母線が入ること。"""
    assert islands(case) == [case.bus_ids]


def test_opening_a_bridge_splits_the_network(case):
    """橋を 1 本開放すると島が 2 個になり、発電機母線が孤立すること。"""
    assert islands(case, removed_branches=[(1, 4)]) == [[1], [2, 3, 4, 5, 6, 7, 8, 9]]
    assert islands(case, removed_branches=[(2, 7)]) == [[1, 3, 4, 5, 6, 7, 8, 9], [2]]
    assert islands(case, removed_branches=[(3, 9)]) == [[1, 2, 4, 5, 6, 7, 8, 9], [3]]


def test_opening_a_non_bridge_keeps_the_network_connected(case):
    """環路の枝を開放しても連結のままであること。"""
    for key in case.contingencies:
        assert len(islands(case, removed_branches=[key])) == 1


def test_two_openings_can_isolate_a_load_bus(case):
    """母線 5 の 2 本を同時に開放すると、母線 5 だけが島になること。

    N-1 では起きないが、N-2 では起きる。島に slack がなければ位相の
    基準も損失の受け皿もなく、潮流計算そのものが定義できない。
    """
    assert islands(case, removed_branches=[(4, 5), (5, 7)]) == [
        [1, 2, 3, 4, 6, 7, 8, 9],
        [5],
    ]


def test_ybus_of_a_split_network_is_singular(case):
    """島に分かれた系統の Ybus が特異になること。

    橋を開放すると母線 1 が孤立し、その行と列が丸ごとゼロになる。
    「ソルバが収束しない」の正体がトポロジーであることの実演である。
    """
    Y = build_ybus(case, removed_branches=[(1, 4)], include_shunts=False)
    i = case.index_of(1)
    assert np.abs(Y[i, :]).max() == 0.0
    assert np.abs(Y[:, i]).max() == 0.0
    assert np.linalg.matrix_rank(Y) < case.n_bus


def test_bridges_on_a_hand_checked_topology():
    """手で数えられる小さなグラフで橋の判定を確かめる。

    三角形 1-2-3 に枝 3-4 と 4-5 をぶら下げた形。環の 3 本は橋でなく、
    ぶら下がりの 2 本が橋である。WSCC 9 母線と違って目で数えられるので、
    low-link の実装そのものの検算になる。
    """
    graph = Case(
        name="triangle-with-tail",
        buses=[
            Bus(id=i, type=BusType.SLACK if i == 1 else BusType.PQ)
            for i in range(1, 6)
        ],
        branches=[
            Branch(from_bus=1, to_bus=2),
            Branch(from_bus=2, to_bus=3),
            Branch(from_bus=3, to_bus=1),
            Branch(from_bus=3, to_bus=4),
            Branch(from_bus=4, to_bus=5),
        ],
    )
    assert bridges(graph) == [(3, 4), (4, 5)]
    assert islands(graph) == [[1, 2, 3, 4, 5]]
    assert islands(graph, removed_branches=[(4, 5)]) == [[1, 2, 3, 4], [5]]


def test_bridges_and_islands_handle_an_already_split_network():
    """最初から 2 つに分かれている系統でも、島ごとに橋を見つけること。

    深さ優先探索の根が複数になる場合の確認である。橋の判定を根 1 つで
    済ませる実装だと、2 つ目の島の枝が丸ごと落ちる。
    """
    split = Case(
        name="two-islands",
        buses=[
            Bus(id=i, type=BusType.SLACK if i == 1 else BusType.PQ)
            for i in range(1, 6)
        ],
        branches=[
            Branch(from_bus=1, to_bus=2),   # 島 A: 橋
            Branch(from_bus=3, to_bus=4),   # 島 B: 環の一部
            Branch(from_bus=4, to_bus=5),
            Branch(from_bus=5, to_bus=3),
        ],
    )
    assert bridges(split) == [(1, 2)]
    assert islands(split) == [[1, 2], [3, 4, 5]]
    # 孤立母線（枝を 1 本も持たない母線）も 1 つの島として数える。
    lonely = replace(split, branches=split.branches[1:])
    assert islands(lonely) == [[1], [2], [3, 4, 5]]


def test_a_bare_pair_is_rejected_with_guidance(case):
    """``removed_branches=(4, 5)`` の書き間違いを日本語で止めること。

    組のリストではなく組そのものを渡すと母線番号 2 つに分解される。
    黙って TypeError を出すより、書き方を示すほうが教材として有益である。
    """
    with pytest.raises(ValueError, match="組のリスト"):
        build_ybus(case, removed_branches=(4, 5))
    with pytest.raises(ValueError, match="長さが 2 でない"):
        islands(case, removed_branches=[(4, 5, 6)])
