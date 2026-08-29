"""直流潮流と感度係数（第 09 回）の検証。

このモジュールの正しさは、**実装の外にある基準**と突き合わせて確かめる。
使う基準は次の 6 つである。

1. 数値微分（母線注入を 1 MW ずらして直流潮流を解き直す）と PTDF の一致
2. 枝を消して直接解いた事故後潮流と、LODF で予測した潮流の一致
3. キルヒホッフの電流則だけで手計算できる事故後潮流
   （例: 枝 4-5 を開放すれば母線 5 の負荷 1.25 は枝 5-7 だけを通る）
4. 三角形の 3 母線系統の解析解（並列した経路にリアクタンスの逆比で分流する）
5. 交流の :math:`Y_{bus}` の虚部（別モジュールの別定式化）
6. 教科書の潮流解から計算した交流の枝潮流

「slack に依存しない」ことだけを assert すると、全部ゼロの実装でも通って
しまう。**「単一列は変わる」と「列の差と LODF は変わらない」の両方**を
固定してあるのはそのためである。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gridops import Branch, Bus, BusType, Case, load_case
from gridops.dc import DCSolution, dc_powerflow, lodf, ptdf, susceptance_matrix
from gridops.ybus import bridges, build_ybus, incidence_matrix

#: WSCC 9 母線の橋（変圧器 3 本）。開放すると発電機母線が島になる。
BRIDGES = [(1, 4), (2, 7), (3, 9)]

#: API 契約の表にある枝 4-5 の交流の値。皮相電力 |S| [p.u.] と負荷率。
AC_APPARENT_4_5 = 0.5614
AC_LOADING_4_5 = 0.561

#: 事故後潮流のうち **キルヒホッフの電流則だけで決まる**もの。
#: 母線 5 / 6 / 8 はいずれも枝 2 本しか持たないので、片方を開放すれば
#: その母線の負荷は残る 1 本を丸ごと通る。リアクタンスの値によらない。
#: ``開放する枝: (観測する枝, その潮流 [p.u.])``
KIRCHHOFF_POST_FLOWS = {
    (4, 5): ((5, 7), -1.25),   # 母線 5 の負荷 1.25 が 7 から流れ込む
    (5, 7): ((4, 5), +1.25),
    (4, 6): ((6, 9), -0.90),   # 母線 6 の負荷 0.90
    (6, 9): ((4, 6), +0.90),
    (7, 8): ((8, 9), -1.00),   # 母線 8 の負荷 1.00
    (8, 9): ((7, 8), +1.00),
}


@pytest.fixture(scope="module")
def case() -> Case:
    return load_case("wscc9")


@pytest.fixture(scope="module")
def base(case: Case) -> DCSolution:
    return dc_powerflow(case)


def _keys(case: Case) -> list[tuple[int, int]]:
    """枝の並び順の :meth:`Branch.key` の一覧。"""
    return [branch.key() for branch in case.branches]


def _position(case: Case, key: tuple[int, int]) -> int:
    """枝の通し番号。"""
    return _keys(case).index(key)


def _with_slack_at(case: Case, bus_id: int) -> Case:
    """slack 母線を付け替えたケースを返す。

    :func:`lodf` は slack を引数に取らない（取る必要がない）ので、
    slack への非依存を公開 API だけで確かめるには母線種別を変える。
    """
    buses = [
        replace(bus, type=BusType.SLACK if bus.id == bus_id else BusType.PV)
        if bus.type is not BusType.PQ
        else bus
        for bus in case.buses
    ]
    return replace(case, buses=buses)


def _triangle() -> Case:
    """手計算できる 3 母線の三角形。リアクタンスはすべて 0.1 p.u.。

    母線 2 に 1 p.u. 注入して slack（母線 1）から引き抜くと、直達路
    ``1-2``（x=0.1）と迂回路 ``2-3-1``（x=0.2）の並列になり、電力は
    リアクタンスの逆比、すなわち 2:1 に分かれる。
    """
    return Case(
        name="triangle",
        buses=[Bus(id=1, type=BusType.SLACK), Bus(id=2), Bus(id=3)],
        branches=[
            Branch(from_bus=1, to_bus=2, x=0.1),
            Branch(from_bus=2, to_bus=3, x=0.1),
            Branch(from_bus=1, to_bus=3, x=0.1),
        ],
    )


# ----------------------------------------------------------------------
# B'（サセプタンス行列）
# ----------------------------------------------------------------------
def test_susceptance_matrix_is_the_imaginary_part_of_a_lossless_ybus(case):
    """抵抗と充電容量を落とせば :math:`B' = -\\mathrm{Im}(Y_{bus})` になること。

    交流の Ybus は枝の 2x2 行列の足し込みで、B' は接続行列の 2 次形式で
    作っており、コードの経路がまったく違う。無損失・無充電容量の極限で
    両者が一致することは、直流近似が「交流の虚部だけを残したもの」で
    あることの確認でもある。許容差 1e-12 は倍精度の丸めの水準。
    """
    lossless = replace(
        case,
        branches=[replace(b, r=0.0, b=0.0) for b in case.branches],
        reference=None,
    )
    expected = -np.imag(build_ybus(lossless, include_shunts=False))
    assert np.abs(susceptance_matrix(lossless) - expected).max() < 1e-12


def test_susceptance_matrix_is_symmetric_with_zero_row_sums(case):
    """B' が対称で、行和がゼロ（したがって特異）であること。

    行和がゼロなのは「全母線を同じ位相にすれば潮流が流れない」ことの
    行列表現である。位相の絶対値に意味がないので、slack の行と列を
    落とさないと解けない。

    特異性は最小特異値を最大特異値との**比**で見る。行列式の絶対値で
    見ると、成分が O(10) の 9x9 行列では特異でも丸めだけで
    :math:`\varepsilon \cdot \sigma_1^{n-1} \sim 10^{-8}` 前後の値になり、
    BLAS の実装（OS）次第で通ったり落ちたりする（Windows の CI で実測）。
    """
    B = susceptance_matrix(case)
    assert np.abs(B - B.T).max() < 1e-12
    assert np.abs(B.sum(axis=1)).max() < 1e-12
    singular_values = np.linalg.svd(B, compute_uv=False)
    assert singular_values[-1] < 1e-12 * singular_values[0]


def test_susceptance_matrix_off_diagonal_is_minus_one_over_x(case):
    """非対角成分がケースファイルの :math:`-1/x` そのものであること。

    期待値をケースファイルの ``x`` から直接組むので、B' の組み立てとは
    独立である。並列回線はないので 1 対 1 に対応する。
    """
    B = susceptance_matrix(case)
    for branch in case.branches:
        i = case.index_of(branch.from_bus)
        j = case.index_of(branch.to_bus)
        assert B[i, j] == pytest.approx(-1.0 / branch.x, rel=1e-12)


def test_susceptance_matrix_uses_the_tap_ratio():
    """タップ比が :math:`1/(\\tau x)` として入ること。

    2 母線の手計算と突き合わせる。``tap=2`` なら結合が半分になる。
    """
    two_bus = Case(
        name="two-bus",
        buses=[Bus(id=1, type=BusType.SLACK), Bus(id=2)],
        branches=[Branch(from_bus=1, to_bus=2, x=0.1, tap=2.0)],
    )
    expected = np.array([[5.0, -5.0], [-5.0, 5.0]])
    assert np.abs(susceptance_matrix(two_bus) - expected).max() < 1e-12


def test_removing_a_branch_matches_a_case_built_without_it(case):
    """開放指定の B' が、その枝を持たないケースの B' と一致すること。

    :meth:`Case.without_branch` はケースそのものから枝を落とす別経路で
    ある。「行をゼロにする」実装と「枝を消す」実装が同じ行列を出す。
    """
    for key in [(4, 5), (7, 8)]:
        removed = susceptance_matrix(case, removed_branches=[key])
        rebuilt = susceptance_matrix(case.without_branch(key))
        assert np.abs(removed - rebuilt).max() < 1e-12


def test_zero_reactance_branch_is_rejected(case):
    """:math:`x=0` の枝が日本語の ValueError で止まること。"""
    broken = replace(
        case,
        branches=[replace(b, x=0.0) if b.key() == (4, 5) else b for b in case.branches],
        reference=None,
    )
    with pytest.raises(ValueError, match="リアクタンスがゼロ"):
        susceptance_matrix(broken)


# ----------------------------------------------------------------------
# 直流潮流
# ----------------------------------------------------------------------
def test_flows_satisfy_kirchhoff_at_every_non_slack_bus(case, base):
    """枝潮流の母線ごとの総和が、指定した注入に一致すること。

    :math:`A^{T} f = P` は直流潮流の解き方とは別の式（節点方程式そのもの）
    であり、期待値は :meth:`Case.bus_injection` から取る。slack 母線だけは
    方程式から外してあるので一致しない（次のテストで扱う）。
    """
    injections = incidence_matrix(case).T @ base.flows
    expected, _ = case.bus_injection()
    slack = case.index_of(base.slack)
    for i, bus in enumerate(case.buses):
        if i == slack:
            continue
        assert injections[i] == pytest.approx(expected[i], abs=1e-12), f"母線 {bus.id}"


def test_no_losses_and_the_slack_absorbs_the_difference(case, base):
    """損失がゼロで、slack 出力が「総需要 - 他機の出力」になること。

    参照解の発電の総和は総需要より損失 0.046410 p.u. だけ多い。直流では
    slack の行を落とすので、その余りは枝潮流に現れず slack の出力から
    差し引かれる。すなわち slack 出力は 3.15 - 1.63 - 0.85 = 0.67 で
    あって、交流の 0.716410 ではない。**この 0.046 の差が「直流には
    損失がない」という近似そのもの**である。
    """
    total_load = sum(bus.pd for bus in case.buses)
    other_generation = sum(
        pg for bus_id, (pg, _) in case.reference.generation.items() if bus_id != base.slack
    )
    slack_output = float((incidence_matrix(case).T @ base.flows)[case.index_of(base.slack)])

    assert total_load == pytest.approx(3.15, abs=1e-12)
    assert slack_output == pytest.approx(total_load - other_generation, abs=1e-12)
    assert slack_output == pytest.approx(0.67, abs=1e-12)
    assert slack_output != pytest.approx(case.reference.checks["slack_p"], abs=1e-3)
    assert abs(case.reference.checks["slack_p"] - slack_output) == pytest.approx(
        case.reference.checks["losses_pu"], abs=1e-6
    )


def test_slack_output_equals_total_load_when_no_one_else_generates(case):
    """他機を全停すると slack 出力が総需要ちょうどになること。

    損失がゼロであることの一番素直な形。号機の出力を dispatch で与えて
    確かめる（slack 母線の号機に何 MW を指定しても、その行は方程式から
    外れるので結果は変わらない）。
    """
    dispatch = {unit.name: 0.0 for unit in case.units}
    solution = dc_powerflow(case, dispatch=dispatch)
    slack_output = float(
        (incidence_matrix(case).T @ solution.flows)[case.index_of(solution.slack)]
    )
    assert slack_output == pytest.approx(3.15, abs=1e-12)


def test_dc_angles_are_close_to_but_not_equal_to_the_reference(case, base):
    """直流の位相が教科書の交流解に近いが一致はしないこと。

    位相の誤差は最大 0.52 deg（母線 2）である。近似が「使えるが正確では
    ない」ことを数値で押さえる。母線 1 は基準なので厳密にゼロ。
    """
    difference = np.abs(base.angle_deg - case.reference.angle_deg)
    assert difference[case.index_of(1)] == 0.0
    assert difference.max() == pytest.approx(0.516, abs=0.02)
    assert difference.max() > 0.1


def test_dc_flow_understates_the_thermal_loading_of_branch_4_5(case, base):
    """枝 4-5 で直流の P が交流の \\|S\\| より 47% 小さいこと。

    **熱容量の判定を直流潮流で行ってはいけない根拠**である。定格
    ``rate_a`` は皮相電力 \\|S\\| の制限であるのに対し、直流が持っている
    のは有効電力 P だけで、無効電力の分がまるごと抜けている。交流の値は
    教科書の潮流解と枝の 2x2 行列から直接計算するので、直流の実装とは
    独立である（許容差 1e-3 は参照解の 4 桁の丸め）。
    """
    branch = next(b for b in case.branches if b.key() == (4, 5))
    v = case.reference.voltage
    f, t = case.index_of(branch.from_bus), case.index_of(branch.to_bus)
    primitive = branch.primitive()
    s_ft = v[f] * np.conj(primitive[0, 0] * v[f] + primitive[0, 1] * v[t])
    s_tf = v[t] * np.conj(primitive[1, 0] * v[f] + primitive[1, 1] * v[t])
    ac_apparent = max(abs(s_ft), abs(s_tf))
    assert ac_apparent == pytest.approx(AC_APPARENT_4_5, abs=1e-3)

    dc_flow = base.flow_of((4, 5))
    assert dc_flow == pytest.approx(0.3803, abs=1e-3)

    error = (ac_apparent - dc_flow) / dc_flow
    assert error == pytest.approx(0.477, abs=0.01), "直流の過小評価は 47% 前後"

    # 負荷率でも同じ差が出る（rate_a = 1.00 なので値がそのまま負荷率）。
    dc_loading = base.loading("rate_a")[_position(case, (4, 5))]
    assert dc_loading == pytest.approx(0.380, abs=1e-3)
    assert AC_LOADING_4_5 - dc_loading > 0.17


def test_flows_are_the_ptdf_times_the_injection(case, base):
    """:math:`f = \\mathrm{PTDF}\\,P` が直流潮流の解と一致すること。

    連立方程式を解く経路と、感度行列を掛ける経路が同じ答えを出す。
    slack 母線の列はゼロなので、注入ベクトルの slack 成分は結果に
    効かない（**slack の注入は使われない**ことの別の見え方）。
    """
    injections, _ = case.bus_injection()
    assert np.abs(ptdf(case) @ injections - base.flows).max() < 1e-12

    tampered = injections.copy()
    tampered[case.index_of(base.slack)] += 99.0
    assert np.abs(ptdf(case) @ tampered - base.flows).max() < 1e-12


def test_phase_shift_is_ignored_by_the_dc_model(case, base):
    """位相調整角を入れても直流潮流が変わらないこと（既知の割り切り）。

    直流モデルは :attr:`Branch.shift_deg` を無視している。docstring に
    明記した割り切りを、テストとしても固定しておく。無視していることを
    忘れて位相調整器つきのケースに使うと、静かに間違った答えが出る。
    """
    shifted = replace(
        case,
        branches=[
            replace(b, shift_deg=5.0) if b.key() == (1, 4) else b for b in case.branches
        ],
    )
    solution = dc_powerflow(shifted)
    assert np.abs(solution.flows - base.flows).max() < 1e-12
    assert np.abs(susceptance_matrix(shifted) - susceptance_matrix(case)).max() < 1e-12


def test_opening_a_bridge_is_rejected_with_a_topological_explanation(case):
    """橋を開放した直流潮流が、島の中身を添えた日本語で止まること。"""
    with pytest.raises(ValueError, match="島に分かれる"):
        dc_powerflow(case, removed_branches=[(1, 4)])


def test_the_slack_can_be_chosen_by_bus_number(case, base):
    """``slack`` に母線番号を渡すと、その母線の位相がゼロになること。

    slack を負荷母線に移すと、位相の基準だけでなく **どの母線の注入を
    方程式から外すか**も変わるので、枝潮流そのものが変わる。「slack は
    位相の原点にすぎない」という誤解を潰すための確認である。
    """
    moved = dc_powerflow(case, slack=5)
    assert moved.slack == 5
    assert moved.theta[case.index_of(5)] == 0.0
    assert np.abs(moved.flows - base.flows).max() > 1e-3

    with pytest.raises(KeyError, match="母線 99"):
        dc_powerflow(case, slack=99)


def test_a_case_without_a_slack_bus_is_rejected(case):
    """slack 母線のないケースが日本語の ValueError で止まること。"""
    without_slack = replace(
        case,
        buses=[
            replace(bus, type=BusType.PV) if bus.type is BusType.SLACK else bus
            for bus in case.buses
        ],
    )
    with pytest.raises(ValueError, match="slack 母線がない"):
        dc_powerflow(without_slack)


def test_flow_of_ignores_the_order_of_the_bus_numbers(case, base):
    """``flow_of`` が母線番号の順序によらず同じ値を返すこと。"""
    assert base.flow_of((4, 5)) == base.flow_of((5, 4))
    assert base.flow_of((4, 5)) == base.flows[_position(case, (4, 5))]
    with pytest.raises(ValueError, match="ケース"):
        base.flow_of((1, 9))


def test_loading_uses_the_requested_rating(case, base):
    """負荷率が \\|P\\|/rate で、rate_a と rate_b を切り替えられること。"""
    rate_a = np.array([b.rate_a for b in case.branches])
    rate_b = np.array([b.rate_b for b in case.branches])
    assert np.abs(base.loading() - np.abs(base.flows) / rate_a).max() < 1e-12
    assert np.abs(base.loading("rate_b") - np.abs(base.flows) / rate_b).max() < 1e-12
    assert np.all(base.loading("rate_b") < base.loading("rate_a"))
    with pytest.raises(ValueError, match="rate_a"):
        base.loading("rate_c")


# ----------------------------------------------------------------------
# PTDF
# ----------------------------------------------------------------------
def test_ptdf_matches_numerical_differentiation(case, base):
    """PTDF が数値微分と一致すること。

    母線ごとに注入を 1 MW（0.01 p.u.、負荷を減らす形）だけずらして直流
    潮流を解き直し、枝潮流の変化を割る。直流潮流は厳密に線形なので
    差分の打切り誤差はゼロで、残るのは倍精度の丸めだけ（実測の最大差は
    3.3e-14）。許容差 1e-8 はそれに対して十分に余裕がある。
    """
    delta = 0.01
    expected = np.empty((case.n_branch, case.n_bus))
    for i, bus in enumerate(case.buses):
        perturbed = replace(
            case,
            buses=[
                replace(b, pd=b.pd - delta) if b.id == bus.id else b for b in case.buses
            ],
        )
        expected[:, i] = (dc_powerflow(perturbed).flows - base.flows) / delta

    assert np.abs(ptdf(case) - expected).max() < 1e-8


def test_ptdf_column_of_the_slack_bus_is_zero(case):
    """slack 母線の列が恒等的にゼロであること。

    自分に注入して自分から引き抜けば何も動かない。PTDF の定義に slack が
    埋め込まれていることの一番わかりやすい現れである。
    """
    for slack in (1, 2, 3):
        matrix = ptdf(case, slack=slack)
        assert np.abs(matrix[:, case.index_of(slack)]).max() < 1e-12


def test_ptdf_single_columns_depend_on_the_slack(case):
    """slack を変えると PTDF の単一列が **変わる**こと。

    これを確かめずに「slack に依存しない」だけを assert すると、全部
    ゼロを返す実装でも通ってしまう。実測では slack を変えると差は
    どの列でもちょうど 1.0 になる（引き抜き先が変わるため）。
    """
    columns = {slack: ptdf(case, slack=slack) for slack in (1, 2, 3)}
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        difference = np.abs(columns[a] - columns[b])
        assert difference.max() > 1e-2
        # どの母線の列も影響を受ける（一部の列だけが動くのではない）。
        assert difference.max(axis=0).min() > 1e-2


def test_ptdf_column_differences_do_not_depend_on_the_slack(case):
    """列の差（母線間の送電）が slack に **依存しない**こと。

    :math:`\\mathrm{PTDF}[:, i] - \\mathrm{PTDF}[:, j]` は「母線 i から
    母線 j へ 1 p.u. 送る」ことに対応し、slack が現れない。許容差 1e-12
    は倍精度の丸めの水準（実測は 1e-15 のオーダー）。
    """
    matrices = [ptdf(case, slack=slack) for slack in (1, 2, 3)]
    for i in range(case.n_bus):
        for j in range(case.n_bus):
            differences = [m[:, i] - m[:, j] for m in matrices]
            assert np.abs(differences[0] - differences[1]).max() < 1e-12
            assert np.abs(differences[0] - differences[2]).max() < 1e-12


def test_ptdf_of_the_triangle_matches_the_hand_calculation():
    """三角形の PTDF が並列回路の解析解に一致すること。

    母線 2 に 1 p.u. 注入して母線 1 から引き抜くと、直達路 ``1-2``
    （x=0.1）と迂回路 ``2-3-1``（x=0.2）にリアクタンスの逆比 2:1 で
    分かれる。枝の向きは from -> to が正なので、``1-2`` は :math:`-2/3`、
    ``2-3`` は :math:`+1/3`、``1-3`` は :math:`-1/3` になる。
    """
    triangle = _triangle()
    column = ptdf(triangle)[:, triangle.index_of(2)]
    assert column == pytest.approx([-2 / 3, 1 / 3, -1 / 3], abs=1e-12)


def test_ptdf_with_a_removed_branch_matches_a_case_built_without_it(case):
    """開放指定の PTDF が、その枝を持たないケースの PTDF と一致すること。

    開放した枝の行はゼロで残る（枝番号がずれない）。その行を除けば、
    最初からその枝がないケースの PTDF と同じである。
    """
    key = (5, 7)
    with_removal = ptdf(case, removed_branches=[key])
    row = _position(case, key)
    assert np.abs(with_removal[row]).max() < 1e-12

    rebuilt = ptdf(case.without_branch(key))
    assert np.abs(np.delete(with_removal, row, axis=0) - rebuilt).max() < 1e-12


# ----------------------------------------------------------------------
# LODF
# ----------------------------------------------------------------------
def test_lodf_rejects_the_bridges_with_a_topological_explanation(case):
    """橋があると LODF が日本語の ValueError で止まること。

    WSCC 9 母線の変圧器 3 本では分母 :math:`1 - \\mathrm{PTDF}[k,(m,n)]`
    が機械精度でゼロになる。これは数値の破綻ではなく「その枝が唯一の
    連絡路である」という位相の事実なので、メッセージにその旨と
    ``gridops.ybus.bridges()`` の案内が入っていることまで固定する。
    """
    assert bridges(case) == BRIDGES

    with pytest.raises(ValueError, match="橋") as excinfo:
        lodf(case)
    message = str(excinfo.value)
    assert "数値の破綻ではなく" in message
    assert "gridops.ybus.bridges()" in message
    for key in BRIDGES:
        assert str(key) in message

    # 分母が実際に機械精度のゼロであること（例外の理由の裏取り）。
    psi = ptdf(case) @ incidence_matrix(case).T
    for key in BRIDGES:
        assert abs(1.0 - psi[_position(case, key), _position(case, key)]) < 1e-12


def test_lodf_rejects_a_bridge_named_in_outages(case):
    """橋を明示的に列に指定した場合も止まること。"""
    with pytest.raises(ValueError, match=r"\(2, 7\)"):
        lodf(case, outages=[(4, 5), (2, 7)])


def test_an_unknown_outage_branch_is_rejected(case):
    """``outages`` にケースにない枝を書いたら止まること。

    黙って無視すると、NaN のままの列を「計算した」と思い込むことになる。
    """
    with pytest.raises(ValueError, match="にない"):
        lodf(case, outages=[(4, 5), (5, 9)])


def test_lodf_columns_that_were_not_requested_are_nan(case):
    """計算しなかった列が NaN で埋まること。

    ゼロで埋めると「事故の影響がゼロ」というもっともらしい誤答になる。
    NaN なら使った瞬間に結果全体が NaN になって気づける。
    """
    matrix = lodf(case, outages=case.contingencies)
    assert matrix.shape == (case.n_branch, case.n_branch)
    for key in BRIDGES:
        assert np.all(np.isnan(matrix[:, _position(case, key)]))
    for key in case.contingencies:
        assert np.all(np.isfinite(matrix[:, _position(case, key)]))


def test_lodf_diagonal_is_minus_one(case):
    """LODF の対角成分が :math:`-1` であること。

    開放した枝自身の潮流は事故前の値をちょうど打ち消してゼロになる。
    """
    matrix = lodf(case, outages=case.contingencies)
    for key in case.contingencies:
        k = _position(case, key)
        assert matrix[k, k] == -1.0


def test_lodf_predicts_the_post_contingency_flows(case, base):
    """LODF の予測が、枝を消して解き直した直流潮流と一致すること。

    :math:`f' = f + \\mathrm{LODF}[:, k] f_k` の行列-ベクトル積 1 回と、
    B' を組み直して解き直した結果を、ケースの N-1 候補 6 本すべてで
    突き合わせる。両者は式の出所が違う（補償定理 と 連立方程式）。
    許容差 1e-12 は倍精度の丸めの水準（実測の最大差は 4.8e-15）。
    """
    matrix = lodf(case, outages=case.contingencies)
    for key in case.contingencies:
        k = _position(case, key)
        predicted = base.flows + matrix[:, k] * base.flows[k]
        direct = dc_powerflow(case, removed_branches=[key]).flows
        assert np.abs(predicted - direct).max() < 1e-12, f"事故 {key}"
        assert predicted[k] == pytest.approx(0.0, abs=1e-12)


def test_post_contingency_flows_match_the_hand_calculation(case, base):
    """事故後潮流が、電流則だけで決まる値に一致すること。

    母線 5 / 6 / 8 は枝を 2 本しか持たない。片方を開放すれば、その母線の
    負荷は残る 1 本を丸ごと通る。**リアクタンスの値によらない**ので、
    LODF の実装からも B' からも独立した基準になる。
    """
    matrix = lodf(case, outages=case.contingencies)
    for outage, (observed, expected) in KIRCHHOFF_POST_FLOWS.items():
        k = _position(case, outage)
        predicted = base.flows + matrix[:, k] * base.flows[k]
        assert predicted[_position(case, observed)] == pytest.approx(
            expected, abs=1e-12
        ), f"事故 {outage} の枝 {observed}"


def test_lodf_does_not_depend_on_the_slack(case):
    """slack を付け替えても LODF が変わらないこと。

    :func:`lodf` は slack を引数に取らないので、母線種別を変えたケースを
    作って公開 API 越しに確かめる。LODF は枝の両端に対する PTDF の
    列の差だけで書けるので、slack の寄与は引き算で消える。許容差 1e-12
    は倍精度の丸めの水準。
    """
    matrices = []
    for bus_id in (1, 2, 3):
        moved = _with_slack_at(case, bus_id)
        assert dc_powerflow(moved).slack == bus_id
        matrices.append(lodf(moved, outages=case.contingencies))

    columns = [_position(case, key) for key in case.contingencies]
    for other in matrices[1:]:
        difference = np.abs(matrices[0][:, columns] - other[:, columns])
        assert difference.max() < 1e-12

    # 「変わらない」だけでは全部ゼロの実装でも通るので、値そのものが
    # 意味のある大きさであることも確かめる。
    assert np.abs(matrices[0][:, columns]).max() > 0.5


def test_lodf_of_the_triangle_matches_the_hand_calculation():
    """三角形の LODF が手計算に一致すること。

    枝 ``1-2`` を開放すれば、母線 2 への注入は迂回路を丸ごと通るしかない。
    向きを考えると ``2-3`` は :math:`-1`、``1-3`` は :math:`+1` である。
    """
    triangle = _triangle()
    matrix = lodf(triangle)
    assert matrix[:, 0] == pytest.approx([-1.0, -1.0, 1.0], abs=1e-12)
    assert np.abs(np.diag(matrix) + 1.0).max() < 1e-12


def test_lodf_does_not_depend_on_the_operating_point(case):
    """LODF が需要（動作点）に依存しないこと。

    LODF はトポロジーとリアクタンスだけで決まる。負荷を 1.7 倍にしても
    同じ行列が出るからこそ、一度作って使い回せる。
    """
    heavier = case.scaled(1.7)
    a = lodf(case, outages=case.contingencies)
    b = lodf(heavier, outages=case.contingencies)
    columns = [_position(case, key) for key in case.contingencies]
    assert np.abs(a[:, columns] - b[:, columns]).max() < 1e-12
