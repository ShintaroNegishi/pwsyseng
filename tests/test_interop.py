"""gridops から genstab への橋渡し（:mod:`gridops.interop`）の検証。

この橋が壊れると「解けているが間違っている」状態になりやすい。潮流解と
発電機の P, Q が食い違っていても計算は最後まで通り、波形も一見それらしく
見えるためである。そこで本ファイルは、実装の出力を実装で確かめるのでは
なく、次の 4 つの独立な基準と突き合わせる。

1. **原典の数値** — Anderson & Fouad の H, x'd, 内部起電力、および
   同じ原典から作られた genstab の ``cases/wscc9.yaml``
2. **手で書ける関係式** — 並列合成 x'd/n、減衰比 D/(2H) の保存
3. **genstab の別経路** — Kron 縮約から求めた事故前の電気出力が、
   こちらが渡した P と一致すること（縮約は S = V (YV)* を一切使わない）
4. **臨界事故除去時間** — genstab が自分のケースファイルから組んだ系と、
   gridops が自力で解いた潮流から組んだ系で CCT が一致すること

D = 0 の運転点では慣性中心の自由回転モードが原点に乗るので、固有値の
実部の符号は数値誤差で決まる。``genstab/eac.py`` はこれを「定態不安定」と
警告することがあるので、CCT を取る箇所では警告を握りつぶしている
（握りつぶしてよい理由そのものを
:func:`test_the_free_rotation_mode_sits_at_the_origin` で固定した）。
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gridops import load_case
from gridops.interop import (
    aggregate_plants,
    check_against_reference,
    to_genstab,
)
from gridops.powerflow import solve
from gridops.ybus import build_ybus

_HAS_GENSTAB = importlib.util.find_spec("genstab") is not None
requires_genstab = pytest.mark.skipif(
    not _HAS_GENSTAB, reason="genstab が入っていない環境では検証できない"
)

#: 原典 (Anderson & Fouad, 2003, Ch.2) の 3 機の諸元。
TEXTBOOK_INERTIA = {"G1": 23.64, "G2": 6.40, "G3": 3.01}
TEXTBOOK_XD_PRIME = {"G1": 0.0608, "G2": 0.1198, "G3": 0.1813}

#: 原典の内部起電力（ケースファイルの solution.checks と同じ値）。
TEXTBOOK_EMF = np.array([1.0566, 1.0502, 1.0170])
TEXTBOOK_EMF_ANGLE_DEG = np.array([2.2717, 19.7315, 13.1752])


def _genstab_case_path() -> Path | None:
    """genstab に同梱された wscc9.yaml を、パッケージの位置から引く。"""
    if not _HAS_GENSTAB:
        return None
    import genstab

    candidate = (
        Path(genstab.__file__).resolve().parents[2] / "cases" / "wscc9.yaml"
    )
    return candidate if candidate.is_file() else None


def _cct(system) -> float:
    """臨界事故除去時間 [s]。D = 0 で出うる「定態不安定」警告は無視する。

    多機系統には慣性中心の自由回転に対応する原点の固有値が必ずあり、
    その実部の符号は数値線形化の誤差で決まる。``eac`` はこれを拾って
    警告することがあるが、CCT の値そのものには影響しない。
    """
    from genstab import eac

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return eac.critical_clearing_time(
            system, t_end=5.0, tolerance=1e-4, upper_bound=1.0
        )


# ----------------------------------------------------------------------
# 共通のフィクスチャ
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def case():
    return load_case("wscc9")


@pytest.fixture(scope="module")
def solution(case):
    """自力で解いた事故前潮流解（参照解は使わない）。"""
    return solve(case)


@pytest.fixture(scope="module")
def system(case, solution):
    """genstab の系（原典に合わせて制動なし）。"""
    return to_genstab(case, solution, damping=0.0)


# ======================================================================
# 1. 号機の集約が原典の 3 機に戻ること
# ======================================================================
def test_aggregation_reproduces_the_textbook_three_machines(case, solution):
    """集約後の H と x'd が原典の値に戻ること。

    比較の相手は Anderson & Fouad の掲載値そのもの（H = 23.64 / 6.40 /
    3.01 s、x'd = 0.0608 / 0.1198 / 0.1813 p.u.）である。ケースファイルの
    号機の諸元はこの値を割り戻して作られているので、戻らなければ
    割り戻しか集約のどちらかが誤っている。許容差 1e-12 は、加算と
    逆数和にしか計算がないことによる（倍精度の丸めしか乗らない）。
    """
    plants = aggregate_plants(case, solution)
    assert [p["name"] for p in plants] == ["G1", "G2", "G3"]
    for plant in plants:
        assert plant["H"] == pytest.approx(
            TEXTBOOK_INERTIA[plant["name"]], rel=1e-12
        )
        assert plant["xd_prime"] == pytest.approx(
            TEXTBOOK_XD_PRIME[plant["name"]], rel=1e-12
        )


@requires_genstab
def test_aggregation_matches_the_genstab_case_file(case, solution):
    """集約後の諸元が genstab の cases/wscc9.yaml と一致すること。

    上のテストが掲載値の書き写しと一致することを見るのに対し、こちらは
    **別のパッケージが別のファイルから読んだ値** と突き合わせる。
    どちらか一方だけを直したときに気づけるようにするための二重化である。
    """
    path = _genstab_case_path()
    if path is None:
        pytest.skip("genstab の cases/wscc9.yaml が見つからない")
    from genstab.multimachine import load_case as genstab_load_case

    reference = genstab_load_case(path)
    plants = aggregate_plants(case, solution)
    assert [p["name"] for p in plants] == [g.name for g in reference.generators]
    for plant, generator in zip(plants, reference.generators):
        assert plant["bus"] == generator.bus
        assert plant["H"] == pytest.approx(generator.H, rel=1e-12)
        assert plant["xd_prime"] == pytest.approx(generator.xd_prime, rel=1e-12)


def test_transient_reactance_is_the_parallel_combination(case, solution):
    """同一諸元の n 台を並列合成すると x'd / n になること。

    ``1/x = Σ 1/x_i`` を実装とは別の形（n 台が同じ値なら x/n）で書いて
    確かめる。wscc9 は発電所ごとに号機の諸元が揃っているのでこの形が使える。
    """
    plants = {p["name"]: p for p in aggregate_plants(case, solution)}
    for name, units in case.plants().items():
        single = units[0].xd_prime
        assert all(u.xd_prime == single for u in units)
        assert plants[name]["xd_prime"] * len(units) == pytest.approx(
            single, rel=1e-15
        )
        assert plants[name]["H"] == pytest.approx(
            units[0].h * len(units), rel=1e-15
        )


def test_damping_is_summed_so_the_damping_ratio_is_preserved(case, solution):
    """制動係数を加算すると 1 台あたりの D/(2H) が保たれること。

    動揺方程式 ``2H dω/dt = Pm - Pe - D ω`` の減衰の効き方は D/(2H) で
    決まる。H を加算したのに D を加算しないと、号機に分けただけで減衰が
    号機数分の 1 に薄まってしまう。ここでは集約後の D/(2H) が 1 台の
    d/(2h) と一致することを確かめる。
    """
    plants = {p["name"]: p for p in aggregate_plants(case, solution)}
    for name, units in case.plants().items():
        single = units[0]
        assert plants[name]["D"] / (2.0 * plants[name]["H"]) == pytest.approx(
            single.d / (2.0 * single.h), rel=1e-12
        )
        assert plants[name]["D"] == pytest.approx(
            sum(u.d for u in units), rel=1e-15
        )


def test_aggregated_power_comes_from_the_power_flow_not_from_the_units(
    case, solution
):
    """集約後の P, Q が潮流解の発電と一致すること。

    slack 母線の発電は損失を引き受けた後の値でなければならない。
    ケースファイルの checks（slack_p = 0.716410、slack_q = 0.270459）と
    突き合わせる。許容差 1e-6 は checks の記載桁数（小数 6 桁）。
    PV 母線の G2, G3 は指定注入そのものなので厳密に 1.630 / 0.850。
    """
    plants = {p["name"]: p for p in aggregate_plants(case, solution)}
    checks = case.reference.checks
    assert plants["G1"]["P"] == pytest.approx(float(checks["slack_p"]), abs=1e-6)
    assert plants["G1"]["Q"] == pytest.approx(float(checks["slack_q"]), abs=1e-6)
    assert plants["G2"]["P"] == pytest.approx(1.630, abs=1e-9)
    assert plants["G3"]["P"] == pytest.approx(0.850, abs=1e-9)
    # MW 表記は base_mva 倍になっているだけ。
    assert plants["G2"]["p_mw"] == pytest.approx(163.0, abs=1e-7)


def test_the_power_balance_closes_on_the_losses(case, solution):
    """発電の合計 - 負荷の合計 = 損失 になること。

    集約が電力を作りも消しもしていないことの検算。損失はケースファイルの
    checks.losses_pu = 0.046410 と比べる（記載桁数 6 桁）。
    """
    plants = aggregate_plants(case, solution)
    generated = sum(p["P"] for p in plants)
    load = sum(bus.pd for bus in case.buses)
    assert generated - load == pytest.approx(
        float(case.reference.checks["losses_pu"]), abs=1e-6
    )


# ======================================================================
# 2. 内部起電力が教科書値に戻ること
# ======================================================================
@requires_genstab
def test_internal_emf_matches_the_textbook(system):
    """自力の潮流解から組んだ E∠δ0 が原典の値と一致すること。

    原典の掲載値は |E| = 1.0566 / 1.0502 / 1.0170 p.u.、
    δ0 = 2.2717 / 19.7315 / 13.1752 deg。位相はラジアンで 1e-3 以内
    （実測の最大差 1.53e-4 rad = 8.8e-3 deg）に収まる。

    度で測ると G3 だけ 8.8e-3 deg 離れるが、これは実装の誤りではなく
    **原典の表そのものの丸め** である。次のテストがその根拠を示す。
    """
    assert system.emf_magnitude == pytest.approx(TEXTBOOK_EMF, abs=1e-3)
    assert system._delta0 == pytest.approx(
        np.radians(TEXTBOOK_EMF_ANGLE_DEG), abs=1e-3
    )
    # 度で見ても genstab 自身のテストと同じ許容差に収まる。
    assert np.degrees(system._delta0) == pytest.approx(
        TEXTBOOK_EMF_ANGLE_DEG, abs=1e-2
    )


def test_the_textbook_table_is_not_self_consistent_to_better_than_1e_2_deg(case):
    """原典の V と P, Q から E を組み直しても掲載の δ0 には戻らないこと。

    E = V + j x'd (S/V)* に **原典の掲載値だけ** を入れて計算すると、
    G3 の位相は 13.1671 deg となり掲載の 13.1752 deg から 8.1e-3 deg
    ずれる。すなわち 1e-2 deg の残差は潮流計算の精度ではなく、掲載表が
    V を 4 桁・P, Q を 3 桁で丸めていることに由来する。

    許容差の根拠を「実装がそこまでしか合わないから」ではなく
    「出典がそこまでしか決まっていないから」に置き換えるためのテスト。
    """
    reference = case.reference
    residual = []
    for k, (name, bus) in enumerate((("G1", 1), ("G2", 2), ("G3", 3))):
        index = case.index_of(bus)
        voltage = reference.voltage[index]
        p, q = reference.generation[bus]
        emf = voltage + 1j * TEXTBOOK_XD_PRIME[name] * np.conj(
            complex(p, q) / voltage
        )
        residual.append(
            abs(math.degrees(np.angle(emf)) - TEXTBOOK_EMF_ANGLE_DEG[k])
        )
    residual = np.array(residual)
    # 掲載値だけで閉じても G3 は 5e-3 deg 以上ずれる。
    assert residual[2] > 5e-3
    # それでも 1e-2 deg には収まる（掲載桁数で説明できる範囲）。
    assert residual.max() < 1e-2


# ======================================================================
# 3. genstab 側から見た整合性
# ======================================================================
@requires_genstab
def test_genstab_ybus_matches_the_gridops_ybus(case, system):
    """genstab が組んだ枝だけの Ybus が gridops の Ybus と一致すること。

    2 つの独立な実装（足し込みの順序も型も違う）を突き合わせる。
    ``include_loads=False`` / ``include_shunts=False`` で条件を揃える。
    倍精度で厳密一致するはずなので許容差は 1e-14。
    """
    expected = build_ybus(case, include_shunts=False)
    actual = system.network.ybus(include_loads=False)
    assert np.abs(actual - expected).max() < 1e-14


@requires_genstab
def test_prefault_electrical_power_equals_the_scheduled_output(system):
    """Kron 縮約から求めた事故前の電気出力が、渡した P と一致すること。

    こちらは S = V (YV)* で P を作ったが、genstab は発電機内部母線を
    足してから Kron 縮約し、``Pe = Σ E_i E_j (G cos + B sin)`` で出力を
    出す。**経路がまったく違う**ので、一致は運転点が真に平衡点である
    ことの独立な証拠になる。一致は解析的に厳密（E (I)* の実部は
    V (I)* の実部に等しい。j x'd |I|^2 が純虚数だから）なので、
    許容差は倍精度の丸めだけを見込んだ 1e-12。
    """
    from genstab.events import Stage
    from genstab.multimachine import electrical_power

    power = electrical_power(
        system.reduced_matrix(Stage.PRE),
        system.emf_magnitude,
        system.initial_state()[0::2],
    )
    scheduled = np.array([g.P for g in system.generators])
    assert power == pytest.approx(scheduled, abs=1e-12)


@requires_genstab
def test_network_injections_reproduce_generation_and_load(case, system):
    """genstab 側で数え直した母線注入が、発電 - 負荷に一致すること。

    genstab の :meth:`MultiMachineNetwork.power_injections` は自前の
    Ybus と潮流解の電圧から注入を計算する。ケースの負荷と、こちらが
    渡した発電機の P, Q だけから組んだ期待値と突き合わせる。
    """
    injections = system.network.power_injections()
    generation = {g.bus: complex(g.P, g.Q) for g in system.generators}
    for bus in case.buses:
        expected = generation.get(bus.id, 0j) - complex(bus.pd, bus.qd)
        assert abs(injections[bus.id] - expected) < 1e-12, f"母線 {bus.id}"


@requires_genstab
def test_the_stability_layer_is_copied_into_the_fault_schedule(case, system):
    """事故スケジュールと基準値がケースの stability 層どおりであること。"""
    fault = case.stability["fault"]
    assert system.fault.t_fault == pytest.approx(float(fault["t_fault"]))
    assert system.fault.t_clear == pytest.approx(float(fault["t_clear"]))
    assert list(system.faulted_buses) == [7]
    assert [tuple(b) for b in system.tripped_branches] == [(5, 7)]
    # WSCC 9 母線は 60 Hz。ここを 50 Hz のまま渡すと CCT が 2 割ずれる。
    assert system.base.frequency_hz == pytest.approx(60.0)
    assert system.base.s_base_mva == pytest.approx(100.0)


# ======================================================================
# 4. 臨界事故除去時間
# ======================================================================
@pytest.fixture(scope="module")
def cct_by_damping(case, solution):
    """制動係数ごとの CCT（重いので 1 度だけ計算する）。"""
    if not _HAS_GENSTAB:
        pytest.skip("genstab が入っていない環境では検証できない")
    return {
        d: _cct(to_genstab(case, solution, damping=d)) for d in (0.0, 0.5, 2.0)
    }


@requires_genstab
def test_cct_matches_the_system_built_by_genstab_itself(cct_by_damping):
    """genstab が自分のケースから組んだ系と CCT が一致すること。

    * 系 A: ``genstab.multimachine.load_case`` が ``cases/wscc9.yaml``
      から組んだ系（潮流解を **入力として** 読む。制動 D = 0）
    * 系 B: gridops が自力で潮流を解き、:func:`to_genstab` で組んだ系
      （``damping=0.0`` で条件を揃える。同梱ケースは D = 2.0 なので
      指定しないと比較にならない）

    2 つの系は入力の出どころも号機の数も違うのに、CCT は一致する。
    許容差 1e-4 秒は :func:`eac.critical_clearing_time` の二分探索の
    打ち切り幅そのもの（実測では両者とも 0.161133 s で完全に一致した）。
    """
    path = _genstab_case_path()
    if path is None:
        pytest.skip("genstab の cases/wscc9.yaml が見つからない")
    from genstab.multimachine import load_case as genstab_load_case

    native = _cct(genstab_load_case(path))
    assert cct_by_damping[0.0] == pytest.approx(native, abs=1e-4)
    # 値が退化していない（0 でも探索上限でもない）ことも確かめる。
    assert 0.10 < native < 0.30


@requires_genstab
def test_damping_extends_the_cct(cct_by_damping):
    """制動係数を上げると CCT が伸びること。

    制動は事故中に得た運動エネルギーを吸収するので、同じ角度に達する
    までの時間が長くなる。D = 0 / 0.5 / 2.0 で単調に増えることを見る
    （実測 0.1611 / 0.1640 / 0.1699 s）。差は探索の打ち切り幅 1e-4 s
    より 1 桁以上大きい。
    """
    assert cct_by_damping[0.0] < cct_by_damping[0.5] < cct_by_damping[2.0]
    assert cct_by_damping[2.0] - cct_by_damping[0.0] > 1e-3


@requires_genstab
def test_the_free_rotation_mode_sits_at_the_origin(case, solution):
    """慣性中心の自由回転モードが原点にあること（D の値によらない）。

    多機系統には無限大母線が無いので、全機が一緒に回るモードの固有値は
    必ず原点に来る。したがって「固有値の実部の最大」は制動の有無に
    かかわらずほぼゼロで、D = 0 では符号が数値誤差で決まる。
    ``genstab/eac.py`` が実部 > 1e-6 で「定態不安定」と警告するのは
    この原点のモードを拾っているためで、CCT の値そのものは正しい。
    テストで D = 0 を使うときに警告を握りつぶしてよい根拠がこれである。
    """
    from genstab.smallsignal import analyze

    for damping in (0.0, 0.5, 2.0):
        eigenvalues = analyze(to_genstab(case, solution, damping=damping)).eigenvalues
        assert np.abs(eigenvalues).min() < 1e-3, f"D={damping} で原点のモードが無い"
        assert abs(np.max(eigenvalues.real)) < 1e-6, f"D={damping}"
    # 振動モードそのものは制動を入れれば確かに左半面へ動く。
    damped = analyze(to_genstab(case, solution, damping=2.0)).eigenvalues
    oscillatory = damped[np.abs(damped.imag) > 1.0]
    assert oscillatory.size >= 2
    assert np.max(oscillatory.real) < -1e-3


# ======================================================================
# 5. dispatch の扱い
# ======================================================================
@requires_genstab
def test_offline_units_do_not_contribute_inertia(case, solution):
    """停止中の号機が慣性にも過渡リアクタンスにも寄与しないこと。

    G1 の 3 号機のうち 1 台を停止させると、H は 23.64 → 15.76 s（2/3）に
    減り、x'd は 0.0608 → 0.0912 p.u.（3/2 倍）に増える。慣性が減って
    背後リアクタンスが増えるので、この母線は事故に弱くなる。
    起動停止計画の結果を安定度に渡すときの要点である。
    """
    dispatch = {
        "G1-1": 40.0, "G1-2": 31.641, "G1-3": 0.0,
        "G2-1": 90.0, "G2-2": 73.0,
        "G3-1": 50.0, "G3-2": 35.0,
    }
    solved = solve(case, dispatch=dispatch)
    plants = {p["name"]: p for p in aggregate_plants(case, solved, dispatch=dispatch)}
    assert plants["G1"]["n_units"] == 2
    assert plants["G1"]["units"] == ("G1-1", "G1-2")
    assert plants["G1"]["H"] == pytest.approx(2.0 * 7.880, rel=1e-12)
    assert plants["G1"]["xd_prime"] == pytest.approx(0.1824 / 2.0, rel=1e-12)
    assert plants["G2"]["H"] == pytest.approx(TEXTBOOK_INERTIA["G2"], rel=1e-12)
    assert plants["G3"]["H"] == pytest.approx(TEXTBOOK_INERTIA["G3"], rel=1e-12)


def test_the_slack_bus_output_comes_from_the_solution_not_from_the_dispatch(
    case, solution
):
    """slack 母線では dispatch の合計ではなく潮流解の発電を使うこと。

    slack は損失を引き受けるので、起動停止計画が置いた出力とは必ず
    ずれる。ここでは dispatch が母線 1 に 120 MW を置いているのに、
    集約後の P が潮流解の 71.64 MW になることを確かめる。dispatch を
    そのまま内部起電力に使うと、事故前から角度が動き出す系ができる。
    """
    dispatch = {
        "G1-1": 60.0, "G1-2": 60.0, "G1-3": 0.0,
        "G2-1": 90.0, "G2-2": 73.0,
        "G3-1": 50.0, "G3-2": 35.0,
    }
    solved = solve(case, dispatch=dispatch)
    plants = {p["name"]: p for p in aggregate_plants(case, solved, dispatch=dispatch)}
    assert plants["G1"]["p_mw"] == pytest.approx(71.641, abs=1e-2)
    assert plants["G1"]["p_mw"] != pytest.approx(120.0, abs=1.0)


def test_a_dispatch_inconsistent_with_the_solution_is_rejected(case, solution):
    """潮流解と食い違う出力表を渡すと日本語の例外で止まること。

    契約の Notes「潮流解と発電機の P, Q は必ずセットで渡すこと」を
    実行時に効かせている。ここでは母線 2 の 2 号機を止めた出力表を、
    2 台とも運転している潮流解に対して渡している。
    """
    dispatch = {
        "G1-1": 60.0, "G1-2": 11.641, "G1-3": 0.0,
        "G2-1": 90.0, "G2-2": 0.0,
        "G3-1": 50.0, "G3-2": 35.0,
    }
    with pytest.raises(ValueError, match="解けているが間違っている"):
        aggregate_plants(case, solution, dispatch=dispatch)


# ======================================================================
# 6. 変換できないデータを止めること
# ======================================================================
@requires_genstab
def test_a_transformer_tap_is_rejected(case, solution):
    """tap != 1 の枝があると日本語の ValueError で止まること。

    genstab の Branch は素の π 型等価回路しか持たない。黙って tap を
    落とすと、事故前潮流と食い違う Ybus で動揺方程式を解くことになる。
    """
    branches = [
        replace(b, tap=1.05) if b.key() == (1, 4) else b for b in case.branches
    ]
    tapped = replace(case, branches=branches)
    with pytest.raises(ValueError, match="タップ比と位相調整角を持たない"):
        to_genstab(tapped, solution)


@requires_genstab
def test_a_phase_shifter_is_rejected(case, solution):
    """shift_deg != 0 の枝も同じ理由で止まること。"""
    branches = [
        replace(b, shift_deg=5.0) if b.key() == (1, 4) else b for b in case.branches
    ]
    shifted = replace(case, branches=branches)
    with pytest.raises(ValueError, match="タップ比と位相調整角を持たない"):
        to_genstab(shifted, solution)


def test_a_unit_without_inertia_is_rejected(case, solution):
    """h を持たない号機があると日本語の ValueError で止まること。

    潮流計算や起動停止計画には h も x'd も要らないので、安定度に
    渡す段になって初めて足りないことが分かる。例外文にその区別を
    書いてある。
    """
    units = [replace(u, h=None) if u.name == "G2-1" else u for u in case.units]
    broken = replace(case, units=units)
    with pytest.raises(ValueError, match="慣性定数 h が設定されていない"):
        aggregate_plants(broken, solution)


def test_a_unit_without_transient_reactance_is_rejected(case, solution):
    """xd_prime を持たない号機も同じく止まること。"""
    units = [
        replace(u, xd_prime=None) if u.name == "G3-2" else u for u in case.units
    ]
    broken = replace(case, units=units)
    with pytest.raises(ValueError, match="過渡リアクタンス xd_prime が設定されていない"):
        aggregate_plants(broken, solution)


def test_a_plant_spread_over_two_buses_is_rejected(case, solution):
    """同じ plant の号機が別の母線にあると止まること。"""
    units = [replace(u, bus=3) if u.name == "G1-3" else u for u in case.units]
    broken = replace(case, units=units)
    with pytest.raises(ValueError, match="複数の母線に分かれている"):
        aggregate_plants(broken, solution)


def test_a_non_converged_solution_is_rejected(case, solution):
    """収束していない潮流解を渡すと止まること。"""
    broken = replace(solution, converged=False)
    with pytest.raises(ValueError, match="収束していない潮流解"):
        aggregate_plants(case, broken)


def test_a_solution_of_the_wrong_length_is_rejected(case):
    """母線数と合わない電圧配列を渡すと止まること。"""
    with pytest.raises(ValueError, match="母線数"):
        aggregate_plants(case, np.ones(5, dtype=complex))


def test_missing_genstab_points_at_the_editable_install(case, solution, monkeypatch):
    """genstab が import できないとき、導入方法を案内する ImportError になること。

    ``sys.modules['genstab'] = None`` は import 機構に「この名前は
    読み込み済みで、しかも中身が無い」と伝える書き方で、genstab を
    アンインストールせずに未導入の環境を再現できる。
    """
    monkeypatch.setitem(sys.modules, "genstab", None)
    with pytest.raises(ImportError, match=re.escape("pip install -e .")):
        to_genstab(case, solution)


def test_importing_interop_does_not_require_genstab(monkeypatch):
    """genstab が無くても gridops.interop 自体は import できること。

    モジュールの先頭で genstab を import してしまうと、安定度を扱わない
    回まで巻き添えで動かなくなる。集約と答え合わせの表は genstab 抜きで
    使えなければならない。
    """
    monkeypatch.setitem(sys.modules, "genstab", None)
    monkeypatch.delitem(sys.modules, "gridops.interop", raising=False)
    import importlib

    module = importlib.import_module("gridops.interop")
    case = load_case("wscc9")
    plants = module.aggregate_plants(case, case.reference)
    assert [p["name"] for p in plants] == ["G1", "G2", "G3"]
    assert module.check_against_reference(case, case.reference)


# ======================================================================
# 7. 母線シャントの読み替え
# ======================================================================
@requires_genstab
def test_bus_shunts_become_equivalent_constant_impedance_loads(case):
    """母線シャントを負荷に読み替えても Ybus が 1 ビットも変わらないこと。

    genstab には母線シャントの入れ物が無いが、シャントはもともと定
    インピーダンスなので ``P = gs|V|^2``, ``Q = -bs|V|^2`` の負荷として
    厳密に等価である。期待値は gridops の Ybus（シャント込み）に、
    ケースの **真の負荷だけ** をアドミタンスに直して足したもの。
    調相コンデンサが負の Q を持つ負荷として現れることも確かめる。
    """
    buses = [
        replace(b, gs=0.05, bs=0.30) if b.id == 5 else b for b in case.buses
    ]
    shunted = replace(case, buses=buses, reference=None)
    solved = solve(shunted)
    system = to_genstab(shunted, solved, damping=0.0)

    voltage = solved.voltage
    expected = build_ybus(shunted).astype(complex)
    for i, bus in enumerate(shunted.buses):
        if bus.pd or bus.qd:
            expected[i, i] += np.conj(complex(bus.pd, bus.qd)) / abs(voltage[i]) ** 2
    assert np.abs(system.network.ybus() - expected).max() < 1e-14

    load5 = next(l for l in system.network.loads if l.bus == 5)
    magnitude_squared = abs(voltage[shunted.index_of(5)]) ** 2
    assert load5.P == pytest.approx(1.25 + 0.05 * magnitude_squared, rel=1e-12)
    assert load5.Q == pytest.approx(0.50 - 0.30 * magnitude_squared, rel=1e-12)
    assert load5.Q < 0.50  # コンデンサが無効電力を供給している


# ======================================================================
# 8. 制動係数の上書き
# ======================================================================
@requires_genstab
def test_damping_can_be_overridden_globally_and_per_plant(case, solution):
    """damping で全機一括にも発電所ごとにも上書きできること。"""
    uniform = to_genstab(case, solution, damping=1.5)
    assert [g.D for g in uniform.generators] == [1.5, 1.5, 1.5]

    per_plant = to_genstab(case, solution, damping={"G1": 1.0, "G2": 2.0, "G3": 3.0})
    assert [g.D for g in per_plant.generators] == [1.0, 2.0, 3.0]

    # 既定はケースの号機の d の合計（G1 は 3 台 x 2.0）。
    default = to_genstab(case, solution)
    assert [g.D for g in default.generators] == [6.0, 4.0, 4.0]

    with pytest.raises(ValueError, match="damping に発電所"):
        to_genstab(case, solution, damping={"G1": 1.0})


# ======================================================================
# 9. 答え合わせの表
# ======================================================================
def test_check_against_reference_returns_an_english_table(case, solution):
    """表が例外なく返り、ヘッダが英語（ASCII）であること。

    作図の軸ラベルを英語にする方針と揃えている。表をそのまま notebook の
    出力や図のキャプションに貼れるようにするためである。
    """
    table = check_against_reference(case, solution)
    lines = table.splitlines()
    headers = [
        line
        for line in lines
        if line.startswith(("Bus ", "Machine ", "Quantity", "Internal EMF",
                            "Scalar checks"))
    ]
    assert len(headers) >= 5
    for line in headers:
        assert line.isascii(), f"ヘッダに非 ASCII 文字がある: {line}"
    for column in ("|V| solved", "|V| ref", "d|V|", "angle solved", "angle ref",
                   "delta solved", "Solved", "Reference", "Difference"):
        assert column in table, column
    # 母線 9 行と発電機 3 行が並ぶ。
    assert sum(1 for line in lines if re.match(r"^\d+\s", line)) == case.n_bus
    assert sum(1 for line in lines if re.match(r"^G\d\s", line)) == 3
    # checks が全部埋まっている（n/a が残っていない）。
    assert "n/a" not in table


def test_check_against_reference_reports_differences_at_the_rounding_scale(
    case, solution
):
    """表に載る差が掲載桁数で説明できる大きさであること。

    参照解は 4 桁なので |V| と位相の差は 5e-5 の桁までしか詰まらない。
    表の数値を読み直す代わりに、同じ量をここで独立に計算して桁を確かめる。
    """
    table = check_against_reference(case, solution)
    reference = case.reference
    magnitude_gap = np.abs(np.abs(solution.voltage) - reference.v).max()
    angle_gap = np.abs(
        np.degrees(np.angle(solution.voltage)) - reference.angle_deg
    ).max()
    # 掲載 4 桁の丸めは |dV| <= 5e-5、位相 <= 5e-5 deg しか決めない。
    assert magnitude_gap < 1e-4
    assert angle_gap < 1e-4

    # 表の max 行に載っている数字が、上で独立に計算した値と一致すること。
    row = next(line for line in table.splitlines() if line.startswith("max"))
    printed = [float(token) for token in row.split()[1:]]
    assert len(printed) == 2
    assert printed[0] == pytest.approx(magnitude_gap, rel=5e-3)
    assert printed[1] == pytest.approx(angle_gap, rel=5e-3)


def test_check_against_reference_requires_a_reference_solution(case, solution):
    """参照解を持たないケースでは日本語の ValueError になること。"""
    bare = replace(case, reference=None)
    with pytest.raises(ValueError, match="次の層がない"):
        check_against_reference(bare, solution)


# ======================================================================
# 10. 参照解をそのまま渡す経路
# ======================================================================
@requires_genstab
def test_the_reference_solution_can_be_used_directly(case):
    """参照解（教科書の潮流解）をそのまま渡しても系が組めること。

    潮流計算をまだ習っていない回でも安定度の話ができるようにするための
    経路である。丸めた電圧から組むので内部起電力は自力解より粗くなるが、
    掲載桁数の範囲では原典に一致する。
    """
    system = to_genstab(case, case.reference, damping=0.0)
    assert system.emf_magnitude == pytest.approx(TEXTBOOK_EMF, abs=1e-3)
    assert np.degrees(system._delta0) == pytest.approx(
        TEXTBOOK_EMF_ANGLE_DEG, abs=1e-2
    )


@requires_genstab
def test_a_mapping_of_bus_voltages_is_accepted(case, solution, system):
    """``{母線番号: 複素電圧}`` の写像でも同じ系が組めること。

    母線の並び順と母線番号を取り違える事故を防ぐための経路。番号で
    引くので、辞書の順序を入れ替えても結果は変わらない。
    """
    voltages = {
        bus.id: complex(solution.voltage[i]) for i, bus in enumerate(case.buses)
    }
    shuffled = dict(reversed(list(voltages.items())))
    other = to_genstab(case, shuffled, damping=0.0)
    assert other.emf_magnitude == pytest.approx(system.emf_magnitude, abs=1e-15)
    assert other._delta0 == pytest.approx(system._delta0, abs=1e-15)
