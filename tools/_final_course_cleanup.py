from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def save(relative: str, text: str) -> None:
    (ROOT / relative).write_text(text, encoding="utf-8")
    print(f"updated: {relative}")


def replace_once(relative: str, old: str, new: str) -> None:
    text = load(relative)
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{relative}: expected exactly one match, found {count}: {old[:120]!r}"
        )
    save(relative, text.replace(old, new, 1))


def replace_all(relative: str, old: str, new: str, *, minimum: int = 1) -> None:
    text = load(relative)
    count = text.count(old)
    if count < minimum:
        raise RuntimeError(
            f"{relative}: expected at least {minimum} matches, found {count}: {old!r}"
        )
    save(relative, text.replace(old, new))


def sub_once(relative: str, pattern: str, replacement: str) -> None:
    text = load(relative)
    updated, count = re.subn(
        pattern,
        lambda _match: replacement,
        text,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(
            f"{relative}: expected exactly one regex match, found {count}: {pattern[:140]!r}"
        )
    save(relative, updated)


# -----------------------------------------------------------------------------
# Notebook 08: distinguish solar PV from PV buses and define inertia aggregation.
# -----------------------------------------------------------------------------
replace_once(
    "notebooks/src/08_reserve_and_vre.py",
    "# - 起動している同期機の **慣性の合計** が減ることを定量化する（第 12 回への橋）",
    "# - 起動している同期機の **共通基準換算済みの慣性寄与の合計**が減ることを定量化する（第 12 回への橋）",
)
replace_all("notebooks/src/08_reserve_and_vre.py", "runs_pv", "runs_solar")
replace_all(
    "notebooks/src/08_reserve_and_vre.py", "capacities", "solar_capacities"
)
replace_all(
    "notebooks/src/08_reserve_and_vre.py",
    'label=f"PV {cap:.0f} MW"',
    'label=f"solar PV {cap:.0f} MW"',
)
replace_all(
    "notebooks/src/08_reserve_and_vre.py",
    'print(f"PV {cap:5.0f} MW:',
    'print(f"太陽光発電 {cap:5.0f} MW:',
)
replace_once(
    "notebooks/src/08_reserve_and_vre.py",
    "# PV 300 MW では 69.4 MW/h に達します。",
    "# 太陽光発電 300 MW では 69.4 MW/h に達します。",
)
replace_once(
    "notebooks/src/08_reserve_and_vre.py",
    "print(f\"{'PV [MW]':>8} {'cost [JPY]':>14} {'spill [MWh]':>12} \"",
    "print(f\"{'solar PV [MW]':>13} {'cost [JPY]':>14} {'spill [MWh]':>12} \"",
)
replace_once(
    "notebooks/src/08_reserve_and_vre.py",
    "# TODO(L3): 下のコードが PV 0 MW と 300 MW の系統慣性 H(t) を計算する。",
    "# TODO(L3): 下のコードが太陽光発電 0 MW と 300 MW の系統慣性 H(t) を計算する。",
)
replace_once(
    "notebooks/src/08_reserve_and_vre.py",
    "#               加速度がどうなるか。臨界事故除去時間 CCT（第 12 回）は",
    "#               加速度がどうなるか。臨界事故除去時間（Critical Clearing Time: CCT、第 12 回）は",
)
replace_once(
    "notebooks/src/08_reserve_and_vre.py",
    "# - この日の最悪の時刻には、母線 2 の LNG コンバインドサイクルが **1 台も",
    "# - この日の最悪の時刻には、母線 2 の液化天然ガス（Liquefied Natural Gas: LNG）コンバインドサイクルが **1 台も",
)

# -----------------------------------------------------------------------------
# Notebook 09: define PI rather than leaving an unexplained abbreviation.
# -----------------------------------------------------------------------------
replace_once(
    "notebooks/src/09_security.py",
    "# - **熱容量だけを見ると最悪の N-1 を見落とす**ことを実データで確かめ、性能指数 PI が",
    "# - **熱容量だけを見ると最悪の N-1 を見落とす**ことを実データで確かめ、性能指数（Performance Index: PI）が",
)
replace_once(
    "notebooks/src/09_security.py",
    "# ## 6. 性能指数 PI とその masking",
    "# ## 6. 性能指数（PI）とマスキング（masking）",
)

# -----------------------------------------------------------------------------
# Notebook 18: LOLE is an expectation, not an event probability or count.
# -----------------------------------------------------------------------------
replace_once(
    "notebooks/src/18_adequacy.py",
    "# - 容量停止確率表（Capacity Outage Probability Table: COPT）を畳み込みで組み、**2 通りの独立な方法で検算する**",
    "# - 強制停止率（Forced Outage Rate: FOR）から容量停止確率表（Capacity Outage Probability Table: COPT）を畳み込みで組み、**2 通りの独立な方法で検算する**",
)
replace_once(
    "notebooks/src/18_adequacy.py",
    "# 第 09 回の答えは「枝 5-7 と 7-8 が拘束する」という**是非**、この回の答えは「年に\n# 何時間足りないか」という**確率**です（本モジュールは**送電網を見ません**）。決定論的な",
    "# 第 09 回の答えは「枝 5-7 と 7-8 が拘束する」という**是非**、この回の答えは「年に\n# 何時間、供給不足状態になると期待されるか」という**期待値**です（本モジュールは**送電網を見ません**）。決定論的な",
)
replace_once(
    "notebooks/src/18_adequacy.py",
    "# 解析解を含むか**で行います。精度の目安は $\\beta = \\sqrt{(1-p)/(pN)}$ で $1/\\sqrt{N}$",
    "# 解析解を含むか**で行います。以下では信頼区間を confidence interval（CI）と表記します。\n# 精度の目安は $\\beta = \\sqrt{(1-p)/(pN)}$ で $1/\\sqrt{N}$",
)
replace_once(
    "notebooks/src/18_adequacy.py",
    "# ## 8. 容量価値 (ELCC)",
    "# ## 8. 等価需要負担能力（Effective Load Carrying Capability: ELCC）",
)

# -----------------------------------------------------------------------------
# Notebook 19: define abbreviations in the actual narrative.
# -----------------------------------------------------------------------------
replace_once(
    "notebooks/src/19_integrated.py",
    "# - 起動停止 → 経済配分 → 潮流 → N-1 → SCED → 過渡安定度 → 年間信頼度 を通しで回し、\n#   各段で新たに判明することと、そのために **前の段へ戻る判断**を体験する\n# - 略語は、セキュリティ制約付き経済配分（SCED）、臨界事故除去時間（CCT）、\n#   供給支障時間期待値（LOLE）として初出時に定義する",
    "# - 起動停止 → 経済配分 → 潮流 → N-1 → セキュリティ制約付き経済配分\n#   （Security-Constrained Economic Dispatch: SCED）→ 過渡安定度 → 年間の供給支障時間期待値\n#   （Loss of Load Expectation: LOLE）を通しで回し、各段で新たに判明することと、そのために\n#   **前の段へ戻る判断**を体験する",
)
replace_once(
    "notebooks/src/19_integrated.py",
    "# `to_genstab` は潮流解と出力を**セットで**受け取り、7 号機を発電所ごとに 3 台へ集約します",
    "# ここでは臨界事故除去時間（Critical Clearing Time: CCT）を指標にします。\n# `to_genstab` は潮流解と出力を**セットで**受け取り、7 号機を発電所ごとに 3 台へ集約します",
)
replace_once(
    "notebooks/src/19_integrated.py",
    "# 浮く」という提案が出たとしましょう。第 18 回（`18_adequacy`）の道具で年間の信頼度を見ます。",
    "# 浮く」という提案が出たとしましょう。第 18 回（`18_adequacy`）の道具で、LOLE と\n# 供給支障電力量期待値（Expected Unserved Energy: EUE）を用いて年間の信頼度を見ます。",
)

# -----------------------------------------------------------------------------
# Course map, plotting labels, and course-case comments.
# -----------------------------------------------------------------------------
replace_once(
    "docs/course_map.md",
    "- **08 → 12**: 起動している同期機の慣性定数の合計が、変動性電源の導入で",
    "- **08 → 12**: 起動している同期機について、共通基準へ換算済みの慣性寄与の合計が、変動性電源の導入で",
)
for old, new in {
    '"genstab 01-03, 08"': '"pwsyseng 10-12, 17"',
    '"genstab 04-07"': '"pwsyseng 13-16"',
    '"gridops 01-04, 09"': '"pwsyseng 01-04, 09"',
    '"gridops 05-08"': '"pwsyseng 05-08"',
    '"gridops 10"': '"pwsyseng 18"',
    "MW peak PV)": "MW peak solar PV)",
}.items():
    replace_once("src/gridops/plotting.py", old, new)
replace_all("src/gridops/plotting.py", "第 10 回", "第 18 回")

replace_once(
    "src/gridops/casedata/wscc9.yaml",
    "# 各号機の h は 100 MVA 共通基準へ換算済みの慣性寄与分として定義している。\n# この寄与分は加算でき、過渡リアクタンスは並列合成で原典の 3 機に戻る。",
    "# 各号機の h と d は 100 MVA 共通基準へ換算済みの慣性・制動の寄与分として定義している。\n# これらの寄与分は加算でき、過渡リアクタンスは並列合成で原典の 3 機に戻る。",
)
replace_once("src/gridops/casedata/wscc9.yaml", "    name: PV", "    name: Solar PV")
replace_once(
    "src/gridops/casedata/wscc9.yaml",
    "# reliability 層 — 第 10 回。",
    "# reliability 層 — 第 18 回。",
)
replace_once(
    "src/gridops/casedata/wscc9.yaml",
    "#    ある。同じ 1 つの事象を第 09 回で静的に（過負荷）、genstab の\n#    第 08 回で動的に（脱調）評価することになる。",
    "#    ある。同じ 1 つの事象を第 09 回で静的に（過負荷）、pwsyseng の\n#    第 17 回で動的に（脱調）評価することになる。",
)

# -----------------------------------------------------------------------------
# Interoperability: state the common-base conversion before summation.
# -----------------------------------------------------------------------------
replace_once(
    "src/gridops/interop.py",
    "genstab が入っていない環境でも gridops の第 01〜10 回は動く。",
    "genstab が入っていない環境でも、gridops の運用・計画パートは動く。",
)
sub_once(
    "src/gridops/interop.py",
    r"    H = \\\\sum_i H_i, \\\\qquad\n"
    r"    \\\\frac\{1\}\{x'_d\} = \\\\sum_i \\\\frac\{1\}\{x'_\{d,i\}\}, \\\\qquad\n"
    r"    D = \\\\sum_i D_i, \\\\qquad",
    "    \\\\bar H_i = H_i^{(unit)} \\\\frac{S_i}{S_{base}}, \\\\qquad\n"
    "    \\\\bar D_i = D_i^{(unit)} \\\\frac{S_i}{S_{base}}, \\\\qquad\n"
    "    H = \\\\sum_i \\\\bar H_i, \\\\qquad\n"
    "    \\\\frac{1}{x'_d} = \\\\sum_i \\\\frac{1}{x'_{d,i}}, \\\\qquad\n"
    "    D = \\\\sum_i \\\\bar D_i, \\\\qquad",
)
sub_once(
    "src/gridops/interop.py",
    r"一般には慣性定数を各機の容量基準のまま足してはならない。本ケースの\n"
    r"``Unit\.h`` は 100 MVA 共通基準へ換算済みの慣性寄与分として定義しているため、\n"
    r"その寄与分を加算できる。過渡リアクタンスは並列につながる枝なので逆数の和になる。制動係数を \*\*加算\*\* するのは\n"
    r"H と揃えるためで、こうすると減衰比",
    "一般には慣性定数 $H$ も制動係数 $D$ も、各機の容量基準のまま足してはならない。\n"
    "本ケースの ``Unit.h`` と ``Unit.d`` は 100 MVA 共通基準へ換算済みの寄与分として\n"
    "定義しているため、その寄与分を加算できる。過渡リアクタンスは並列につながる枝なので\n"
    "逆数の和になる。共通基準上の制動寄与を加算すると、減衰比",
)
replace_once(
    "src/gridops/interop.py",
    "制動係数 D も H と同じく加算する（減衰比を保つため。モジュールの",
    "共通基準上の制動寄与 D も加算する（減衰比を保つため。モジュールの",
)
replace_once(
    "src/gridops/interop.py",
    "``D``         制動係数 [p.u.]（加算）",
    "``D``         100 MVA 共通基準上の制動寄与 [p.u.]（加算）",
)

# -----------------------------------------------------------------------------
# Model-assumption table: align statements with what is and is not implemented.
# -----------------------------------------------------------------------------
replace_once(
    "docs/model_assumptions.md",
    "**制動係数 $D$ も $H$ と同じく加算します**",
    "各号機の $D$ も共通基準上の制動寄与として定義し、発電所ごとに加算します",
)
replace_once(
    "docs/model_assumptions.md",
    "| 運転予備力は「同期並列した未負荷容量」$\\sum(P_{max}u - p) \\ge R$ | 「起動していれば即座に全出力が出せる」という仮定を含みます。瞬動予備力はランプ律速で別に扱います |",
    "| 予備力は「同期並列した未負荷容量」$\\sum(P_{max}u - p) \\ge R$ の簡易モデル | 応答時間、ランプ率、最大単一事故、送電制約、一次・二次・待機予備力の区別を同時には表しません。制度上の調整力商品と同一視できません |",
)
replace_once(
    "docs/model_assumptions.md",
    "| 起動費は単一（熱間／冷間の区別なし） | 停止時間に応じた段階的起動費には追加の 0-1 変数か区分定式化が必要です。教材では割り切っています |",
    "| 起動費は単一（熱間／冷間の区別なし） | 停止時間に応じた段階的起動費には追加の 0-1 変数か区分定式化が必要です。教材では割り切っています |\n| 24 時間の終端状態を拘束しない | 最終時刻付近の起動・停止は翌日の最低運転停止時間や起動費を負担しない終端効果を持ちます。実務では look-ahead または終端条件が必要です |",
)
replace_once(
    "docs/model_assumptions.md",
    "| 交流最適潮流 (ACOPF) は扱わない | 非凸性・局所解・初期値依存という重要な論点を落としています。直流近似が何を捨てているかは交流との誤差として定量化しますが、「直流最適潮流の解が交流で実行可能とは限らない」という核心には触れません |",
    "| 交流最適潮流（Alternating Current Optimal Power Flow: ACOPF）は扱わない | 直流最適潮流や SCED の解を交流潮流で再検査し、交流では実行不可能になり得ることは示しますが、非凸な交流最適化そのもの、局所解、初期値依存は扱いません |",
)

replace_once(
    "docs/references.md",
    "    本教材の第 10 回はこの本に従っています。",
    "    本教材の第 18 回はこの本に従っています。",
)
replace_all("tests/test_adequacy.py", "第 10 回", "第 18 回")
replace_all("tests/test_plotting.py", "第 10 回", "第 18 回")

# -----------------------------------------------------------------------------
# Regression tests for the final consistency fixes.
# -----------------------------------------------------------------------------
test_file = "tests/test_course_consistency.py"
text = load(test_file)
old_imports = "from gridops import load_case\nfrom gridops.security import screen_n1"
new_imports = (
    "from gridops import load_case\n"
    "from gridops.plotting import COURSE_THEMES\n"
    "from gridops.security import screen_n1"
)
if old_imports not in text:
    raise RuntimeError("tests/test_course_consistency.py: import insertion point not found")
text = text.replace(old_imports, new_imports, 1)
text += '''


def test_timescale_map_uses_integrated_course_numbers() -> None:
    labels = [theme[2] for theme in COURSE_THEMES]
    assert labels == [
        "pwsyseng 10-12, 17",
        "pwsyseng 13-16",
        "pwsyseng 01-04, 09",
        "pwsyseng 05-08",
        "pwsyseng 18",
    ]


def test_adequacy_notebook_describes_lole_as_an_expectation() -> None:
    text = (ROOT / "notebooks" / "src" / "18_adequacy.py").read_text(
        encoding="utf-8"
    )
    assert "供給不足状態になると期待されるか" in text
    assert "何時間足りないか」という**確率**" not in text
'''
save(test_file, text)

print("Final course cleanup completed.")
