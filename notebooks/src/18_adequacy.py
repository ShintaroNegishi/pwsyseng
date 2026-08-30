# %% [markdown]
# # 18 アデカシー — 設備は足りているか
#
# ## この回のねらい
#
# - **アデカシー**と**セキュリティ**を、問いの立て方と時間スケールで区別する
# - 強制停止率（Forced Outage Rate: FOR）から容量停止確率表（Capacity Outage Probability Table: COPT）を畳み込みで組み、**2 通りの独立な方法で検算する**
# - 供給支障確率（Loss of Load Probability: LOLP）、供給支障時間期待値（Loss of Load Expectation: LOLE）、
#   供給支障電力量期待値（Expected Unserved Energy: EUE）が別のものを測ることを同じ系統の数値で見る
# - モンテカルロを**点推定ではなく信頼区間**で語る
#
# ## アデカシーとセキュリティは別の問いである
#
# | | アデカシー | セキュリティ |
# |---|---|---|
# | 問い | そもそも設備は足りているか | いま起きた 1 事故に耐えられるか |
# | 時間スケール | 年（設備計画） | 分〜時（運用） |
# | 不確かさ | 多重停止の**確率分布** | 決められた N-1 リスト（決定論） |
# | 答えの形 | 期待値（年に何時間足りないか） | 是か非か |
#
# 第 09 回の答えは「枝 5-7 と 7-8 が拘束する」という**是非**、この回の答えは「年に
# 何時間、供給不足状態になると期待されるか」という**期待値**です（本モジュールは**送電網を見ません**）。決定論的な
# 設備予備率 $P_{inst}/D_{max}-1$ は**台数と停止率の情報を捨てます。**

# %%
import itertools
import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

import gridops
from gridops.adequacy import (annual_load, capacity_outage_table, elcc, eue,
                              load_duration_curve, lole, lolp, monte_carlo_adequacy)
from gridops.plotting import (plot_capacity_outage_table, plot_lolp_convergence,
                              use_gridops_style)

use_gridops_style()
case = gridops.load_case("wscc9")
installed = sum(unit.p_max_mw for unit in case.units)
peak = case.commitment["peak_mw"]
print(f"設備容量 {installed:.0f} MW / 最大需要 {peak:.0f} MW"
      f"  ->  設備予備率 {100 * (installed / peak - 1):.1f}%")
print("  ".join(f"{u.name} {u.p_max_mw:.0f}MW(FOR {u.forced_outage_rate:.2f})"
                for u in case.units))

# %% [markdown]
# ## 1. 容量停止確率表を畳み込みで作る
#
# 号機を 1 台ずつ足します。停止容量 $x$ の分布に容量 $C_n$、停止率 $p_n$ の号機を足すと
# $P_{n}(x) = (1-p_n)\,P_{n-1}(x) + p_n\,P_{n-1}(x - C_n)$ です。「健全なら分布はそのまま、
# 停止なら $C_n$ 右へずれる」を重ね合わせるだけで $2^n$ 通りと**同じ分布**が得られます。

# %%
def convolve_one(states, capacity, rate):
    """停止容量の分布 states に、容量 capacity・停止率 rate の号機を 1 台足す。"""
    new: dict[float, float] = {}
    for outage, probability in states.items():
        # TODO(L1): 健全側（確率 1-rate、停止容量そのまま）と
        #           停止側（確率 rate、停止容量 +capacity）を new に足し込むこと。
        # BEGIN SOLUTION
        new[outage] = new.get(outage, 0.0) + probability * (1.0 - rate)
        new[outage + capacity] = new.get(outage + capacity, 0.0) + probability * rate
        # END SOLUTION
    return new


states = {0.0: 1.0}
for unit in case.units:
    states = convolve_one(states, unit.p_max_mw, unit.forced_outage_rate)
copt = capacity_outage_table(case.units)
hand = np.array([states.get(x, 0.0) for x in copt.outage_mw])
print(f"手で畳み込んだ状態数 {len(states)} / gridops {copt.outage_mw.size}、"
      f"確率の最大差 {np.abs(hand - copt.probability).max():.3e}\n")
print(copt.summary())
plot_capacity_outage_table(copt)
plt.show()

# %% [markdown]
# **7 号機すべてが動いている確率は 69.1% しかありません。** 縦軸が対数なのは確率が
# $10^{-20}$ の桁まで落ちるからで、線形軸では稀な多重停止が見えなくなります。
#
# ## 2. 同じ量を 2 通りに計算して突き合わせる
#
# ひとつは**全列挙**（$2^7 = 128$ 通り）。畳み込みと同じ仮定から出発する実装の検算です。
# もうひとつは**二項分布**で、同一容量・同一停止率の号機 $n$ 台なら停止容量 $kC$ の確率は
# $\binom{n}{k}p^k(1-p)^{n-k}$。gridops の外（`scipy.stats.binom`）にある**独立な基準**
# なので、実装だけでなく「独立な 2 状態」という仮定ごと検算できます。

# %%
enumerated: dict[float, float] = {}
# TODO(L2): itertools.product([0, 1], repeat=号機数) で健全/停止の組合せを
#           全列挙し、停止容量ごとに確率を足し上げて enumerated を作ること。
# BEGIN SOLUTION
for pattern in itertools.product([0, 1], repeat=len(case.units)):
    probability = 1.0
    outage = 0.0
    for down, unit in zip(pattern, case.units):
        probability *= unit.forced_outage_rate if down else 1.0 - unit.forced_outage_rate
        outage += unit.p_max_mw if down else 0.0
    enumerated[outage] = enumerated.get(outage, 0.0) + probability
# END SOLUTION

gap = max(abs(enumerated.get(x, 0.0) - p)
          for x, p in zip(copt.outage_mw, copt.probability))
print(f"全列挙 {2 ** len(case.units)} 通り -> {len(enumerated)} 状態、"
      f"畳み込みとの最大差 {gap:.3e}")
g1 = [unit for unit in case.units if unit.plant == "G1"]      # 60 MW x 3、FOR 0.04
for same in (g1, [replace(g1[0], name=f"S{i}") for i in range(12)]):
    table = capacity_outage_table(same)
    exact = binom.pmf(np.arange(len(same) + 1), len(same), same[0].forced_outage_rate)
    print(f"同一容量 {len(same):2d} 台: 二項分布との最大差 "
          f"{np.abs(table.probability - exact).max():.3e}")

# %% [markdown]
# 差は倍精度の丸め（$10^{-16}$）の桁です。**畳み込みは近似ではありません。**
#
# ## 3. 年間需要と需要持続曲線
#
# 第三者の実需要は再配布しない方針なので `annual_load` は seed 固定の**合成**系列です。

# %%
load = annual_load(case)
ldc = load_duration_curve(load)
available = copt.available_mw()
print(f"年間 {load.size} 時間、最大 {load.max():.1f} MW、平均 {load.mean():.1f} MW、"
      f"負荷率 {load.mean() / load.max():.3f}")
fig, ax = plt.subplots(figsize=(9.0, 4.6))
ax.plot(np.arange(ldc.size), ldc, color="tab:blue", lw=2.0, label="load duration curve")
for target, color in ((460.0, "tab:green"), (370.0, "tab:orange"), (310.0, "tab:red")):
    i = int(np.argmin(np.abs(available - target)))
    ax.axhline(available[i], color=color, ls="--", lw=1.4,
               label=f"{available[i]:.0f} MW available (outage "
                     f"{copt.outage_mw[i]:.0f} MW, p = {copt.probability[i]:.4f})")
ax.set(xlabel="Hours per year above the level [h]", ylabel="Load [MW]",
       title="Load duration curve against capacity states")
ax.legend(loc="upper right", fontsize=8)
plt.show()
print(f"需要が 310 MW を超える時間: {int((load > 310.0).sum())} 時間/年")

# %% [markdown]
# 供給支障は**二重の偶然**です。150 MW 以上が同時に止まる（確率 1.5%）だけでは足りず、
# その時刻の需要が 310 MW を超えていなければならない。それは年間 **1 時間**だけです。
#
# ## 4. LOLP・LOLE・EUE
#
# $$ \mathrm{LOLP}(D) = P(A < D), \quad \mathrm{LOLE} = \sum_t \mathrm{LOLP}(D_t),
#    \quad \mathrm{EUE} = \sum_t \sum_i P_i \max(D_t - A_i, 0) $$
#
# $A_i$ は状態 $i$ の利用可能容量、$P_i$ はその確率。**LOLE の「日/年」は「時間/年」の
# 1/24 ではありません。** `unit="days"` は**その日の最大需要**に対する LOLP を 1 日 1 回
# 足す別の指標で、北米の「0.1 日/年」を「2.4 時間/年」と読んではいけません。

# %%
lole_hours = lole(copt, load)
lole_days = lole(copt, load, unit="days")
eue_manual = float("nan")
# TODO(L1): EUE の定義 sum_t sum_i P_i * max(D_t - A_i, 0) を行列で組み立て、
#           eue_manual に入れること（load, available, copt.probability を使う）。
# BEGIN SOLUTION
shortfall = np.maximum(load[:, None] - available[None, :], 0.0)
eue_manual = float((shortfall @ copt.probability).sum())
# END SOLUTION

print(f"LOLP(最大需要 {load.max():.0f} MW) = {lolp(copt, load.max()):.6f}")
print(f"LOLE = {lole_hours:.4f} 時間/年 (教材の目標 "
      f"{case.reliability['target_lole_hours']:.1f} 時間/年)")
print(f"LOLE = {lole_days:.4f} 日/年 <- LOLE[時間]/24 = {lole_hours / 24:.4f} の"
      f" {lole_days / (lole_hours / 24):.2f} 倍")
print(f"EUE  = {eue(copt, load):.3f} MWh/年 (定義式から {eue_manual:.3f})")

# %% [markdown]
# **設備予備率 46% は十分に見えるのに、LOLE は目標の 3.0 時間/年を超えています。**
# 予備率という 1 つの数字では、90 MW 機 2 台が同時に止まる確率 $0.05^2$ を表せません。
#
# ## 5. LOLE は不足の「深さ」を見ない
#
# 設備容量 200 MW、需要 110 MW 一定・10 時間の系統を考えます。確実な号機（FOR = 0）
# を $F$ MW、停止しうる号機（FOR = 0.05）を $200-F$ MW とすると、利用可能容量は
# 200 MW（確率 0.95）か $F$ MW（確率 0.05）です。$F$ を動かすとどうなるでしょうか。

# %%
def firm_split(firm_mw, rate=0.05, load_mw=110.0, hours=10):
    """確実な号機 firm_mw MW と、停止しうる号機 200-firm_mw MW の系統を評価する。"""
    table = capacity_outage_table(
        [replace(case.units[0], p_max_mw=firm_mw, forced_outage_rate=0.0),
         replace(case.units[0], p_max_mw=200.0 - firm_mw, forced_outage_rate=rate)])
    profile = np.full(hours, load_mw)
    return lole(table, profile), eue(table, profile)


# TODO(L3): 設備容量 200 MW・需要 110 MW を保ったまま確実容量 F を振ると、
#           LOLE と EUE はどう動くか。EUE をちょうど 3 倍にする構成はどれか。
#           なぜ LOLE は動かないのかを、自分の言葉で説明すること。
firm_grid = np.arange(100.0, 15.0, -10.0)
pairs = np.array([firm_split(firm) for firm in firm_grid])
fig, ax = plt.subplots(figsize=(9.0, 4.4))
twin = ax.twinx()
ax.plot(firm_grid, pairs[:, 0], "o-", color="tab:blue", label="LOLE [h]")
twin.plot(firm_grid, pairs[:, 1], "s--", color="tab:red", label="EUE [MWh]")
ax.set(xlabel="Firm capacity $F$ [MW]   (variable unit = $200-F$ MW)",
       ylabel="LOLE [h]", ylim=(0.0, 1.0),
       title="Same installed capacity, same LOLE, very different EUE")
twin.set_ylabel("EUE [MWh]", color="tab:red")
twin.grid(False)
ax.legend(handles=ax.lines + twin.lines, loc="center left", fontsize=9)
plt.show()
for firm, (hours_lost, energy) in zip(firm_grid, pairs):
    print(f"F = {firm:5.0f} MW : LOLE {hours_lost:.3f} h, "
          f"EUE {energy:6.2f} MWh ({energy / pairs[0, 1]:.1f} 倍)")

# %% [markdown]
# **LOLE の線はぴくりとも動かず 0.5 時間のままなのに、EUE は直線で増えます。**
# 不足の確率は「停止しうる号機が止まる確率 0.05」だけで決まり $F$ によりませんが、
# 不足の**深さ**は $110-F$ MW だからです。LOLE は不足状態にある時間の期待値を数えますが、
# その時間に何 MW 足りないかは区別しません。
#
# **`# TODO(L3)` の模範解答（の一例）**: $\mathrm{EUE} = 0.05\,(110-F)\times 10$ なので
# $F = 100$ MW の 5 MWh の 3 倍（15 MWh）になるのは $F = 80$ MW、すなわち**確実 80 MW +
# 変動 120 MW**。**指標を 1 つに絞ると、設備計画は必ずどこかを見落とします。**
#
# ## 6. モンテカルロ — 点推定ではなく信頼区間で語る
#
# 厳密に解ける問題をあえてサンプリングでも解くのは、**答え合わせができる題材で
# モンテカルロの振る舞いを学ぶため**です。判定は点推定の一致ではなく **95% 信頼区間が
# 解析解を含むか**で行います。以下では信頼区間を confidence interval（CI）と表記します。
# 精度の目安は $\beta = \sqrt{(1-p)/(pN)}$ で $1/\sqrt{N}$
# でしか縮まず、**稀な事象ほど $p$ が小さく必要な標本数が増えます**。

# %%
fast = bool(os.environ.get("GRIDOPS_FAST"))
sizes = (10_000, 100_000) if fast else (10_000, 100_000, 1_000_000, 4_000_000)
reference = lole_hours / load.size            # 年平均の LOLP（解析解）
trials = [monte_carlo_adequacy(case.units, load, n_samples=n, seed=0) for n in sizes]
for n, trial in zip(sizes, trials):
    low, high = trial.lolp_interval()
    print(f"N = {n:>9,}  LOLP = {trial.lolp:.6f}  95% CI [{low:.6f}, {high:.6f}]"
          f"  beta = {trial.coefficient_of_variation():.4f}"
          f"  EUE = {trial.eue:6.1f} MWh  解析解を含む: {low <= reference <= high}")
print(f"解析解    : LOLP = {reference:.6f}  EUE = {eue(copt, load):.1f} MWh")
plot_lolp_convergence(trials, reference=reference)
plt.show()

# %% [markdown]
# 標本数を 4 倍にすると $\beta$ がほぼ半分になります。逆に言えば $p \approx
# 4.7\times10^{-4}$ の LOLP を 5% の精度で出すには $N \approx 8\times10^5$ 標本が要る。
# **畳み込みがミリ秒で出す答えに、モンテカルロは 100 万標本かけて近づきます。**
# それでも使うのは、畳み込みでは扱えない相関・時系列・送電制約を持ち込めるからです。
# なお非逐次法は**頻度と継続時間を区別できません**。FOR = 0.05 は「年 1 回 438 時間」でも
# 「年 50 回 8.8 時間」でも成り立ちます（区別には逐次法が要る。発展課題）。
#
# ## 7. 丸めは「整理」ではなく「近似」である
#
# 停止容量を格子に載せ替えると状態数が減ります。gridops は近い方に丸めるのではなく両隣に
# **内分**するので確率の総和と期待停止容量は厳密に保存されます。**では LOLE は？**

# %%
for step in (None, case.reliability["rounding_mw"], 25.0, 50.0):
    table = capacity_outage_table(case.units, rounding_mw=step)
    print(f"rounding = {str(step):>5} MW : 状態数 {table.outage_mw.size:3d}"
          f"  期待停止容量 {table.expected_outage_mw():.4f} MW"
          f"  LOLE {lole(table, load):.4f} h/年"
          f"  EUE {eue(table, load):8.3f} MWh/年")

# %% [markdown]
# ケース既定の `rounding_mw = 5.0` は**何も丸めません**（号機容量 50/60/90 MW がすべて
# 5 の倍数だからです）。25 MW にすると状態数は 33 から 21 に減り、**期待停止容量が 23.2 MW
# のまま動かないのに LOLE は 4.15 から 5.16 時間/年へ 24% も増えます。** 60 と 90 が
# 25 の倍数でないためで、**偏りの向きすら保証されません。**
#
# ## 8. 等価需要負担能力（Effective Load Carrying Capability: ELCC）
#
# 新電源の価値は**「信頼度を保ったまま何 MW の需要を追加で背負えるか」**で測ります。
# $\mathrm{LOLE}(既設+新設, D+\Delta) = \mathrm{LOLE}(既設, D)$ を満たす $\Delta$ が ELCC。

# %%
print(f"{'容量':>6} {'FOR':>6} {'ELCC':>9} {'容量比':>8}   素朴な見積り P(1-FOR)")
for capacity, rate in itertools.product((20.0, 50.0, 100.0), (0.05, 0.30)):
    candidate = replace(case.units[0], name="NEW", p_max_mw=capacity,
                        forced_outage_rate=rate)
    value = elcc(case.units, load, candidate)
    print(f"{capacity:6.0f} {rate:6.2f} {value:8.2f} MW {100 * value / capacity:7.1f}%"
          f"{capacity * (1.0 - rate):20.1f} MW")

# %% [markdown]
# 20 MW 機なら容量の 92%（素朴な見積り 95% に近い）ですが、**100 MW 機では 71%** しか
# ありません。大きい号機ほど 1 MW あたりの価値が下がる、容量価値の**逓減**です。
#
# ## まとめ
#
# - **アデカシー**は年の問い、**セキュリティ**は分〜時の問い。片方では信頼度を語れない
# - COPT は畳み込みで厳密に作れる。**全列挙**と**二項分布**の 2 通りで検算した
# - LOLP は時点の確率、LOLE は不足状態となる時間・日数の期待値、EUE は不足の**深さ**。LOLE が同じで EUE が
#   3 倍の系統が作れる以上、指標は 1 つに絞れない。「日/年」は「時間/年」の 1/24 でもない
# - モンテカルロは**信頼区間**で語る。$\beta \propto 1/\sqrt{N}$、稀な事象ほど重い
# - 丸めは近似であって整理ではない。確率と期待値は保存されても LOLE はずれる
#
# ## 次回へ
#
# 最終回の総合演習では、起動停止計画 → 経済負荷配分 → 潮流計算 → N-1 と SCED → 過渡安定度
# → 年間 LOLE を **1 日の運用として通し**、5 つの答えが矛盾しないかを確かめます。
