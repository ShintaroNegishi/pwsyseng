# %% [markdown]
# # 19 総合演習 — 1 日の運用を通しで回す
#
# ## この回のねらい
#
# - 起動停止 → 経済配分 → 潮流 → N-1 → セキュリティ制約付き経済配分
#   （Security-Constrained Economic Dispatch: SCED）→ 過渡安定度 → 年間の供給支障時間期待値
#   （Loss of Load Expectation: LOLE）を通しで回し、各段で新たに判明することと、そのために
#   **前の段へ戻る判断**を体験する
# - 費用・セキュリティ・安定度・アデカシーが運転点を **別々の向きに引っ張る**のを 1 枚の図で見る
#
# ```
#   (07)起動停止 -> (05)経済配分 -> (02)潮流 -> (09)N-1 -> (09)SCED -> 安定度 -> (18)LOLE
#        ^              ^                                    |
#        +---- 戻る ----+---------------- 戻る --------------+
# ```
#
# 各段は前の段の答えを **所与**として受け取ります。ところが後ろの段でしか分からないことが
# あり、そのとき前の段へ戻らねばなりません。この回の主題は個々の道具ではなく、**この矢印の
# 向きと戻り方**です。割り切りは `docs/model_assumptions.md` に。とくに熱容量 `rate_a` /
# `rate_b` は交流の N-1 潮流から逆算した自作値、需要は合成データです。**絶対値ではなく比較を。**

# %%
import os
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

import gridops
from genstab import eac
from gridops.plotting import timescale_map, use_gridops_style

use_gridops_style()
FAST = bool(os.environ.get("GRIDOPS_FAST"))

case = gridops.load_case("wscc9")
demand = gridops.demand_profile(case, "summer_weekday")
uc = gridops.unit_commitment(case, demand, reserve_rate=0.10)
peak = int(np.argmax(demand))
online = [u.name for u in case.units if uc.schedule[u.name][peak] > 0.5]
print(uc.summary())
print(f"\nピーク時刻 t = {peak} 時、需要 {demand[peak]:.1f} MW / 運転中 {online} / "
      f"停止中 {[u.name for u in case.units if u.name not in online]}")
timescale_map()
plt.show()

# %% [markdown]
# ## 1. ピーク時刻を切り出す
#
# 起動停止計画は「どの号機を回すか」しか決めていません。**いくら出すか**を決めるのは経済
# 負荷配分で、渡す `committed=online` が第 07 回の答えです。出力が決まると母線の種別が
# 変わります。母線 3 の号機が 2 台とも停止しているので、母線 3 は電圧を支持できず **PV から
# PQ に落ちます** (`Case.effective_bus_types`)。第 01 回で数えた未知数も動きます。

# %%
ed = gridops.economic_dispatch(case, float(demand[peak]), committed=online)
dispatch_a = dict(ed.dispatch)
print(ed.summary())
types = case.effective_bus_types(dispatch_a)
print(f"\nPV -> PQ に落ちた母線 {[b.id for b in case.buses if types[b.id] is not b.type]}、"
      f"未知数 2n_PQ + n_PV: {case.n_unknowns()} -> {case.n_unknowns(dispatch_a)} 個\n")
# TODO(L1): 決めた出力 dispatch_a を、そのまま交流潮流に流すこと。
# BEGIN SOLUTION
flow_a = gridops.solve_powerflow(case, dispatch=dispatch_a)
# END SOLUTION
print(flow_a.summary())

# %% [markdown]
# **事故を起こす前から枝 4-6 が常時容量 `rate_a` の 116.7%** です。経済負荷配分は系統を見て
# いないので当然ですが、第 06 回の「経済配分の解を直流潮流に流すと 104.7%」よりさらに悪い。
# G3 を止めたぶん安い G1（母線 1）へ寄せたからです。
#
# ## 2. N-1 を掛ける
#
# 基準解に `flow_a` を渡すのが要点で、これで **この運転点に対する**判定になります。

# %%
report_a = gridops.screen_n1(case, flow_a)
print(f"{'事故':>7} {'最悪枝':>7} {'負荷率':>8} {'最低電圧':>9} {'母線':>4}  判定")
for r in report_a.results:
    load = f"{r.worst_loading:7.1%} " if r.converged else "収束せず"
    volt = f"{r.v_min:9.4f} {r.v_min_bus:>4}" if r.converged else f"{'-':>9} {'-':>4}"
    print(f"{str(r.outage):>7} {str(r.worst_branch) if r.converged else '-':>7} {load}{volt}  "
          f"{'健全' if r.is_secure else '逸脱'}{'' if r.converged else '（解けない）'}")
print(f"\nis_secure = {report_a.is_secure}（逸脱 {len(report_a.insecure())} 件 / 6 件中）")

# %% [markdown]
# 6 件中 4 件が不合格で、うち **2 件は事故後潮流が収束すらしません**。第 03 回でやったとおり、
# 収束しないことは「計算が下手」ではなく **その運転点ではその事故に耐える定常状態が存在
# しない**ことの表れで、判定以前の異常です。
#
# ## 3. 是正しようとすると、起動停止まで戻される
#
# `sced` は「全事故に耐える点のうち最も安いもの」を、まず **起動中の 5 台のまま**探します。

# %%
case_online = replace(case, units=[u for u in case.units if u.name in online])
try:
    gridops.sced(case_online, mode="preventive")
except ValueError as exc:
    print("5 台のまま SCED:", str(exc).splitlines()[0])

def hourly_cost(dispatch):
    """1 時間あたり費用 [円/h]（無負荷費 + 1 次項 + 2 次項）。"""
    total = 0.0
    for unit in case.units:
        p = dispatch.get(unit.name, 0.0)
        # TODO(L2): 出力が正の号機だけ a + b P + c P^2 を total に足すこと。
        # BEGIN SOLUTION
        if p > 0.0:
            total += unit.noload_cost + unit.var_cost * p + unit.quadratic * p**2
        # END SOLUTION
    return total

def plant_mw(dispatch):
    """発電所ごとの合計出力 (P_G1, P_G2, P_G3) [MW]。"""
    return tuple(sum(dispatch.get(u.name, 0.0) for u in case.units if u.plant == p)
                 for p in ("G1", "G2", "G3"))

points = {"A": dispatch_a,                                             # 経済最適・G3 停止
          "C": dict(gridops.economic_dispatch(case, 315.0).dispatch),  # 経済配分・G3 起動
          "B'": dict(gridops.sced(case, mode="corrective").dispatch),  # 是正的 SCED
          "B": dict(gridops.sced(case, mode="preventive").dispatch)}   # 予防的 SCED
print(f"\n{'点':<4}{'G1':>8}{'G2':>8}{'G3':>8}{'費用 [円/h]':>15}")
for name, d in points.items():
    print(f"{name:<4}" + "".join(f"{v:8.1f}" for v in plant_mw(d)) + f"{hourly_cost(d):15,.0f}")
print(f"\nセキュリティの値段 A -> B: "
      f"{hourly_cost(points['B']) - hourly_cost(points['A']):,.0f} 円/h")

# %% [markdown]
# **5 台のままでは実行不可能です。** 再給電をどう工夫しても N-1 に耐える点が存在しません。
# セキュリティの問題が、経済配分ではなく **起動停止の段**の問題として跳ね返ってきました。
# 第 07 回の入切表は第 09 回の制約を知らずに作られていたからです。G3 を起動する（起動費
# 240,000 円、無負荷費 50,000 円/h）ところまで戻さないとこの時刻は運用できません。なお
# `DispatchResult.total_cost` は無負荷費と 2 次項込み、`SCEDResult.total_cost` は
# $\sum_i b_i P_i$ だけなので **直接比べてはいけません**。4 点を同じ物差しで測る
# `hourly_cost` を自分で書いたのはそのためです。
#
# ## 4. その運転点は、事故の瞬間に耐えるか
#
# ここでは臨界事故除去時間（Critical Clearing Time: CCT）を指標にします。
# `to_genstab` は潮流解と出力を**セットで**受け取り、7 号機を発電所ごとに 3 台へ集約します
# （$H$ は和、$1/x'_d$ は逆数の和）。事故はケースの標準事故 **母線 7 の三相地絡・枝 5-7
# 開放**、第 09 回で拘束した枝そのものです。

# %%
def cct_of(dispatch):
    """臨界事故除去時間 [s]。潮流を解き直してから genstab へ渡す。"""
    flow = gridops.solve_powerflow(case, dispatch=dispatch)
    system = gridops.to_genstab(case, flow, dispatch=dispatch)
    return eac.critical_clearing_time(system, t_end=3.0, tolerance=2e-3, upper_bound=0.6)

cct = {name: cct_of(d) for name, d in points.items()}
t_prot = case.stability["protection_time"]
for name, value in cct.items():
    print(f"{name:<3} CCT = {value * 1e3:6.1f} ms（保護動作時間 "
          f"{t_prot * 1e3:.0f} ms の {value / t_prot:.1f} 倍）")

fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.bar(list(cct), [v * 1e3 for v in cct.values()],
       color=["tab:red", "tab:red", "tab:blue", "tab:blue"])   # 赤 = N-1 不合格
ax.axhline(t_prot * 1e3, color="k", ls="--", label="protection time 80 ms")
ax.set(ylabel="CCT [ms]", xlabel="operating point (red = fails N-1)",
       title="Transient stability margin")
ax.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# CCT はどの点でも保護動作時間 80 ms の 3 倍以上あり、**過渡安定度はこの系統・この事故では
# 縛りません**。しかし図の並びを読んでください。C（408 ms）→ B'（362 ms）→ B（318 ms）と、
# **N-1 を厳しく満たすほど余裕が削れています。** 理由は第 6 節の図で見ます。
#
# ## 5. 年間で見ると、その G3 はどう見えるか
#
# G3 は起動しない時間が長く燃料費も最も高い電源です。「廃止すればピーク時に毎時 270,200 円
# 浮く」という提案が出たとしましょう。第 18 回（`18_adequacy`）の道具で、LOLE と
# 供給支障電力量期待値（Expected Unserved Energy: EUE）を用いて年間の信頼度を見ます。

# %%
load_year = gridops.annual_load(case)
copt_all = gridops.capacity_outage_table(case.units)
copt_no3 = gridops.capacity_outage_table([u for u in case.units if u.plant != "G3"])
print(f"{'構成':<8}{'設備 [MW]':>11}{'予備率':>9}{'LOLE [h/年]':>14}{'EUE [MWh/年]':>15}")
for label, copt in (("全 7 台", copt_all), ("G3 廃止", copt_no3)):
    print(f"{label:<8}{copt.installed_mw:11.0f}{copt.installed_mw / load_year.max() - 1:9.1%}"
          f"{gridops.lole(copt, load_year):14.2f}{gridops.eue(copt, load_year):15.1f}")

# TODO(L3): G3 を廃止する提案に賛成か反対か。費用・N-1・年間信頼度の 3 つの物差しの値を
#           それぞれ数値で挙げ、どれを決め手にしたかを次の markdown に書くこと。

# %% [markdown]
# **模範解答（L3）**: 反対。値は (1) 費用: ピーク時の経済配分で 270,200 円/h の節約、
# (2) N-1: G3 を止めた運転点では予防的 SCED が **実行不可能**（第 3 節）、(3) 年間信頼度:
# LOLE が 4.15 → 103.89 時間/年（25 倍）、EUE が 94 → 2,702 MWh/年。決め手は (2) で、費用の
# 議論以前に **運用できない**からです。**メリットオーダーの最後尾にいる号機が系統の要になる
# ことは珍しくありません。** ただしこの LOLE は送電網を見ていない（第 18 回の割り切り）ので、
# 「母線 3 に電源があること」の価値は入っていません。
#
# ## 6. 3 つの要求を 1 枚の平面に重ねる
#
# 需要 315 MW が固定なので $P_{G1} = 315 - P_{G2} - P_{G3}$、つまり **自由度は 2 つ**です。
# $(P_{G2}, P_{G3})$ 平面に、(a) **N-1 の実行可能領域**（第 04 回の PTDF と LODF で事故後潮流
# $f' = f + \mathrm{LODF}[:,k]\,f_k$ を作り、事故前 `rate_a`・事故後 `rate_b` を全事故で満たす
# 点。`sced` が探す領域そのもの）、(b) **CCT**（各点で潮流を解き直し `to_genstab` に渡して
# 測る）、(c) **経済最適点** を重ねます。CCT は 1 点約 0.5 秒なので粗いグリッドに留めます。
#
# ## 7. 締めくくり — 三者はどこで会うか

# %%
ptdf = gridops.ptdf(case)
lodf = gridops.lodf(case, outages=case.contingencies)   # 橋を含めると ValueError
outs = [[b.key() for b in case.branches].index(c) for c in case.contingencies]
lim_a = np.array([b.rate_a for b in case.branches])
lim_b = np.array([b.rate_b for b in case.branches])
load_pu = np.array([b.pd for b in case.buses])          # 母線負荷 [p.u.]

def dc_secure(p2, p3):
    """直流の N-1 判定（事故前 rate_a / 事故後 rate_b。LODF[k,k]=-1 で開放枝は 0）。"""
    p1 = 315.0 - p2 - p3
    if not 72.0 <= p1 <= 180.0:            # 運転中号機の ΣPmin / ΣPmax
        return False
    inj = -load_pu.copy()
    for bus, mw in ((1, p1), (2, p2), (3, p3)):
        inj[case.index_of(bus)] += mw / case.base_mva
    f = ptdf @ inj
    if np.any(np.abs(f) > lim_a + 1e-9):
        return False
    return all(np.all(np.abs(f + lodf[:, k] * f[k]) <= lim_b + 1e-9) for k in outs)

g2, g3 = np.linspace(72.0, 180.0, 55), np.linspace(30.0, 100.0, 36)
region = np.array([[float(dc_secure(x, y)) for x in g2] for y in g3])

n = 3 if FAST else 5
c2, c3 = np.linspace(100.0, 160.0, n), np.linspace(40.0, 95.0, n)
cct_grid = np.full((n, n), np.nan)
for j, y in enumerate(c3):
    for i, x in enumerate(c2):
        share = {"G1": (315.0 - x - y) / 3.0, "G2": x / 2.0, "G3": y / 2.0}
        if 72.0 <= 315.0 - x - y <= 180.0:
            cct_grid[j, i] = cct_of({u.name: share[u.plant] for u in case.units})

m2, m3, sel = *np.meshgrid(g2, g3), region > 0.5
print(f"N-1 実行可能領域: P_G3 >= {m3[sel].min():.0f} / P_G2 <= {m2[sel].max():.0f} / P_G1 <="
      f" {(315.0 - m2 - m3)[sel].max():.0f} MW、CCT は {np.nanmin(cct_grid) * 1e3:.0f} 〜"
      f" {np.nanmax(cct_grid) * 1e3:.0f} ms")

# %%
fig, ax = plt.subplots(figsize=(8.4, 5.4))
filled = ax.contourf(c2, c3, cct_grid * 1e3, levels=8, cmap="RdYlGn", alpha=0.9)
fig.colorbar(filled, ax=ax, label="CCT [ms]")
ax.contour(c2, c3, cct_grid * 1e3, levels=[3e3 * t_prot],   # 保護動作時間の 3 倍
           colors="k", linewidths=2.0, linestyles="--")
ax.contourf(g2, g3, region, levels=[0.5, 1.5], colors="none", hatches=["///"])
ax.contour(g2, g3, region, levels=[0.5], colors="k", linewidths=1.6)
for key, label, marker, color in (("C", "C: economic dispatch", "*", "tab:orange"),
                                  ("B", "B: preventive SCED", "o", "tab:red"),
                                  ("A", "A: G3 off (off-plane)", "s", "dimgray")):
    _, x, y = plant_mw(points[key])
    ax.plot(x, y, marker=marker, ms=14, mec="k", ls="none", color=color, label=label)
ax.annotate("", xy=plant_mw(points["B"])[1:], xytext=plant_mw(points["C"])[1:],
            arrowprops=dict(arrowstyle="->", lw=1.8, color="k"))
ax.set(xlabel="$P_{G2}$ [MW]  (bus 2)", ylabel="$P_{G3}$ [MW]  (bus 3)",
       xlim=(72, 180), ylim=(-4, 104),
       title="Peak hour: hatched = N-1 feasible (DC), dashed = CCT 240 ms")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
plt.show()

# TODO(L3): 斜線の領域の中で運転点をどこに置くか。費用が最も安い角と CCT が最も長い角の
#           どちらへ寄せるか。領域の左上へ動かすと CCT は何 ms 伸び費用は何円/h 増えるかを
#           hourly_cost と cct_of で実際に測り、判断を次の markdown に書くこと。

# %% [markdown]
# **模範解答（L3）**: 一意の答えはありません。経済最適点 C は斜線の外にあります。矢印は SCED が
# 押し出した向きで、**右上、すなわち安い G1 を諦めて高い G2・G3 を焚く向き**です。ところが色を
# 見ると、右へ行くほど CCT が短くなります。事故点が母線 7、つまり G2 の高圧側だからです。
# **N-1 は右上へ、過渡安定度も費用も左下へ**引っ張り、N-1 だけが独りで逆を向いています。
# 「安全側に倒す」が、どの安全のことか言わないと意味を持たないのはこのためです。
#
# 領域の中で最も安い角が B $(125, 70)$ で CCT 318 ms、最も左上の $(106, 90)$ へ寄せると約
# 375 ms に伸びますが費用は 10 万円/h の単位で増えます。**黒い破線（CCT 240 ms = 保護動作時間の
# 3 倍。教材で置いた目安であってデータではない）は、領域の右上の角を切り落としています。**
# 運用の窓はもともと狭いのです。なお斜線は **直流の意味での**実行可能領域で、同じ点を交流で
# `screen_n1` に掛けるとこの平面には合格する点が 1 つもありません（第 09 回のとおり予防的 SCED
# の点でも 4 件の逸脱が残る）。運転点を動かすだけでは足りず、設備の増強が要るのが正直な結論です。

# %% [markdown]
# ## まとめ
#
# 1. **起動停止 → 経済配分 → 潮流**は素直に流れます。ただし停止した号機の母線は PV から PQ に
#    落ち、未知数の数まで変わります（14 → 15）。費用は 4 点とも同じ物差しで測ること。
# 2. **N-1 は経済最適点を失格にします。** 6 件中 4 件が逸脱し、2 件は収束すらしません。
# 3. **是正は経済配分の段では終わりません。** 5 台のままでは SCED が実行不可能で、起動停止まで
#    戻って G3 を起動する必要があります。セキュリティの値段は 833,456 円/h（22.5%）でした。
# 4. **N-1 が押し出す向きは、過渡安定度の余裕が薄くなる向きです。** CCT は 408 → 318 ms へ 22%
#    減りました（80 ms への余裕はまだ 4 倍ですが、減る向きに動いたことは記録に値します）。
# 5. **年間で見ると、最も高い電源が外せません。** G3 の廃止で LOLE は 4.15 → 103.89 h/年。
#
# ここで科目は一周しました。ミリ秒（CCT）から年（LOLE）まで扱った時間スケールは 10 桁以上に
# 及びますが、問いは終始 1 つ、「需要と供給を釣り合わせ続けられるか」でした。次に進むなら、この
# 教材が意図的に落としたところ——交流最適潮流（直流の解が交流で実行可能とは限らないという第 7
# 節の結論を正面から扱う）、逐次モンテカルロ、別ケース（IEEE 14 母線）——が入口になります。
