# %% [markdown]
# # 08 予備力と変動性再生可能エネルギー（Variable Renewable Energy: VRE）— 下げ代と出力抑制
#
# ## この回のねらい
#
# - 運転予備力を **同期並列している未負荷容量** として書く理由を、式の形でつかむ
# - 予備力率を振って「安心の値段」を測る
# - 太陽光を差し引いた **純需要**（ダックカーブ）で起動停止計画を解き直す
# - 夕方のランプが急になること、下げ代が尽きて **出力抑制** が立つことを確かめる
# - 起動している同期機の **慣性の合計** が減ることを定量化する（第 12 回への橋）
#
# ## 予備力の 2 つの書き方
#
# 第 07 回の起動停止計画には、需給バランス
#
# $$ \sum_i p_{it} + \mathit{shed}_t - \mathit{spill}_t = D_t $$
#
# のほかに予備力の制約が入っていました。本教材はこれを
#
# $$ \textrm{(A)}\quad \sum_i \bigl(P^{max}_i u_{it} - p_{it}\bigr) \ge R_t,
# \qquad R_t = r\,D_t $$
#
# と書きます。「**いま同期並列していて、まだ出していない容量**」が $R_t$ 以上
# あること、という意味です。停止中の号機は $u = 0$ なので 1 MW も数えません。
# 起動に何時間もかかる容量を予備力に数えない、というのがこの式の要点です。
#
# ただし、これは **同期並列中の上げ余力に基づく簡易な予備力容量モデル**です。
# 所定時間内の応答、ランプ率、最大単一事故、送電制約、一次・二次・待機予備力の区別は
# この制約だけでは表していません。実制度上の調整力商品と同一視しないでください。
#
# 教科書によっては $\textrm{(B)}\ \sum_i P^{max}_i u_{it} \ge (1 + r)\,D_t$ と
# 書いてあります。こちらのほうが覚えやすいのですが、**(A) と同じ制約ではありません。**
# 需給バランスを代入して、2 つの制約の余裕（左辺 − 右辺）を引き算すると
#
# $$ \bigl[\textrm{(B) の余裕}\bigr] - \bigl[\textrm{(A) の余裕}\bigr]
# = \mathit{spill}_t - \mathit{shed}_t $$
#
# が出ます。緩和変数が 0 のあいだは一致しますが、**出力抑制が立った瞬間に (B) は
# 「捨てている電力」を予備力として数え始めます。** 予備力は需要の何割かではなく
# 「いま出していない容量」に対する要求です。この回は $\mathit{spill}$ が実際に
# 立つ状況を作るので、この差が数字で出ます。

# %%
import numpy as np
import matplotlib.pyplot as plt

import gridops
from gridops.plotting import use_gridops_style, plot_commitment, plot_duck_curve

use_gridops_style()

case = gridops.load_case("wscc9")
demand = gridops.demand_profile(case, "summer_weekday")
hours = np.arange(demand.size)
vre = case.commitment["vre"]

print(f"需要 {demand.min():.1f} 〜 {demand.max():.1f} MW / 号機 {len(case.units)} 台 / "
      f"ΣPmin {sum(u.p_min_mw for u in case.units):.0f} MW / "
      f"ΣPmax {sum(u.p_max_mw for u in case.units):.0f} MW")
print(f"ケースの VRE: {vre['name']} 設備容量 {vre['capacity_mw']:.0f} MW")

# %% [markdown]
# ## 1. 予備力を自分で数える
#
# まず VRE 無し・予備力率 10% で解き、予備力を定義の式そのままで計算して
# `CommitmentResult.reserve_mw()` と突き合わせます。式が読めることと式が
# 書けることの間には、いつも溝があります。

# %%
base = gridops.unit_commitment(case, demand, reserve_rate=0.10)

# TODO(L1): 運転予備力 Σ_i (Pmax_i * u_it - p_it) を sum と内包表記の 1 文で書くこと。
#           base.schedule[名前] が u_it、base.dispatch[名前] が p_it の (24,) 配列。
# BEGIN SOLUTION
reserve = sum(
    unit.p_max_mw * base.schedule[unit.name] - base.dispatch[unit.name]
    for unit in case.units
)
# END SOLUTION

print(base.summary())
print(f"\n定義どおりに数えた予備力と reserve_mw() の差 = "
      f"{np.abs(np.asarray(reserve) - base.reserve_mw()).max():.2e} MW")

# %% [markdown]
# ## 2. 予備力率を振る — 安心の値段
#
# 予備力率 $r$ を 0 / 5 / 10 / 20% と振って、総費用がどう増えるかを見ます。
# 予備力を積むと費用が増える経路は 2 つあります。**（a）** 高い機を最低出力で
# 走らせて未負荷容量を作る、**（b）** そのために 1 台余分に起動する。
# (b) が起きた瞬間に費用は階段状に飛びます。

# %%
rates = [0.0, 0.05, 0.10, 0.20]
runs_rate: dict[float, gridops.CommitmentResult] = {}

for rate in rates:
    # TODO(L2): 予備力率 rate で起動停止計画を解き、runs_rate[rate] に入れること。
    # BEGIN SOLUTION
    runs_rate[rate] = gridops.unit_commitment(case, demand, reserve_rate=rate)
    # END SOLUTION

for rate in rates:
    run = runs_rate[rate]
    print(f"r = {rate:4.0%}: 総費用 {run.total_cost:12,.0f} 円  "
          f"起動 {run.n_startups()} 回  予備力の最小 {run.reserve_mw().min():6.1f} MW")
extra = runs_rate[0.20].total_cost - runs_rate[0.0].total_cost
print(f"\nr = 0% から 20% への上乗せ = {extra:,.0f} 円/日 "
      f"（{extra / runs_rate[0.0].total_cost:.2%}）")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
axes[0].plot([100 * r for r in rates],
             [runs_rate[r].total_cost / 1e6 for r in rates], "o-")
axes[0].set_xlabel("Reserve rate [%]")
axes[0].set_ylabel("Total cost [million JPY/day]")
axes[0].set_title("Cost of carrying reserve")
for rate in rates:
    run = runs_rate[rate]
    axes[1].step(hours, run.reserve_mw() - np.asarray(run.options["reserve_mw"]),
                 where="mid", label=f"r = {rate:.0%}")
axes[1].axhline(0.0, color="k", ls="--", lw=1.2, label="constraint binds")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Reserve minus requirement [MW]")
axes[1].set_title("Slack of the reserve constraint")
axes[1].legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# 右図は制約の余裕（予備力 − 要求）です。$r = 20\%$ で初めて 0 まで下がる時刻が
# 現れ、そこで予備力が入切を動かし始めます。$r = 0\%$ の線がゼロに張り付かないのは、
# 最低出力の刻みのせいで **予備力の一部がただで手に入っている** からです。
#
# ## 3. 純需要 — ダックカーブ
#
# 起動停止計画が見るのは需要そのものではなく、変動性電源を差し引いた **純需要**
# $D^{net}_t = D_t - P^{VRE}_t$ です。太陽光は昼の需要を押し下げますが、夕方には
# 急速に消えます。効いてくるのは需要の大きさではなく、その **傾き** です。

# %%
capacities = [0.0, 120.0, 180.0, 240.0, 300.0]
coal_ramp = sum(u.ramp_up for u in case.units if u.plant == "G1")

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4))
plot_duck_curve(case, ax=axes[0])
for cap in capacities:
    net = gridops.net_demand(case, demand, vre_mw=cap)
    ramp = np.diff(net)
    k = int(np.argmax(ramp))
    axes[1].plot(hours[1:], ramp, label=f"PV {cap:.0f} MW")
    print(f"PV {cap:5.0f} MW: 純需要の最小 {net.min():6.1f} MW、"
          f"最大ランプ {ramp.max():5.1f} MW/h（{k} 時 → {k + 1} 時）、"
          f"朝 5→6 時 {ramp[5]:5.1f} MW/h")
axes[1].axhline(coal_ramp, color="k", ls="--", lw=1.2,
                label=f"3 coal units: {coal_ramp:.0f} MW/h")
axes[1].axhline(0.0, color="0.6", lw=0.8)
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Net demand ramp [MW/h]")
axes[1].set_title("Hourly ramp of net demand")
axes[1].legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()

# %% [markdown]
# **朝のランプは太陽光が肩代わりし、夕方のランプは太陽光が作ります。**
# 最大ランプの時刻が朝（5→6 時）から夕方（15→16 時）へ移り、
# PV 300 MW では 69.4 MW/h に達します。石炭 3 台の増出力率の合計 60 MW/h では
# 追いつきません。**足りない分は「もう 1 台起動する」以外に作れません。**
#
# ## 4. 下げ代不足と出力抑制
#
# 設備容量を振って解き直します。見るのは費用ではなく `spill_mw` です。

# %%
runs_pv: dict[float, gridops.CommitmentResult] = {}
for cap in capacities:
    runs_pv[cap] = gridops.unit_commitment(case, demand, reserve_rate=0.10, vre_mw=cap)

print(f"{'PV [MW]':>8} {'cost [JPY]':>14} {'spill [MWh]':>12} "
      f"{'min net [MW]':>13} {'min SumPmin(on)':>16} {'starts':>7}")
for cap in capacities:
    run = runs_pv[cap]
    on_pmin = sum(u.p_min_mw * run.schedule[u.name] for u in case.units)
    print(f"{cap:8.0f} {run.total_cost:14,.0f} {run.spill_mw.sum():12.1f} "
          f"{run.demand_mw.min():13.1f} {on_pmin.min():16.1f} {run.n_startups():7d}")

# %% [markdown]
# ## 5. 上げ代はあるのに下げられない
#
# 抑制が立った時刻を分解します。運転中の号機は $P^{min}$ を下回れないので、
# 火力の出力は $\sum_i P^{min}_i u_{it}$ より下がりません。これが純需要を
# 上回った分が、そのまま出力抑制です。
#
# $$ \mathit{spill}_t = \max\Bigl\{0,\ \sum_i P^{min}_i u_{it} - D^{net}_t\Bigr\} $$
#
# **この $\mathit{spill}$ という変数が無ければ、需給バランスの等式は成立せず、
# モデルは実行不可能になります。**「解けない」ではなく「抑制した」と答えるのが
# 正しい。第 07 回で $\mathit{spill}$ を既定で常に入れていた理由がここにあります。

# %%
heavy = runs_pv[300.0]
rate = heavy.options["reserve_rate"]
net = heavy.demand_mw
on_pmin = sum(u.p_min_mw * heavy.schedule[u.name] for u in case.units)
up_reserve = heavy.reserve_mw()
down_reserve = sum(heavy.dispatch[u.name] - u.p_min_mw * heavy.schedule[u.name]
                   for u in case.units)
committed = np.array([heavy.committed_mw(t) for t in hours])
naive_slack = committed - (1.0 + rate) * net
true_slack = up_reserve - np.asarray(heavy.options["reserve_mw"])

print(f"{'hour':>4} {'net [MW]':>9} {'SumPmin':>8} {'spill':>7} "
      f"{'up':>7} {'down':>7} {'(B)-(A)':>9} {'spill-shed':>11}")
for t in range(9, 16):
    print(f"{t:>4} {net[t]:9.1f} {on_pmin[t]:8.1f} {heavy.spill_mw[t]:7.1f} "
          f"{up_reserve[t]:7.1f} {down_reserve[t]:7.1f} "
          f"{naive_slack[t] - true_slack[t]:9.1f} "
          f"{heavy.spill_mw[t] - heavy.shortfall_mw[t]:11.1f}")

plot_commitment(heavy)
plt.show()

# %% [markdown]
# 図の 10〜14 時では、積み上げが $\sum P^{min} = 72$ MW で止まり、黒い純需要の線が
# その下をくぐります。その差が斜線（curtailed）です。表の右 2 列も一致していて、
# 冒頭の恒等式がそのまま出ました。12 時の $\mathit{spill} = 64.5$ MW は、
# **$(1+r)D$ 型なら捨てている 64.5 MW を予備力として数えていた**ことを意味します。
#
# そして 12 時の **上げ代は 108.0 MW もあるのに、下げ代は 0.0 MW** です。
# 予備力制約は上げ代しか見ていません。変動性電源が主役になった系統で最初に
# 尽きるのは、上げ代ではなく下げ代のほうです。
#
# ## 6. 起動している同期機の慣性
#
# 昼に火力を止めるということは、**回っている回転体を減らす**ということです。
# 一般に、各発電機の銘板容量基準の慣性定数をそのまま足すことはできず、共通の
# 系統基準容量へ換算して $H_{eq}=\sum_i H_i S_i/S_{base}$ とします。本教材の
# `Unit.h` は、あらかじめ 100 MVA 共通基準へ換算した各号機の慣性寄与分です。
# したがって、運転中の号機について次のように加算できます。
#
# $$ H^{sys}_t = \sum_i \bar H_i \, u_{it} \quad [\mathrm{s\ on\ 100\ MVA\ base}] $$

# %%
# TODO(L3): 下のコードが PV 0 MW と 300 MW の系統慣性 H(t) を計算する。
#           (1) 最小値と減少率、そのとき動いている発電所を読み取ること。
#           (2) 動揺方程式 2H dΔω/dt = Pm - Pe から、慣性が減ると事故直後の
#               加速度がどうなるか。臨界事故除去時間 CCT（第 12 回）は
#               長くなるか短くなるか。理由とともに次の markdown セルに書くこと。
#           (3) 昼の慣性を確保する方策を 1 つ挙げ、その費用を gridops の
#               どの関数でいくらと見積もれるかを書くこと。
inertia = {cap: sum(u.h * runs_pv[cap].schedule[u.name] for u in case.units)
           for cap in (0.0, 300.0)}

plt.figure(figsize=(9.0, 3.6))
plt.fill_between(hours, inertia[300.0], inertia[0.0], step="mid", alpha=0.25,
                 label="inertia lost at midday")
for cap, series in inertia.items():
    plt.step(hours, series, where="mid", lw=2.0, label=f"PV {cap:.0f} MW")
plt.ylim(0.0, None)
plt.xlabel("Hour")
plt.ylabel("Committed inertia $H^{sys}$ [s on 100 MVA]")
plt.title("Synchronous inertia left online")
plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

worst = int(np.argmax(heavy.spill_mw))          # 抑制が最大の時刻
for cap in (0.0, 300.0):
    on = [u.name for u in case.units if runs_pv[cap].schedule[u.name][worst] > 0.5]
    print(f"PV {cap:5.0f} MW: H = {inertia[cap].min():5.2f} 〜 {inertia[cap].max():5.2f} s、"
          f"{worst} 時は {inertia[cap][worst]:5.2f} s（{', '.join(on)}）")
print(f"{worst} 時の慣性の減少 = "
      f"{1 - inertia[300.0][worst] / inertia[0.0][worst]:.1%}")

# %% [markdown]
# ### 考えるための材料
#
# - $2H \, d\Delta\omega/dt = P_m - P_e$ なので、同じ不平衡 $P_m - P_e$ に対する
#   角加速度は $H$ に反比例します。$H$ が 21% 減れば加速度は約 27% 増えます。
#   等面積法（第 11 回）で言えば、**同じ加速面積に達するまでの時間が
#   短くなる**ということです。第 12 回でこれを CCT として測ります。
# - この日の最悪の時刻には、母線 2 の LNG コンバインドサイクルが **1 台も
#   回っていません**。第 17 回の標準事故（母線 7 の三相地絡を線路 5-7 の開放で
#   除去）で最も大きく振れるのがこの機です。**同じ事故を、同じ日の違う時刻に
#   起こしてみると答えが変わる** —— これが第 12 回の入口です。
# - 方策の例: 最低出力の低い号機を昼に残す（must-run 制約）、蓄電池で谷を埋める、
#   同期調相機を入れる。前 2 つは「入切を固定して `unit_commitment` を解き直し、
#   総費用の差を取る」で値段が付きます。第 08 回の道具だけで評価できます。
#
# ## まとめ
#
# 予備力は「需要の何割か」ではなく「**いま同期並列していて出していない容量**」と
# して書きます。両者は緩和変数がゼロのあいだだけ一致し、抑制が立った瞬間に
# $\mathit{spill}$ の分だけ食い違いました。純需要は、昼の谷と夕方の急峻なランプと
# いう 2 つの負担を同時に持ち込みます。谷では起動中の号機の $\sum P^{min}$ が
# 純需要を上回り、上げ代が 100 MW 以上残っているのに下げ代が尽きて抑制が立ち、
# ランプ側では石炭 3 台の増出力率では追いつかず余分な起動が要りました。
#
# そして昼に火力を止めた結果、系統に残る同期機の慣性は 30.04 s から 23.64 s へ
# 減りました。ここから先は時間スケールが一気に変わります。**分〜時の計画問題として
# 決めた入切表が、ミリ秒〜秒の安定度の初期条件になる。** 第 12 回
# （`12_cct`）では、この慣性の違いが臨界事故除去時間を何ミリ秒縮めるかを測り、
# 保護装置の動作時間と比べます。計画の話が、そこで保護の話に変わります。
