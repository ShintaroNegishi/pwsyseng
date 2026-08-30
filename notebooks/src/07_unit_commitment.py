# %% [markdown]
# # 07 発電機起動停止計画 — 混合整数計画
#
# ## この回のねらい
#
# - 経済負荷配分（第 05 回）を **24 個並べただけでは足りない**ことを確かめる
# - 0-1 変数 $u_{it}$、状態遷移 $u_{it}-u_{i,t-1}=v_{it}-w_{it}$、最低運転停止
#   時間の **窓和**と **初期状態からの持ち越し**を書けるようになる
# - 区分線形近似の誤差の向きと大きさを知り、優先順位法との費用の差を測る
# - 混合整数計画の整数解に線形計画と同じ影価格を直接対応させず、限界費用を 2 段階で取る
#
# ## 経済負荷配分が決めていないこと
#
# 第 05 回の等 $\lambda$ 法が決めるのは「**運転中の**号機に需要をどう割り振るか」
# だけで、次の 3 つを決めていません。
#
# 1. **起動費** — 止まっている機を起動するのに 12 万〜120 万円かかる
# 2. **最低運転／停止時間** — 一度起動したら 8 時間は止められない
# 3. **最低出力** — $P^{min}$ より下には絞れない。下げたければ止めるしかない
#
# どれも「運転しているかどうか」に紐づく量で、連続量の最適化では表せません。
# そこで 0-1 の $u_{it}$（運転）、$v_{it}$（起動）、$w_{it}$（停止）を入れます。
#
# $$
# \min \sum_i \sum_t \bigl[ c_{0i} u_{it} + \widehat{C}_i(p_{it}) + SU_i v_{it} \bigr],
# \quad u_{it} - u_{i,t-1} = v_{it} - w_{it}, \quad
# P^{min}_i u_{it} \le p_{it} \le P^{max}_i u_{it}, \quad \sum_i p_{it} = D_t,
# \quad \sum_i \bigl(P^{max}_i u_{it} - p_{it}\bigr) \ge R_t, \quad
# \sum_{s=t-TU_i+1}^{t} v_{is} \le u_{it}, \quad
# \sum_{s=t-TD_i+1}^{t} w_{is} \le 1 - u_{it}
# $$
#
# 予備力は **同期並列している未負荷容量**の形で書きます。$\sum_i P^{max}_i u_{it}
# \ge (1+r)D_t$ と書くと、需要を捨てているのに予備力を満たす解が出るからです。

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

import gridops
from gridops import plotting as gplt

gplt.use_gridops_style()
case = gridops.load_case("wscc9")
demand = gridops.demand_profile(case, "summer_weekday")   # 夏の平日
light = gridops.demand_profile(case, "light_load")        # 軽負荷日
print(f"号機 {len(case.units)} 台 / 総容量 {sum(u.p_max_mw for u in case.units):.0f}"
      f" MW / 最低出力の合計 {sum(u.p_min_mw for u in case.units):.0f} MW"
      f" / 予備力率 {case.commitment['reserve_rate']:.0%}")
print(f"夏の平日 {demand.min():.1f} 〜 {demand.max():.1f} MW（1 日 {demand.sum():.0f}"
      f" MWh）/ 軽負荷日 {light.min():.1f} 〜 {light.max():.1f} MW")

# %% [markdown]
# ## 1. 経済負荷配分を 24 回並べてみる
#
# 7 号機を全部運転したまま各時刻で独立に等 $\lambda$ 法を解きます。

# %%
ed_cost = sum(gridops.economic_dispatch(case, d).total_cost for d in demand)
print(f"経済負荷配分 24 回の総費用 : {ed_cost:,.0f} 円")
print(f"  うち無負荷費 {sum(u.noload_cost for u in case.units) * 24:,.0f} 円"
      "（7 台 x 24 時間ぶんを、まるごと払っている）")

try:                                     # 軽負荷日の谷では解が存在しない
    gridops.economic_dispatch(case, light.min())
except ValueError as error:
    print("\n軽負荷日の最小需要で解こうとすると:\n", error)

# %% [markdown]
# 上の例外が **3 番目の欠落そのもの**です。$\sum P^{min} = 174$ MW は軽負荷日の谷
# を上回るので、**どれかを止めない限り需給は釣り合いません**。
#
# ## 2. では 1 時間ごとに入切も選べばよいのでは
#
# 入切の組合せは各時刻 $2^7 = 128$ 通りしかありません。**全部試して、その時刻で
# いちばん安い組合せを選ぶ**ことにします。

# %%
names = [u.name for u in case.units]
myopic = {n: np.zeros(24) for n in names}
myopic_cost = 0.0
for t, d in enumerate(light):
    best, best_on = None, None
    for mask in range(1, 1 << len(names)):
        on = [n for i, n in enumerate(names) if mask >> i & 1]
        chosen = [u for u in case.units if u.name in on]
        if (sum(u.p_max_mw for u in chosen) < d * 1.1        # 需要 + 予備力 10%
                or sum(u.p_min_mw for u in chosen) > d):     # 下げ切れない
            continue
        cost = gridops.economic_dispatch(case, d, committed=on).total_cost
        if best is None or cost < best:
            best, best_on = cost, on
    myopic_cost += best
    for n in best_on:
        myopic[n][t] = 1.0

startup_cost = 0.0
for unit in case.units:
    u = myopic[unit.name]
    previous = np.concatenate(([float(unit.u0)], u[:-1]))   # 直前の状態は u0
    # TODO(L1): 起動回数 sum_t max(u_t - u_{t-1}, 0) に起動費 SU を掛けて
    #           startup_cost に足し込むこと（1 行）。
    # BEGIN SOLUTION
    startup_cost += unit.startup_cost * float(np.maximum(u - previous, 0.0).sum())
    # END SOLUTION

for name in names:
    print(f"{name:5s} " + "".join("#" if v > 0.5 else "." for v in myopic[name]))
print(f"\n毎時最適の燃料費+無負荷費 {myopic_cost:,.0f} 円 / あとから足した起動費 "
      f"{startup_cost:,.0f} 円 / 合計 {myopic_cost + startup_cost:,.0f} 円")

# %% [markdown]
# 入切表がぎざぎざになりました。1 時間ごとの最適化は次の時刻を見ていないので、
# **安くなるなら 1 時間で止めて 1 時間で起こします**。起動費は決めたあとに足す
# しかなく、最低運転時間は満たしているかどうかも見ていません。
#
# ## 3. 窓和で最低運転／停止時間を数える
#
# 起動 $v_t = \max(u_t-u_{t-1},0)$、停止 $w_t = \max(u_{t-1}-u_t,0)$ と置けば
# $TU$・$TD$ の拘束は **窓和**で書けます（「直近 $TU$ 時間に起動があったなら、
# いまも運転していること」）。
#
# **窓和だけでは足りません。** 窓は期間の頭で切り詰まるので、期間の前から運転
# している号機の拘束が消えます。最低運転時間 8 時間の石炭機を 1 時間目に止める解
# が、窓和を 1 本も破らずに書けてしまう。持ち越し分は `Unit.remaining_min_up()`
# / `remaining_min_down()` が返します。
#
# **計画期間の終端にも注意が必要です。** 本教材は 24 時間の外側を評価しないため、
# 最終時刻付近の起動・停止が翌日の最低運転停止時間や起動費を負担しない終端効果があります。
# 実務では翌日を含む look-ahead、終端状態、または 48 時間計算の中央 24 時間を用います。

# %%
def check_min_times(unit, u):
    """入切表 u が最低運転／停止時間を満たすか調べ、破れた (時刻, 種別) を返す。"""
    previous = np.concatenate(([float(unit.u0)], u[:-1]))
    v = np.maximum(u - previous, 0.0)
    w = np.maximum(previous - u, 0.0)
    bad = []
    for t in range(u.size):
        # TODO(L2): 起動の窓和 sum_{s=t-TU+1}^{t} v_s <= u_t を確かめ、
        #           破れていたら bad.append((t, "min_up")) すること。
        # BEGIN SOLUTION
        if v[max(0, t - unit.min_up + 1):t + 1].sum() > u[t] + 1e-9:
            bad.append((t, "min_up"))
        # END SOLUTION
        # TODO(L2): 停止の窓和 sum_{s=t-TD+1}^{t} w_s <= 1 - u_t も同様に。
        # BEGIN SOLUTION
        if w[max(0, t - unit.min_down + 1):t + 1].sum() > 1.0 - u[t] + 1e-9:
            bad.append((t, "min_down"))
        # END SOLUTION
    # 窓和では見えない、初期状態からの持ち越し
    bad += [(t, "carry TU") for t in range(unit.remaining_min_up()) if u[t] < 0.5]
    bad += [(t, "carry TD") for t in range(unit.remaining_min_down()) if u[t] > 0.5]
    return bad


for unit in case.units:
    bad = check_min_times(unit, myopic[unit.name])
    if bad:
        print(f"{unit.name} TU={unit.min_up} TD={unit.min_down}: "
              f"{len(bad)} 件、はじめの 4 件 {bad[:4]}")

# %% [markdown]
# 毎時最適の入切表は **実行不可能**でした。安く見えた計画はそもそも運転できません。
# これが「24 個並べただけでは足りない」の中身です。
#
# ## 4. 混合整数計画で解く
#
# **ここで需要を夏の平日に戻します**（第 1 節と同じ日。第 2・3 節の軽負荷日とは別の
# 日なので金額を直接比べないこと）。後半では持ち越しも効かせます。G2-2 が「1 時間前に
# 起動したばかり」なら、最低運転時間 4 時間なので、あと 3 時間は止められないはずです。

# %%
result = gridops.unit_commitment(case, demand)   # PuLP + CBC
print(result.summary())
print("\n" + result.to_table())
print(f"\n経済負荷配分 24 回との差 {ed_cost - result.total_cost:,.0f} 円")

warm = replace(case, units=[replace(u, u0=1, hours_in_state=1)
                            if u.name == "G2-2" else u for u in case.units])
warm_result = gridops.unit_commitment(warm, demand)
g22 = [u for u in warm.units if u.name == "G2-2"][0]
print(f"\nG2-2 の remaining_min_up = {g22.remaining_min_up()} 時間")
for tag, r in (("もとの計画", result), ("持ち越し後", warm_result)):
    print(f"{tag} : " + "".join("#" if v > .5 else "." for v in r.schedule["G2-2"]))
print(f"総費用 {result.total_cost:,.0f} -> {warm_result.total_cost:,.0f} 円")

gplt.plot_commitment(result); plt.show()

# %% [markdown]
# **3 時間の拘束が、6 時間ぶん余分な運転を生みました。** $t = 0,1,2$ を運転すれば最低
# 運転時間は満たされますが、そこで止めると最低停止時間 4 時間が明けるのは $t = 7$ で、
# 需要が立ち上がる $t = 6$ に間に合いません。もう一度起動費 40 万円を払うより運転を
# 続けるほうが安い、と解が言っています。**制約は独立に効くのではなく、連鎖します。**
#
# 図の破線（同期並列容量）と黒線（需要）の差が運転予備力です。**停止中の号機は
# この図に 1 MW も現れません**――起動に何時間もかかる容量は予備力に数えない。
#
# ## 5. 区分線形費用 — 近似はどちら側に外れるか
#
# PuLP が扱えるのは線形と混合整数線形だけなので、2 次の燃料費
# $C(P) = c_2P^2 + c_1P + c_0$ は書けません。$[P^{min}, P^{max}]$ を $K$ 等分し、
# 各区間を **割線**で置き換えます。$C$ が凸なので割線の傾き
# $s_k = c_1 + c_2(x_{k-1}+x_k)$ は狭義単調増加で、線形計画は放っておいても安い
# セグメントから埋めます（**SOS2 も追加の 0-1 変数も要りません**）。割線は曲線の
# 上側に来るので近似は **必ず過大評価**側で、誤差の上界は $c_2L^2/4$ です。

# %%
unit = case.units[0]                     # 誤差の上界を見る代表の号機
span = unit.p_max_mw - unit.p_min_mw
# TODO(L3): 分割数 K を振り、(1) 誤差がどちら側に出るか (2) K を増やすと単調に
#           減るか (3) 実務でどう K を選ぶか を下の出力から述べること。
for k in (1, 2, 4, 8):
    trial = gridops.unit_commitment(case, demand, n_segments=k)
    exact = sum(float(u.fuel_cost(trial.dispatch[u.name][t])) for u in case.units
                for t in range(24) if trial.schedule[u.name][t] > 0.5)
    modelled = trial.cost_breakdown["fuel"] + trial.cost_breakdown["noload"]
    print(f"K = {k}: 総費用 {trial.total_cost:12,.0f} 円 / 厳密な費用との差 "
          f"{modelled - exact:8,.0f} 円 / 1 セグメントの上界 "
          f"{unit.quadratic * (span / k) ** 2 / 4:7.1f} 円/h")

# %% [markdown]
# **L3 の模範解答**: 差はすべて正、近似は必ず過大評価側です（割線が曲線の上側に
# 来る凸性の帰結）。$K$ を倍にすると上界はきっかり 1/4 になり実際の差もおおむね
# 1/4 ずつ減りますが、厳密にはそうなりません。$K$ を変えると **出力配分そのもの
# が動く**ので、比べているのが同じ点ではないからです。実務では上界が「意思決定を
# 変えない大きさ」――たとえば起動費の 1/100 ――を下回る $K$ で十分で、**入切表が
# 変わらなくなればそれ以上細かくしても意思決定は変わりません。**
#
# ## 6. 優先順位法との比較
#
# 混合整数計画が実用になる前は、**全負荷平均費用 $C(P^{max})/P^{max}$ の安い順に
# 需要と予備力を満たすまで起動する**優先順位法が使われました。夏の平日ではこの
# 2 つは同じ入切表に着くので、差の出る軽負荷日で比べます。

# %%
milp = gridops.unit_commitment(case, light)
heur = gridops.priority_list(case, light)
for label, r in (("優先順位法", heur), ("混合整数計画", milp)):
    print(f"{label:12s} {r.total_cost:12,.0f} 円  起動 {r.n_startups()} 回"
          f"（起動費 {r.cost_breakdown['startup']:9,.0f} 円）")
print(f"差 {heur.total_cost - milp.total_cost:,.0f} 円 = 1 日の費用の "
      f"{100 * (heur.total_cost / milp.total_cost - 1):.1f}%")

fig, axes = plt.subplots(2, 1, figsize=(10, 6.2))
for ax, r in zip(axes, (heur, milp)):
    gplt.plot_commitment_schedule(r, ax=ax)
fig.tight_layout(); plt.show()

# %% [markdown]
# **答え合わせは入切表ではなく総費用で行ってください。** 同じ費用の別解が多数
# あり（縮退）、入切表はソルバの版で変わります。優先順位法が負ける理由は順位づけ
# の側で、全負荷平均費用は起動費も最低運転時間も見ていません。なお燃料費・起動費・
# 需要形状は教材用の合成データです（`docs/model_assumptions.md`）。
#
# ## 7. 限界費用 — 混合整数計画に双対は無い
#
# 「その時刻に 1 MWh 追加で要るとしたら、いくらか」。線形計画なら需給バランス
# 制約の双対がそれですが、**整数変数を含む問題の最適値は右辺について階段状に
# 動く**ので、微分＝双対がそもそも存在しません。そこで混合整数計画で入切 $u$ を
# 決め、その $u$ を **定数として固定**して線形計画に落とし直し、双対を取ります。

# %%
price = gridops.marginal_prices(case, result)
print(f"{'unit':6s}{'電力量収入':>14s}{'費用':>14s}{'差':>14s}")
for u in case.units:
    on, p = result.schedule[u.name], result.dispatch[u.name]
    if on.sum() == 0:
        continue
    starts = np.maximum(on - np.concatenate(([float(u.u0)], on[:-1])), 0.0)
    cost = sum(float(u.fuel_cost(p[t])) for t in range(24) if on[t] > 0.5)
    cost += u.startup_cost * float(starts.sum())
    income = float((price * p).sum())
    print(f"{u.name:6s}{income:14,.0f}{cost:14,.0f}{income - cost:+14,.0f}")
print(f"\n収入の合計 {float((price * demand).sum()):,.0f} 円 / "
      f"総費用 {result.total_cost:,.0f} 円")
plt.figure(figsize=(9, 3.0))
plt.step(np.arange(24), price, where="mid", lw=2, color="tab:red")
plt.xlabel("Hour"); plt.ylabel("Marginal price [JPY/MWh]")
plt.title("Hourly marginal prices (commitment fixed, LP duals)"); plt.show()

# %% [markdown]
# **限界価格で精算すると、動かすと決めた発電機の一部が赤字になります。** G2-1 と
# G2-2 は多くの時刻で限界的な号機なので増分費用ぶんしか受け取れず、無負荷費と起動
# 費を回収できません。全体では収入が費用を上回っているのに個々の号機では足りない。
# これが卸電力市場の古典的な論点（missing money）で、実際の市場は不足分を上乗せ
# 精算（make-whole payment）で埋めます。**起動費と無負荷費は、限界費用という
# 1 つの数には収まりません。**
#
# ## まとめ
#
# - 24 回並べても起動費・最低運転停止時間・最低出力が抜け、毎時の最適化は
#   **実行不可能な計画**を作る
# - 0-1 の $u$、状態遷移、窓和、そして **初期状態からの持ち越し**で問題が閉じる
# - 区分線形近似は凸性のおかげで追加の 0-1 変数が要らず、誤差は必ず過大評価側
# - 解は縮退する。答え合わせは入切表ではなく総費用で
# - 限界費用は入切を固定した線形計画から取る。それは固定費を回収しない
#
# 次の第 08 回では、この計画に **変動性電源**を入れます。太陽光は昼の需要を押し
# 下げて夕方に急速に消えるので、純需要は今日見た形より **立ち上がりが急**になり
# ます。効くのは需要の大きさではなく傾きで、今日は使わなかったランプ制約と出力
# 抑制がそこで主役になります。運転する同期機が減れば系統の慣性も減り、それが
# 第 12 回の「慣性が減ると臨界事故除去時間が短くなる」につながります。
