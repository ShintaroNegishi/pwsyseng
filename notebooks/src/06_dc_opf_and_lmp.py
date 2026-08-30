# %% [markdown]
# # 06 系統制約つき経済配分 — 直流最適潮流（Direct Current Optimal Power Flow: DC-OPF）と地点別限界価格（Locational Marginal Price: LMP）
#
# ## この回のねらい
#
# - 第 05 回の「最も安い出力の組」が、**送電線を守れているとは限らない**ことを見る
# - 系統制約を入れた最適化（直流最適潮流）で、混雑が費用をいくら押し上げるかを測る
# - **ひとつだった $\lambda$ が母線ごとに割れる**こと（ノード価格 LMP）を双対から読む
# - 双対の符号の規約がなぜ要るのかを、わざと逆向きに書いて確かめ、混雑レントを 2 通りで検算する
#
# ## 前回の定式化に足りなかったもの
#
# 第 05 回の経済負荷配分は
#
# $$ \min_{P} \sum_i C_i(P_i) \quad \text{s.t.} \quad \sum_i P_i = D, \quad
#    P_i^{min} \le P_i \le P_i^{max} $$
#
# でした。答えは「増分燃料費が全機で等しい」— **ひとつの $\lambda$** です。よく見るとこの式には
# **母線も線路も出てきません**。電線が無限に太い系統を解いていたのです。今回は線路の容量制約を
# 足します。**ひとつだった $\lambda$ が母線ごとに割れ、最も高い発電機の限界費用を超えさえします。**

# %%
import math
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

import gridops
from gridops import solvers
from gridops.plotting import use_gridops_style, plot_lmp, plot_network_flows

use_gridops_style()

case = gridops.load_case("wscc9")
base = case.base_mva          # [MVA] p.u. と MW の換算に何度も使う
print(case.describe())

# %% [markdown]
# ## 1. 前回の解を、そのまま潮流に流してみる
#
# 需要 315 MW の経済負荷配分を解き、その出力をそのまま直流潮流に流します。**経済的に最適な運転点が
# 系統として実行可能かを、第 05 回では誰も確かめていなかった**のです。

# %%
ed = gridops.economic_dispatch(case, 315.0)
print(ed.summary())

flow = gridops.dc_powerflow(case, dispatch=ed.dispatch)
print(f"\n{'branch':>7} {'flow [MW]':>10} {'rate_a [MW]':>12} {'loading':>9}")
for branch, f_pu in zip(case.branches, flow.flows):
    loading = abs(f_pu) / branch.rate_a
    print(f"{branch.label:>7} {f_pu * base:10.2f} {branch.rate_a * base:12.1f} "
          f"{loading:8.1%}{'  <== over' if loading > 1.0 else ''}")

plot_network_flows(flow, limit="rate_a")
plt.show()

# %% [markdown]
# 枝 **4-6 が 104.7%** で容量を超えています。第 05 回の解は「安いものから使う」だけを見て母線 1 の
# 石炭 3 台を上限まで出し切りました。その 180 MW は母線 4 で 4-5 と 4-6 に分かれますが、分かれ方を
# 決めるのは費用ではなく**リアクタンス**です。経済最適が容量の内側に収まる理由はどこにもありません。
#
# > **断っておくこと**: `rate_a` は皮相電力 $|S|$ の制限で、直流の $P$ と比べるのは近似です（第 04
# > 回のとおり枝 4-5 では交流が 47.6% 大きい）。交流で確かめ直すのは第 09 回の仕事です。
#
# ## 2. 制約を式に入れる — 直流最適潮流
#
# 単一の需給バランスをやめ、**母線ごとに 1 本ずつ**注入等式を書きます。
#
# $$ \min_{p,\theta} \sum_i b_i p_i \;\; \text{s.t.} \;\;
# \underbrace{\sum_{i \in b} p_i - S_{base}(B\theta)_b = d_b}_{\text{双対 } \pi_b}, \;\;
# \underbrace{-\bar f_k \le f_k \le \bar f_k}_{\text{双対 } \mu_k}, \;\;
# P_i^{min} \le p_i \le P_i^{max} $$
#
# 母線別の需給等式で定式化すると、各等式の双対として LMP を直接読み取れます。
# 一方、系統全体の需給バランスを 1 本だけ置く PTDF 定式化でも、系統限界費用と
# 線路容量制約の双対から母線別 LMP を再構成できます。本教材は前者を採用します。比較のため、全枝の熱容量を無限大にした
# 系統も解きます。なお費用は 1 次項 `var_cost` だけなので（CBC は線形計画しか扱えない）、
# `DCOPFResult.total_cost` は無負荷費も 2 次項も含む `DispatchResult.total_cost` と**比べられません**。

# %%
unlimited = replace(case, branches=[replace(b, rate_a=math.inf) for b in case.branches])
free = gridops.dc_opf(unlimited, demand_mw=315.0)
tight = gridops.dc_opf(case, demand_mw=315.0)

print(f"no limits   : {free.total_cost:12,.1f} 円/h  congested={free.is_congested()}")
print(f"with limits : {tight.total_cost:12,.1f} 円/h  congested={tight.is_congested()}")
print(f"混雑による費用増 : {tight.total_cost - free.total_cost:12,.1f} 円/h\n")
for unit in case.units:
    delta = tight.dispatch[unit.name] - free.dispatch[unit.name]
    if abs(delta) > 1e-6:
        print(f"  {unit.name}: {free.dispatch[unit.name]:7.3f} ->"
              f" {tight.dispatch[unit.name]:7.3f} MW"
              f" ({delta:+7.3f} @ {unit.var_cost:,.0f} 円/MWh)")

# %% [markdown]
# 動いたのは 2 台だけです。母線 1 の G1-3 が 10.402 MW 下がり、母線 2 の G2-1 が同じだけ上がり、
# 費用増はちょうど $10.402 \times (12000 - 8200) = 39529$ 円/h です。**混雑の代償とは、安い機を
# 止めて高い機を焚くこと**であり、その量は系統が決めます。この操作を**再給電**と呼びます。
#
# ## 3. 双対の符号は規約である
#
# LMP は注入等式の双対です。ところが**双対の符号は制約をどちら向きに書いたかで反転します**。
# `gridops` の規約は「**右辺に需要を正の符号で置く**」＝ `lp_sum(...) == demand`（`gridops.solvers`
# の docstring と `docs/solver_notes.md`）。この向きなら双対は限界費用 $\partial C/\partial D$ です。

# %%
def lambda_lp(demand_mw: float, *, flipped: bool = False) -> solvers.Solution:
    """単一母線の線形経済負荷配分。バランス制約の向きだけを差し替える。"""
    prob = solvers.problem("ed_sign")
    p = {u.name: solvers.variable(f"p_{u.name}", u.p_min_mw, u.p_max_mw) for u in case.units}
    prob += solvers.lp_sum(u.var_cost * p[u.name] for u in case.units), "fuel_cost"
    if flipped:
        prob += demand_mw - solvers.lp_sum(p.values()) == 0.0, "balance"
    else:
        # TODO(L1): 需給バランスを規約どおりの向きで 1 行書くこと。
        #           右辺に demand_mw を正の符号で置き、制約名は "balance" にする。
        # BEGIN SOLUTION
        prob += solvers.lp_sum(p.values()) == demand_mw, "balance"
        # END SOLUTION
    return solvers.solve(prob, context="符号の規約の実験")


forward = lambda_lp(315.0)
print(f"目的関数 : {forward.objective:12,.1f} 円/h")
print(f"双対 pi  : {forward.duals.get('balance', float('nan')):12,.1f} 円/MWh")
print(f"混雑が無いときの LMP: {sorted({round(v, 6) for v in free.lmp.values()})} 円/MWh")

# %% [markdown]
# 手で組んだ線形計画の双対 12,000 円/MWh が、熱容量を外した直流最適潮流の**全母線の LMP と一致**
# しました。混雑が無ければ LMP は母線によらず等しく、その値が系統 $\lambda$ です。母線 2 の G2-1
# （12,000 円/MWh）が限界機で、その限界費用がそのまま系統全体の価格になっています。（第 05 回の
# 2 次費用の解は $\lambda = 13{,}090$ 円/MWh。違うのは費用関数の形のせいです。）

# %%
# TODO(L3): 下の 2 通りは同じ問題を違う向きで書いたものである。実行して数値を見たうえで、
#           (a) パッケージはどちらの向きを規約に選ぶべきか、
#           (b) 規約を決めずに済ませると下流で何が起きるか、
#           (c) 双対の絶対値を比べるテストがなぜ役に立たないか、
#           を次の markdown セルに 3 行程度で述べること。コードは完成している。
flipped = lambda_lp(315.0, flipped=True)
print(f"{'向き':<26} {'目的関数':>14} {'双対 pi':>12}")
for label, sol in (("lp_sum(p) == demand", forward), ("demand - lp_sum(p) == 0", flipped)):
    print(f"{label:<26} {sol.objective:14,.1f} {sol.duals.get('balance', float('nan')):12,.1f}")

# %% [markdown]
# **模範解答**: 最適値はどちらも 3,343,500 円/h で同じなのに、双対だけが $+12{,}000$ と $-12{,}000$
# に分かれます。等価変形で価格の符号が変わるのですから、バグではなく「**規約が要る**」のです。
# (a) 限界費用が正で読める `lp_sum(p) == demand` を選ぶ。(b) 規約が無いと混雑レントが負（送電権の
# 収入がマイナス）になり、高い母線と安い母線が入れ替わったまま下流に流れる。(c) $|\pi|$ を比べる
# テストは $\pm 12{,}000$ を区別できない。**テストは符号込みで書くこと。**
#
# ## 4. ひとつの $\lambda$ が母線ごとに割れる
#
# 熱容量を戻した解を見ます。混雑した枝の影値 $\mu_k$ と第 04 回の PTDF があれば、価格の割れ方は
# 完全に説明できます。参照母線（slack）を $r$ として、
#
# $$ \pi_b = \pi_r - \sum_k \mu_k \, \mathrm{PTDF}_{k,b} $$
#
# です。**価格差の原因は費用ではなく送電制約である**ことが、この式に出ています。

# %%
print(tight.summary())

H = gridops.ptdf(case)
k46 = next(i for i, b in enumerate(case.branches) if b.key() == (4, 6))
mu46 = tight.congestion_price[(4, 6)]
print(f"\n{'bus':>4} {'LMP':>10} {'pi_ref - mu*PTDF':>18} {'PTDF(4-6)':>11}")
for i, bus_id in enumerate(case.bus_ids):
    print(f"{bus_id:>4} {tight.lmp[bus_id]:10.2f} "
          f"{tight.lmp[1] - mu46 * H[k46, i]:18.2f} {H[k46, i]:11.4f}")

plot_lmp(tight)
plt.show()

# %% [markdown]
# 母線 6 の PTDF が $-0.865$ と大きい（母線 6 に注入すると枝 4-6 の潮流が強く減る）ので、そこの価格が
# 最も高くなります。母線 1 と 4 は PTDF がゼロなので $\pi_r = 8{,}200$ 円/MWh のまま。母線 1 の 8,200 は
# G1-3 の、母線 2 の 12,000 は G2-1 の限界費用です。**混雑すると限界機が母線ごとに立ちます。**
#
# ## 5. 混雑レントを 2 通りで計算する
#
# 損失のない直流の世界では、負荷が払う額と発電が受け取る額は一致しません。その差が**混雑レント**です。
#
# $$ R = \sum_k (\pi_{t(k)} - \pi_{f(k)})\, f_k \qquad\text{および}\qquad
#    R = \sum_k |\mu_k| \, \bar f_k $$
#
# 前者は「安い側で買って高い側で売る差額」、後者は「拘束した枝の影値 × 容量」です。**必ず 2 通り
# 計算して突き合わせること。** 片方だけを信じると符号の取り違えに気づきません。単位に注意。
# `flows` と `rate_a` は **p.u.**、`lmp` は **円/MWh** なので `base` を掛けて MW に直します。

# %%
# TODO(L2): rent_price（価格差形式）と rent_shadow（影値形式）を計算すること。
#           flows[k] と rate_a は p.u. なので base を掛けて MW にすること。
# BEGIN SOLUTION
rent_price = sum(
    (tight.lmp[b.to_bus] - tight.lmp[b.from_bus]) * tight.flows[k] * base
    for k, b in enumerate(case.branches)
)
rent_shadow = sum(tight.congestion_price[b.key()] * b.rate_a * base for b in case.branches)
# END SOLUTION
print(f"price form  : {rent_price:14,.1f} 円/h")
print(f"shadow form : {rent_shadow:14,.1f} 円/h")
print(f"gridops     : {tight.congestion_rent(method='price'):14,.1f} / "
      f"{tight.congestion_rent(method='shadow'):,.1f} 円/h")

payment = sum(tight.lmp[b] * load for b, load in tight.loads_mw.items())
revenue = sum(tight.lmp[u.bus] * tight.dispatch[u.name] for u in case.units)
print(f"\n負荷の支払い : {payment:14,.1f} 円/h")
print(f"発電の受取り : {revenue:14,.1f} 円/h")
print(f"差            : {payment - revenue:14,.1f} 円/h")

# %% [markdown]
# 3 通りが 841,314 円/h で一致します（数円の差は CBC の許容誤差）。影値形式は $10{,}516.42 \times 80$ の
# 掛け算 1 回です（混雑していない枝は $\mu_k = 0$）。混雑レントは**送電容量が足りないことの値段**で、これを線路の増強費用と比べるのが設備計画の入口になります。
#
# ## 6. どこで価格が割れ始めるか
#
# 枝 4-6 の熱容量を 100 MW から絞り、母線ごとの LMP を追いかけます。制約が効いていない間は
# 全母線が同じ高さに並び、**拘束した瞬間に割れます。**

# %%
def with_rating(cap_mw: float) -> gridops.Case:
    """枝 4-6 の常時許容容量だけを差し替えたケースを返す。"""
    return replace(case, branches=[replace(b, rate_a=cap_mw / base)
                                   if b.key() == (4, 6) else b for b in case.branches])


caps = np.arange(100.0, 44.0, -1.0)
curves = {bus: [] for bus in (1, 5, 6, 8)}
for cap in caps:
    solution = gridops.dc_opf(with_rating(cap), demand_mw=315.0)
    for bus in curves:
        curves[bus].append(solution.lmp[bus])

split = abs(free.flows[k46]) * base
fig, ax = plt.subplots(figsize=(8.5, 4.6))
for bus, values in curves.items():
    ax.step(caps, values, where="post", lw=1.8, label=f"bus {bus}")
ax.axvline(split, color="0.35", ls="--", lw=1.3, label=f"free flow = {split:.1f} MW")
ax.axhline(max(u.var_cost for u in case.units), color="crimson", ls=":", lw=1.5,
           label="most expensive unit (20,500 JPY/MWh)")
ax.set(xlabel="Thermal rating of branch 4-6 [MW]", ylabel="LMP [JPY/MWh]",
       title="Where the single lambda splits into nodal prices")
ax.invert_xaxis()
ax.legend(fontsize=9)
plt.show()

# %% [markdown]
# 容量が 83.8 MW（制約を外したときの潮流）を下回った瞬間に 4 本の線がばらけます。それより右では
# 4 本が重なって 12,000 円/MWh に張り付いています。**制約は、拘束するまでは価格に何の影響も与えません。**
# 階段状なのは線形計画の最適基底が区間ごとに一定だからで、段差は発電機が上下限に達した点です。
#
# ## 7. LMP が最も高い発電機の限界費用を超える
#
# 図の赤い点線（20,500 円/MWh = 最も高い G3-2 の限界費用）を、母線 6 の価格が**上に突き抜けています**。
# 「価格は最も高い限界費用で頭打ちのはず」という直感に反しますが、バグではありません。確かめましょう。

# %%
narrow = with_rating(55.0)
narrow_opf = gridops.dc_opf(narrow, demand_mw=315.0)
shifted = dict(narrow_opf.loads_mw)
shifted[6] += 1.0
perturbed = gridops.dc_opf(narrow, loads=shifted)

print(f"最も高い限界費用 : {max(u.var_cost for u in case.units):12,.1f} 円/MWh")
print(f"母線 6 の LMP    : {narrow_opf.lmp[6]:12,.1f} 円/MWh (双対)")
print(f"実際の費用差     : {perturbed.total_cost - narrow_opf.total_cost:12,.1f} 円/MWh"
      "  <- 母線 6 の負荷を +1 MW して解き直した\n")
for unit in case.units:
    delta = perturbed.dispatch[unit.name] - narrow_opf.dispatch[unit.name]
    if abs(delta) > 1e-6:
        print(f"  {unit.name}: {delta:+7.3f} MW @ {unit.var_cost:,.0f} 円/MWh")

# %% [markdown]
# 双対と、負荷を実際に 1 MW 増やして解き直した費用差が小数点以下まで一致します。**LMP は本当に
# $\partial(\text{総費用})/\partial d_b$ である**ことが確かめられました。中身を見ると理由が分かり
# ます。母線 6 で 1 MW 余計に要るとき、G3-1（20,000 円/MWh）が **+1.406 MW**、G1-1（8,000 円/MWh）が
# **−0.406 MW** 動きます。合計は +1 MW ですが、**高い機を 1 MW より多く焚かないと枝 4-6 が守れない**
# のです。$1.406 \times 20000 - 0.406 \times 8000 = 24871$ 円。同じ理屈で **LMP は負にもなります**。
#
# > **あっ、と思うところ**: 価格の上限を決めるのは「一番高い発電機」ではなく**系統**です。
#
# ## まとめ
#
# - 第 05 回の解は枝 4-6 を 104.7% に過負荷させていました。系統制約を入れると再給電が要り、費用は
#   39,529 円/h 増えます。経済最適は、それだけでは実行可能ですらありません。
# - 母線ごとに注入等式を書くと双対が LMP になります。混雑が無ければ全母線が系統 $\lambda$
#   （12,000 円/MWh）に一致し、枝 1 本が拘束した瞬間に 8,200 〜 17,295 円/MWh に割れました。割れ方は
#   $\pi_b = \pi_r - \sum_k \mu_k \mathrm{PTDF}_{k,b}$、第 04 回の PTDF そのものです。
# - 混雑レントは 2 通りの式で一致し（841,314 円/h）、「負荷の支払い − 発電の受取り」に等しい。
#   双対の符号は制約の向きで反転するので、規約を 1 箇所に決め、テストは符号込みで書きます。
#
# ## 次回へ
#
# ここまでは「いまの 1 時点」の話でした。第 07 回では時間の軸を入れて、**どの発電機を起動しておくか**
# を決めます。0-1 変数が入って混合整数計画になり、今回いちばん働いた道具が使えなくなります。
# **混合整数計画に双対は存在しません。** 入切を決めた後に価格を出すには、それを固定して線形計画に
# 落とし直す 2 段階が要ります（`gridops.commitment.marginal_prices`）。LMP を「需要を 1 MW 増やした
# ときの費用差」として実際に測った今回が、なぜその 2 段階が要るのかを理解する足場になります。
