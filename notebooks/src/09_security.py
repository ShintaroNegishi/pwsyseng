# %% [markdown]
# # 09 セキュリティ — N-1 スクリーニングと SCED
#
# ## この回のねらい
#
# - **「いま成り立っている」だけでは不十分**であること（N-1 基準）を理解する
# - LODF で全事故を行列-ベクトル積 1 回ずつ掃き、交流の判定と突き合わせる
# - **熱容量だけを見ると最悪の N-1 を見落とす**ことを実データで確かめ、性能指数 PI が
#   順位を誤ること（masking）も見る
# - 予防的と是正的の SCED の費用差 ＝ **セキュリティの値段**を測る
#
# ## N-1 基準
#
# 要求されるのは「いま制約を守っていること」ではなく、**想定した設備が 1 つ失われても
# 守り続けられること**です。想定事故の集合を $\mathcal{C}$ とすると
#
# $$ |f_\ell^{c}| \le f_\ell^{\max}, \quad
# V_i^{\min} \le |V_i^{c}| \le V_i^{\max}
# \qquad \forall c \in \mathcal{C},\ \forall \ell,\ \forall i $$
#
# $f_\ell^{c}$ は事故 $c$ 後の枝 $\ell$ の潮流、$|V_i^{c}|$ は母線 $i$ の電圧。事故前は
# 常時容量 `rate_a`、事故後は緊急時容量 `rate_b` で見ます（熱容量は原典に無く、この
# 教材のために逆算した自作の値です）。

# %%
import time

import numpy as np
import matplotlib.pyplot as plt

import gridops
from gridops import plotting, solvers
from gridops.plotting import use_gridops_style

use_gridops_style()

case = gridops.load_case("wscc9")
base = gridops.solve_powerflow(case)
print(base.summary())

# %% [markdown]
# ## 1. 想定事故をどう作るか
#
# 枝は 9 本あるのに N-1 候補 `case.contingencies` は **6 件**しかありません。
# **問い: 抜けている 3 本はどれで、なぜ候補から外れているのでしょうか。**

# %%
keys = [branch.key() for branch in case.branches]
print("枝       :", keys)
print("N-1 候補 :", list(case.contingencies))
print("差       :", [k for k in keys if k not in case.contingencies])
print("bridges():", gridops.bridges(case))

try:
    gridops.lodf(case)            # 全枝の LODF を作ろうとすると
except ValueError as error:       # 分母 1 - Psi[k,k] がゼロになる
    print("\nlodf(case) ->", str(error).splitlines()[0][:64], "...")

print("枝 1-4 を開いた後の島 :", gridops.islands(case.without_branch((1, 4))))

# %% [markdown]
# 抜けているのは変圧器 3 本 `(1,4) (2,7) (3,9)`、すなわち **橋**です。第 01 回でトポロジーとして
# （開けば母線 1 が孤立して島が 2 つになる）、第 04 回で代数として（LODF の分母 $1-\Psi[k,k]$ が
# 厳密にゼロ＝両端の間で送る電力の 100% がその枝を通る）見た同じ 3 本で、**事故後潮流という概念
# そのものが成り立ちません**。**ただし「除外」は「健全」ではありません。** なお本教材の N-1 は
# **線路 1 回線の開放だけ**で、発電機事故や多重事故 (N-2) は扱っていません。
#
# ## 2. LODF による高速スクリーニング
#
# 事故を 1 件ずつ交流で解けば正確ですが、実系統では候補が数千件あり運用の周期（数分）に
# 間に合いません。直流近似は線形なので、事故後潮流は $f' = f + \mathrm{LODF}[:,k] f_k$
# と **行列とベクトルの積 1 回**で出ます。橋を避けるため列は候補に限定します。

# %%
rate_b = np.array([branch.rate_b for branch in case.branches])
base_dc = gridops.dc_powerflow(case)
L = gridops.lodf(case, outages=case.contingencies)

started, dc_worst = time.perf_counter(), {}
for key in case.contingencies:
    k = keys.index(key)
    # TODO(L2): 枝 k を開放したときの事故後潮流 post を LODF から作り（開放した
    #           枝はゼロ）、rate_b に対する最大負荷率を dc_worst[key] に入れること。
    # BEGIN SOLUTION
    post = base_dc.flows + L[:, k] * base_dc.flows[k]
    post[k] = 0.0
    dc_worst[key] = float(np.max(np.abs(post) / rate_b))
    # END SOLUTION
dc_seconds = time.perf_counter() - started
print(f"{len(dc_worst)} 件の事故後潮流を {dc_seconds * 1e3:.2f} ms で得た")

# %% [markdown]
# ## 3. 直流の絞り込みと交流の判定を突き合わせる
#
# `screen_n1` に事故前の解を渡すと、候補すべての交流潮流を解き直します。直流の予測と比べます。

# %%
started = time.perf_counter()
report = gridops.screen_n1(case, base, method="ac")
ac_seconds = time.perf_counter() - started

print(f"{'outage':>7}{'DC':>9}{'AC':>9}{'diff':>8}   worst  verdict")
for result in report.ranked(by="worst_loading"):
    dc = dc_worst.get(result.outage, float("nan")) * 100.0
    ac, tag = result.worst_loading * 100.0, f"{result.outage[0]}-{result.outage[1]}"
    worst = f"{result.worst_branch[0]}-{result.worst_branch[1]}"
    print(f"{tag:>7}{dc:8.1f}%{ac:8.1f}%{ac - dc:+8.1f}   {worst:>5}  "
          f"{'secure' if result.is_secure else 'INSECURE'}")
print(f"\n直流 {dc_seconds * 1e3:.2f} ms / 交流 {ac_seconds * 1e3:.2f} ms "
      f"= {ac_seconds / dc_seconds:.0f} 倍")

# %% [markdown]
# 直流はふつう **過小評価**します。定格は皮相電力 $|S|$ の制限なのに直流が持っているのは
# 有効電力 $P$ だけだからで、枝 4-5 の開放では直流 86.2% 対 交流 101.5% と 15 ポイント
# 違います。**しかし誤差は安全側に偏っていません。** 枝 6-9 だけは直流 102.1% 対 交流
# 101.5% で直流が大きく出ます。「一律の余裕を足せば安全側」とは言えないのです。
#
# ## 4. どの設備が運転点を縛っているか
#
# 「どの事故が重いか」ではなく **「どの枝が縛るか」**——各枝の全事故にわたる最大負荷率です。

# %%
worst_by_branch = report.worst_loading_by_branch()
binding = report.binding_branches()

fig, ax = plt.subplots(figsize=(8.0, 3.4))
ax.bar([f"{a}-{b}" for a, b in worst_by_branch],
       [100.0 * v for v in worst_by_branch.values()], edgecolor="k",
       color=["tab:red" if k in binding else "tab:blue" for k in worst_by_branch])
ax.axhline(100.0, color="k", linestyle="--", linewidth=1.0)
ax.set_xlabel("Branch")
ax.set_ylabel("Worst N-1 loading [%]")
ax.set_title("Worst post-contingency loading per branch (rate_b)")
plt.show()

slack = max(v for k, v in worst_by_branch.items() if k not in binding)
print("拘束する枝 :", binding, f"/ 非拘束枝の最大 {slack * 100:.1f}%")

# %% [markdown]
# 拘束するのは **5-7 (112.4%) と 7-8 (112.5%) のちょうど 2 本**で、非拘束枝の最大は 4-5 の
# 91.1%。100% までに 9 ポイントの空きがあるので、この 2 本は「たまたま境界に近い」のでは
# なく **系統の設計上の隘路**です。「どの設備が効いているか」を数えられることが、N-1 解析
# が単なる合否判定でない理由です。
#
# ## 5. この回の山場 — 熱容量だけを見ると何を見落とすか
#
# 電圧を判定に使わない表 (`check_voltage=False`) と、使う表を並べます。括弧つきの `v_min` は
# 「値は計算してあるが判定に使っていない」印です。

# %%
# TODO(L3): 下の 2 つの表を見比べて、電圧を見た瞬間に判定がひっくり返る事故を特定し、
#           なぜ熱容量では捕まらないのかを次の markdown セルに述べること。
thermal_only = gridops.screen_n1(case, base, method="ac", check_voltage=False)
print(thermal_only.to_table())
print(f"\nis_secure（電圧を見ない）: {thermal_only.is_secure}")
print("\n" + report.to_table())
print(f"\nis_secure（電圧を見る）  : {report.is_secure}")

# %% [markdown]
# ### 模範解答
#
# **枝 4-6 の開放**です。熱容量は最悪でも枝 7-8 の **75.7%**（`rate_b` 基準）で 24 ポイント
# の余裕があり、6 件中 5 番目に軽い、まったく健全に見える事故です。ところが母線 6 の電圧は
# **0.9418 p.u.** まで落ちて下限 0.95 を割り、`check_voltage=False` では `secure`、`True`
# では `INSECURE` と判定が反転します。**数字が表に出ているのに、判定は「健全」**なのです。
#
# 捕まらない理由は実装の粗さではありません。直流潮流は $|V|=1$, $r=0$ と置いて無効電力を
# 捨てた近似なので、**電圧という量を原理的に持っていない**からです。閾値をどれだけ下げても
# LODF からこの事故は出てきません。母線 6 は 90 MW の負荷を 4-6 と 6-9 の 2 経路で支えて
# おり、片方を失うと残る経路の無効電力損失が増えて電圧が沈みます。**スクリーニングは
# 絞り込みであって判断ではない** ——これが第 09 回の結論であり、`screen_n1` が「直流で
# 並べ、交流で判定する」2 段構えである理由です。
#
# ## 6. 性能指数 PI とその masking
#
# 事故の順位づけによく使われるのが性能指数
# $\mathrm{PI} = \frac{1}{2n}\sum_\ell w_\ell (f_\ell/f_\ell^{\max})^{2n}$ です。偶数乗なので
# 向きに依らず定格超過が指数関数的に効きますが、**PI ランキングは順位を誤ります**。

# %%
capacity = np.ones(10)
light = np.full(10, 0.95)                            # 95% が 10 本（健全）
heavy = np.concatenate([[2.0], np.full(9, 0.1)])     # 200% が 1 本（危険）
for n in (1, 4):
    a = gridops.performance_index(light, capacity, n=n)
    b = gridops.performance_index(heavy, capacity, n=n)
    print(f"n={n}（{2 * n} 乗）: PI(軽い多数)={a:7.4f}  PI(重い 1 本)={b:7.4f}"
          f"  -> 上位は {'軽い多数' if a > b else '重い 1 本'}")

plotting.plot_contingency_ranking(report)
plt.show()

# %% [markdown]
# $n=1$（2 乗）では、1 本も定格を超えていない「軽い多数」の PI 4.5125 が、200% の過負荷を
# 含む「重い 1 本」の 2.0450 を **上回ります**。重大な 1 本が軽い多数に隠されるので
# **masking** と呼びます。$n=4$（8 乗）なら 0.83 対 32.0 で順位が入れ替わりますが、今度は
# 105% と 110% の差が潰れます。上の図でも PI が下から 2 番目の **4-6（PI 1.033）が赤い
# （不合格）**なのに最下位の 8-9（PI 1.002）は青い（合格）です。
#
# ## 7. SCED — 制約生成でセキュリティを買う
#
# 逸脱が見つかったら運転点を動かします。「全事故に耐える運転点のうち最も安いもの」を直流で解く
# のが SCED です。54 本（事故 6 × 枝 9）の制約を並べる代わりに、**違反した制約だけを足します**
# （制約生成）。

# %%
units, H = list(case.units), gridops.ptdf(case)
load_pu = np.array([bus.pd for bus in case.buses])
G = np.column_stack([H[:, case.index_of(u.bus)] / case.base_mva for u in units])
f0, demand = -(H @ load_pu), float(case.to_mw(load_pu.sum()))
rate_a = np.array([branch.rate_a for branch in case.branches])
post_coeff = {k: (G + np.outer(L[:, i], G[i]), f0 + L[:, i] * f0[i])
              for k, i in ((key, keys.index(key)) for key in case.contingencies)}

active = []                                     # 足した (事故, 枝, 向き)
for sweep in range(1, 6):
    problem = solvers.problem("sced-lite")
    p = {u.name: solvers.variable(f"p_{u.name}", u.p_min_mw, u.p_max_mw) for u in units}
    problem += solvers.lp_sum(u.var_cost * p[u.name] for u in units), "cost"
    problem += solvers.lp_sum(p.values()) == demand, "balance"
    for ell in range(case.n_branch):            # 事故前 (N-0) は rate_a で見る
        e = solvers.lp_sum(G[ell, j] * p[u.name] for j, u in enumerate(units)) + f0[ell]
        problem += (e <= rate_a[ell], f"base_p{ell}")
        problem += (-e <= rate_a[ell], f"base_n{ell}")
    for key, ell, sign in active:               # 事故後 (N-1) は rate_b で見る
        Gc, fc0 = post_coeff[key]
        e = solvers.lp_sum(Gc[ell, j] * p[u.name] for j, u in enumerate(units)) + fc0[ell]
        # TODO(L1): 違反した向き sign (= +1 か -1) の側だけを rate_b 以内に抑える制約を書く
        # BEGIN SOLUTION
        problem += (sign * e <= rate_b[ell], f"post{key[0]}{key[1]}_{ell}_{sign:+.0f}")
        # END SOLUTION
    solution = solvers.solve(problem)
    pv = np.array([solution.values[f"p_{u.name}"] for u in units])
    found = [(key, ell, float(np.sign((Gc @ pv + fc0)[ell])))
             for key, (Gc, fc0) in post_coeff.items() for ell in range(case.n_branch)
             if ell != keys.index(key) and abs((Gc @ pv + fc0)[ell]) > rate_b[ell] + 1e-6]
    added = [item for item in found if item not in active]
    active += added
    print(f"周回 {sweep}: 費用 {solution.objective:11,.0f} 円/h / "
          f"制約 {len(active):2d} 本 / 今回追加 {len(added)} 本")
    if not added:
        break

# %% [markdown]
# 2 周で止まり、**54 本の候補制約のうち実際に足したのは 4 本だけ**です。1 周目（N-0
# の制約しかない）の 3,383,029 円/h が 2 周目で 3,896,400 円/h に上がった差が、N-1 に
# 耐えるために払った額です。足さなかった 50 本は「無視した」のではなく **「効かない
# ことを確かめた」**ものです。緩和問題の最適値は下界、違反ゼロを確認した解は実行
# 可能なので上界。両者が同じ 1 点で一致するので、それが最適解です。
#
# ## 8. セキュリティの値段
#
# 予防的 (preventive) は「事故が起きても発電機を動かさずに耐える」、是正的 (corrective) は
# 「事故後にランプの範囲で再給電できる」前提です。是正的の実行可能領域は予防的を含むので、
# 費用は必ず 予防的 $\ge$ 是正的 $\ge$ 制約なし です。

# %%
preventive = gridops.sced(case)
corrective = gridops.sced(case, mode="corrective")
free = preventive.unconstrained_cost

print(f"制約なし : {free:11,.0f} 円/h")
for name, plan in (("是正的", corrective), ("予防的", preventive)):
    print(f"{name}   : {plan.total_cost:11,.0f} 円/h "
          f"（セキュリティの値段 {plan.cost_of_security(free):+11,.0f}）")
print("\n予防的 / 是正的 = "
      f"{preventive.cost_of_security(free) / corrective.cost_of_security(free):.0f} 倍")

check = gridops.screen_n1(case, gridops.solve_powerflow(case, dispatch=preventive.dispatch),
                          method="ac")
print(f"\n予防的 SCED の運転点を交流で検査 -> is_secure = {check.is_secure}")
print("  残る逸脱 :", [f"{r.outage[0]}-{r.outage[1]}" for r in check.insecure()])

# %% [markdown]
# 是正的は +39,529 円/h（第 06 回の「混雑による費用増」と同じ数です。是正的 SCED の事故前制約は
# 直流最適潮流そのものだから）、予防的は +552,900 円/h で **14 倍**。同じ系統・同じ需要・同じ事故の
# 集合でも、事故後に再給電できるかどうかでセキュリティの値段が桁で変わります。しかも **SCED の
# 答えを交流で検査すると、まだ逸脱が残っています。** 得られたのは「直流の意味で N-1 に耐える運転
# 点」であり、交流の $|S|$ は直流の $P$ より大きく出るのがふつう、電圧に至っては直流の視野の外
# です。**解いて終わりではなく、交流で確かめるところまでが 1 つの作業です。**
#
# ## まとめ
#
# - N-1 基準は「1 つ失っても健全」を要求する。**橋は候補から外す**が、記録には残す
# - LODF なら事故後潮流が積 1 回で出る。直流は $|S|$ を **過小にも過大にも**外す
# - 拘束するのは 9 本中 **5-7 と 7-8 の 2 本だけ**。非拘束枝の最大は 91.1%
# - **枝 4-6 の開放は熱容量では 75.7% で健全なのに、母線 6 が 0.9418 p.u. まで落ちて
#   失格になる。** 熱容量だけのスクリーニングはこれを見落とす
# - PI は masking で順位を誤る。**スクリーニングは絞り込みであって判断ではない**
# - 予防的と是正的の費用差がセキュリティの値段。ここでは 14 倍違う
#
# ### 次回への橋渡し
#
# 枝 5-7 の開放は、静的には「拘束した」——緊急時容量を 12.4% 超える——という結論でした。
# **では、その事故が起きた瞬間はどうでしょうか。** 5-7 は第 17 回の標準事故（母線 7 の三相
# 地絡を線路 5-7 の開放で除去する）と **同じ枝**です。地絡から遮断までの 0.1 秒足らずの
# あいだ、発電機は電力を送り出せずに加速します。ここまで扱ってきた「分〜時」の世界の下に
# 「ミリ秒〜秒」の世界があり、次は同じ枝を **同期を保てるか**で見直します。過負荷は数分
# あれば是正できますが、脱調は数百ミリ秒で決まります。
