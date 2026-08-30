# %% [markdown]
# # 04 直流潮流と感度係数 — 送電電力分布係数（Power Transfer Distribution Factor: PTDF）と線路開放分布係数（Line Outage Distribution Factor: LODF）
#
# ## この回のねらい
#
# - 交流潮流を最適化に載せられる **線形の形**に落とし、捨てたものを交流との差で測る
# - **PTDF** を解析式と数値微分の両方で作り、一致を確かめる
# - **LODF を公式として覚えず、補償定理から導く**
# - 分母がゼロになる枝が、数値の破綻ではなく **トポロジーの事実**だと知る
#
# 第 02 回の Newton 法は解くたびに反復が要ります。N-1 では「事故 1 件につき潮流計算 1 回」を
# 何百回と繰り返し、第 06 回の「潮流の制約つきで費用を最小にする」も非線形の等式制約が
# ある限り線形計画には載りません。**直流近似の値打ちは精度ではなく線形性にあります。** 下の
# 4 つを置くと、枝潮流と母線注入は位相 $\theta$ について完全に線形になります（$A$ は枝-母線
# 接続行列。枝の行の送り出し側に $+1$、受け側に $-1$。`gridops.incidence_matrix(case)`）。
#
# | 仮定 | 捨てるもの |
# |---|---|
# | $\|V_i\| \simeq 1$ p.u. | 電圧の情報。電圧逸脱は原理的に見えない |
# | $r \ll x$（抵抗を無視） | **損失**。枝の両端の $P$ が符号だけ逆になる |
# | $\sin\theta_{ij} \simeq \theta_{ij}$, $\cos\theta_{ij} \simeq 1$ | 重潮流時の非線形性 |
# | 充電容量・母線シャントを無視 | 無効電力の授受。$Q$ という量が消える |
#
# $$ f_\ell = \frac{\theta_f - \theta_t}{\tau_\ell x_\ell}, \qquad
# P = B'\theta, \qquad B' = A^{\mathsf T}\operatorname{diag}(1/\tau x)\,A $$

# %%
import time
import numpy as np
import matplotlib.pyplot as plt

import gridops
from gridops.plotting import use_gridops_style

use_gridops_style()

case = gridops.load_case("wscc9")
ac = gridops.solve_powerflow(case)
dc = gridops.dc_powerflow(case)
keys = [branch.key() for branch in case.branches]
labels = [f"{a}-{b}" for a, b in keys]
print(f"交流: Newton {ac.iterations} 反復 / slack 出力 {ac.slack_power:.6f} p.u.")
print(f"直流: 反復なし（連立 1 次方程式を 1 回）/ 位相の基準は母線 {dc.slack}")

# %% [markdown]
# ## 1. 交流と直流はどれだけ違うか
#
# 直流潮流は $P$ しか持ちませんが、熱容量 `rate_a` は **皮相電力** $|S|$ の制限です。

# %%
apparent = ac.apparent_flows()
s_ac = np.array([apparent[key] for key in keys])
p_dc = np.abs(dc.flows)
gap = 100.0 * (s_ac - p_dc) / p_dc
print(f"{'branch':>8}{'AC |S|':>10}{'DC |P|':>10}{'差 [%]':>9}")
for name, a, d, g in zip(labels, s_ac, p_dc, gap):
    print(f"{name:>8}{a:10.4f}{d:10.4f}{g:9.1f}" + ("  <--" if g > 40.0 else ""))

y = np.arange(case.n_branch)[::-1]
fig, ax = plt.subplots(figsize=(8.0, 4.2))
ax.barh(y + 0.19, s_ac, height=0.36, label="AC apparent power |S|")
ax.barh(y - 0.19, p_dc, height=0.36, label="DC active power |P|")
ax.set_yticks(y, labels)
ax.set_xlabel("Branch flow [p.u.]")
ax.set_ylabel("Branch")
ax.set_title("What the DC approximation throws away")
ax.legend()
plt.show()

# %% [markdown]
# **枝 4-5 で交流の $|S| = 0.5614$ に対し直流の $P = 0.3803$。交流のほうが 47.6%
# 大きい。** `rate_a` は 1.00 p.u. なので負荷率は交流 56%、直流 38% です。**直流の値
# をそのまま定格と比べると過負荷を「健全」と誤判定します。** 差の正体は無効電力で、
# 負荷母線へ $Q$ を送り込む枝（4-5, 8-9）ほど $|S|$ が $P$ から離れます。
#
# ## 2. 直流潮流を自分で組む
#
# $B'$ は行和がゼロなので特異です（全母線を同じ位相にすれば潮流は流れない）。slack の
# 行と列を落として初めて正則になります。「位相の絶対値には意味がなく差にだけ意味が
# ある」という物理が行列の階数に現れたものです。解くのは $B'_{rr}\theta_r = P_r$
# （$\theta_{slack}=0$）だけ。反復も収束判定もありません。

# %%
A = gridops.incidence_matrix(case)
b_series = np.array([1.0 / (br.tap * br.x) for br in case.branches])
slack_index = case.index_of(dc.slack)
keep = np.delete(np.arange(case.n_bus), slack_index)

def dc_flows(p_injection):
    """母線注入 [p.u.] から (枝潮流, 母線位相) を返す。"""
    theta, flow = np.zeros(case.n_bus), np.zeros(case.n_branch)
    # TODO(L2): B' = A^T diag(1/(tau x)) A を組み、slack の行と列を落とした
    #           B'_rr theta_r = P_r を解いて theta[keep] を埋め、枝潮流を作ること。
    # BEGIN SOLUTION
    bprime = A.T @ (b_series[:, None] * A)
    theta[keep] = np.linalg.solve(bprime[np.ix_(keep, keep)], p_injection[keep])
    flow = b_series * (A @ theta)
    # END SOLUTION
    return flow, theta

B = gridops.susceptance_matrix(case)  # 上で組んだ bprime と一致する
p_spec, _ = case.bus_injection()
f_mine, theta_mine = dc_flows(p_spec)
p_slack_dc = (B @ theta_mine)[slack_index]
print(f"dc_powerflow との最大の差 {np.abs(f_mine - dc.flows).max():.2e} / "
      f"B' の階数 {np.linalg.matrix_rank(B)}（母線数 {case.n_bus}）")
print(f"参照解が slack に与えた注入 {p_spec[slack_index]:+.6f}（使われない）/ "
      f"直流の方程式が返す値 {p_slack_dc:+.6f}")
print(f"交流の slack 出力 {ac.slack_power.real:.6f} との差 "
      f"{ac.slack_power.real - p_slack_dc:.6f} = 交流の総損失 {ac.losses:.6f}")

# %% [markdown]
# **slack に与えた注入は使われません。** slack の行を落としたので、その母線の $P$ は方程
# 式に入らず、残り全部の帳尻として決まります。直流に損失は無いのでその値は「総需要 -
# 他機の出力」ちょうど 0.670000 p.u. になり、交流の 0.716410 p.u. との差 **0.046410
# p.u. がそのまま総損失**です。交流ではこの 4.64 MW を slack が引き受けますが、**実際
# の系統に「損失を全部引き受ける発電機」はありません。** 分担の仕方が第 05 回の題材です。
#
# ## 3. PTDF — 注入を動かすと潮流はどう動くか
#
# $\mathrm{PTDF}[\ell, i]$ は「母線 $i$ に 1 p.u. 注入し、**slack から 1 p.u. 引き
# 抜いた**ときの枝 $\ell$ の潮流」で、線形なので偏微分 $\partial f_\ell/\partial P_i$
# でもあります（$X$ は slack の行と列にゼロを詰め戻した行列）。解析式と数値微分の
# 両方で作ります。「合っている」ことより **どちらかが間違っていれば気づける**ことに
# 意味があります。
#
# $$ \mathrm{PTDF} = \operatorname{diag}(1/\tau x)\,A\,X, \qquad
# X = \begin{pmatrix} 0 & 0 \\ 0 & (B'_{rr})^{-1} \end{pmatrix} $$

# %%
X = np.zeros((case.n_bus, case.n_bus))
X[np.ix_(keep, keep)] = np.linalg.inv(B[np.ix_(keep, keep)])
# TODO(L1): 上の式の PTDF を 1 行で書くこと（b_series, A, X を使う）。
# BEGIN SOLUTION
H_manual = (b_series[:, None] * A) @ X
# END SOLUTION

H = gridops.ptdf(case)
eps, H_numeric = 1e-4, np.zeros_like(H)
for i in range(case.n_bus):
    bump = np.zeros(case.n_bus)
    bump[i] += eps
    bump[slack_index] -= eps
    H_numeric[:, i] = (dc_flows(p_spec + bump)[0] - f_mine) / eps
print(f"解析式 vs gridops.ptdf : {np.abs(H_manual - H).max():.2e}")
print(f"解析式 vs 数値微分     : {np.abs(H_numeric - H_manual).max():.2e}")
print(f"母線 5 に 1 p.u. 注入すると枝 4-5 は {H[keys.index((4, 5)), 4]:+.4f} p.u. 動く")

# %% [markdown]
# ## 4. slack を変えると何が変わるか
#
# PTDF の **1 列は slack の取り方に依存します**。slack 母線の列は恒等的にゼロです（自分
# に注入して自分から引き抜けば何も動かない）。「母線 6 の PTDF」は slack を言わなければ
# 意味を持ちません。一方 **列の差** $\mathrm{PTDF}[:,i]-\mathrm{PTDF}[:,j]$ は「母線 $i$
# から $j$ へ 1 p.u. 送る」という slack を含まない取引なので依存しません。下の表で
# 第 2・3 列は食い違い、第 4・5 列は一致します。

# %%
H1, H5 = gridops.ptdf(case, slack=1), gridops.ptdf(case, slack=5)
src, dst = case.index_of(6), case.index_of(9)
print(f"{'branch':>8}{'col6|s=1':>10}{'col6|s=5':>10}{'6->9|s=1':>10}{'6->9|s=5':>10}")
for i, name in enumerate(labels):
    print(f"{name:>8}{H1[i, src]:10.4f}{H5[i, src]:10.4f}"
          f"{H1[i, src] - H1[i, dst]:10.4f}{H5[i, src] - H5[i, dst]:10.4f}")
print(f"列そのものの差 {np.abs(H1[:, src] - H5[:, src]).max():.4f} / 列の差の差 "
      f"{np.abs((H1[:, src] - H1[:, dst]) - (H5[:, src] - H5[:, dst])).max():.2e}")

# %% [markdown]
# ## 5. LODF を補償定理から導く
#
# 枝 $k$（母線 $m$-$n$）の開放とは「その枝の潮流をゼロにすること」です。ならば **枝を残し
# たまま**、両端に等価注入対 $+\Delta$（母線 $m$）/ $-\Delta$（母線 $n$）を置いて打ち消せ
# ばよい。これが補償定理です。母線 $m$ から $n$ へ 1 p.u. 送ったときの枝 $\ell$ の変化を
# $\Psi[\ell,k]=\mathrm{PTDF}[\ell,m]-\mathrm{PTDF}[\ell,n]$ と書くと（行列では
# $\Psi = \mathrm{PTDF}\,A^{\mathsf T}$）、注入対のうち枝 $k$ を通るのは $\Psi[k,k]\Delta$、
# 残りが迂回して枝 $k$ の潮流を減らすので
#
# $$ f_k + \Psi[k,k]\,\Delta - \Delta = 0 \;\Longrightarrow\;
#    \Delta = \frac{f_k}{1 - \Psi[k,k]}, \qquad
#    \mathrm{LODF}[\ell,k] = \frac{\Psi[\ell,k]}{1 - \Psi[k,k]} $$
#
# となります（$\ell \ne k$、対角は $-1$）。**slack が式のどこにも出てきません。**

# %%
Psi = H @ A.T
k = keys.index((4, 5))
delta, column = 0.0, np.zeros(case.n_branch)
# TODO(L2): 上の導出をそのまま書くこと。等価注入 delta と LODF の列 column
#           （対角は下で -1 に上書きするので触らなくてよい）。
# BEGIN SOLUTION
delta = dc.flows[k] / (1.0 - Psi[k, k])
column = Psi[:, k] / (1.0 - Psi[k, k])
# END SOLUTION
column[k] = -1.0

post = dc.flows + column * dc.flows[k]
exact = gridops.dc_powerflow(case, removed_branches=[(4, 5)]).flows
psi5 = H5 @ A.T   # slack を母線 5 に変えて作り直した Psi
print(f"枝 4-5: 事故前 f_k = {dc.flows[k]:+.4f} / 等価注入 Delta = {delta:+.4f} p.u. /"
      f" slack=5 の列との差 "
      f"{np.abs(np.delete(column - psi5[:, k] / (1.0 - psi5[k, k]), k)).max():.2e}")
print(f"{'branch':>8}{'LODF':>9}{'補償定理':>11}{'解き直し':>11}")
for name, c, p, e in zip(labels, column, post, exact):
    print(f"{name:>8}{c:+9.4f}{p:+11.4f}{e:+11.4f}")
print(f"補償定理と解き直しの最大の差 : {np.abs(post - exact).max():.2e}")

# %% [markdown]
# ### ここで手を止めること
#
# **LODF の値が $0$ か $\pm 1$ しかありません。** 感度係数と言われて期待した「30% が
# こっち、70% があっち」が出てこない。理由は WSCC 9 母線の網目部分が **たった 1 つの
# 環状路** 4-5-7-8-9-6-4 だからです。環の 1 本を切れば、そこを流れていた電力は反対回り
# に全部行くしかなく、分岐先がありません。変圧器 3 本の行がゼロなのも同じで、発電機の
# 出力を据え置く限り変圧器の潮流は環の切り方に左右されません。形の情報は
# 分母に残り、環の枝では $1-\Psi[k,k]=x_k/\sum_{\ell\in\text{loop}} x_\ell$ です。
#
# ## 6. 分母がゼロになる枝

# %%
bridge_keys = gridops.bridges(case)
loop_x = sum(br.x for br in case.branches if br.key() not in bridge_keys)
denominator = 1.0 - np.diag(Psi)
ratio = np.array([np.nan if b.key() in bridge_keys else b.x / loop_x for b in case.branches])
print(f"橋: {bridge_keys} / 分母と x_k/sum(x) の差 {np.nanmax(abs(denominator - ratio)):.1e}")
# TODO(L3): 橋の分母がなぜ厳密にゼロなのかを、下の markdown セルに 3 行で
#           述べること。「数値誤差だから tolerance を緩める」は誤りである。
#           「橋」「島」「両端の間で送る電力がどこを通るか」を必ず使うこと。
try:
    gridops.lodf(case)
except ValueError as exc:
    print("lodf(case) は例外 :", str(exc).split("。")[0], "…")
L = gridops.lodf(case, outages=case.contingencies)
print("outages= なら通る。橋の列は NaN:", bool(np.isnan(L[:, 0]).all()))

mask = np.isnan(ratio)
fig, ax = plt.subplots(figsize=(8.0, 3.4))
ax.bar(labels, denominator, color="tab:blue")
ax.plot(labels, ratio, "kd", ms=7, label="x_k / sum(x) over the mesh loop")
ax.plot(np.array(labels)[mask], denominator[mask], "rx", ms=12, mew=2.5,
        label="bridge: denominator is exactly 0")
ax.set(xlabel="Outaged branch k", ylabel="1 - PTDF[k, (m,n)]", ylim=(-0.02, 0.30),
       title="LODF denominator: exactly zero on the three bridges")
ax.legend(fontsize=9)
plt.show()

# %% [markdown]
# ### L3 の模範解答
#
# $\Psi[k,k]$ は「枝 $k$ の両端の間で 1 p.u. 送ったとき、そのうち枝 $k$ を通る割合」です。
# 枝 $k$ が **橋**なら両端を結ぶ道はその枝しかないので割合は必ず 1、分母は厳密にゼロ。
# 橋を開放すれば系統は 2 つの **島**に分かれ、島ごとに需給が独立するので「事故後潮流」
# という量そのものが定義できません。ゼロ割りは数値の破綻ではなく答えが存在しないことの
# 正しい報告で、`tolerance` を緩めても意味のある値は出ません。
#
# ## 7. N-1 を行列とベクトルの積で掃く
#
# $f' = f + \mathrm{LODF}[:,k]\,f_k$ で、潮流を解き直さずに N-1 が掃けます。同じ
# ことを交流でやると事故ごとに Newton 法を回すことになります。両方比べます。

# %%
rate_b = np.array([br.rate_b for br in case.branches])
start = time.perf_counter()
dc_worst = []
for outage in case.contingencies:
    j = keys.index(outage)
    after = dc.flows + L[:, j] * dc.flows[j]
    after[j] = 0.0
    dc_worst.append(float((np.abs(after) / rate_b).max()))
t_dc = time.perf_counter() - start
start = time.perf_counter()
ac_worst = []
for outage in case.contingencies:
    trial = gridops.solve_powerflow(case.without_branch(outage, keep_generation=True))
    ac_worst.append(max(trial.loading("rate_b").values()))
t_ac = time.perf_counter() - start
print(f"直流 {t_dc * 1000:.2f} ms / 交流 {t_ac * 1000:.1f} ms（{t_ac / t_dc:.0f} 倍）")
for (m, n), d, a in zip(case.contingencies, dc_worst, ac_worst):
    print(f"  {m}-{n} 開放: 直流 {d * 100:5.1f}%  交流 {a * 100:5.1f}%"
          + ("  <-- 直流は見逃す" if d < 1.0 <= a else ""))
# %% [markdown]
# **枝 4-5 の開放で、直流は最悪 86.2%、交流は 101.5% と言います。** 直流の値に 100% の
# しきい値をかけただけのスクリーニングは、この事故を「健全」として捨ててしまう。だから
# 直流は **絞り込み**に使い、残った候補を交流で **判定**します。枝 4-6 の開放はさらに厄介
# で、熱容量は最悪 75.7% と余裕なのに **母線 6 の電圧が 0.9418 p.u. まで落ちて下限 0.95
# を割ります**。$|V|=1$ と置いた直流には見えません（第 09 回）。
#
# ## まとめ
#
# - 直流近似が捨てるのは電圧・無効電力・損失。**枝 4-5 では交流の $|S|$ が直流の
#   $P$ より 47.6% 大きい**ので、熱容量の判定を直流で行ってはいけない
# - 損失 0.046410 p.u. はすべて slack に乗る。誰が分担すべきかが第 05 回
# - PTDF の 1 列は slack に依存するが、**列の差と LODF は依存しない**
# - LODF は覚える公式ではなく補償定理から出る。環が 1 つの本系統では値が $\pm 1$ と $0$
# - **分母がゼロになる枝は橋である。**`lodf(case, outages=...)` で候補を限定する
#
# 次回（第 05 回）は、いま「帳尻」として slack に押しつけた 0.046410 p.u. の損失と発電機
# の費用に目を向けます。同じ 315 MW をどう分担すれば費用が最小になるか。そこで出る
# ペナルティファクタは、**本回の感度係数と同じ「注入を動かしたときの系統の応答」**を、
# 潮流ではなく損失について測ったものです。第 06 回ではこの PTDF が直流最適潮流の系統
# 制約になり、ひとつの $\lambda$ が母線ごとのノード価格に割れます。
