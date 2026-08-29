# %% [markdown]
# # 02 潮流計算 — Newton-Raphson
#
# ## この回のねらい
#
# - 第 01 回で「一致するか確かめた」式を、今度は **ゼロにする方程式**として解く
# - ヤコビアン 4 ブロックを自分で組み、$|N|,|M| \ll |H|,|L|$ を実際に **測る**
# - 収束判定を **ミスマッチ**で行い、修正量で行ってはいけない理由を実例で見る
# - Newton / Gauss-Seidel / Fast Decoupled の収束の違いを 1 枚に描く
#
# ## 解くべき方程式
#
# $$
# P_i = \sum_j |V_i||V_j|(G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij}),
# \quad
# Q_i = \sum_j |V_i||V_j|(G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij})
# $$
#
# 第 01 回はこの式に参照解の電圧を入れ、発電と負荷に一致するかを **検算**しました
# （$\theta_{ij} = \theta_i - \theta_j$）。今回は逆向きに、指定注入 $(P^{sp}, Q^{sp})$
# に対し $\Delta P_i = P_i^{sp} - P_i = 0$、$\Delta Q_i = Q_i^{sp} - Q_i = 0$ を満たす
# 電圧を探す。**検算の式と解くべき方程式は同じもの**で、違うのはどちらを既知とみなす
# かだけ。slack の $P$ 式と slack・PV の $Q$ 式は「未知の $P, Q$ を後から決める式」
# なので落とし、残る $n_{PV} + 2n_{PQ}$ 本が未知数の数に一致します。

# %%
import numpy as np
import matplotlib.pyplot as plt

import gridops
from gridops.plotting import plot_convergence, plot_network_flows, plot_voltage_profile
from gridops.powerflow import jacobian, jacobian_blocks, mismatch
from gridops.ybus import build_ybus

gridops.use_gridops_style()

case = gridops.load_case("wscc9")
Y = build_ybus(case)
_, pv_idx, pq_idx = case.type_indices()
non_slack = np.array(sorted(list(pv_idx) + list(pq_idx)))
print(case.describe())

# %% [markdown]
# ## 1. ミスマッチ — 「まだどれだけ釣り合っていないか」
#
# 初期値は **フラットスタート**（PQ 母線は $|V|=1.0$、位相は全部 0）。位相差がゼロ
# なら線路に有効電力はほとんど流れないので、そのミスマッチは **指定注入そのもの**に
# ほぼ等しい。並びは `[ΔP(slack 以外); ΔQ(PQ のみ)]` です。

# %%
v_flat = np.array([bus.v_set for bus in case.buses])
theta_flat = np.zeros(case.n_bus)

# TODO(L1): フラットスタートでのミスマッチを 1 行で求めること。
#           引数は (case, Y, |V|, theta) の順である。
# BEGIN SOLUTION
residual = mismatch(case, Y, v_flat, theta_flat)
# END SOLUTION

p_ids = [case.buses[i].id for i in non_slack]
q_ids = [case.buses[i].id for i in pq_idx]
peak = float(np.abs(residual).max())
print(f"ミスマッチの長さ {residual.size} / 未知数 {case.n_unknowns()}")
print("  dP:", ", ".join(f"{b}:{x:+.3f}" for b, x in zip(p_ids, residual[:len(p_ids)])),
      "\n  dQ:", ", ".join(f"{b}:{x:+.3f}" for b, x in zip(q_ids, residual[len(p_ids):])))
print(f"  最大 |ミスマッチ| = {peak:.4f} p.u. ({case.to_mw(peak):.1f} MW) — 母線 "
      f"{(p_ids + q_ids)[int(np.argmax(np.abs(residual)))]}")

# %% [markdown]
# 最大は母線 2 の $\Delta P = 1.63$ p.u.、**163 MW ぶん釣り合っていない**。G2 の発電
# そのものです。ミスマッチは p.u. の電力そのものだから `tol=1e-10` を「どの母線でも
# $10^{-8}$ MW 以内で釣り合っている」と訳せる。**この読みやすさが後で効きます。**
#
# ## 2. ヤコビアン — 4 つのブロック
#
# 未知数を $[\Delta\theta;\ \Delta|V|/|V|]$ の順に並べると
#
# $$
# J = \begin{bmatrix} H & N \\ M & L \end{bmatrix}
#   = \begin{bmatrix} \partial P/\partial\theta & (\partial P/\partial|V|)|V| \\
#     \partial Q/\partial\theta & (\partial Q/\partial|V|)|V| \end{bmatrix}
# $$
#
# $N, L$ に $|V|$ を掛けるのは修正量を $\Delta|V|/|V|$（無次元）で取るためで、4 ブ
# ロックがすべて電力の次元になり成分の大きさが揃います。**この規約は教科書によって
# 違う**ので、文献と数値を比べる前に確認すること。$P_i = \sum_j a_{ij}$、$a_{ij} =
# |V_i||V_j|(G_{ij}\cos\theta_{ij} + B_{ij}\sin\theta_{ij})$ を $\theta_j$ で
# 微分すると、対角だけ形が変わります。
#
# $$
# H_{ij} = |V_i||V_j|(G_{ij}\sin\theta_{ij} - B_{ij}\cos\theta_{ij})\ \ (i \ne j),
# \qquad H_{ii} = -Q_i - B_{ii}|V_i|^2
# $$
#
# 4 ブロックを個別に取り出せるのは **$|N|,|M| \ll |H|,|L|$ を測らせるため**です。送電
# 線は $x \gg r$ なので有効電力は主に位相差が、無効電力は主に電圧差が決める。

# %%
G, B = Y.real, Y.imag
v_complex = v_flat * np.exp(1j * theta_flat)
Q_flat = (v_complex * np.conj(Y @ v_complex)).imag
H_full = np.zeros((case.n_bus, case.n_bus))
for i in range(case.n_bus):
    for j in range(case.n_bus):
        # TODO(L2): 上の 2 式のとおりに H_full[i, j] を埋めること。
        #           対角 (i == j) と非対角で式が違うことに注意。
        # BEGIN SOLUTION
        if i == j:
            H_full[i, j] = -Q_flat[i] - B[i, i] * v_flat[i] ** 2
        else:
            angle = theta_flat[i] - theta_flat[j]
            H_full[i, j] = v_flat[i] * v_flat[j] * (
                G[i, j] * np.sin(angle) - B[i, j] * np.cos(angle))
        # END SOLUTION

H_ref = jacobian_blocks(case, Y, v_flat, theta_flat)[0]
print(f"自作 H と jacobian_blocks の H の差 = "
      f"{np.abs(H_full[np.ix_(non_slack, non_slack)] - H_ref).max():.2e}")

solution = gridops.solve_powerflow(case)
H, N, M, L = jacobian_blocks(case, Y, solution.v, solution.theta)
print("  " + "  ".join(f"max|{n}| = {np.abs(b).max():6.2f}"
                       for n, b in (("H", H), ("N", N), ("M", M), ("L", L))))
print(f"  max|N|/max|H| = {np.abs(N).max() / np.abs(H).max():.3f}"
      f"      max|M|/max|L| = {np.abs(M).max() / np.abs(L).max():.3f}")

J = jacobian(case, Y, solution.v, solution.theta)   # [[H, N], [M, L]] を 1 つに

# %% [markdown]
# 比は 0.08 と 0.10。**非対角ブロックは対角ブロックの 1 割しかない**。この小ささが
# **Fast Decoupled 法の唯一の根拠**です。$N = M = 0$ と置けば $H, L$ を定数行列で近似
# できる。逆に $r \sim x$ の配電系統では前提が崩れるので、「Fast Decoupled は収束が
# 遅い」という観察は **系統の性質か XB / BX という版の選択の話**です。
#
# ## 3. Newton の 1 歩を自分で回す
#
# $J\,\Delta x = \Delta S$ を解いて $\theta \leftarrow \theta + \Delta\theta$、
# $|V| \leftarrow |V|(1 + \Delta|V|/|V|)$ と更新します。**2 番目が掛け算**（$L$ に
# 掛けた $|V|$ を戻している）ことに注意。

# %%
def newton_trace(target, max_iter=12, tol=1e-10):
    """Newton 法を 1 歩ずつ回し、ミスマッチ・修正量・条件数を記録する。"""
    Yt, theta = build_ybus(target), np.zeros(target.n_bus)
    v = np.array([bus.v_set for bus in target.buses])
    _, pv, pq = target.type_indices()
    ns = np.array(sorted(list(pv) + list(pq)))
    rows = []
    for _ in range(max_iter):
        r, Jk = mismatch(target, Yt, v, theta), jacobian(target, Yt, v, theta)
        step = np.linalg.solve(Jk, r)
        rows.append((np.abs(r).max(), np.abs(step).max(), np.linalg.cond(Jk)))
        if rows[-1][0] < tol:
            break
        theta[ns] += step[:ns.size]
        v[pq] *= 1.0 + step[ns.size:]
    return np.array(rows)


def show(trace, tag):
    print(f"[{tag}] k {'|dS|inf [pu]':>15}{'|dx|inf':>13}{'|dx|/|dS|':>11}{'cond(J)':>10}")
    for k, (ds, dx, cond) in enumerate(trace):
        print(f"      {k:>3} {ds:>15.3e}{dx:>13.3e}{dx / ds:>11.3f}{cond:>10.1f}")


base = newton_trace(case)
show(base, "base")

# %% [markdown]
# ミスマッチの指数が $10^{0} \to 10^{-1} \to 10^{-3} \to 10^{-7} \to 10^{-14}$ と
# **倍々に増えて**います。これが二次収束です。一番上の行をもう一度見てください。
# ミスマッチは 1.63 p.u.（163 MW）もあるのに **修正量は 0.17 しかない**。修正量の
# 小ささは釣り合いを意味しません。両者は $\Delta x = J^{-1}\Delta S$ で結ばれ、比
# $|dx|/|dS|$ は $\|J^{-1}\|$ そのもの、条件数しだいで動く量だからです。この系統は
# 条件数が 58 前後で安定なので比も 0.07〜0.13 に収まり、どちらで判定しても同じ答えに
# なる。**易しい問題ではこの誤りは表に出ません。** 負荷を 2.4 倍したら？

# %%
heavy = newton_trace(case.scaled(2.4, keep_generation=True))
show(heavy, "x2.4")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
for ax, trace, title in ((axes[0], base, "base case"), (axes[1], heavy, "load x 2.4")):
    ax.semilogy(trace[:, 0], "o-", label=r"mismatch $\|\Delta S\|_\infty$")
    ax.semilogy(trace[:, 1], "s--", label=r"step $\|\Delta x\|_\infty$")
    ax.set(title=title, xlabel="Iteration")
    ax.legend()
axes[0].set_ylabel("Infinity norm")
plt.tight_layout()
plt.show()

# %% [markdown]
# **ここが今日の山です。** 重負荷ケースの $k=2 \to 3$ で修正量は $0.130 \to 0.116$ と
# **小さくなっています**。修正量で判定していれば「動きが止まってきた」と読むところ。
# ところが同じ時点のミスマッチは 0.10 p.u.、**10 MW 不足したまま**で、その先も減りま
# せん（$k=4$ で修正量は逆に 0.29 へ跳ね上がる）。条件数が 59 から 900 超へ悪化し、
# 悪条件のヤコビアンが修正量とミスマッチの対応を壊したのです。だから **収束判定は
# ミスマッチで行います。** 解が無いのか解き方が悪いのかの切り分けは第 03 回で。
#
# ## 4. 3 つの解法 — 同じ方程式、違う近づき方

# %%
solutions = {name: gridops.solve_powerflow(case, method=name)
             for name in ("newton", "fast_decoupled", "gauss_seidel")}
for name, sol in solutions.items():
    print(f"  {name:15s} 反復 {sol.iterations:4d} 回 / 最終ミスマッチ "
          f"{sol.mismatch_history[-1]:.2e} p.u. / 総損失 {sol.losses:.6f} p.u.")

plot_convergence(solutions)
plt.show()

# %% [markdown]
# 総損失が 6 桁一致しました。3 つは **同じ方程式を解いている**のだから当然で、違うの
# は「どう近づくか」だけ。Newton は下に折れ曲がり（二次収束）、Gauss-Seidel はほぼ直線
# （一次収束）で、61 回かかるのは収束次数の帰結。Fast Decoupled は中間ですが 1 反復が
# 最も軽い（$B', B''$ を 1 度 LU 分解）ので、**反復回数だけで速さを語れません。**
#
# ## 5. 教科書解との突き合わせ、そして解を読む
#
# **どこまでの差なら丸めで説明できるかを先に見積もってから**表を読むこと。参照解は
# 出典の掲載桁数（4 桁）で丸めてあるので $5\times10^{-5}$ の差は消せません。

# %%
print(gridops.check_against_reference(case, solution))
print(solution.summary())

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
plot_voltage_profile(solution, ax=axes[0])
plot_network_flows(solution, ax=axes[1])
plt.tight_layout()
plt.show()
print(f"枝 4-5 の皮相電力 |S| = {solution.apparent_flows()[(4, 5)]:.4f} p.u.")

# %% [markdown]
# 最大差は $|V|$ で 4.7e-5、位相で 4.9e-5 deg。丸めの範囲内なので **実装は正しい**と
# 判断できます。内部起電力の表で G3 の位相だけ 8.8e-3 deg と 2 桁大きいのも誤りでは
# ありません。G3 は $Q$ が小さく $x'_d$ が大きいので、出典が $Q$ を 3 桁で丸めた影響が
# 増幅されている。**「G1, G2 は合うのに G3 だけずれる」と読めれば、原因を実装でなく
# データの丸めに絞り込めます。** 電圧プロファイルでは slack と PV 母線の $|V|$ が
# 設定値に並び、**PQ 母線だけが解として動いている**のが見えます。
#
# **ここで得た $(|V|,\theta)$ が、この科目の残り全部の入力になります。** 第 04 回は枝
# 4-5 の交流 $|S| = 0.5614$ を直流近似の $P = 0.3803$ と比べ（**47.6% の差**。熱容量の
# 判定を直流で行ってはいけない根拠）、第 05 回は総損失 0.046410 p.u. をペナルティ
# ファクタの源にし、第 09 回はこの電圧から N-1 の事故後潮流を解き直します。
# 第 17 回で `to_genstab(case, solution)` に渡せば **自分の潮流解**で安定度が解けます。
#
# ## 6. 演習 (L3): 解析ヤコビアンを信用してよいか
#
# 本モジュールのヤコビアンは **解析式**です（`genstab` の数値線形化とは方針が違う）。
# 速く丸め誤差も入りませんが、**写し間違いを実行時に教えてくれる仕組みがありません。**
# 正しさは中心差分 $-[\Delta S(x + h e_c) - \Delta S(x - h e_c)] / 2h$ と突き合わせま
# す。符号が負なのは $\Delta S = S^{sp} - S(x)$ だから。$|V|$ の列は $\Delta|V|/|V|$
# に対応するので $|V_j|$ を $|V_j|(1 \pm h)$ と **相対的に**動かします。

# %%
def numeric_jacobian(target, Yt, v_mag, theta_rad, h):
    """中心差分でヤコビアンを組む（解析式の答え合わせ用）。"""
    _, pv, pq = target.type_indices()
    columns = []
    for i in np.array(sorted(list(pv) + list(pq))):
        plus, minus = theta_rad.copy(), theta_rad.copy()
        plus[i], minus[i] = plus[i] + h, minus[i] - h
        columns.append(-(mismatch(target, Yt, v_mag, plus)
                         - mismatch(target, Yt, v_mag, minus)) / (2 * h))
    for i in pq:
        plus, minus = v_mag.copy(), v_mag.copy()
        plus[i], minus[i] = plus[i] * (1 + h), minus[i] * (1 - h)
        columns.append(-(mismatch(target, Yt, plus, theta_rad)
                         - mismatch(target, Yt, minus, theta_rad)) / (2 * h))
    return np.column_stack(columns)


# TODO(L3): 下の h の候補を眺め、(a) どの h を選ぶか、(b) どの指標で「一致した」と
#           報告するか（絶対誤差か max|J| で割った相対誤差か）、(c) その根拠を述べよ。
for h in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
    err = np.abs(J - numeric_jacobian(case, Y, solution.v, solution.theta, h)).max()
    print(f"  h = {h:.0e}   最大絶対誤差 {err:.3e}   "
          f"max|J| に対する比 {err / np.abs(J).max():.3e}")

# %% [markdown]
# **模範解答 (L3)**: 誤差は $h$ を小さくするほど減る……のは $h = 10^{-5}$ までで、そこ
# から先は **逆に増えます**。中心差分の誤差は打ち切り誤差 $O(h^2)$ と桁落ち
# $O(\varepsilon/h)$ の和で、$h \sim \varepsilon^{1/3} \approx 6\times10^{-6}$ で最小に
# なるから。**「刻み幅は小さいほど正確」は嘘です。** これが解析式には無い悩みであり、
# 解析式を採った理由でもある。報告は `max|J|` に対する相対誤差（最良で
# $2\times10^{-11}$）が妥当。$J$ の成分は 40 まであり、絶対誤差では相対精度が読めない。
#
# ## まとめ
#
# - 第 01 回の検算式を **ゼロにする方程式**として解いたのが潮流計算である
# - $\max|N|/\max|H| \approx 0.08$ の小ささが Fast Decoupled 法の根拠になる
# - **収束判定はミスマッチの無限大ノルムで行う。** 修正量は $J^{-1}$ を通した後の量
#   なので、悪条件のとき「動きが小さいだけ」を収束と誤判定する
# - 3 つの解法は同じ解に着く。違うのは収束次数と 1 反復の重さだけである
#
# 次回は、この Newton 法が **収束しなくなる**ところへ進みます。今日わざと悪条件にした
# 「負荷 2.4 倍」で見たとおり、負荷を増やすとヤコビアンは特異に近づき、ある点から先は
# 解が存在しません。その点が **電圧崩壊**の限界で、$J$ の最小特異値が余裕の指標になる。
# 「ソルバが悪いのか、解が無いのか」を切り分けられると、潮流計算は診断装置に変わります。
