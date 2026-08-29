# %% [markdown]
# # 03 潮流が解けないとき — Q 制限・P-V 曲線・電圧崩壊
#
# ## この回のねらい
#
# - **「解が存在しない」と「手法が見つけられない」を切り分ける**
# - 2 母線なら限界が閉じた式で書けることを確かめ、答え合わせの足場にする
# - 負荷余裕と最小特異値という 2 つの「限界までの距離」を読む
# - 無効電力の制限と調相設備が、その距離をどう動かすかを見る
#
# 第 02 回で Newton 法は 4 回の反復で解に着きました。二次収束は気持ちがよいので「収束しな
# ければ反復回数を増やせばよい」と思いがちですが、この回で見るのは **どんなソルバでも解け
# ない領域がある**ことです。潮流方程式は非線形で、ミスマッチをゼロにする実数解が **1 つも
# 無い**ことがあります。反復回数や許容誤差をいじるのは、実根を持たない 2 次方程式に解の
# 公式をより丁寧に当てはめるようなものです。

# %%
import math
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

import gridops
from gridops.plotting import plot_pv_curve, use_gridops_style
from gridops.voltage import pv_curve, two_bus_nose, two_bus_voltages

use_gridops_style()
case = gridops.load_case("wscc9")
E, X, P0 = 1.0, 0.10, 0.50   # 無限大母線電圧・線路リアクタンス・基準負荷 [p.u.]

# %% [markdown]
# ## 1. 2 母線なら限界は閉じた式で書ける
#
# 無限大母線（電圧 $E$ 固定）にリアクタンス $X$ だけの線路をつなぎ、その先に力率一定の
# 負荷 $S = P + jQ$ を置くと、負荷母線の電圧は
#
# $$ |V|^4 + (2QX - E^2)\,|V|^2 + X^2 (P^2 + Q^2) = 0 $$
#
# を満たします。$u = |V|^2$ の **2 次方程式**と見れば、実数解の存在条件は判別式
# $D = (2QX - E^2)^2 - 4 X^2 (P^2 + Q^2) \ge 0$ です。$D > 0$ なら解が 2 つ（上枝＝運用解と
# 下枝＝低電圧解）、**$D = 0$ がノーズ点**、$D < 0$ では解が存在しません。$Q = P\tan\phi$ を
# 入れて $D = 0$ を解けば $P_{max} = \frac{E^2}{2X}\cdot\frac{1-\sin\phi}{\cos\phi}$、
# $|V|_{crit} = \frac{E}{\sqrt{2(1+\sin\phi)}}$ です。符号は `gridops` の規約に合わせ、
# **力率が正なら遅れ（$Q > 0$、負荷が無効電力を消費）、負なら進み**とします。

# %%
def discriminant(p, q, e=E, x=X):
    """4 次方程式を $u=|V|^2$ の 2 次式と見たときの判別式 D。"""
    d = float("nan")   # 下の TODO を埋めると上書きされる（穴埋め版でも図は出る）
    # TODO(L1): D = (2QX - E^2)^2 - 4 X^2 (P^2 + Q^2) を 1 行で書くこと。
    # BEGIN SOLUTION
    d = (2.0 * q * x - e**2) ** 2 - 4.0 * x**2 * (p**2 + q**2)
    # END SOLUTION
    return d


labels = {0.95: "pf 0.95 lagging", 1.0: "pf 1.00", -0.95: "pf 0.95 leading"}
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for pf, colour in zip(labels, ("tab:red", "tab:blue", "tab:green")):
    p_max, v_crit = two_bus_nose(E, X, power_factor=pf)
    tan_phi = math.copysign(math.sqrt(1.0 - pf**2), pf) / abs(pf)
    print(f"{labels[pf]:18s}: P_max = {p_max:.6f}  |V|_crit = {v_crit:.6f}  "
          f"D(P_max) = {discriminant(p_max, p_max * tan_phi):+.2e}  "
          f"D(1.01 P_max) = {discriminant(1.01 * p_max, 1.01 * p_max * tan_phi):+.2e}")
    grid = np.linspace(1e-6, p_max * 0.999999, 400)
    upper, lower = np.array([two_bus_voltages(E, X, p, power_factor=pf) for p in grid]).T
    ax.plot(grid, upper, color=colour, label=labels[pf])
    ax.plot(grid, lower, color=colour, ls="--", lw=1.0)
    ax.plot([p_max], [v_crit], "k*", ms=13, zorder=5)
ax.set_xlabel("Load active power $P$ [p.u.]")
ax.set_ylabel("Voltage magnitude $|V|$ [p.u.]")
ax.set_title("Two-bus P-V curves (solid: upper branch, dashed: lower branch)")
ax.legend(fontsize=9)
plt.show()

# %% [markdown]
# 星印が $D = 0$ の点で、そこで上枝と下枝が合流します。印字した $D$ は $P_{max}$ でちょうど
# ゼロ、その 1% 先で **負**です。$P > P_{max}$ で潮流計算が収束しないのは、ソルバの都合では
# なく判別式が負だからです。**進み力率ほどノーズが遠い**（0.95 遅れ 3.62、力率 1 で 5.00、
# 0.95 進みで 6.91 p.u.）ことも読んでください。これが調相設備の効能です。
#
# ## 2. 継続法は解析解に一致するか
#
# 多母線では閉じた式が書けないので、負荷倍率を上げながら潮流を解き **解けなくなった点**を
# 限界と呼ぶしかありません（継続法）。しかしこの手続きは「収束しない = 解が無い」と決めつけ、
# **自分の答えを自分で定義してしまいます**。まず 2 母線で答え合わせをします。

# %%
def two_bus(pf, e=E, x=X, p0=P0):
    """無限大母線 + 純リアクタンス + 力率一定負荷。r = b = 0 で解析解の前提を満たす。"""
    q0 = p0 * math.copysign(math.sqrt(1.0 - pf**2), pf) / abs(pf)
    load = gridops.Bus(id=2, type=gridops.BusType.PQ, pd=p0, qd=q0, v_min=0.0, v_max=2.0)
    return gridops.Case(
        name=f"two-bus (pf={pf})", branches=[gridops.Branch(1, 2, x=x)],
        buses=[gridops.Bus(id=1, type=gridops.BusType.SLACK, v_set=e), load])


print(f"{'':18s}  {'numeric':>12s} {'analytic':>12s} {'rel.err':>9s}   "
      f"{'|V| num':>9s} {'|V| ana':>9s} {'rel.err':>9s}")
for pf in labels:
    f_num, v_num = pv_curve(two_bus(pf), step=0.05, max_factor=40.0).nose(2)
    f_ana, v_ana = two_bus_nose(E, X, P0, power_factor=pf)
    print(f"{labels[pf]:18s}  {f_num:12.8f} {f_ana:12.8f} {abs(f_num / f_ana - 1):9.1e}   "
          f"{v_num:9.6f} {v_ana:9.6f} {abs(v_num / v_ana - 1):9.1e}")

# %% [markdown]
# 倍率は $10^{-10}$ の水準で一致するのに電圧は $10^{-5}$ でしか一致しません。**これは実装の粗さではなくノーズ点の
# 性質です。** そこでは $dV/d\lambda \to \infty$ なので、倍率の誤差 $\varepsilon$ に対し電圧の誤差は $\sqrt{\varepsilon}$ の速さでしか縮まないからです。
#
# ## 3. WSCC 9 母線の P-V 曲線と最小特異値
#
# 負荷だけを一律に倍にし、**発電は基準値に据え置きます**（`Case.scaled` の `keep_generation=True`。忘れると
# 発電ゼロの別系統を解くことになります）。増分と損失は slack 母線が引き受けます。一緒に見るのはヤコビアンの
# **最小特異値** $\sigma_{min}(J)$ で、ノーズ点で $J$ が特異になるので限界までの近さが連続量として読めます。

# %%
curve = pv_curve(case)
print(curve.summary())

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
plot_pv_curve(curve, 5, ax=axes[0])
axes[0].axhline(0.95, color="tab:red", ls="--", lw=1.2)
ok = curve.converged
axes[1].plot(curve.factors[ok], curve.min_singular_values[ok], "o-", ms=4)
axes[1].set_yscale("log")
axes[1].set_xlabel("Load scaling factor [-]")
axes[1].set_ylabel(r"$\sigma_{min}(J)$ [-]")
axes[1].set_title("Smallest singular value of the Jacobian")
plt.tight_layout()
plt.show()

# %% [markdown]
# 負荷余裕は 1.374、つまり基準負荷の 137% 増までしか解が存在しません。電圧が最も低いのは
# 負荷の最も大きい母線 5 で、ノーズ点では 0.668 p.u. です。**運用上の限界（赤い破線
# $v_{min}=0.95$）はノーズ点のずっと手前**、倍率 1.51 で先に来ます。最小特異値は 0.961
# （倍率 1.0）から単調に減り、ノーズ点の直前で $10^{-5}$ の桁に落ちます。ただし **絶対値
# そのものに意味はありません**。基準容量や $\Delta|V|/|V|$ を未知数に取る規約に依存する
# ので、比べてよいのは同じケースの値どうしです。
#
# ## 4. 「収束しない」と「解が無い」は違う
#
# 同じ「収束しませんでした」が、2 つのまったく違う事情から出ます。

# %%
def try_solve(factor, **kwargs):
    """解ければ反復回数、解けなければ None を返す。"""
    with np.errstate(all="ignore"):
        try:
            scaled = case.scaled(factor, keep_generation=True)
            return gridops.solve_powerflow(scaled, **kwargs).iterations
        except RuntimeError:
            return None


trials = [
    ("2.00 (ノーズ点の内側) 既定の初期値", 2.00, {}),
    ("2.00 (ノーズ点の内側) 初期値 |V| = 0.5", 2.00, {"v0": np.full(case.n_bus, 0.5)}),
    ("2.40 (ノーズ点の外側) 既定", 2.40, {}),
    ("2.40 (ノーズ点の外側) 反復 500 回", 2.40, {"max_iter": 500}),
    ("2.40 (ノーズ点の外側) tol を 1e-3 に緩める", 2.40, {"tol": 1e-3}),
    ("2.40 (ノーズ点の外側) Gauss-Seidel 5000 回", 2.40,
     {"method": "gauss_seidel", "max_iter": 5000}),
]
for text, factor, kwargs in trials:
    iterations = try_solve(factor, **kwargs)
    print(f"{'収束する' if iterations else '収束せず'}（反復 {iterations or '-'} 回）: {text}")

# %% [markdown]
# 上 2 行は **同じ倍率 2.00** です。解は確かに存在するのに、初期値を $|V| = 0.5$ にした
# だけで Newton は失敗します。これが「手法が見つけられない」場合で、初期値を直せば解けます。
# 下 4 行は倍率 2.40、ノーズ点 2.374 の外側です。反復を 500 回にしても、許容誤差を
# $10^{-3}$ に緩めても、解法を変えても解けません。**判別式が負のときと同じで、探す対象が
# 存在しません。** 継続法の warm start は前者の失敗を減らす工夫ですが、2 つは原理的には
# 区別できず、`pv_curve` のノーズ点は **限界を下から押さえた推定**です。
#
# ## 5. 無効電力の制限 — 電圧を支える力は無限ではない
#
# 第 3 節では発電機の無効電力に上限を置きませんでした。「PV 母線は何があっても設定電圧を
# 保つ」という**楽観的な**仮定です。実際には $Q_{gen}$ が上限に達した瞬間、その母線は電圧を
# 保てず **PV 母線から PQ 母線に落ちます**（`enforce_q_limits=True`）。同梱ケースの `q_max`
# は有効出力によらない一定値で、これも楽観側です（実機は有効出力が大きいほど無効電力の余力が
# 減る）。ここでは全号機の `q_max` を 0.6 p.u. に絞った「余力の乏しい運用」を考えます。
# `pv_curve` には `enforce_q_limits` を渡す口が無いので、継続法のループを自分で書きます。

# %%
tight = replace(case, units=[replace(u, q_max=min(u.q_max, 0.6)) for u in case.units])
factors, v2, v5, switched = [], [], [], []
warm = None

# TODO(L2): 負荷倍率を 1.0 から 0.02 刻みで 2.6 まで上げる継続法のループを書くこと。
#   - tight.scaled(factor, keep_generation=True) を enforce_q_limits=True で解く
#   - 前の点の解を初期値に使う（warm start）。RuntimeError が出たら break
#   - factors, v2, v5, switched に倍率・母線 2 の |V|・母線 5 の |V|・PV→PQ 切替が
#     起きたか（solution.q_limited が空でないか）を追加する
# BEGIN SOLUTION
for factor in np.round(np.arange(1.0, 2.60, 0.02), 10):
    try:
        solution = gridops.solve_powerflow(
            tight.scaled(factor, keep_generation=True), enforce_q_limits=True,
            v0=None if warm is None else warm.v,
            theta0=None if warm is None else warm.theta)
    except RuntimeError:
        break
    warm = solution
    factors.append(factor)
    v2.append(solution.v[case.index_of(2)])
    v5.append(solution.v[case.index_of(5)])
    switched.append(bool(solution.q_limited))
# END SOLUTION

first = switched.index(True)
print(f"PV -> PQ の切替 : 倍率 {factors[first]:.2f}（母線 {warm.q_limited}）")
print(f"負荷余裕        : {factors[-1] - 1.0:.3f} (第 3 節は {curve.loading_margin:.3f})")

plt.figure(figsize=(7.2, 4.2))
plt.plot(factors, v2, "o-", ms=3, label="bus 2 (PV generator)")
plt.plot(factors, v5, "o-", ms=3, label="bus 5 (largest load)")
plt.axvline(factors[first], color="tab:red", ls="--", lw=1.2, label="Q limit reached")
plt.xlabel("Load scaling factor [-]")
plt.ylabel("Voltage magnitude $|V|$ [p.u.]")
plt.title("PV to PQ switching when the reactive limit binds")
plt.legend(fontsize=9)
plt.show()

# %% [markdown]
# 母線 2 の電圧は倍率 2.18 まで設定値 1.025 に**張り付いています**。そこで $Q_{gen}$ が
# 上限に達すると以後は電圧を支えられず、3 刻みで 0.992 まで落ち、系統は 2.24 で崩壊します。
# 無効電力に上限を置いただけで負荷余裕が 1.374 から 1.240 に縮みました。**ただしこの扱いは
# 厳密ではありません。** `gridops` は PV→PQ の切替を反復の外で判定し、しかも **一方向にしか
# 切り替えません**。正しくは電圧が設定値まで戻せる母線を PQ→PV に戻す必要がありますが、
# 両方向にすると切替が振動して止まらないため割り切っています
# （`docs/model_assumptions.md`）。`q_limited` を必ず確認してください。
#
# ## 6. 調相設備 — ノーズは伸びる。では安全になったのか
#
# 母線 5 に容量性サセプタンス `Bus.bs` を置くと、その母線から見た負荷の力率が進み側に
# 寄ります。第 1 節の $P_{max} \propto (1-\sin\phi)/\cos\phi$ が効くはずです。3 通り試します。

# %%
index5 = case.index_of(5)
rows = []
plt.figure(figsize=(7.2, 4.4))
for shunt, colour in zip((0.0, 0.3, 0.6), ("tab:blue", "tab:orange", "tab:green")):
    buses = list(case.buses)
    buses[index5] = replace(buses[index5], bs=shunt)
    shunted = pv_curve(replace(case, buses=buses))
    ok = shunted.converged
    f, v = shunted.factors[ok], shunted.voltages[ok, index5]
    nose_f, nose_v = shunted.nose(5)
    cross = float(np.interp(0.95, v[::-1], f[::-1]))   # |V| が 0.95 を割る倍率
    rows.append((shunt, v[0], cross, nose_f, nose_f - cross))
    plt.plot(f, v, "-", color=colour, label=f"$b_s$ = {shunt} p.u.")
    plt.plot([nose_f], [nose_v], "*", color=colour, ms=14)
plt.axhline(0.95, color="tab:red", ls="--", lw=1.2, label="$v_{min}$ = 0.95")
plt.xlabel("Load scaling factor [-]")
plt.ylabel("Voltage magnitude at bus 5 [p.u.]")
plt.title("Shunt compensation at bus 5")
plt.legend(fontsize=9)
plt.show()

print(f"{'b_s':>5s} {'|V| at base':>12s} {'|V|<0.95 at':>12s} {'nose':>8s} {'warning gap':>12s}")
for row in rows:
    print("{:5.1f} {:12.4f} {:12.3f} {:8.3f} {:12.3f}".format(*row))

# TODO(L3): 母線 5 に b_s = 0.6 p.u. を入れる案に賛成か反対かを決め、次の markdown セルに
#   理由を書くこと。(a) 基準負荷での母線 5 の電圧は運用上の上限 1.05 に収まっているか、
#   (b) ノーズ点はどれだけ伸びたか、(c) 電圧が 0.95 を割ってからノーズ点までの距離
#   (warning gap) はどう変わったか、の 3 点に必ず触れること。

# %% [markdown]
# ### 逆説を読む（L3 の模範解答の一例）
#
# 調相設備は確かに効きました。ノーズ点は 2.374 から 2.443 へ伸び、基準負荷での電圧も
# 0.996 から 1.053 に上がります。**しかし警報が鳴るのが遅くなりました。** 母線 5 の電圧が
# 下限 0.95 を割るのは無補償なら倍率 1.51、$b_s = 0.6$ では 1.91 です。ノーズ点は 2.443
# なので、**電圧が下限を割ってから崩壊するまでの距離が 0.862 から 0.533 に縮みました**。
# 図の緑の曲線も鼻先が鋭く、最後に一気に落ちています。つまり調相設備は「限界を遠ざける」と
# 同時に「限界までの残りを電圧で測れなくする」のです。運転員が電圧計だけを見ていると、
# 健全に見えたまま突然崩壊します。**電圧の大きさは電圧安定余裕の指標ではありません。**
#
# 判断としては、基準負荷での 1.053 p.u. が上限 1.05 を超えるのでそのままでは投入できない、と
# 書けます（段階投入にするか 0.3 p.u. に留める。0.3 なら電圧 1.023、ノーズ点 2.408、warning
# gap 0.681）。**「ノーズが伸びたから安全になった」の一言で済ませてはいけません。**

# %% [markdown]
# ## まとめ
#
# - 2 母線の限界は判別式 $D = (2QX-E^2)^2 - 4X^2(P^2+Q^2)$ がゼロになる点で
#   **閉じた式で書ける**。継続法はこれと $10^{-10}$ の水準で一致した
# - ノーズ点の外側では解が存在せず、反復回数・許容誤差・解法を変えても解けない。内側でも
#   初期値が悪ければ Newton は失敗する。**同じ「収束しません」が 2 つの事情から出る**
# - WSCC 9 母線の負荷余裕は 1.374。運用限界 $v_{min}=0.95$ は倍率 1.51 で先に来る
# - 無効電力に上限を置くと余裕は 1.240 に縮む。PV→PQ 切替は一方向のみの近似である
# - 調相設備はノーズを伸ばすが鼻先を鋭くする。電圧の大きさを余裕の代わりにしない
#
# 次の第 04 回では **直流潮流**という線形近似に移ります。$|V| = 1$、$r = 0$ と置くので、この回で
# 見た電圧の話は**すべて消えます**。消えることを承知で使う近似で、感度係数（PTDF・LODF）が高速に
# 得られるという見返りがあります。ただし第 09 回で、熱容量では健全なのに母線 6 の電圧が
# 0.9418 p.u. まで落ちる事故を扱い、そこでこの回の話が戻ってきます。
