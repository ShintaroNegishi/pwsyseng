# %% [markdown]
# # 01 系統をデータで表す — 単位法（per unit: p.u.）と母線アドミタンス行列（bus admittance matrix: Ybus）
#
# ## この回のねらい
#
# - 系統図を **1 枚の行列** に落とす手順を、枝 1 本から積み上げて理解する
# - p.u. と MW のどちらで書かれた量かを、識別子の名前から読み取れるようになる
# - **解く前にデータを疑う**作法を身につける。参照解の電圧から注入を計算し直し、
#   残差が「誤り」か「掲載桁数の丸め」かを自分で判断する
#
# ## 扱う系統: Western Systems Coordinating Council（WSCC）3 機 9 母線系統（科目を通してこの 1 系統で押し通します）
#
# ```
#   G2-(2)-[T2]-(7)---(8)---(9)-[T3]-(3)-G3     [T] = 変圧器
#                 |           |                 (n) = 母線
#                (5)         (6)
#                 |           |
#                 +----(4)----+---[T1]---(1)-G1 (slack)
# ```
#
# ## Ybus は何を表す行列か
#
# 母線への **注入電流** $\bar I$ と母線電圧 $\bar V$ は線形に結ばれます。その係数行列が
# 母線アドミタンス行列 $Y$ で、母線の複素電力は次の形になります。
#
# $$ \bar I = Y \bar V, \qquad
#    \bar S_i = \bar V_i \left( \sum_j Y_{ij} \bar V_j \right)^{*} $$
#
# 右の $\bar S_i$ の式が電力系統の計算のほぼすべての出発点です。この回は $\bar V$ を与えて計算し、
# 次回はこれを **未知数 $\bar V$ の方程式**として解きます。式は同じで、どちらを既知にするか
# だけが違います。枝 1 本の寄与は、タップ比 $\tau$ と位相調整角 $\phi$ を $\bar a = \tau e^{j\phi}$ と
# まとめると 2x2 に畳み込めます（$y_s = 1/(r+jx)$、$b$ は全充電サセプタンス）。
#
# $$ Y_{ff} = \frac{y_s + jb/2}{\tau^2}, \quad
#    Y_{ft} = -\frac{y_s}{\bar a^{*}}, \quad
#    Y_{tf} = -\frac{y_s}{\bar a}, \quad
#    Y_{tt} = y_s + \frac{jb}{2} $$

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

import gridops
from gridops.plotting import use_gridops_style

use_gridops_style()

case = gridops.load_case("wscc9")
print(case.describe())
print(f"\nデータの不整合: {case.check() or '検出なし'}")

# %% [markdown]
# `case.check()` の役割（解く前にデータを疑う）は第 00 回で述べたとおりです。
#
# ## 1. p.u. と MW — 規約を守らないと何が起きるか
#
# 単位の規約そのもの（**電力・容量の末尾が `_mw` / `_mvar` なら実単位、ネットワーク量は原則 p.u.**、換算は
# `Case.to_mw` / `Case.to_pu` だけを通す）は第 00 回の第 7 節で置きました。ここでは
# その規約が **何を防いでいるのか**を、実際に取り違えて確かめます。

# %%
total_load = sum(bus.pd for bus in case.buses)
capacity_mw = sum(u.p_max_mw for u in case.units)
print(f"基準容量 {case.base_mva} MVA / 総負荷 {total_load:.2f} p.u. = "
      f"{case.to_mw(total_load):.0f} MW / 設備容量 {capacity_mw:.0f} MW = "
      f"{case.to_pu(capacity_mw):.2f} p.u.")

unit = case.units[0]
print(f"\n号機 {unit.name}: p_max_mw = {unit.p_max_mw} MW、"
      f"quadratic = {unit.quadratic} 円/(MW^2 h)")
print(f"  全負荷での二次項:  MW で代入 {unit.quadratic * unit.p_max_mw**2:>10,.2f} 円/h")
print(f"                    p.u. で代入 {unit.quadratic * case.to_pu(unit.p_max_mw)**2:>10,.2f} 円/h")

# %% [markdown]
# 最後の 2 行が単位の混在の怖さです。**p.u. の出力を 円/(MW$^2$ h) の係数にそのまま代入
# すると、二次項が $10^4$ 倍小さくなります。** 例外も警告も出ず「もっともらしい額」が出て
# くるだけなので気づけません。接尾辞の規約はこの事故を構造的に防ぐためにあります。
#
# ## 2. 枝 1 本の寄与
#
# 線路 4-5 を取り出します。`Branch.primitive()` が上の 4 つの式をそのまま計算します。

# %%
line45 = next(b for b in case.branches if b.key() == (4, 5))
print(f"枝 {line45.label}: r={line45.r}, x={line45.x}, b={line45.b}, tap={line45.tap}")
print(f"直列アドミタンス y_s = {line45.series_admittance():.4f}")
print(np.array2string(line45.primitive(), precision=4))

f = case.index_of(line45.from_bus)
t = case.index_of(line45.to_bus)
Y_one = np.zeros((case.n_bus, case.n_bus), dtype=complex)

# TODO(L1): 枝 4-5 の 2x2 行列を Y_one の (f, t) の位置に足し込むこと（np.ix_ を使えば 1 行）。
# BEGIN SOLUTION
Y_one[np.ix_([f, t], [f, t])] += line45.primitive()
# END SOLUTION

print(f"\n非ゼロ要素 {np.count_nonzero(Y_one)} 個: "
      f"Y[4,4] = {Y_one[f, f]:.4f},  Y[4,5] = {Y_one[f, t]:.4f}")

# %% [markdown]
# 対角には $y_s + jb/2$ が **足し込まれ**、非対角には $-y_s$ が入ります。対角が「その母線に
# つながる枝の合計」になるのは、電流則を母線ごとに書いているからです。
#
# ## 3. 全部の枝を足す — 9 本で同じことを繰り返すだけ

# %%
Y_mine = np.zeros((case.n_bus, case.n_bus), dtype=complex)

# TODO(L2): すべての枝について 2x2 行列を足し込むループを書くこと。母線番号から
#           行列の添字への変換は case.index_of() を通す（両者は一致しない）。
# BEGIN SOLUTION
for branch in case.branches:
    ix = [case.index_of(branch.from_bus), case.index_of(branch.to_bus)]
    Y_mine[np.ix_(ix, ix)] += branch.primitive()
# END SOLUTION

Y = gridops.build_ybus(case)
print(f"build_ybus との差: {np.abs(Y_mine - Y).max():.2e}")

# 安い検算: 充電容量を外せば行和はゼロ（全母線が同電圧なら電流は流れない）。
b_only = replace(case, branches=[replace(b, b=0.0) for b in case.branches])
print(f"充電容量を外した Ybus の行和: {np.abs(gridops.build_ybus(b_only).sum(axis=1)).max():.2e}")

# %% [markdown]
# ## 4. タップと位相調整 — Ybus が非対称になるとき

# %%
def modified(key, **kwargs):
    """枝 key だけを差し替えたケースの Ybus。"""
    branches = [replace(b, **kwargs) if b.key() == key else b for b in case.branches]
    return gridops.build_ybus(replace(case, branches=branches))


Y_shift = modified((1, 4), shift_deg=5.0)
i1, i4 = case.index_of(1), case.index_of(4)
for label, M in (("公称タップ ", Y), ("tap=1.05   ", modified((1, 4), tap=1.05)),
                 ("shift=5 deg", Y_shift)):
    print(f"{label}: Y[1,1]={M[i1, i1]:>17.4f}  Y[1,4]={M[i1, i4]:>17.4f}  "
          f"Y[4,1]={M[i4, i1]:>17.4f}  max|Y-Y^T|={np.abs(M - M.T).max():.2e}")

labels = [str(b.id) for b in case.buses]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, matrix, cmap, title in (
    (axes[0], np.abs(Y), "viridis", "Magnitude of Ybus (base case)"),
    (axes[1], np.abs(Y_shift - Y_shift.T), "magma",
     "Asymmetry $|Y - Y^\\mathsf{T}|$ (5 deg shift on 1-4)"),
):
    fig.colorbar(ax.imshow(matrix, cmap=cmap), ax=ax, label="[p.u.]")
    ax.set_xticks(range(case.n_bus), labels)
    ax.set_yticks(range(case.n_bus), labels)
    ax.set(title=title, xlabel="Bus", ylabel="Bus")
plt.tight_layout()
plt.show()

# %% [markdown]
# 左の図は **行列がそのまま結線図**であることを示します。非ゼロの位置が枝の位置で、
# 9 母線 9 枝なので行列はすかすかです（実系統ではさらに疎になります）。右の図が重要
# です。タップ比だけなら $Y_{ff}$ が $1/\tau^2$ 倍になるだけで **行列は対称のまま**ですが、
# 位相調整角を入れると $Y_{ft} \ne Y_{tf}$ となり **Ybus は非対称になります**（大きさは同じで
# 角度だけが違う）。位相調整器は潮流を能動的に押し込む装置で、その非相反性が非対称と
# して現れます。`Y == Y.T` を仮定してはいけません。
#
# ## 5. 未知数と方程式の数合わせ
#
# 母線ごとに $(P, Q, |V|, \theta)$ の 4 つのうち 2 つを与え、残り 2 つを求めます。どの 2 つを
# 与えるかが母線種別です。
#
# | 種別 | 与える量 | 求める量 | 個数 |
# |---|---|---|---|
# | slack | $\lvert V \rvert, \theta$ | $P, Q$ | 必ず 1 |
# | PV | $P, \lvert V \rvert$ | $Q, \theta$ | $n_{PV}$ |
# | PQ | $P, Q$ | $\lvert V \rvert, \theta$ | $n_{PQ}$ |

# %%
slack, pv, pq = case.type_indices()
equations = (len(pv) + len(pq)) + len(pq)
print(f"slack {len(slack)} 母線, PV {len(pv)} 母線, PQ {len(pq)} 母線")
print(f"未知数 = 2*n_PQ + n_PV = 2*{len(pq)} + {len(pv)} = {case.n_unknowns()}")
print(f"方程式 = P 式 (n_PV+n_PQ) + Q 式 (n_PQ) = ({len(pv)}+{len(pq)}) + {len(pq)} = {equations}")

# %% [markdown]
# 14 = 14 で一致しました。**slack 母線で P と Q の式を立てないのは、そこが損失の受け皿
# だからです。** 損失は解いてみるまで分からないので slack の $P$ を先に与えられません。
#
# ## 6. データを疑う — 参照解から注入を計算し直す
#
# ここからがこの回の山場です。ケースには教科書の潮流解が `solution` 層として入って
# います。**これは答え合わせ専用で、入力には使いません。** その電圧を上の
# $\bar S_i = \bar V_i (\sum_j Y_{ij} \bar V_j)^{*}$ に代入して得た注入が、ケースの発電と負荷に
# 一致するかを見ます。一致しなければ、潮流解か線路データが間違っています。

# %%
v = case.reference.voltage
s = v * np.conj(Y @ v)
p_spec, q_spec = case.bus_injection()
residual = np.abs(s - (p_spec + 1j * q_spec))

print(f"{'bus':>4}{'S = V (YV)* [p.u.]':>24}{'発電 - 負荷 [p.u.]':>24}{'|差|':>11}")
for i, bus in enumerate(case.buses):
    print(f"{bus.id:>4}{s[i].real:>+12.4f}{s[i].imag:>+11.4f}j"
          f"{p_spec[i]:>+12.4f}{q_spec[i]:>+11.4f}j{residual[i]:>11.2e}")
print(f"\n最大残差 = {residual.max():.3e} p.u.（母線 {case.bus_ids[int(residual.argmax())]}）")

# %% [markdown]
# 残差はゼロではありません。$1.1 \times 10^{-3}$ p.u.、100 MVA 基準で 0.11 MW です。**これは
# データの誤りでしょうか、それとも丸めでしょうか。** 材料は、参照解が **小数点以下 4 桁**で
# 載っている事実です。丸め $|\delta \bar V| \le \tfrac12 10^{-4}$ が注入に響く量は、1 次の変化
#
# $$ \delta \bar S \approx \delta \bar V \odot \overline{Y \bar V}
#    + \bar V \odot \overline{Y\, \delta \bar V} $$
#
# から上界として押さえられます。

# %%
step = 0.5 * 10.0 ** (-case.reference.digits)
bound = step * (np.abs(Y @ v).max() + np.abs(v).max() * np.abs(Y).sum(axis=1).max())

# TODO(L3): 残差 1.1e-03 は「線路データの誤り」か「掲載桁数の丸め」か。下の 2 つの数値を
#           根拠に、どちらだと判断したかを次の markdown セルに 2〜3 行で書くこと。
#           「1e-3 を超えたら異常」という閾値で判定していたらどうなるかにも触れること。
print(f"実際の最大残差 = {residual.max():.3e} p.u.")
print(f"丸めだけの上界 = {bound:.3e} p.u.")
print(f"比 = {residual.max() / bound:.2f}（1 未満なら丸めで説明がつく）")

# %% [markdown]
# **L3 の模範解答**: 残差 $1.119 \times 10^{-3}$ は丸めの上界 $4.20 \times 10^{-3}$ の 0.27 倍で、掲載
# 4 桁の丸めだけで説明がつく。よって線路データと潮流解は整合していると判断する。「$10^{-3}$
# を超えたら異常」という切りのよい閾値なら **正しいデータを壊れていると誤判定していた**。
#
# ## 7. では、本当に壊れているデータはどう見えるか
#
# 線路 4-5 の x を 10% 小さく書き間違えた場合と、充電容量を入れ忘れた場合で測ります。

# %%
def residuals(c):
    """ケース c の線路データと参照解の電圧が食い違う量 |dS| [p.u.]。"""
    p_c, q_c = c.bus_injection()
    return np.abs(v * np.conj(gridops.build_ybus(c) @ v) - (p_c + 1j * q_c))


typo = replace(case, branches=[replace(b, x=b.x * 0.9) if b.key() == (4, 5) else b
                               for b in case.branches])
trials = {"published": case, "x(4-5) 10% low": typo, "charging ignored": b_only}

x_pos = np.arange(case.n_bus)
fig, ax = plt.subplots(figsize=(9, 4))
for k, (label, trial) in enumerate(trials.items()):
    values = residuals(trial)
    ax.bar(x_pos + (k - 1) * 0.27, values, width=0.26, label=label)
    print(f"{label:16s}: 最大残差 {values.max():.3e} p.u.")
ax.axhline(bound, color="k", ls="--", lw=1.4, label="rounding bound")
ax.set_xticks(x_pos, [str(bus.id) for bus in case.buses])
ax.set(yscale="log", xlabel="Bus", ylabel=r"$|\Delta S|$ [p.u.]",
       title="Injection residual: rounding vs. wrong branch data")
ax.legend(fontsize=9, ncol=2)
plt.show()

# %% [markdown]
# 書き間違いは丸めの上界を 1 桁以上、充電容量の入れ忘れは 2 桁近く超えます。注目すべきは
# **出方**です。枝 1 本の誤りは両端の母線 4, 5 だけを持ち上げ（$S_i$ に効くのは Ybus の第 $i$
# 行だけ）、全枝に及ぶ誤りは全母線を持ち上げます。**残差の出た母線が疑う枝を教えます。**
#
# ## 8. 橋 — 開放してはいけない枝

# %%
print(f"橋: {gridops.bridges(case)}")
for key in [(1, 4), (4, 5)]:
    parts = gridops.islands(case, removed_branches=[key])
    print(f"枝 {key} を開放 -> 島 {len(parts)} 個: {parts}")
print(f"\nN-1 の想定事故 ({len(case.contingencies)} 件): {case.contingencies}")

# %% [markdown]
# 開放すると系統が 2 つの島に分かれる枝を **橋 (bridge)** と呼びます。ここでは変圧器 3 本で、
# 開放すると発電機母線が単独の島になり、位相の基準も損失の受け皿もありません。Ybus は
# 特異になり交流潮流は解を持ちません。だから想定事故は 6 本でこの 3 本が入っていません。
# **除外の理由がトポロジーから説明できる**ことが大事です（第 04 回で再登場します）。
#
# ## まとめ
#
# - Ybus は枝 1 本の 2x2 行列を母線番号の位置に足し込むだけで組める
# - 位相調整角があると **Ybus は非対称**になる。対称性を仮定しない
# - 未知数 $2n_{PQ} + n_{PV} = 14$ と方程式の数は一致する。slack で式を立てないのは、
#   そこが損失の受け皿だから
# - **許容差は勘ではなくデータの素性から計算する。** 残差 1.119e-03 は掲載 4 桁の丸めの
#   上界 4.20e-03 に収まっており、データは整合している。1e-3 の直書きなら誤警報だった
#
# ## 次回へ
#
# 今日は $\bar V$ を与えて注入 $\bar S = \bar V (Y \bar V)^{*}$ を計算し、指定値との差を眺めました。
# **次回はこの同じ式をゼロにします。** すなわち $\bar S(\bar V) - \bar S^{sp} = 0$ を未知数 $\bar V$ に
# ついて解きます。今日「検算の残差」と呼んだ量が、次回は「ミスマッチ」と名を変えて
# Newton 法が潰しにいく対象になります。未知数を数えたのは、ヤコビアンが 14x14 だから。
