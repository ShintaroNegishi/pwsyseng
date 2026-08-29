# %% [markdown]
# # 16 周波数制御 — ガバナと負荷周波数制御 (LFC)
#
# ## この回のねらい
#
# - 系統周波数が負荷と発電のバランスで決まることを理解する
# - **ガバナ（一次調整）** と **LFC（二次調整）** の役割の違いを確認する
# - 速度調定率 $R$ と系統定数の意味をつかむ
#
# ## 無限大母線を外す
#
# これまで使ってきた SMIB では、無限大母線が周波数を固定してしまうため
# 周波数制御の問題を扱えません。そこで系統全体を 1 台の等価発電機で
# 表した **孤立系モデル** を使います。
#
# $$ 2H \frac{d\Delta\omega}{dt} = \Delta P_m - \Delta P_L - D\,\Delta\omega $$
#
# - $\Delta P_L$: 負荷の変化
# - $D$: 負荷の周波数特性（周波数が下がると負荷も減る効果）
#
# ## なぜ周波数が下がるのか
#
# 負荷が増えたのに発電が変わらなければ、その差は回転体の運動エネルギーから
# 供給されます。エネルギーを吐き出した回転体は減速し、周波数が下がります。
# 日本では 50 Hz / 60 Hz を ±0.2 Hz 程度に保つことが求められます。

# %%
import numpy as np
import matplotlib.pyplot as plt

import genstab
from genstab import IsolatedSystem, StepLoad
from genstab.controllers.governor import Governor, LoadFrequencyControl
from genstab.plotting import use_genstab_style

use_genstab_style()

# %% [markdown]
# ## 1. 制御なし — 周波数は落ちっぱなし

# %%
load = StepLoad(magnitude=0.1, time=1.0)   # t=1 s に負荷が 0.1 p.u. 増える
plain = IsolatedSystem(H=5.0, D=1.0, controllers=[], load=load)
print(plain.describe())

result = genstab.simulate(plain, t_end=60.0, dt=0.01)
print(f"\n定常周波数偏差 = {result.omega[-1]:+.6f} p.u. "
      f"-> {result.frequency_hz[-1]:.4f} Hz")
print(f"解析値         = {plain.steady_state_deviation(0.1):+.6f} p.u.")

# %% [markdown]
# 制御がなくても周波数はどこかで落ち着きます。負荷自身の周波数特性 $D$ が
# 効いて、$\Delta\omega = -\Delta P_L / D$ で釣り合うためです。
# ただし偏差が大きすぎて実用になりません（50 Hz が 45 Hz まで落ちます）。
#
# ## 2. ガバナ（一次調整）
#
# ガバナは速度偏差に比例して機械入力を増やします。
#
# $$ T_g \frac{d\Delta P_m}{dt} = -\frac{1}{R}\Delta\omega - \Delta P_m $$
#
# $R$ は **速度調定率** で、実機では 0.03〜0.05（3〜5 %）です。
# $R = 0.05$ は「周波数が 5 % 下がったら出力を 100 % 増やす」という意味です。

# %%
with_governor = IsolatedSystem(
    H=5.0, D=1.0, controllers=[Governor(R=0.05, Tg=0.2)], load=load
)
result_gov = genstab.simulate(with_governor, t_end=60.0, dt=0.01)
print(f"定常周波数偏差 = {result_gov.omega[-1]:+.6f} p.u. "
      f"-> {result_gov.frequency_hz[-1]:.4f} Hz")
print(f"解析値 -ΔPL/(D + 1/R) = {with_governor.steady_state_deviation(0.1):+.6f} p.u.")

# %% [markdown]
# 偏差はぐっと小さくなりましたが、**ゼロにはなりません**。
# 比例制御だから当然です。この残った偏差を消すのが二次調整です。
#
# $D + 1/R$ を **系統定数** と呼びます。周波数が 1 p.u. ずれたときに
# 系統全体で何 p.u. の電力が自然に応答するかを表し、
# 大きいほど周波数が動きにくい丈夫な系統ということになります。
#
# ## 3. LFC（二次調整）
#
# LFC は速度偏差を積分します。積分項があるかぎり、定常状態では
# 必ず $\Delta\omega = 0$ になります。

# %%
with_lfc = IsolatedSystem(
    H=5.0, D=1.0,
    controllers=[Governor(R=0.05, Tg=0.2), LoadFrequencyControl(Ki=0.3)],
    load=load,
)
result_lfc = genstab.simulate(with_lfc, t_end=300.0, dt=0.02)
print(f"定常周波数偏差 = {result_lfc.omega[-1]:+.3e} p.u. "
      f"-> {result_lfc.frequency_hz[-1]:.6f} Hz")

# %% [markdown]
# ## 4. 3 つを並べて見る

# %%
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for label, result in (("no control", result),
                      ("governor only", result_gov),
                      ("governor + LFC", result_lfc)):
    mask = result.t <= 60.0
    axes[0].plot(result.t[mask], result.frequency_hz[mask], label=label)
    axes[1].plot(result.t[mask], result.Pm[mask], label=label)

axes[0].axhline(50.0, color="gray", lw=1.0)
axes[0].axhspan(49.8, 50.2, color="tab:green", alpha=0.12, label="±0.2 Hz band")
axes[0].set_ylabel("Frequency [Hz]")
axes[0].set_ylim(44.5, 50.5)
axes[1].set_ylabel("$\\Delta P_m$ [p.u.]")
axes[1].set_xlabel("Time [s]")
axes[0].legend(fontsize=9)
axes[0].set_title("Frequency response to a 0.1 p.u. load step")
plt.tight_layout()
plt.show()

# %% [markdown]
# 下段を見ると、ガバナが素早く出力を上げ（数秒）、
# その後 LFC がゆっくり残りを埋めている（数十秒）ことが分かります。
# **速い一次調整と遅い二次調整の役割分担** がはっきり見えます。
#
# ## 5. 演習: 速度調定率 R の影響
#
# $R$ を変えて、周波数の落ち込みと定常偏差がどうなるか調べてください。

# %%
plt.figure(figsize=(9, 4))
for R in (0.03, 0.05, 0.10, 0.20):
    # TODO: 速度調定率 R のガバナだけを持つ孤立系をシミュレーションし、
    #       周波数を重ね描きすること。定常偏差も表示すること。
    # BEGIN SOLUTION
    system = IsolatedSystem(H=5.0, D=1.0, controllers=[Governor(R=R, Tg=0.2)], load=load)
    r = genstab.simulate(system, t_end=40.0, dt=0.01)
    plt.plot(r.t, r.frequency_hz, label=f"R = {R:.2f}")
    print(f"R = {R:.2f}: 系統定数 D+1/R = {1.0 + 1/R:6.2f}, "
          f"定常偏差 {r.omega[-1]:+.6f} p.u. ({r.frequency_hz[-1]:.4f} Hz), "
          f"最低 {r.frequency_hz.min():.4f} Hz")
    # END SOLUTION

plt.axhline(50.0, color="gray", lw=1.0)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Effect of governor droop $R$")
plt.legend(fontsize=9)
plt.show()

# %% [markdown]
# $R$ を小さくすると周波数はよく保たれますが、その分ガバナが敏感になり、
# 複数機が並列運転するときは出力の奪い合い（ハンチング）が起きやすくなります。
# 実務で $R$ を 3〜5 % に取るのはこのバランスからです。
#
# ## 6. 演習: 慣性と周波数の落ち込み
#
# 慣性 $H$ を変えて、周波数の最低値（nadir）と落ち込みの速さ
# （RoCoF: Rate of Change of Frequency）を調べてください。
# 再生可能エネルギーの大量導入で系統慣性が下がると何が問題になるかが分かります。

# %%
plt.figure(figsize=(9, 4))
for H in (2.0, 5.0, 10.0):
    # TODO: 慣性 H を変えてシミュレーションし、周波数の最低値と
    #       負荷変化直後の RoCoF [Hz/s] を表示すること。
    # BEGIN SOLUTION
    system = IsolatedSystem(H=H, D=1.0, controllers=[Governor(R=0.05, Tg=0.2)], load=load)
    r = genstab.simulate(system, t_end=30.0, dt=0.005)
    plt.plot(r.t, r.frequency_hz, label=f"H = {H} s")
    just_after = (r.t > 1.0) & (r.t < 1.5)
    rocof = np.gradient(r.frequency_hz[just_after], r.t[just_after]).min()
    print(f"H = {H:5.1f} s: 最低周波数 {r.frequency_hz.min():.4f} Hz, "
          f"RoCoF {rocof:+.4f} Hz/s")
    # END SOLUTION

plt.axhline(50.0, color="gray", lw=1.0)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Lower inertia means a faster and deeper frequency drop")
plt.legend(fontsize=9)
plt.show()

# %% [markdown]
# 慣性が下がると **落ち込みが速く、深く** なります。
# 定常偏差は同じ（ガバナが決める）ですが、過渡的な落ち込みが問題です。
# 周波数低下リレーが動作して負荷遮断に至る恐れがあるため、
# 慣性低下への対策（疑似慣性制御、高速な周波数応答）が研究されています。
#
# ## 7. 演習: 積分ゲインの選び方

# %%
plt.figure(figsize=(9, 4))
for Ki in (0.1, 0.3, 1.0, 3.0):
    system = IsolatedSystem(
        H=5.0, D=1.0,
        controllers=[Governor(R=0.05, Tg=0.2), LoadFrequencyControl(Ki=Ki)],
        load=load,
    )
    r = genstab.simulate(system, t_end=60.0, dt=0.01)
    plt.plot(r.t, r.frequency_hz, label=f"$K_I$ = {Ki}")
    overshoot = r.frequency_hz.max() - 50.0
    print(f"Ki = {Ki:4.1f}: 最低 {r.frequency_hz.min():.4f} Hz, "
          f"行き過ぎ {overshoot:+.4f} Hz")

plt.axhline(50.0, color="gray", lw=1.0)
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Effect of the LFC integral gain")
plt.legend(fontsize=9)
plt.show()

# %% [markdown]
# 積分ゲインを大きくすると復帰は速くなりますが、行き過ぎ（オーバーシュート）や
# 振動が出ます。実際の LFC は数分オーダーでゆっくり動かすのが普通で、
# 速い変動はガバナに任せます。
#
# ## まとめ
#
# | | ガバナ（一次調整） | LFC（二次調整） |
# |---|---|---|
# | 制御方式 | 比例（ドループ） | 積分 |
# | 応答時間 | 数秒 | 数十秒〜数分 |
# | 定常偏差 | 残る | ゼロ |
# | 目的 | 周波数の急落を止める | 定格周波数に戻す |
#
# 次回は最終回、複数の発電機がある系統を扱います。
