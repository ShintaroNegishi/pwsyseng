# %% [markdown]
# # 13 定態安定性 — 線形化と固有値
#
# ## この回のねらい
#
# - 非線形の動揺方程式を動作点まわりで **線形化** する
# - 状態行列の **固有値** から安定性・振動周波数・減衰比を読む
# - `python-control` の道具（伝達関数・ボード線図）を発電機系に適用する
#
# ## 過渡安定性と定態安定性の違い
#
# | | 過渡安定性 | 定態安定性 |
# |---|---|---|
# | 対象とする擾乱 | 大きな事故（三相地絡など） | 微小な変動（負荷のゆらぎなど） |
# | 使うモデル | 非線形の動揺方程式 | 動作点まわりの線形モデル |
# | 調べ方 | 時間積分 | 固有値解析 |
# | 判定 | 同期を失わないか | すべての固有値が左半面にあるか |
#
# **どちらか一方だけでは足りません。** 大事故に耐えられても、
# 定常運転中に振動が育つ系は使えません。
#
# ## 線形化
#
# 動作点 $(\delta_0, 0)$ のまわりで $\delta = \delta_0 + \Delta\delta$ と
# おいて 1 次の項だけ残すと、$\sin\delta \approx \sin\delta_0 + \cos\delta_0 \Delta\delta$
# より
#
# $$
# \frac{d}{dt}\begin{bmatrix}\Delta\delta \\ \Delta\omega\end{bmatrix}
# = \begin{bmatrix} 0 & \omega_s \\ -\dfrac{K_s}{2H} & -\dfrac{D}{2H} \end{bmatrix}
#   \begin{bmatrix}\Delta\delta \\ \Delta\omega\end{bmatrix}
# $$
#
# ここで $K_s = P_{max}\cos\delta_0$ を **同期化力係数** と呼びます。
# 位相角がわずかにずれたとき、それを引き戻す向きに働く力の強さです。

# %%
import numpy as np
import matplotlib.pyplot as plt
import control as ct

import genstab
from genstab import linearize as lin
from genstab import smallsignal as ss
from genstab.plotting import use_genstab_style, plot_eigenvalues

use_genstab_style()

machine = genstab.ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
network = genstab.SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.4)
system = genstab.SMIBSystem(machine, network, Pe0=0.8)
print(system.describe())

# %% [markdown]
# ## 1. 状態行列を作る
#
# `genstab` は数値微分（中心差分）で状態行列を求めます。こうしておくと、
# あとで制御器を追加してもコードを変えずに線形化できます。

# %%
A = lin.state_matrix(system)
print("状態行列 A =")
print(np.array2string(A, precision=5, suppress_small=True))
print(f"\n状態変数: {system.state_names}")

# %% [markdown]
# 手計算の値と合っているか確かめましょう。
# $A_{12} = \omega_s$、$A_{21} = -K_s/(2H)$、$A_{22} = -D/(2H)$ のはずです。

# %%
K_s = ss.synchronizing_coefficient(system)
omega_s = system.base.omega_s
H, D = machine.H, machine.D

print(f"A[0,1] = {A[0,1]:12.6f}   理論値 omega_s     = {omega_s:12.6f}")
print(f"A[1,0] = {A[1,0]:12.6f}   理論値 -K_s/(2H)   = {-K_s/(2*H):12.6f}")
print(f"A[1,1] = {A[1,1]:12.6f}   理論値 -D/(2H)     = {-D/(2*H):12.6f}")
print(f"\n同期化力係数 K_s = Pmax * cos(delta0) = {K_s:.6f} p.u./rad")

# %% [markdown]
# ## 2. 固有値を読む
#
# 固有値 $\lambda = \sigma \pm j\omega_d$ から次が読めます。
#
# - $\sigma < 0$ なら安定（振動が減衰する）
# - 振動周波数 $f = \omega_d / 2\pi$ [Hz]
# - 減衰比 $\zeta = -\sigma / |\lambda|$

# %%
modes = ss.analyze(system)
print(modes.table())

# %% [markdown]
# **参加係数 (participation factor)** は、そのモードにどの状態変数が
# 効いているかを表す無次元量です。単位系に依存しないので、
# 「この振動は何の振動なのか」を判断するのに使えます。
#
# ## 3. 解析式と照合する
#
# 2 次系なので、特性方程式から $\omega_n$ と $\zeta$ が手で求まります。
#
# $$
# \lambda^2 + \frac{D}{2H}\lambda + \frac{K_s\omega_s}{2H} = 0
# \;\Rightarrow\;
# \omega_n = \sqrt{\frac{K_s\omega_s}{2H}}, \quad
# \zeta = \frac{D}{2\sqrt{2HK_s\omega_s}}
# $$

# %%
analytic = ss.classical_mode_analytic(system)
print(f"固有角振動数 omega_n = {analytic.natural_frequency:.6f} rad/s "
      f"= {analytic.natural_frequency_hz:.6f} Hz")
print(f"減衰比       zeta    = {analytic.damping_ratio:.6f}")
print(f"\n固有値 (解析) : {np.sort_complex(analytic.eigenvalues)}")
print(f"固有値 (数値) : {np.sort_complex(modes.eigenvalues)}")
print(f"最大差        : {np.max(np.abs(np.sort_complex(analytic.eigenvalues) - np.sort_complex(modes.eigenvalues))):.3e}")

# %% [markdown]
# 電力系統の動揺モードは典型的に **0.7〜2 Hz**（局所モード）に現れます。
# 上で得た値もその範囲に入っています。
#
# 減衰比については、実務上 **$\zeta \ge 0.05$（5 %）** が一つの目安とされます。
# 上の系はこれをかなり下回っており、制動が足りません。

# %% [markdown]
# ## 4. 演習: 動作点と固有値
#
# 送電電力 $P_{e0}$ を変えると固有値がどう動くか調べてください。

# %%
powers = np.arange(0.3, 1.05, 0.1)
eigenvalue_sets, labels = [], []
for Pe0 in powers:
    # TODO: Pe0 を変えた系の固有値を求めて eigenvalue_sets に追加すること。
    # BEGIN SOLUTION
    trial = genstab.SMIBSystem(machine, genstab.SMIBNetwork(0.4, np.inf, 0.4), Pe0=Pe0)
    eigenvalue_sets.append(ss.analyze(trial).eigenvalues)
    labels.append(f"Pe0 = {Pe0:.1f}")
    # END SOLUTION

plot_eigenvalues(eigenvalue_sets, labels, title="Eigenvalue locus versus power transfer")
plt.show()

for Pe0, values in zip(powers, eigenvalue_sets):
    trial = genstab.SMIBSystem(machine, genstab.SMIBNetwork(0.4, np.inf, 0.4), Pe0=Pe0)
    mode = ss.classical_mode_analytic(trial)
    print(f"Pe0 = {Pe0:.1f}: K_s = {mode.K_s:.4f}, f = {mode.natural_frequency_hz:.4f} Hz, "
          f"zeta = {mode.damping_ratio:.4f}")

# %% [markdown]
# 重負荷になるほど $\delta_0$ が大きくなり、$K_s = P_{max}\cos\delta_0$ が
# 小さくなります。その結果、**振動周波数が下がり、減衰比は上がります**
# （$\zeta \propto 1/\sqrt{K_s}$ のため）。
#
# ただし $\delta_0 = 90°$ を超えると $K_s < 0$ になり、
# 振動する間もなく静的に脱調します。これが **定態安定限界** です。

# %% [markdown]
# ## 5. python-control につなぐ
#
# 線形化した系を `control.StateSpace` に変換すれば、
# 制御工学の道具がそのまま使えます。

# %%
G = lin.state_space(system, inputs=("Pm",), outputs=("delta", "omega", "Pe"))
print(G)
print(f"極 : {ct.poles(G)}")

# %% [markdown]
# ### ステップ応答
#
# 機械入力を 0.05 p.u. だけ増やしたときの応答です。

# %%
T = np.linspace(0, 20, 1000)
response = ct.step_response(G * 0.05, T)

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(T, np.degrees(response.outputs[0].ravel()))
axes[0].set_ylabel("$\\Delta\\delta$ [deg]")
axes[1].plot(T, response.outputs[2].ravel())
axes[1].set_ylabel("$\\Delta P_e$ [p.u.]")
axes[1].set_xlabel("Time [s]")
axes[0].set_title("Step response to a 0.05 p.u. increase in mechanical power")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 周波数応答（ボード線図）
#
# 動揺モードの共振がはっきり見えます。

# %%
G_delta = lin.state_space(system, inputs=("Pm",), outputs=("delta",))
plt.figure(figsize=(8, 6))
ct.bode_plot(G_delta, np.logspace(-1, 2, 500), dB=True)
plt.suptitle("Bode plot: mechanical power to rotor angle", y=1.01)
plt.show()

# %% [markdown]
# ## 6. 線形モデルはどこまで正しいか
#
# 線形化は「微小な擾乱」に対してのみ正しい近似です。
# 擾乱の大きさを変えて、線形モデルと非線形シミュレーションを比べましょう。

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=False)
for ax, epsilon in zip(axes, (0.001, 0.1, 0.6)):
    nonlinear = genstab.simulate(
        system, t_end=10.0, dt=0.005,
        x0=system.initial_state() + np.array([epsilon, 0.0]),
    )
    _, y = ct.initial_response(G, T=nonlinear.t, X0=np.array([epsilon, 0.0]))
    ax.plot(nonlinear.t, np.degrees(nonlinear.delta - system.operating_point.delta),
            label="nonlinear")
    ax.plot(nonlinear.t, np.degrees(y[0]), "--", label="linear")
    ax.set_title(f"disturbance = {np.degrees(epsilon):.1f} deg")
    ax.set_xlabel("Time [s]")
    ax.legend(fontsize=8)
axes[0].set_ylabel("$\\Delta\\delta$ [deg]")
plt.tight_layout()
plt.show()

# %% [markdown]
# 小さな擾乱では両者が重なりますが、大きくなるとずれます。
# **過渡安定性の評価に線形モデルを使ってはいけない** のはこのためです。
#
# 次回は、この線形モデルに自動電圧調整器 (AVR) を追加します。
# 端子電圧を保つための装置が、なぜ動揺の制動を悪化させるのかを見ます。
