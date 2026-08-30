# %% [markdown]
# # 10 動揺方程式 — 発電機はなぜ揺れるのか
#
# ## この回のねらい
#
# - 同期発電機の回転子の動きを表す **動揺方程式** を理解する
# - 送電線事故に対する応答を計算し、**過渡安定性** という概念をつかむ
# - 慣性定数 H と制動係数 D が応答をどう変えるかを確かめる
#
# ## 動揺方程式
#
# 同期発電機の回転子は、原動機（タービン）から機械的な力を受け、
# 電気的な負荷から反対向きの力を受けます。この 2 つが釣り合っている限り
# 回転子は同期速度で回り続けますが、釣り合いが崩れると加速・減速します。
# これを表すのが動揺方程式です。
#
# $$
# \frac{d\delta}{dt} = \omega_s \Delta\omega, \qquad
# 2H \frac{d\Delta\omega}{dt} = P_m - P_e - D\,\Delta\omega
# $$
#
# | 記号 | 意味 | 単位 |
# |---|---|---|
# | $\delta$ | 回転子の位相角（同期回転座標から見た角度） | rad |
# | $\Delta\omega$ | 速度偏差 $(\omega - \omega_s)/\omega_s$ | p.u. |
# | $H$ | 慣性定数（回転体が蓄えるエネルギーの指標） | s |
# | $D$ | 制動係数 | p.u. |
# | $P_m$ | 機械入力（タービンの出力） | p.u. |
# | $P_e$ | 電気出力 | p.u. |
#
# 力学の $m\ddot{x} = F$ とまったく同じ形です。$2H$ が質量、
# $P_m - P_e$ が力、$D\Delta\omega$ が摩擦にあたります。
#
# ## 電気出力 $P_e$
#
# 1 機無限大母線系統（Single-Machine Infinite-Bus: SMIB）では、発電機の内部起電力 $E\angle\delta$ と
# 無限大母線 $V_\infty\angle 0$ が総リアクタンス $X$ を介してつながります。
# このとき電気出力は
#
# $$ P_e = \frac{E V_\infty}{X}\sin\delta $$
#
# となります。$\sin\delta$ が現れるのが電力系統の特徴で、この非線形性が
# 過渡安定性の問題を生みます。

# %%
import numpy as np
import matplotlib.pyplot as plt

import genstab
from genstab.plotting import use_genstab_style, plot_swing, plot_power_angle

use_genstab_style()

# %% [markdown]
# ## 1. 系を組み立てる
#
# 部品は 3 つです。
#
# - `ClassicalMachine` : 発電機（古典モデル）
# - `SMIBNetwork` : 送電線（事故前・事故中・事故後のリアクタンス）
# - `FaultSchedule` : いつ事故が起きて、いつ除去されるか
#
# 事故中のリアクタンスを `np.inf` にしているのは、
# 「事故のあいだ電力をまったく送れない」状況を表すためです。

# %%
machine = genstab.ClassicalMachine(
    H=5.0,          # 慣性定数 [s]
    D=2.0,          # 制動係数 [p.u.]
    x_d_prime=0.3,  # 過渡リアクタンス [p.u.]
    E=1.1,          # 内部起電力 [p.u.]
)
network = genstab.SMIBNetwork(
    x_pre=0.4,      # 事故前の外部リアクタンス [p.u.]
    x_fault=np.inf, # 事故中（電力を送れない）
    x_post=0.6,     # 事故後（1 回線が開放されて増える）
)
fault = genstab.FaultSchedule(t_fault=1.0, t_clear=1.15)

system = genstab.SMIBSystem(machine, network, fault, Pe0=0.8)
print(system.describe())

# %% [markdown]
# `Pe0=0.8` と指定すると、事故前にその電力を送るための位相角 $\delta_0$ が
# 逆算されます。$P_e = P_{max}\sin\delta$ を $\delta$ について解いているだけです。

# %%
delta0 = system.operating_point.delta
pmax = network.max_power(genstab.Stage.PRE, machine.E)
print(f"事故前の送電可能最大電力 Pmax = {pmax:.4f} p.u.")
print(f"初期位相角 delta0 = {np.degrees(delta0):.3f} deg")
print(f"検算: Pmax * sin(delta0) = {pmax * np.sin(delta0):.4f} p.u.")

# %% [markdown]
# ## 2. シミュレーションする

# %%
result = genstab.simulate(system, t_end=10.0, dt=0.005)
plot_swing(result, ("delta_deg", "omega", "Pe"), title="Response to a cleared fault")
plt.show()

print(f"最大位相角        : {result.delta_deg.max():.2f} deg")
print(f"安定判定          : {'安定' if result.is_stable() else '脱調'}")

# %% [markdown]
# 赤い網掛けが事故期間です。この間 $P_e = 0$ になるので、
# 機械入力 $P_m$ がまるごと加速に使われて位相角が急に増えます。
# 事故が除去されると $P_e$ が復活し、$P_e > P_m$ の領域で減速します。
#
# ## 3. P-δ 平面で見る
#
# 同じ現象を横軸 $\delta$、縦軸 $P$ で見ると、次回学ぶ等面積法につながります。

# %%
plot_power_angle(system, result, title="Trajectory on the power-angle plane")
plt.show()

# %% [markdown]
# ## 4. 演習: 慣性定数の影響
#
# 慣性定数 $H$ を変えると応答がどう変わるか調べましょう。
# $H$ は回転体の「重さ」にあたる量です。重い回転子ほど、
# 同じ力を受けたときに角度の変化はゆっくりになります。
#
# 下のセルの `# TODO` を埋めて、$H = 3, 5, 10$ の 3 通りを比較してください。

# %%
results = []
labels = []
for H in (3.0, 5.0, 10.0):
    # TODO: 慣性定数 H だけを変えた系を作り、シミュレーションして
    #       results と labels に追加すること。
    # BEGIN SOLUTION
    trial = genstab.SMIBSystem(
        genstab.ClassicalMachine(H=H, D=2.0, x_d_prime=0.3, E=1.1),
        genstab.SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6),
        genstab.FaultSchedule(t_fault=1.0, t_clear=1.15),
        Pe0=0.8,
    )
    results.append(genstab.simulate(trial, t_end=10.0, dt=0.005))
    labels.append(f"H = {H} s")
    # END SOLUTION

from genstab.plotting import compare_results

compare_results(results, labels, ("delta_deg",), title="Effect of inertia constant H")
plt.show()

for label, r in zip(labels, results):
    print(f"{label}: 最大位相角 {r.delta_deg.max():6.2f} deg")

# %% [markdown]
# **考えてみよう**: 慣性が大きいほど第 1 波の振れは小さくなりますが、
# 振動の周期は長くなります。なぜでしょうか。
# （ヒント: ばね-質量系の固有振動数 $\omega_n = \sqrt{k/m}$）

# %% [markdown]
# ## 5. 演習: 制動係数の影響
#
# 制動係数 $D$ を $0, 2, 8$ と変えて比較してください。

# %%
results = []
labels = []
for D in (0.0, 2.0, 8.0):
    # TODO: 制動係数 D だけを変えて比較すること。
    # BEGIN SOLUTION
    trial = genstab.SMIBSystem(
        genstab.ClassicalMachine(H=5.0, D=D, x_d_prime=0.3, E=1.1),
        genstab.SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6),
        genstab.FaultSchedule(t_fault=1.0, t_clear=1.15),
        Pe0=0.8,
    )
    results.append(genstab.simulate(trial, t_end=20.0, dt=0.005))
    labels.append(f"D = {D}")
    # END SOLUTION

compare_results(results, labels, ("delta_deg",), title="Effect of damping coefficient D")
plt.show()

# %% [markdown]
# $D = 0$ では振動がいつまでも減衰しません。これは摩擦のない振り子と同じで、
# エネルギーが保存されるからです。実際の発電機には制動巻線があり、
# ある程度の制動が自然に備わっています。
#
# ただし **制動が正であることは自明ではありません**。第 14 回（AVR）で、
# 自動電圧調整器 (AVR) を付けると制動が負になりうることを見ます。

# %% [markdown]
# ## 6. 事故除去が遅れるとどうなるか

# %%
for clearing in (0.15, 0.25, 0.35):
    trial = genstab.SMIBSystem(
        machine,
        genstab.SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6),
        genstab.FaultSchedule(t_fault=1.0, t_clear=1.0 + clearing),
        Pe0=0.8,
    )
    r = genstab.simulate(trial, t_end=10.0)
    verdict = "安定" if r.is_stable() else "脱調"
    print(f"事故除去時間 {clearing*1000:5.0f} ms -> {verdict}")

# %% [markdown]
# 除去が遅れるとある時点で脱調します。この境目を **臨界事故除去時間 (CCT)**
# と呼び、第 12 回で正確に求めます。その前に、次回は
# **等面積法** という図式的な考え方でこの境目を理解します。
