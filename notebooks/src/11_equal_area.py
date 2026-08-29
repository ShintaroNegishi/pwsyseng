# %% [markdown]
# # 11 等面積法 — 図で安定性を判定する
#
# ## この回のねらい
#
# - **等面積法 (Equal Area Criterion)** の考え方を理解する
# - 加速面積と減速面積を計算し、安定・不安定の境目を求める
# - 図式的な判定と、前回の時間領域シミュレーションが一致することを確かめる
#
# ## 考え方
#
# 動揺方程式に $\Delta\omega$ を掛けて積分すると、
# **回転子の運動エネルギーの変化** が
#
# $$ \int_{\delta_0}^{\delta} (P_m - P_e)\, d\delta $$
#
# に等しいことが導けます（制動 $D=0$ の場合）。つまり P-δ 平面上の
# **面積がエネルギー** を表します。
#
# - 事故中は $P_m > P_e$ なので回転子は加速し、運動エネルギーを蓄える
#   → この面積を **加速面積 $A_1$** と呼ぶ
# - 事故除去後は $P_e > P_m$ の領域で減速し、エネルギーを吐き出す
#   → この面積を **減速面積 $A_2$** と呼ぶ
#
# 蓄えたエネルギーを全部吐き出せれば回転子は行き過ぎずに戻ってきます。
# 吐き出しきれなければ、そのまま加速し続けて同期を失います。
#
# $$ A_1 \le A_2 \;\Rightarrow\; \text{安定}, \qquad A_1 > A_2 \;\Rightarrow\; \text{脱調} $$

# %%
import numpy as np
import matplotlib.pyplot as plt

import genstab
from genstab import eac
from genstab.plotting import use_genstab_style

use_genstab_style()

# %% [markdown]
# ## 1. 制動のない系を用意する
#
# 等面積法はエネルギー保存を前提にしているので、
# **制動 $D = 0$ でないと厳密には成り立ちません**。まずその条件で確かめます。

# %%
machine = genstab.ClassicalMachine(H=5.0, D=0.0, x_d_prime=0.3, E=1.1)
network = genstab.SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6)
system = genstab.SMIBSystem(
    machine, network, genstab.FaultSchedule(t_fault=1.0, t_clear=1.1), Pe0=0.8
)
print(system.describe())

# %% [markdown]
# ## 2. 3 本の P-δ 曲線
#
# 事故前・事故中・事故後でリアクタンスが変わるので、P-δ 曲線も 3 本になります。
# 波高値は $P_{max} = E V_\infty / X$ で、$X$ が大きいほど低くなります。

# %%
from genstab.plotting import plot_power_angle

plot_power_angle(system, title="Three power-angle curves")
plt.show()

for stage in (genstab.Stage.PRE, genstab.Stage.FAULT, genstab.Stage.POST):
    x = network.transfer_reactance(stage)
    print(f"{stage.value:6s}: X = {x:6.3f} p.u.  ->  Pmax = "
          f"{network.max_power(stage, machine.E):.4f} p.u.")

# %% [markdown]
# 事故中は $X = \infty$ なので $P_{max} = 0$、つまり曲線が横軸に潰れます。
# 事故後は 1 回線が開放されてリアクタンスが増えるため、
# 事故前より低い曲線になります。**事故後の曲線が $P_m$ より低くなってしまうと、
# どんなに早く事故を除去しても安定運転に戻れません。**
#
# ## 3. 3 つの重要な角度
#
# | 記号 | 意味 |
# |---|---|
# | $\delta_0$ | 事故前の安定平衡点。$P_m = P_{max,pre}\sin\delta_0$ |
# | $\delta_u$ | 事故後の不安定平衡点。$\delta_u = \pi - \arcsin(P_m/P_{max,post})$ |
# | $\delta_c$ | 臨界事故除去角。ここまでに除去すればぎりぎり安定 |

# %%
outcome = eac.evaluate(system)
print(outcome.summary())

# %% [markdown]
# ## 4. 面積を目で見る
#
# 赤が加速面積 $A_1$、緑が減速面積 $A_2$ です。
# 臨界事故除去角ではこの 2 つがちょうど等しくなります。

# %%
eac.plot_equal_area(system, title="Equal area criterion at the critical angle")
plt.show()

# %% [markdown]
# ## 5. 演習: 除去角を変えて面積を比べる
#
# 事故除去角 $\delta_c$ を臨界値より小さく・大きくしたときに、
# 加速面積と減速面積がどうなるか確かめてください。

# %%
delta_critical = eac.critical_clearing_angle(system)

for factor in (0.8, 1.0, 1.2):
    # TODO: 臨界角の factor 倍を除去角として eac.evaluate に渡し、
    #       A1, A2, 余裕, 判定を表示すること。
    # BEGIN SOLUTION
    trial = eac.evaluate(system, delta_c=factor * delta_critical)
    print(f"δc = 臨界角の {factor:.1f} 倍 ({np.degrees(trial.delta_c):6.2f} deg): "
          f"A1 = {trial.accelerating_area:.4f}, A2 = {trial.decelerating_area:.4f}, "
          f"余裕 = {trial.margin:+.3f} -> {'安定' if trial.is_stable else '脱調'}")
    # END SOLUTION

# %% [markdown]
# ## 6. 時間領域シミュレーションとの照合
#
# 等面積法は「事故除去角」で判定しますが、実際に制御できるのは
# 「事故除去**時間**」です。両者を結びつけるため、シミュレーションで
# 位相角が $\delta_c$ に達する時刻を調べます。

# %%
from dataclasses import replace

analytic_cct = eac.critical_clearing_time_analytic(system)
print(f"等面積法から求めた臨界事故除去時間 = {analytic_cct*1000:.2f} ms")

for factor, name in ((0.95, "臨界より少し短い"), (1.05, "臨界より少し長い")):
    trial = replace(
        system, fault=genstab.FaultSchedule(1.0, 1.0 + factor * analytic_cct)
    )
    result = genstab.simulate(trial, t_end=20.0)
    limit = outcome.delta_u - outcome.delta_0
    verdict = "安定" if result.is_stable(angle_limit=limit) else "脱調"
    print(f"  {name} ({factor*analytic_cct*1000:.1f} ms) -> {verdict}")

# %% [markdown]
# ## 7. 演習: なぜ制動があると等面積法がずれるのか
#
# 制動 $D > 0$ の系で、等面積法の予測と実際の CCT を比べてください。

# %%
damped = genstab.SMIBSystem(
    genstab.ClassicalMachine(H=5.0, D=4.0, x_d_prime=0.3, E=1.1),
    genstab.SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=0.6),
    genstab.FaultSchedule(1.0, 1.1),
    Pe0=0.8,
)

# TODO: 制動ありの系について、数値的に求めた CCT を
#       eac.critical_clearing_time で計算し、上の解析値と比べること。
# BEGIN SOLUTION
numerical_cct = eac.critical_clearing_time(damped, t_end=20.0, tolerance=1e-4)
print(f"制動なしの解析値      : {analytic_cct*1000:.2f} ms")
print(f"制動あり (D=4) の数値解: {numerical_cct*1000:.2f} ms")
print(f"差                    : {(numerical_cct - analytic_cct)*1000:+.2f} ms "
      f"({(numerical_cct/analytic_cct - 1)*100:+.1f} %)")
# END SOLUTION

# %% [markdown]
# 制動があると CCT は **長くなります**。制動が回転子の運動エネルギーを
# 熱として吸収してくれるぶん、余裕が増えるからです。
#
# つまり等面積法は制動を無視している点で **保守的（安全側）** な評価に
# なっています。実務でも、こうした簡便法は安全側に外れるように使います。
#
# 次回は、この CCT を数値的に精度よく求める方法を扱います。
