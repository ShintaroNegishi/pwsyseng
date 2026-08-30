# %% [markdown]
# # 12 臨界事故除去時間（Critical Clearing Time: CCT） — 保護はどれだけ速くあるべきか
#
# ## この回のねらい
#
# - **臨界事故除去時間 (CCT)** を数値的に求める
# - CCT が系のパラメータや運転状態にどう依存するかを調べる
# - 保護リレーと遮断器の動作時間という実務的な要求に結びつける
#
# ## なぜ CCT が重要か
#
# 送電線に事故が起きたとき、保護リレーが事故を検出し、遮断器が回線を
# 切り離します。この一連の動作にかかる時間が CCT を超えると、
# 発電機は同期を失って系統から解列され、大規模な停電につながりかねません。
#
# 実際の系統では、送電線の保護は
#
# - リレーの検出時間: 約 20〜30 ms
# - 遮断器の動作時間: 約 30〜50 ms
#
# 合わせて **50〜80 ms 程度** で事故を除去できるよう設計されています。
# CCT がこれを下回るような運転点は、そもそも許容されません。
#
# ## 求め方
#
# 等面積法には解析解がありますが、制動や制御器があると使えません。
# そこで **事故除去時間を変えながらシミュレーションを繰り返し、
# 安定・不安定の境目を二分探索する** のが実用的な方法です。

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace

import genstab
from genstab import eac
from genstab.plotting import use_genstab_style, compare_results

use_genstab_style()

# %%
def make_system(H=5.0, D=2.0, Pe0=0.8, x_post=0.6, E=1.1, clearing=0.15):
    """パラメータを変えた SMIB を作るための補助関数。"""
    return genstab.SMIBSystem(
        genstab.ClassicalMachine(H=H, D=D, x_d_prime=0.3, E=E),
        genstab.SMIBNetwork(x_pre=0.4, x_fault=np.inf, x_post=x_post),
        genstab.FaultSchedule(t_fault=1.0, t_clear=1.0 + clearing),
        Pe0=Pe0,
    )

system = make_system()
print(system.describe())

# %% [markdown]
# ## 1. 境目を目で見る
#
# 事故除去時間を少しずつ変えて、どこで挙動が変わるか見てみましょう。

# %%
results, labels = [], []
for clearing in (0.15, 0.20, 0.22, 0.25):
    result = genstab.simulate(make_system(clearing=clearing), t_end=8.0)
    results.append(result)
    labels.append(f"{clearing*1000:.0f} ms ({'安定' if result.is_stable() else '脱調'})")

compare_results(results, labels, ("delta_deg",), title="Effect of clearing time")
plt.ylim(-180, 540)
plt.show()

# %% [markdown]
# ## 2. 二分探索で CCT を求める

# %%
cct = eac.critical_clearing_time(system, t_end=20.0, tolerance=1e-4)
print(f"CCT = {cct*1000:.2f} ms")
print(f"     = {cct*50:.2f} サイクル (50 Hz)")

# %% [markdown]
# 求まった値の前後で判定が変わることを確かめます。

# %%
for offset in (-0.002, +0.002):
    trial = replace(system, fault=genstab.FaultSchedule(1.0, 1.0 + cct + offset))
    result = genstab.simulate(trial, t_end=20.0)
    limit = eac.unstable_equilibrium_angle(system) - system.operating_point.delta
    print(f"除去時間 {(cct+offset)*1000:6.2f} ms -> "
          f"{'安定' if result.is_stable(angle_limit=limit) else '脱調'}")

# %% [markdown]
# ## 3. 演習: 運転状態と CCT
#
# 送電電力 $P_{e0}$ を変えると CCT はどう変わるでしょうか。
# 予想を立ててから計算してください。

# %%
powers = np.arange(0.4, 1.05, 0.1)
ccts = []
for Pe0 in powers:
    # TODO: 送電電力 Pe0 を変えた系の CCT を求めて ccts に追加すること。
    # BEGIN SOLUTION
    ccts.append(eac.critical_clearing_time(make_system(Pe0=Pe0), t_end=20.0, tolerance=1e-4))
    # END SOLUTION

plt.figure(figsize=(7, 4))
plt.plot(powers, np.array(ccts) * 1000, "o-")
plt.axhspan(50, 80, color="tab:green", alpha=0.15,
            label="typical protection clearing time")
plt.xlabel("Pre-fault power transfer $P_{e0}$ [p.u.]")
plt.ylabel("Critical clearing time [ms]")
plt.title("CCT versus operating point")
plt.legend()
plt.show()

for Pe0, t in zip(powers, ccts):
    print(f"Pe0 = {Pe0:.1f} p.u. -> CCT = {t*1000:6.1f} ms")

# %% [markdown]
# **重負荷ほど CCT が短くなります。** 送電量が多いほど事故前の位相角
# $\delta_0$ が大きく、不安定平衡点 $\delta_u$ までの余裕（減速面積）が
# 少ないためです。
#
# これは実務上とても重要な帰結です。系統を目一杯使って送電すると、
# 経済的には有利でも安定性の余裕を削ることになります。
# **経済性と安定性はトレードオフの関係にあります。**

# %% [markdown]
# ## 4. 演習: 慣性と CCT
#
# 近年、火力発電が減って太陽光・風力が増えると、系統全体の慣性が
# 下がることが問題になっています（慣性低下問題）。
# 慣性定数 $H$ を変えて CCT への影響を調べてください。

# %%
inertias = np.array([2.0, 3.0, 5.0, 8.0, 12.0])
ccts_h = []
for H in inertias:
    # TODO: 慣性定数 H を変えた系の CCT を求めること。
    # BEGIN SOLUTION
    ccts_h.append(eac.critical_clearing_time(make_system(H=H), t_end=20.0, tolerance=1e-4))
    # END SOLUTION

plt.figure(figsize=(7, 4))
plt.plot(inertias, np.array(ccts_h) * 1000, "s-", color="tab:orange")
plt.axhspan(50, 80, color="tab:green", alpha=0.15,
            label="typical protection clearing time")
plt.xlabel("Inertia constant $H$ [s]")
plt.ylabel("Critical clearing time [ms]")
plt.title("CCT versus inertia (why inertia matters for renewables)")
plt.legend()
plt.show()

# 理論的には CCT は sqrt(H) に比例するはず（等面積法の解析解より）。
reference = ccts_h[2] * np.sqrt(inertias / inertias[2])
for H, t, ref in zip(inertias, ccts_h, reference):
    print(f"H = {H:5.1f} s -> CCT = {t*1000:6.1f} ms   "
          f"(sqrt(H) 則の予測 {ref*1000:6.1f} ms)")

# %% [markdown]
# 等面積法の解析解
#
# $$ t_{cr} = \sqrt{\frac{4H(\delta_c - \delta_0)}{\omega_s P_m}} $$
#
# から、CCT は $\sqrt{H}$ に比例するはずです。数値解もほぼその通りに
# なっています（制動があるぶん少しずれます）。
#
# **慣性が半分になると CCT は $1/\sqrt{2} \approx 0.71$ 倍になります。**
# 再生可能エネルギーの導入で系統の慣性が下がると、保護に要求される
# 速度が上がる、という関係がここから読み取れます。

# %% [markdown]
# ## 5. 演習: 事故後のネットワーク構成
#
# 事故後リアクタンス `x_post` は「事故で何回線が失われたか」を表します。
# これを変えて CCT への影響を調べてください。
# 事故後に平衡点が存在しなくなる（$P_{max,post} \le P_m$）とどうなるかも
# 確かめましょう。

# %%
for x_post in (0.4, 0.6, 0.8, 1.0, 1.4):
    trial = make_system(x_post=x_post)
    pmax_post = trial.network.max_power(genstab.Stage.POST, trial.machine.E)
    try:
        # BEGIN SOLUTION
        t = eac.critical_clearing_time(trial, t_end=20.0, tolerance=1e-4)
        print(f"x_post = {x_post:.1f} (Pmax_post = {pmax_post:.3f}) -> CCT = {t*1000:6.1f} ms")
        # END SOLUTION
    except ValueError as error:
        print(f"x_post = {x_post:.1f} (Pmax_post = {pmax_post:.3f}) -> 計算不能: {error}")

# %% [markdown]
# ## まとめ
#
# - CCT は保護システムへの設計要求そのものである
# - 重負荷ほど、慣性が小さいほど、事故後の系統が弱いほど CCT は短い
# - 事故後に $P_{max,post} \le P_m$ となる構成では、そもそも安定運転に戻れない
#
# ここまでは **大きな事故に耐えられるか（過渡安定性）** を見てきました。
# 次回からは **小さな擾乱に対して運転点が保てるか（定態安定性）** を扱います。
