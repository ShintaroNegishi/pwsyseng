# %% [markdown]
# # 14 自動電圧調整器 (AVR) の功罪
#
# ## この回のねらい
#
# - **1 軸モデル** を使って界磁回路のダイナミクスを扱う
# - AVR が端子電圧を保つ働きを確認する
# - **高応答 AVR が動揺モードの制動を悪化させる** ことを固有値で確かめる
#
# ## なぜ古典モデルでは AVR を扱えないのか
#
# 前回まで使った古典モデルは、内部起電力 $E'$ を **定数** と仮定していました。
# AVR は界磁電圧 $E_{fd}$ を操作する装置ですが、$E'$ が定数だと
# 界磁電圧を変えても何も起きません。
#
# そこで界磁磁束の変化を状態量として持つ **1 軸モデル（3 次モデル）** を使います。
#
# $$
# \frac{d\delta}{dt} = \omega_s\Delta\omega, \quad
# 2H\frac{d\Delta\omega}{dt} = P_m - P_e - D\Delta\omega, \quad
# T'_{d0}\frac{dE'_q}{dt} = E_{fd} - E'_q - (x_d - x'_d)I_d
# $$
#
# 第 3 式の右辺最後の項が **電機子反作用** で、負荷が重いほど
# 内部起電力を押し下げます。AVR はこれを補うために界磁電圧を上げます。

# %%
import numpy as np
import matplotlib.pyplot as plt
import control as ct

import genstab
from genstab import linearize as lin
from genstab import smallsignal as ss
from genstab.plotting import use_genstab_style, plot_eigenvalues, plot_swing

use_genstab_style()

# %% [markdown]
# ## 1. 重負荷・長距離送電の条件を作る
#
# AVR の悪影響は、**重負荷で外部リアクタンスが大きい**（＝長距離送電）
# 条件で顕著になります。ここでは Kundur の教科書に近い設定を使います。

# %%
def build(controllers=None, D=0.0, Pe0=0.9):
    machine = genstab.OneAxisMachine(
        H=3.5, D=D,
        xd=1.81,          # d 軸同期リアクタンス
        xd_prime=0.30,    # d 軸過渡リアクタンス
        Td0_prime=8.0,    # 界磁の時定数 [s]
        Vt0=1.0,          # 端子電圧の目標値
    )
    network = genstab.SMIBNetwork(x_pre=0.65, x_fault=np.inf, x_post=0.65, V_inf=0.995)
    return genstab.SMIBSystem(
        machine, network, controllers=list(controllers or []), Pe0=Pe0
    )

plant = build()
print(plant.describe())
print(f"\n内部起電力 E'q0 = {plant.initial_state()[2]:.4f} p.u.")
print(f"界磁電圧   Efd0 = {plant.operating_point.Efd:.4f} p.u.")

# %% [markdown]
# 界磁電圧が内部起電力より大きいことに注意してください。
# その差 $(x_d - x'_d)I_d$ が、電機子反作用に打ち勝つために必要な分です。

# %% [markdown]
# ## 2. 界磁のモードが増える
#
# 状態が 1 つ増えたので、固有値も 1 つ増えます。

# %%
modes_open = ss.analyze(plant)
print(modes_open.table())

# %% [markdown]
# 動揺モード（$\delta$, $\Delta\omega$ が主役）に加えて、
# **界磁磁束の遅いモード**（$E'_q$ が主役、時定数は数秒）が現れました。
#
# ## 3. AVR を付ける
#
# AVR は端子電圧を測って界磁電圧を操作します。
#
# $$ T_a \frac{dE_{fd}}{dt} = K_a (V_{ref} - V_t) - E_{fd} $$
#
# `SimpleExciter` を `controllers` に渡すだけで接続できます。
# 渡さなければ界磁電圧は定数のままです。

# %%
avr = genstab.SimpleExciter(Ka=200.0, Ta=0.05)
with_avr = build([avr])

print(f"AVR の基準電圧 v_ref = {avr.v_ref:.6f} p.u.")
print(f"（定常状態で Efd が元の値と一致するよう自動設定される）")
print(f"動作点の残差 = {ss.residual_at_operating_point(with_avr):.2e}  <- ほぼ 0 なら正しく初期化されている")

# %% [markdown]
# ## 4. AVR の「功」— 電圧を保つ
#
# 無限大母線の電圧が下がったとき、AVR があると端子電圧をどれだけ
# 保てるかを見ます。ここでは基準電圧に微小なステップを与えて
# 応答の速さを比べます。

# %%
G_with = lin.state_space(with_avr, inputs=("Vref",), outputs=("Vt", "Efd"))
T = np.linspace(0, 5, 1000)
response = ct.step_response(G_with * 0.01, T)

fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
axes[0].plot(T, response.outputs[0].ravel())
axes[0].set_ylabel("$\\Delta V_t$ [p.u.]")
axes[1].plot(T, response.outputs[1].ravel(), color="tab:orange")
axes[1].set_ylabel("$\\Delta E_{fd}$ [p.u.]")
axes[1].set_xlabel("Time [s]")
axes[0].set_title("Response to a 0.01 p.u. step in the AVR reference voltage")
plt.tight_layout()
plt.show()

print(f"端子電圧の直流ゲイン = {float(np.squeeze(ct.dcgain(G_with))[0]):.4f}")
print("（1 に近いほど、基準電圧の指令によく追従している）")

# %% [markdown]
# ## 5. AVR の「罪」— 制動を壊す
#
# ここが本教材の山場です。AVR のゲインを変えて動揺モードの固有値を追います。

# %%
def swing_mode(modes):
    """δ と Δω が主役の振動モードを取り出す。"""
    candidates = [
        i for i in range(modes.eigenvalues.size)
        if modes.eigenvalues[i].imag > 1e-6
        and modes.participation[0, i] + modes.participation[1, i] > 0.4
    ]
    return candidates[0]

gains = [0.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0]
records = []
for Ka in gains:
    system = build() if Ka == 0.0 else build([genstab.SimpleExciter(Ka=Ka, Ta=0.05)])
    modes = ss.analyze(system)
    i = swing_mode(modes)
    records.append((Ka, modes.eigenvalues[i], modes.damping_ratios[i]))

print(f"{'Ka':>6}  {'固有値':>24}  {'減衰比':>9}  判定")
print("-" * 60)
for Ka, eigenvalue, zeta in records:
    label = "AVR なし" if Ka == 0 else ""
    verdict = "安定" if eigenvalue.real < 0 else "★不安定★"
    print(f"{Ka:>6.0f}  {eigenvalue.real:>+11.5f}{eigenvalue.imag:>+11.5f}j  "
          f"{zeta:>+9.5f}  {verdict} {label}")

# %%
plot_eigenvalues(
    [np.array([r[1], np.conj(r[1])]) for r in records],
    [f"Ka = {r[0]:.0f}" for r in records],
    title="Swing mode moves to the right half plane as AVR gain increases",
)
plt.show()

# %% [markdown]
# **AVR を強くすると動揺モードが右半面に移り、系が不安定になります。**
#
# ### なぜこうなるのか
#
# 回転子が揺れると位相角 $\delta$ が変わり、端子電圧 $V_t$ も変わります。
# AVR はこれを打ち消そうと界磁電圧を動かしますが、界磁回路には時定数
# $T'_{d0}$（数秒）があるため、その効果は **遅れて** 現れます。
#
# 遅れて到着した電気トルクが速度偏差と逆位相になると、
# それは **負の制動トルク** として働きます。AVR が速く強いほど、
# この負制動が大きくなります。
#
# この現象は Heffron-Phillips モデルの係数 $K_5$ が負になる条件、すなわち
# **重負荷・大きな外部リアクタンス** で起こります。

# %% [markdown]
# ## 6. 演習: どんな運転条件で起こるのか
#
# 送電電力と外部リアクタンスを変えて、AVR ありの動揺モードの減衰比が
# どこで負になるか調べてください。

# %%
powers = np.linspace(0.3, 1.0, 8)
reactances = [0.3, 0.5, 0.65, 0.8]

plt.figure(figsize=(8, 4.5))
for x_e in reactances:
    zetas = []
    for Pe0 in powers:
        # TODO: 外部リアクタンス x_e、送電電力 Pe0 の系に AVR(Ka=200) を付け、
        #       動揺モードの減衰比を zetas に追加すること。
        # BEGIN SOLUTION
        machine = genstab.OneAxisMachine(H=3.5, D=0.0, xd=1.81, xd_prime=0.30,
                                         Td0_prime=8.0, Vt0=1.0)
        network = genstab.SMIBNetwork(x_pre=x_e, x_fault=np.inf, x_post=x_e, V_inf=0.995)
        system = genstab.SMIBSystem(
            machine, network, Pe0=Pe0,
            controllers=[genstab.SimpleExciter(Ka=200.0, Ta=0.05)],
        )
        modes = ss.analyze(system)
        zetas.append(modes.damping_ratios[swing_mode(modes)])
        # END SOLUTION
    plt.plot(powers, zetas, "o-", label=f"$x_e$ = {x_e}")

plt.axhline(0.0, color="k", lw=1.2)
plt.axhline(0.05, color="tab:green", ls="--", lw=1.2, label="practical target (5 %)")
plt.xlabel("Power transfer $P_{e0}$ [p.u.]")
plt.ylabel("Damping ratio of the swing mode")
plt.title("Where does the AVR destabilise the swing mode? (Ka = 200)")
plt.legend(fontsize=9)
plt.show()

# %% [markdown]
# **重負荷かつ外部リアクタンスが大きいほど、減衰比が負に落ち込みます。**
# つまり系統を目一杯使って遠くへ送電するときほど危険です。
#
# ## 7. 時間応答で確かめる

# %%
results, labels = [], []
for label, system in (("AVR なし", build(D=0.0)),
                      ("AVR あり (Ka=200)", build([genstab.SimpleExciter(Ka=200.0, Ta=0.05)]))):
    disturbed = system.initial_state().copy()
    disturbed[0] += np.radians(1.0)   # 1 度だけずらす
    results.append(genstab.simulate(system, t_end=20.0, dt=0.005, x0=disturbed))
    labels.append(label)

fig, ax = plt.subplots(figsize=(9, 4))
for result, label in zip(results, labels):
    ax.plot(result.t, np.degrees(result.delta - result.system.operating_point.delta),
            label=label)
ax.set_xlabel("Time [s]")
ax.set_ylabel("$\\Delta\\delta$ [deg]")
ax.set_title("A 1-degree disturbance: the AVR makes the oscillation grow")
ax.legend()
plt.show()

# %% [markdown]
# AVR なしでは振動がゆっくり減衰しますが、AVR ありでは **成長** します。
# 事故がなくても、わずかな擾乱から振動が育ってしまう状態です。
#
# ## では AVR を弱くすればよいのか
#
# いいえ。AVR を弱めると電圧品質が悪化し、過渡安定性（第 1 波）も
# 悪くなります。AVR は必要な装置です。
#
# **必要なのは、AVR を保ったまま制動だけを取り戻す仕組みです。**
# それが次回扱う電力系統安定化装置 (PSS) です。
