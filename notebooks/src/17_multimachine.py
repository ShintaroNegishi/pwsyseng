# %% [markdown]
# # 17 多機系統 — 発電機どうしの動揺
#
# ## この回のねらい
#
# - 複数の発電機を持つ系統の過渡安定性を計算する
# - **Kron 縮約** で発電機内部母線だけの問題に縮める考え方を理解する
# - SMIB では見えない **機器間動揺モード** を観察する
#
# ## 扱う系統: WSCC 3 機 9 母線系統
#
# 過渡安定性の教材として世界的に使われているベンチマーク系統です。
# 3 台の発電機、9 つの母線、3 つの負荷からなります。
#
# ```
#        G2                         G3
#         |                          |
#     (2)-+-[T]-(7)---(8)---(9)-[T]-+-(3)
#                |           |
#               (5)         (6)
#                |           |
#                +----(4)----+
#                      |
#                     [T]
#                      |
#                     (1)
#                      |
#                     G1 (slack)
# ```
#
# ## 計算の流れ
#
# 1. 線路・変圧器から母線アドミタンス行列 Ybus を組む
# 2. 負荷を **定インピーダンス** に変換して Ybus に足しこむ
# 3. 各発電機の内部母線（背後 $x'_d$ の先）を追加する
# 4. 発電機内部母線以外を **Kron 縮約** で消す
# 5. 縮約行列から各機の電気出力を求める
#
# $$
# P_{e,i} = \sum_j E_i E_j \bigl(G_{ij}\cos(\delta_i-\delta_j) + B_{ij}\sin(\delta_i-\delta_j)\bigr)
# $$
#
# 負荷を定インピーダンスとみなすのは、負荷母線を縮約で消すために
# 必要な仮定です。実際の負荷は電圧に依存するので、これは近似です。

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import replace
from pathlib import Path

import genstab
from genstab import eac
from genstab.multimachine import load_case
from genstab.plotting import use_genstab_style

use_genstab_style()

case_path = Path("..") / "cases" / "wscc9.yaml"
if not case_path.exists():
    case_path = Path("cases") / "wscc9.yaml"

system = load_case(case_path)
print(system.describe())

# %% [markdown]
# ## 1. まずデータを疑う
#
# **数値計算がおかしいとき、ソルバの設定より先にデータを確認します。**
# ケースファイルの潮流解と線路データが整合していなければ、
# 以降の計算はすべて「解けているが間違っている」状態になります。
#
# 母線電圧から計算した注入電力が、発電量・負荷と一致するか確かめます。

# %%
injections = system.network.power_injections()
expected = {1: 0.716+0.270j, 2: 1.630+0.067j, 3: 0.850-0.109j,
            4: 0j, 5: -(1.25+0.50j), 6: -(0.90+0.30j), 7: 0j,
            8: -(1.00+0.35j), 9: 0j}

print(f"{'母線':>4}  {'計算値':>22}  {'期待値':>22}  {'差':>10}")
print("-" * 66)
for bus in sorted(injections):
    diff = abs(injections[bus] - expected[bus])
    print(f"{bus:>4}  {injections[bus].real:>+10.4f}{injections[bus].imag:>+10.4f}j  "
          f"{expected[bus].real:>+10.4f}{expected[bus].imag:>+10.4f}j  {diff:>10.2e}")

# %% [markdown]
# 差は $10^{-3}$ 程度で、ケースファイルに書いた電圧位相の桁数
# （小数点以下 4 桁）で決まる丸めの範囲です。データは整合しています。
#
# ## 2. 縮約アドミタンス行列
#
# 9 母線 + 3 内部母線 = 12 次元の問題が、3 次元まで縮みます。

# %%
for stage in (genstab.Stage.PRE, genstab.Stage.FAULT, genstab.Stage.POST):
    Y = system.reduced_matrix(stage)
    print(f"\n{stage.value} の縮約行列 (3x3):")
    print(np.array2string(Y, precision=4, suppress_small=True))

# %% [markdown]
# 事故中（母線 7 が地絡）の行列を見ると、成分が大きく変わっています。
# 特に母線 7 に直結している G2 は、ほとんど電力を送り出せなくなります。

# %%
from genstab.multimachine import electrical_power

for stage in (genstab.Stage.PRE, genstab.Stage.FAULT, genstab.Stage.POST):
    power = electrical_power(system.reduced_matrix(stage),
                             system.emf_magnitude, system._delta0)
    labels = [g.name for g in system.generators]
    print(f"{stage.value:6s}: " +
          "  ".join(f"{n} = {p:+.4f}" for n, p in zip(labels, power)))

# %% [markdown]
# ## 3. 標準ケースのシミュレーション
#
# 母線 7 で三相地絡が起き、5 サイクル（83 ms）後に線路 5-7 を
# 開放して除去します。

# %%
result = genstab.simulate(system, t_end=5.0, dt=0.002)
print(f"安定判定: {'安定' if result.is_stable() else '脱調'}")

fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
for generator in system.generators:
    axes[0].plot(result.t, np.degrees(result[f"delta_{generator.name}"]),
                 label=generator.name)
    axes[1].plot(result.t, np.degrees(result[f"delta_coi_{generator.name}"]),
                 label=generator.name)
    axes[2].plot(result.t, result[f"Pe_{generator.name}"], label=generator.name)

axes[0].set_ylabel("Rotor angle [deg]")
axes[0].set_title("WSCC 9-bus: fault at bus 7, cleared in 5 cycles")
axes[1].set_ylabel("Angle w.r.t. COI [deg]")
axes[2].set_ylabel("Electrical power [p.u.]")
axes[2].set_xlabel("Time [s]")
for ax in axes:
    ax.axvspan(1.0, system.fault.t_clear, color="tab:red", alpha=0.12)
    ax.legend(fontsize=9, ncol=3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 慣性中心 (COI) 基準で見る理由
#
# 上段（絶対角度）では 3 台がまとまって回転しているように見えます。
# しかし系全体が一緒に回っても同期は失われません。問題なのは
# **機どうしの角度差** です。
#
# そこで慣性で重み付けした平均
#
# $$ \delta_{COI} = \frac{\sum_i H_i \delta_i}{\sum_i H_i} $$
#
# を基準に取り直すと（中段）、機器間の相対運動だけが見えます。

# %%
print(f"機器間の角度差の最大 = {np.degrees(result['max_separation'].max()):.2f} deg")
for generator in system.generators:
    swing = np.ptp(np.degrees(result[f"delta_coi_{generator.name}"]))
    print(f"  {generator.name}: H = {generator.H:5.2f} s, P = {generator.P:.3f} p.u., "
          f"振れ幅 {swing:6.2f} deg")

# %% [markdown]
# **慣性の小さい G3 と、出力の大きい G2 が大きく振れています。**
# 慣性が最大の G1（H = 23.64 s）はほとんど動きません。
#
# ## 4. 臨界事故除去時間
#
# 多機系統では「機器間の角度差が開ききるか」で判定します。

# %%
cct = eac.critical_clearing_time(system, t_end=5.0, tolerance=1e-4, upper_bound=1.0)
print(f"CCT = {cct*1000:.1f} ms = {cct*60:.2f} サイクル (60 Hz)")
print(f"標準ケースの除去時間 = {system.fault.clearing_time*1000:.1f} ms "
      f"(余裕 {(cct - system.fault.clearing_time)*1000:.1f} ms)")

# %% [markdown]
# ## 5. 演習: 脱調の様子を見る
#
# 事故除去を CCT より遅らせると、どの発電機が最初に同期を失うでしょうか。

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, clearing in zip(axes, (cct - 0.01, cct + 0.02)):
    # TODO: 除去時間 clearing のケースをシミュレーションし、
    #       各機の COI 基準角度を描くこと。
    # BEGIN SOLUTION
    trial = replace(system, fault=genstab.FaultSchedule(1.0, 1.0 + clearing))
    r = genstab.simulate(trial, t_end=3.0, dt=0.002)
    for generator in system.generators:
        ax.plot(r.t, np.degrees(r[f"delta_coi_{generator.name}"]), label=generator.name)
    ax.set_title(f"clearing = {clearing*1000:.0f} ms "
                 f"({'安定' if r.is_stable() else '脱調'})")
    # END SOLUTION
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Angle w.r.t. COI [deg]")
    ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# 脱調するとき、**発電機は 2 つのグループに分かれて離れていきます**。
# ここでは事故点に近い G2, G3 が一方、遠い G1 が他方のグループです。
# この分かれ方を「脱調モード」と呼び、系統分割（島運転）の設計に使われます。
#
# ## 6. 演習: 機器間動揺モードの周波数
#
# 事故後の振動をスペクトル解析して、動揺モードの周波数を求めてください。

# %%
result_long = genstab.simulate(system, t_end=10.0, dt=0.002)
after = result_long.t > 1.5

plt.figure(figsize=(9, 4))
for generator in system.generators:
    # TODO: 各機の COI 基準角度から平均を引いて FFT し、
    #       振幅スペクトルを描くこと（横軸は Hz、0〜3 Hz の範囲）。
    # BEGIN SOLUTION
    signal = result_long[f"delta_coi_{generator.name}"][after]
    signal = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(signal.size, d=0.002)
    band = frequencies < 3.0
    plt.plot(frequencies[band], spectrum[band] / spectrum[band].max(),
             label=generator.name)
    peak = frequencies[band][np.argmax(spectrum[band][1:]) + 1]
    print(f"{generator.name}: 主要な振動周波数 = {peak:.3f} Hz")
    # END SOLUTION

plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalised amplitude")
plt.title("Spectrum of inter-machine oscillations")
plt.legend()
plt.show()

# %% [markdown]
# 3 機系統には原理的に **2 つの機器間動揺モード** があります
# （$n$ 機なら $n-1$ 個）。いずれも 0.7〜2 Hz の局所モード帯にあります。
#
# ## 7. 演習: 制動を加える
#
# 全機に制動係数 D を与えて、振動の減衰を確かめてください。

# %%
from dataclasses import replace as dc_replace

plt.figure(figsize=(9, 4))
for D in (0.0, 2.0, 5.0):
    # TODO: 全発電機の制動係数を D にした系を作り、
    #       G2 の COI 基準角度を描くこと。
    #       ヒント: dc_replace(generator, D=D) で発電機データを差し替える。
    # BEGIN SOLUTION
    damped = replace(
        system,
        generators=[dc_replace(g, D=D) for g in system.generators],
    )
    r = genstab.simulate(damped, t_end=8.0, dt=0.002)
    plt.plot(r.t, np.degrees(r["delta_coi_G2"]), label=f"D = {D}")
    # END SOLUTION

plt.xlabel("Time [s]")
plt.ylabel("G2 angle w.r.t. COI [deg]")
plt.title("Effect of damping on inter-machine oscillation")
plt.legend()
plt.show()

# %% [markdown]
# ## 7. 前半とつながる — 自分で解いた潮流解を渡す
#
# ここまでの潮流解はケースファイルに **数値として** 入っていたものですが、第 02 回で
# **自分の手で** 解いたものと同じです。`gridops.to_genstab` に自分の潮流解を渡すと
# 同じ系が立ち上がります（同梱ケースは D=2 なので `damping=0.0` で揃えます）。
# この連鎖を最後まで通すのが第 19 回の総合演習です。

# %%
import gridops

my_flow = gridops.solve_powerflow(gridops.load_case("wscc9"))   # 第 02 回と同じ計算
my_system = gridops.to_genstab(gridops.load_case("wscc9"), my_flow, damping=0.0)
for mine, given in zip(my_system.emf_magnitude, system.emf_magnitude):
    print(f"  内部起電力  自力 {mine:.4f} / 所与 {given:.4f}  差 {abs(mine - given):.1e}")

# %% [markdown]
# ## まとめ
#
# - 多機系統は Kron 縮約で発電機の数だけの問題に縮められる
# - **自分で解いた潮流解**からも同じ系が立ち上がる（前半と後半の接続点）
# - 安定性は絶対角度ではなく **機器間の角度差** で判定する
# - 慣性の小さい機、出力の大きい機ほど大きく振れる
# - 脱調はグループに分かれる形で起こる
#
# ## 安定度パート（第 10〜17 回）のまとめ
#
# 動揺方程式（10）→ 等面積法（11）→ CCT（12）→ 固有値（13）→ AVR の功罪（14）→
# PSS（15）→ 周波数制御（16）→ 多機系統（17）と進んできました。
#
# **過渡安定性と定態安定性は別の性質であり、両方を確認しないと
# 系統の安定性は語れません。** そして制御装置は、片方を良くしながら
# もう片方を悪くすることがあります（AVR がその代表例です）。
