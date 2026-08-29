# %% [markdown]
# # 15 電力系統安定化装置 (PSS) の設計
#
# ## この回のねらい
#
# - PSS の構成（washout + 進み遅れ補償）を理解する
# - **位相補償** の考え方で PSS を設計する
# - 固有値と時間応答の両方で効果を確認する
#
# ## PSS とは
#
# 前回、高応答 AVR が動揺モードの制動を壊すことを見ました。
# PSS は **速度偏差 $\Delta\omega$ を測って、AVR の基準に補助信号 $V_s$ を
# 重ねる** 装置です。狙いは、速度偏差と **同相** の電気トルクを作ること。
#
# 動揺方程式 $2H\,d\Delta\omega/dt = P_m - P_e - D\Delta\omega$ を見ると、
# $P_e$ が $\Delta\omega$ と同相であれば、それは $-D\Delta\omega$ と
# 同じ働き、すなわち **正の制動** になります。
#
# ## 構成
#
# ```
#  Δω ──▶ ゲイン K_s ──▶ washout ──▶ 進み遅れ 1 ──▶ 進み遅れ 2 ──▶ V_s
#                        T_w s          1+T_1 s        1+T_3 s
#                       ───────        ───────        ───────
#                       1+T_w s        1+T_2 s        1+T_4 s
# ```
#
# - **washout**: 定常的な速度偏差に反応して端子電圧を狂わせないためのハイパスフィルタ。
#   時定数 $T_w$ を 5〜10 s と大きく取れば、動揺周波数帯の位相はほぼ変えません。
# - **進み遅れ補償**: 励磁系と界磁回路が作る位相遅れを打ち消します。ここが設計の本体です。

# %%
import numpy as np
import matplotlib.pyplot as plt
import control as ct

import genstab
from genstab import linearize as lin
from genstab import smallsignal as ss
from genstab.controllers.pss import open_loop_gep
from genstab.plotting import use_genstab_style, plot_eigenvalues

use_genstab_style()

def build(controllers=None, D=0.0, Pe0=0.9):
    machine = genstab.OneAxisMachine(H=3.5, D=D, xd=1.81, xd_prime=0.30,
                                     Td0_prime=8.0, Vt0=1.0)
    network = genstab.SMIBNetwork(x_pre=0.65, x_fault=np.inf, x_post=0.65, V_inf=0.995)
    return genstab.SMIBSystem(machine, network, controllers=list(controllers or []), Pe0=Pe0)

def swing_mode(modes):
    candidates = [i for i in range(modes.eigenvalues.size)
                  if modes.eigenvalues[i].imag > 1e-6
                  and modes.participation[0, i] + modes.participation[1, i] > 0.4]
    return candidates[0]

avr_only = build([genstab.SimpleExciter(Ka=200.0, Ta=0.05)])
modes_avr = ss.analyze(avr_only)
i = swing_mode(modes_avr)
swing_frequency = float(modes_avr.eigenvalues[i].imag)
print(f"AVR のみ: 動揺モード λ = {modes_avr.eigenvalues[i]:.5f}, "
      f"減衰比 {modes_avr.damping_ratios[i]:+.5f}")
print(f"補償の中心にする角周波数 = {swing_frequency:.4f} rad/s "
      f"({swing_frequency/(2*np.pi):.4f} Hz)")

# %% [markdown]
# ## 1. 補償すべき位相遅れを測る — GEP(s)
#
# PSS の出力 $V_s$ から電気出力 $P_e$ までの伝達関数を **GEP(s)** と呼びます。
# これが持つ位相遅れを、PSS の進み遅れ補償で打ち消します。
#
# ### 重要な注意: 動揺ループを開いて評価する
#
# GEP を求めるとき、$\delta$ と $\Delta\omega$ を **状態から取り除いて**
# 評価します。残したままだと動揺モードの共振がそのまま伝達関数に現れ、
# ちょうど設計したい周波数で位相が急変してしまい、正しい補償量が読めません。
#
# 物理的には「回転子を固定した状態で、界磁と励磁系だけの応答を見る」
# ことに相当します。

# %%
gep_open = open_loop_gep(avr_only)            # 動揺ループを開いた GEP（正しい）
gep_closed = lin.state_space(avr_only, inputs=("Vref",), outputs=("Pe",))  # 開いていない（誤り）

omega = np.logspace(-1, 2, 600)
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for label, sys_ in (("open loop (correct)", gep_open), ("closed loop (misleading)", gep_closed)):
    values = np.array([complex(np.squeeze(sys_(1j * w))) for w in omega])
    axes[0].semilogx(omega, 20 * np.log10(np.abs(values)), label=label)
    axes[1].semilogx(omega, np.degrees(np.angle(values)), label=label)
axes[0].set_ylabel("Magnitude [dB]")
axes[1].set_ylabel("Phase [deg]")
axes[1].set_xlabel("Frequency [rad/s]")
for ax in axes:
    ax.axvline(swing_frequency, color="tab:red", ls=":", label="swing mode")
axes[0].legend(fontsize=8)
axes[0].set_title("GEP(s): why the swing loop must be opened")
plt.tight_layout()
plt.show()

phase_open = np.degrees(np.angle(complex(np.squeeze(gep_open(1j * swing_frequency)))))
phase_closed = np.degrees(np.angle(complex(np.squeeze(gep_closed(1j * swing_frequency)))))
print(f"動揺周波数における位相:")
print(f"  動揺ループを開いた場合 : {phase_open:+.2f} deg  <- これを補償する")
print(f"  開いていない場合       : {phase_closed:+.2f} deg  <- 共振に飲まれて使えない")

# %% [markdown]
# ## 2. 進み遅れ補償の設計
#
# 1 段の進み遅れ補償 $(1+T_1 s)/(1+T_2 s)$ は、
# $\omega_m = 1/\sqrt{T_1 T_2}$ で最大の位相進み $\phi_m$ を与えます。
#
# $$
# \alpha = \frac{T_2}{T_1} = \frac{1 - \sin\phi_m}{1 + \sin\phi_m}, \qquad
# T_2 = \frac{\sqrt{\alpha}}{\omega_m}, \qquad T_1 = \frac{T_2}{\alpha}
# $$
#
# 必要な位相進みを段数で等分して、各段をこの式で決めます。
# `design_pss` がこの手順を自動化しています。

# %%
pss = genstab.design_pss(avr_only, Ks=10.0, n_stages=2, Tw=10.0)
print(f"設計結果:")
print(f"  T1 = T3 = {pss.T1:.4f} s")
print(f"  T2 = T4 = {pss.T2:.4f} s")
print(f"  T1/T2 = {pss.T1/pss.T2:.3f}")
print(f"\n動揺周波数での位相進み = {np.degrees(pss.phase_lead(swing_frequency)):+.2f} deg")
print(f"補償すべき位相遅れ     = {-phase_open:+.2f} deg")
print("この 2 つが一致していれば設計成功です。")

# %% [markdown]
# ## 3. 効果を固有値で確かめる

# %%
records = []
for Ks in (0.0, 1.0, 3.0, 5.0, 10.0, 20.0):
    if Ks == 0.0:
        system = avr_only
    else:
        system = build([genstab.SimpleExciter(Ka=200.0, Ta=0.05),
                        genstab.design_pss(avr_only, Ks=Ks)])
    modes = ss.analyze(system)
    j = swing_mode(modes)
    records.append((Ks, modes.eigenvalues[j], modes.damping_ratios[j],
                    modes.damped_frequencies_hz[j], modes.is_stable))

print(f"{'Ks':>6}  {'固有値':>24}  {'減衰比':>9}  {'周波数[Hz]':>10}  判定")
print("-" * 72)
for Ks, eigenvalue, zeta, freq, stable in records:
    label = "(PSS なし)" if Ks == 0 else ""
    print(f"{Ks:>6.1f}  {eigenvalue.real:>+11.5f}{eigenvalue.imag:>+11.5f}j  "
          f"{zeta:>+9.5f}  {freq:>10.4f}  {'安定' if stable else '★不安定★'} {label}")

# %%
plot_eigenvalues(
    [np.array([r[1], np.conj(r[1])]) for r in records],
    [f"Ks = {r[0]:.0f}" for r in records],
    title="PSS gain moves the swing mode back to the left half plane",
)
plt.show()

# %% [markdown]
# **PSS は振動周波数をほとんど変えずに、制動だけを強くしています。**
# これは位相補償が正しく効いている証拠です。位相が合っていないと、
# 制動ではなく周波数（同期化力）のほうが変わってしまいます。
#
# ## 4. 演習: 位相補償を意図的に外す
#
# わざと間違った時定数を与えると何が起こるか確かめてください。

# %%
from genstab.controllers.pss import PowerSystemStabilizer

for label, (T1, T2) in (
    ("正しい設計",      (pss.T1, pss.T2)),
    ("補償なし (T1=T2)", (0.05, 0.05)),
    ("過剰補償",         (0.60, 0.02)),
    ("逆向き（遅れ）",   (0.02, 0.60)),
):
    # TODO: 与えられた T1, T2 の PSS を作り、動揺モードの減衰比を表示すること。
    #       Ks=10.0, Tw=10.0, T3=T1, T4=T2 とする。
    # BEGIN SOLUTION
    trial_pss = PowerSystemStabilizer(Ks=10.0, Tw=10.0, T1=T1, T2=T2, T3=T1, T4=T2)
    system = build([genstab.SimpleExciter(Ka=200.0, Ta=0.05), trial_pss])
    modes = ss.analyze(system)
    j = swing_mode(modes)
    print(f"{label:16s} 位相進み {np.degrees(trial_pss.phase_lead(swing_frequency)):+7.2f} deg  "
          f"減衰比 {modes.damping_ratios[j]:+.5f}  "
          f"{'安定' if modes.is_stable else '★不安定★'}")
    # END SOLUTION

# %% [markdown]
# 位相を合わせないと制動が改善しないどころか、悪化することもあります。
# **PSS はゲインを上げれば効く装置ではなく、位相を合わせて初めて効く装置です。**
#
# ## 5. 時間応答で確かめる

# %%
scenarios = [
    ("制御なし",            build()),
    ("AVR のみ",            avr_only),
    ("AVR + PSS (Ks=10)",   build([genstab.SimpleExciter(Ka=200.0, Ta=0.05),
                                   genstab.design_pss(avr_only, Ks=10.0)])),
]

fig, ax = plt.subplots(figsize=(9, 4.5))
for label, system in scenarios:
    disturbed = system.initial_state().copy()
    disturbed[0] += np.radians(1.0)
    result = genstab.simulate(system, t_end=15.0, dt=0.005, x0=disturbed)
    ax.plot(result.t,
            np.degrees(result.delta - system.operating_point.delta), label=label)
ax.set_xlabel("Time [s]")
ax.set_ylabel("$\\Delta\\delta$ [deg]")
ax.set_title("Response to a 1-degree disturbance")
ax.legend()
plt.show()

# %% [markdown]
# ## 6. 演習: PSS は過渡安定性も改善するか
#
# PSS は微小擾乱に対する制動を狙った装置ですが、大きな事故に対しても
# 効果があるでしょうか。CCT を比べてみましょう。
#
# **モデルの限界についての注意**
#
# 事故中リアクタンスに有限値を使っていますが、これは「事故中もある程度は
# 電力を送れる」状況を表すためであって、端子電圧を正しく再現するためでは
# ありません。このモデルは事故を転送リアクタンスの変化だけで表すので、
# **事故中の端子電圧は物理的に正しくありません**（実際の地絡では電圧が
# 下がりますが、このモデルではむしろ上がることがあります。詳細は
# `docs/model_assumptions.md`）。
#
# したがって以下の CCT の比較は、AVR と PSS が過渡安定性に与える効果の
# **定性的な傾向**を見るものと理解してください。事故中の端子電圧に AVR が
# どう応答するかを定量的に評価するには、事故点の分路を含むモデルが必要です。
#
# **この演習には仕掛けがあります。** 3 つの構成すべてで数字は出ますが、
# そのうち 1 つは **信用してはいけない値** です。どれか、そしてなぜかを
# 考えてください。

# %%
from genstab import eac

def build_faulted(controllers=None):
    machine = genstab.OneAxisMachine(H=3.5, D=0.0, xd=1.81, xd_prime=0.30,
                                     Td0_prime=8.0, Vt0=1.0)
    network = genstab.SMIBNetwork(x_pre=0.65, x_fault=2.5, x_post=0.70, V_inf=0.995)
    return genstab.SMIBSystem(
        machine, network, genstab.FaultSchedule(1.0, 1.1),
        controllers=list(controllers or []), Pe0=0.9,
    )

reference = build_faulted([genstab.SimpleExciter(Ka=200.0, Ta=0.05)])
configurations = [
    ("制御なし",          build_faulted()),
    ("AVR のみ",          build_faulted([genstab.SimpleExciter(Ka=200.0, Ta=0.05)])),
    ("AVR + PSS",         build_faulted([genstab.SimpleExciter(Ka=200.0, Ta=0.05),
                                         genstab.design_pss(reference, Ks=10.0)])),
]

import warnings

for label, system in configurations:
    # TODO: 各構成について、定態安定性（固有値）と CCT の両方を表示すること。
    #       CCT は eac.critical_clearing_time に t_end=10.0, tolerance=1e-4,
    #       upper_bound=1.0 を渡して求める。警告が出た場合はその内容も見ること。
    # BEGIN SOLUTION
    modes = ss.analyze(system)
    verdict = "定態安定" if modes.is_stable else "★定態不安定★"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cct = eac.critical_clearing_time(system, t_end=10.0, tolerance=1e-4, upper_bound=1.0)
    print(f"{label:14s} {verdict:14s} CCT = {cct*1000:6.1f} ms")
    for warning in caught:
        print(f"    警告: {warning.message}")
    # END SOLUTION

# %% [markdown]
# ### 数字が出ることと、正しいことは別
#
# 3 つとも数値は返ってきますが、**「AVR のみ」の値には意味がありません**。
# この運転点は定態不安定で、事故がなくても振動が成長するからです。
# 事故がなくても壊れる系に対して「どこまで事故に耐えられるか」を
# 問うこと自体が成立していません。
#
# `genstab` はこの状況を警告として知らせますが、警告を読まずに
# 数値だけを表に並べれば、そのまま誤った結論になります。
#
# > **教訓 1: 定態安定性が確保されていない運転点では、過渡安定性を
# > 論じる前提が崩れている。** 順序として、まず定態安定性を確保し、
# > そのうえで過渡安定性を評価する。
#
# > **教訓 2: 計算が通ったことは、結果が正しいことを意味しない。**
# > 数値が出たら、その前提が満たされているかを必ず確認する。
#
# 「制御なし」と「AVR + PSS」を比べると、PSS は制動を改善するだけでなく
# **CCT も延ばしている** ことが分かります。AVR が事故中に界磁電圧を上げて
# 内部起電力を支え、PSS が事故後の動揺を抑えるためです。

# %% [markdown]
# ## まとめ
#
# - AVR は電圧品質と送電能力のために必要だが、動揺の制動を悪化させうる
# - PSS は速度偏差から同相の電気トルクを作り、制動を取り戻す
# - 設計の要は **位相補償**。ゲインだけ上げても効かない
# - 設計後は必ず固有値を再計算し、他のモードが悪化していないか確認する
#
# 次回は視点を変えて、**系統周波数** を保つ制御（ガバナと LFC）を扱います。
