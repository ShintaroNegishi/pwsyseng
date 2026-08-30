# %% [markdown]
# # 00 環境の確認とこの科目の地図
#
# 最初にこの notebook を **上から順に最後まで** 実行してください。ここが通れば、以降の回が動きます。
#
# **エラーが出たら、そのセルのエラーメッセージをそのまま担当教員に見せてください。**
# 環境構築の問題は初回に片づけるのが一番早く、第 07 回で「ソルバが無い」と
# 気づくのが一番遅いです。
#
# ## この回のねらい
#
# - 全員が同じ環境で動いていることを確かめる（Python・パッケージ・**最適化ソルバ**）
# - 手で解ける線形計画と混合整数計画を 1 つずつ解き、答えが手計算と合うことを見る
# - 5 つのテーマが **1 つの問いを違う時間スケールで見たもの**だと知る
# - 単位法（per unit: p.u.）とメガワット（MW）の使い分けを確認し、今学期ずっと使う系統に触れる
#
# ## 環境の作り方（まだの人向け）
#
# ターミナル（Windows なら Anaconda Prompt）で `pwsyseng` フォルダに移動して実行します。
#
# ```bash
# conda env create -f environment.yml
# conda activate pwsyseng
# pip install -e .
# ```

# %% [markdown]
# ## 1. 正しい環境で動いているか
#
# このリポジトリには 2 つのパッケージが入っています。前半（潮流・経済運用・
# 信頼度）が `gridops`、後半（安定度）が `genstab` で、`pip install -e .` 1 回で
# **両方**が入ります。第 17 回で「自分で解いた潮流解を過渡安定度計算に渡す」
# ところまで行くので、両方が同じ Python から import できる必要があります。
#
# `conda activate pwsyseng` を忘れて Jupyter を起動すると `base` 環境の Python が
# 使われ、パッケージが見つからない・版が違うといった問題が起きます。

# %%
import os
import sys
import platform

executable = sys.executable
environment = os.environ.get("CONDA_DEFAULT_ENV", "(不明)")

print(f"Python 実行ファイル : {executable}")
print(f"conda 環境名        : {environment}")
print(f"Python / OS         : {sys.version.split()[0]} / "
      f"{platform.system()} {platform.release()}")

if environment == "pwsyseng" or "pwsyseng" in executable:
    print("\n-> 正しい環境で動いています。")
else:
    print("\n-> 【注意】pwsyseng 環境ではない可能性があります。")
    print("   ターミナルで次を実行してから Jupyter を起動し直してください:")
    print("     conda activate pwsyseng")
    print("     jupyter lab")

# %% [markdown]
# ## 2. パッケージが読み込めるか
#
# `gridops`（前半）と `genstab`（後半）の **両方** を確認します。どちらかが
# 見つからない場合は `pip install -e .` をやり直してください（両方いっぺんに入ります）。

# %%
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import pulp

import gridops
from gridops.plotting import use_gridops_style

use_gridops_style()

print(f"numpy      : {np.__version__}")
print(f"scipy      : {scipy.__version__}")
print(f"matplotlib : {matplotlib.__version__}")
print(f"pulp       : {pulp.__version__}")
print(f"gridops    : {gridops.__version__}  ({gridops.__file__})")

try:
    import genstab

    print(f"genstab    : {genstab.__version__}  ({genstab.__file__})")
except ImportError:
    print("genstab    : 見つかりません。pip install -e . をやり直してください。")

# %% [markdown]
# ## 3. 最適化ソルバが使えるか — **ここで一番詰まります**
#
# 第 05 回以降の経済負荷配分・起動停止計画は、線形計画と **混合整数計画** を解きます。
# 計算そのものは `PuLP`（定式化を書く道具）と COIN-OR Branch-and-Cut（CBC、実際に解くソルバ）が行います。
# **`CBC` は `PuLP` に同梱されていないことがあり**、ここが最大の関門です。
#
# **動かなかったときの対処**: 下のセルが `RuntimeError: CBC ソルバが見つからない` で
# 止まったら、ターミナルで順に試してください。
#
# ```bash
# conda activate pwsyseng                         # 1. 環境の有効化を忘れていないか
# conda install -c conda-forge coin-or-cbc        # 2. CBC 本体を入れる
# python -c "import pulp; print(pulp.listSolvers(onlyAvailable=True))"
# ```
#
# `COIN_CMD` か `PULP_CBC_CMD` の **どちらか一方** が出れば動きます
# （`gridops.solvers` は両方に対応しています）。詳しくは `docs/solver_notes.md` へ。
# なぜ商用ソルバを使わないのか、という判断の理由もそこに書いてあります。

# %%
from gridops import solvers

# listSolvers は Gurobi も探して gurobi.log を残すので、選ばれた CBC だけ表示する。
print(f"gridops が選んだソルバ : {type(solvers.available_solver()).__name__}")
print("\n-> ソルバの準備ができました。")

# %% [markdown]
# ## 4. 手で解ける線形計画を 1 つ
#
# ソルバが「正しく」動くことは、動くことより大事です。手計算と突き合わせます。
#
# 100 MW の需要を 2 台で分担します。A は安いが小さく、B は高いが大きい。
#
# $$ \min_{p_A,\,p_B}\; 10\,p_A + 20\,p_B \quad\text{s.t.}\quad
#    p_A + p_B = 100,\quad 0 \le p_A \le 60,\quad 0 \le p_B \le 80 $$
#
# **手計算**: 安い A を上限まで使い、残りを B が埋めます。
# $p_A = 60$, $p_B = 40$, 費用 $= 600 + 800 = 1400$ 円/h。
#
# バランス制約は **右辺に需要を正の符号で置く向き**（`lp_sum(...) == 需要`）で
# 書いてください。この向きなら制約の双対がそのまま
# $\partial(\text{費用})/\partial(\text{需要})$、すなわち **限界費用** になります。
# 逆向きだと最適解は同じまま符号だけ反転します。第 06 回のノード価格まで効き続ける規約です。

# %%
cost = {"A": 10.0, "B": 20.0}      # 燃料費 [円/MWh]
p_max = {"A": 60.0, "B": 80.0}     # 上限 [MW]
demand_mw = 100.0

prob = solvers.problem("toy-lp")
p = {name: solvers.variable(f"p_{name}", 0.0, p_max[name]) for name in cost}
prob += solvers.lp_sum(cost[name] * p[name] for name in cost), "fuel_cost"

# TODO(L1): 需給バランス制約を 1 行で書くこと。右辺に demand_mw を正で置く向き。
# BEGIN SOLUTION
prob += solvers.lp_sum(p.values()) == demand_mw, "balance"
# END SOLUTION

lp = solvers.solve(prob, context="第 00 回の練習用の線形計画")
print(f"状態     : {lp.status}")
print(f"総費用   : {lp.objective:.1f} 円/h   (手計算 1400.0)")
for name in cost:
    print(f"  p_{name} : {lp.values[f'p_{name}']:6.1f} MW")
price = lp.duals.get("balance", float("nan"))   # 制約名で引く。混合整数計画では空になる
print(f"限界費用 : {price:.1f} 円/MWh   (手計算 20.0 = 高いほうの B が限界機)")

# %% [markdown]
# ## 5. 手で解ける混合整数計画を 1 つ
#
# 発電機には **最低出力** があります。「B を 20 MW で回す」ことはできず、
# 動かすなら 50 MW 以上、止めるなら 0 です。この **飛び** は不等式では書けません。
# 0-1 変数 $u_B \in \{0, 1\}$（起動していれば 1）を入れて、無負荷費 200 円/h も足します。
#
# $$ \min\; 10\,p_A + 20\,p_B + 200\,u_B \quad\text{s.t.}\quad
#    p_A + p_B = 100,\quad 0 \le p_A \le 60,\quad 50\,u_B \le p_B \le 80\,u_B $$
#
# **手計算**: $u_B = 0$ なら $p_B = 0$ で $p_A = 100 > 60$ となり不可能。
# よって $u_B = 1$。目的関数は $10(100-p_B) + 20 p_B + 200 = 1200 + 10\,p_B$ なので
# $p_B$ は小さいほどよく、最低出力に張り付いて $p_B = 50$。
# $p_A = 50$, $p_B = 50$, 費用 $= 500 + 1000 + 200 = 1700$ 円/h。
#
# 同じ問題を、$u_B$ の 0-1 条件だけ外して $0 \le u_B \le 1$ の **連続変数** にしたもの
# （線形緩和）も一緒に解きます。答えを見る前に予想してください。

# %%
def toy_commitment(u_kind: str) -> solvers.Solution:
    """B に最低出力と無負荷費を与えた問題を解く。u_kind で 0-1 か連続かを切り替える。"""
    lp_prob = solvers.problem(f"toy-uc-{u_kind}")
    q = {name: solvers.variable(f"p_{name}", 0.0, None) for name in cost}
    u_b = solvers.variable("u_B", 0.0, 1.0, cat=u_kind)
    lp_prob += solvers.lp_sum(cost[n] * q[n] for n in cost) + 200.0 * u_b, "cost"
    lp_prob += solvers.lp_sum(q.values()) == demand_mw, "balance"
    lp_prob += q["A"] <= p_max["A"], "cap_A"
    lp_prob += q["B"] <= p_max["B"] * u_b, "cap_B"
    lp_prob += q["B"] >= 50.0 * u_b, "min_B"
    return solvers.solve(lp_prob, context=f"第 00 回の練習用 ({u_kind})")


for kind, label in (("Binary", "0-1 変数 (混合整数計画)"), ("Continuous", "連続変数 (線形緩和)")):
    sol = toy_commitment(kind)
    print(f"費用 {sol.objective:7.1f} 円/h   p_A {sol.values['p_A']:5.1f} MW   "
          f"p_B {sol.values['p_B']:5.1f} MW   u_B {sol.values['u_B']:.2f}   "
          f"{sol.seconds*1000:3.0f} ms   <- {label}")

# %% [markdown]
# **緩和のほうが安い。しかも $u_B = 0.5$ です。**
#
# 「B を 0.5 台だけ起動する」運用は存在しません。にもかかわらず数式としては完全に正しく、
# 無負荷費も半額の 100 円しか払っていません。この 1500 円は **どうやっても実現できない
# 安さ** で、実際に払うのは 1700 円、差の 200 円が整数ギャップです。0-1 条件を外した瞬間に
# 答えが物理から離れる。これが起動停止計画（第 07 回）を線形計画では書けない理由です。
#
# ついでに、混合整数計画の整数最適解には、線形計画と同じ意味で直接解釈できる影価格を一般には
# 対応させられません（`sol.duals` は空です）。第 07 回で
# 時間別の限界費用が要るときは、入切を固定して線形計画に組み直します。
#
# ## 6. この科目の地図
#
# 潮流計算・経済負荷配分・起動停止計画・アデカシー・安定度。一見ばらばらですが、すべて
# **「需要と供給を、どの時間スケールでも釣り合わせ続けられるか」** という 1 つの問いです。
# 釣り合わせる手段と時定数だけが違います。

# %%
from gridops.plotting import timescale_map

timescale_map()
plt.show()

# %% [markdown]
# 図が出れば作図も動いています。**軸ラベル・凡例・タイトルはすべて英語です。**
# 日本語フォントは OS によって入っていたりいなかったりして、□（豆腐）になる事故が
# 起きやすいためです。説明はこのような markdown セルに日本語で書きます。
#
# 帯が隣と接しているのは、時間スケールが連続でテーマの境目に壁が無いからです。
# 各回の冒頭でこの図に戻り、「いまどこにいるか」を確認してください。
#
# ## 7. p.u. と MW — その量はどちらの単位か
#
# 電力系統では基準容量（本教材では 100 MVA）で規格化した **単位法 (per unit)** を使います。
# 変圧器の巻数比が消え、電圧が 1.0 前後の読みやすい数になるためです。一方、燃料費や需要は
# MW で書くほうが自然です。そこで本教材は次の規約を置きます。
#
# > **有効電力・無効電力・設備容量について、末尾が `_mw` / `_mvar` なら実単位、**
# > **ネットワークの `pd` / `qd` / 枝潮流は、特に断らない限り p.u. とします。**
#
# `p_max_mw` は MW、`Bus.pd` は p.u.、`economic_dispatch` が返す `dispatch` は「号機名 → MW」。
# 換算は `case.to_mw()` と `case.to_pu()` です。**単位の取り違えはこの科目で最も多い誤りです。**

# %%
case = gridops.load_case("wscc9")
print(case.describe())
print(f"\nデータの不整合: {case.check() or '検出なし'}")

total_pd_pu = sum(bus.pd for bus in case.buses)
total_pd_mw = float("nan")   # 下の TODO を埋めると上書きされます

# TODO(L1): p.u. の合計負荷 total_pd_pu を MW に直すこと（case のメソッドを使う）。
# BEGIN SOLUTION
total_pd_mw = case.to_mw(total_pd_pu)
# END SOLUTION

print(f"\n合計負荷 : {total_pd_pu:.3f} p.u. = {total_pd_mw:.1f} MW  "
      f"(基準 {case.base_mva:.0f} MVA)")
print(f"設備容量 : {sum(u.p_max_mw for u in case.units):.0f} MW")
print(f"原典     : {case.source}")

# %% [markdown]
# `case.check()` は母線番号の重複・slack の個数・ゼロインピーダンス枝・非連結な島を一括で調べます。
# **数値的に解けないとき、ソルバの設定より先にここを通してください。** 原因の大半はデータ側です。
#
# 今学期は Western Systems Coordinating Council（WSCC）3 機 9 母線系統 1 つで押し通します。系統を覚え直さずに済むことを優先したためです。
# ただし熱容量・燃料費・信頼度データ・号機への分解は原典になく、**教材用に著者が設定した自作の値**
# です。どこを変えたか、何を意図的に落としてあるかは `case.modifications` と
# `docs/model_assumptions.md` にあるので、各回で「なぜそう置いたのか」を説明できるようにしてください。
#
# なお設備容量 460 MW に対してピーク需要は 315 MW。予備率 46% は一見潤沢ですが、
# これで足りているかどうかは第 18 回で計算します。
#
# ## 8. 各回の並び
#
# | 回 | notebook | 主題 |
# |---|---|---|
# | 01〜04 | `01_ybus` … `04_dc_and_sensitivity` | 系統をデータで表し、潮流を解き、感度で近似する |
# | 05〜08 | `05_economic_dispatch` … `08_reserve_and_vre` | どの発電機をいくら動かすか（経済運用・起動停止） |
# | 09 | `09_security` | 1 回線失っても大丈夫か（N-1・SCED） |
# | 10〜17 | `10_swing_equation` … `17_multimachine` | 事故の瞬間に同期を保てるか（安定度） |
# | 18 | `18_adequacy` | そもそも設備は足りているか（LOLE・EUE） |
# | 19 | `19_integrated` | 5 テーマを 1 日の運用として通す総合演習 |
#
# **「第 NN 回」はこの表の左の通し番号で、notebook のファイル名の数字と一致します。**
#
# 全 20 回の一覧と、**回どうしの受け渡し**（前の回の出力が次の回の入力になる仕掛け）は
# `docs/course_map.md` にあります。たとえば第 09 回で N-1 が拘束する枝 5-7 は、第 17 回で過渡
# 安定度を見る **同じ枝** です。同じ 1 つの事象を、静的には過負荷、動的には脱調として評価します。
#
# ## まとめ
#
# - 環境は `pwsyseng` 1 つ。`gridops`（運用・計画）と `genstab`（安定度）の両方が import できること
# - **混合整数計画のソルバ (CBC) が動くこと**をいま確認しておく。詰まったら
#   `conda activate pwsyseng` → `conda install -c conda-forge coin-or-cbc` → `docs/solver_notes.md`
# - 0-1 条件を外すと「0.5 台起動する」1500 円の解が出る。実現できるのは 1700 円
# - `_mw` / `_mvar` の接尾辞は実単位の目印。接尾辞のない量（`dispatch` の値など）は
#   関数の説明で単位を確認する癖をつけること。5 テーマは 1 つの問いの時間スケール違い
#
# ## 次回へ
#
# 次回は、いま `load_case` で読み込んだ系統を **1 枚の行列** に落とします。線路と変圧器の
# データから母線アドミタンス行列 $Y$ を組み、参照解の電圧から注入
# $\bar S_i = \bar V_i (\sum_j Y_{ij} \bar V_j)^{*}$ を計算し直して、発電と負荷に一致するかを
# 確かめます。今日 `case.check()` でやった「解く前にデータを疑う」作法を、もう一段深いところで
# 繰り返すことになります。そしてその検算の式は、第 02 回で **ゼロにすべき方程式** に姿を変えます。
