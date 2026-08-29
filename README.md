# pwsyseng — 電力システム工学のシミュレーション教材

電力システム工学の講義・自習用 Python 教材です。**1 つの標準系統
（WSCC 3 機 9 母線）**を全 20 回で使い回し、

- **潮流計算**（いま系統に何がどう流れているか）
- **経済負荷配分**（同じ需要をいくらで作るか）
- **発電機起動停止計画**（どの発電機を動かしておくか）
- **アデカシーとセキュリティ**（設備は足りているか、1 回線失っても大丈夫か）
- **過渡安定度・定態安定度**（事故の瞬間に同期を保てるか、AVR・ガバナ・PSS）

を一貫した枠組みで扱います。5 つのテーマは「需要と供給を、どの時間スケールでも
釣り合わせ続けられるか」という 1 つの問いの、時間スケール違いです。

リポジトリには 2 つのパッケージが入っており、`pip install -e .` 1 回で両方が
入ります。前半（運用・計画）が `gridops`、後半（安定度）が `genstab` です。
**自分で解いた潮流解をそのまま過渡安定度計算に渡せる**ので、計画から事故時の
挙動までが 1 本の線でつながります。

```python
import gridops

case = gridops.load_case("wscc9")          # 今学期ずっと使う系統

# 1. 潮流計算 — いま何が流れているか
solution = gridops.solve_powerflow(case)
print(solution.summary())

# 2. 経済負荷配分 — 同じ需要をいくらで作るか
plan = gridops.economic_dispatch(case, demand_mw=315.0)
print(f"lambda = {plan.lam:,.0f} JPY/MWh")

# 3. 系統制約を入れると、ひとつの価格が母線ごとに割れる
opf = gridops.dc_opf(case)
print(opf.lmp)                              # ノード価格

# 4. 1 回線失っても大丈夫か
report = gridops.screen_n1(case, solution)
print(report.to_table())

# 5. 自分で解いた潮流解を過渡安定度に渡す（前半と後半の接続点）
system = gridops.to_genstab(case, solution)
```

安定度パートは `genstab` を直接使います。制御器（AVR・ガバナ・PSS・
状態フィードバック）はすべて **オプション**で、接続しなければ素の
動揺方程式に縮退します。

```python
import genstab

machine = genstab.ClassicalMachine(H=5.0, D=2.0, x_d_prime=0.3, E=1.1)
network = genstab.SMIBNetwork(x_pre=0.4, x_fault=float("inf"), x_post=0.6)
fault   = genstab.FaultSchedule(t_fault=1.0, t_clear=1.15)

system = genstab.SMIBSystem(machine, network, fault, Pe0=0.8)   # 制御なし
result = genstab.simulate(system, t_end=10.0)
print(result.is_stable())
```

## 環境構築

Anaconda（または Miniconda）が入っていることが前提です。Windows での
Anaconda の導入手順は、別途配布する導入ドキュメントを参照してください。

```bash
git clone https://github.com/ShintaroNegishi/pwsyseng.git
cd pwsyseng
conda env create -f environment.yml
conda activate pwsyseng
pip install -e .
```

**教員・学生ともにこの `environment.yml` 1 つで同じ環境を作ります。**
環境が分かれると「手元では動くのに」という問題の切り分けに時間を取られるので、
全員が `pwsyseng` という同じ名前の環境を使う運用にしています。

Jupyter を起動するときは、必ず `conda activate pwsyseng` してからにしてください。

```bash
conda activate pwsyseng
jupyter lab notebooks/00_setup.ipynb
```

環境ができたら `notebooks/00_setup.ipynb` を最後まで実行して確認します。
このノートブックは、正しい環境で動いているか、**混合整数計画のソルバ (CBC)
が使えるか**も表示します。

### 依存パッケージ

`numpy` / `scipy` / `matplotlib` / `python-control` / `slycot` / `pyyaml` /
`pulp` / `coin-or-cbc` と、notebook 用に `jupyterlab` / `ipywidgets` /
`pandas` / `sympy` を使います。

**商用ソルバは使いません。** 学生の手元でライセンスの問題を起こさないことを
最優先し、教材コードから `gurobipy` の import を排除しています。CBC は
conda-forge にビルド済みバイナリがあり、Windows でも conda 経由で入ります。
conda-forge の `pulp` は CBC を**同梱していない**ので、`environment.yml` は
`coin-or-cbc` を別行で並べています。詳細と動かないときの切り分けは
[docs/solver_notes.md](docs/solver_notes.md) を参照してください。

`slycot` も conda-forge のビルド済みバイナリを使います（`pip install slycot`
は Fortran のビルドが必要で失敗しやすいので使わないでください）。

## 教材の構成

「第 NN 回」の番号は notebook のファイル名と一致します。

| # | notebook | 主題 | 鍵となる概念 |
|---|---|---|---|
| 00 | `00_setup` | 環境の確認と科目の地図 | 動作確認、p.u. と MW、時間スケールの階層 |
| 01 | `01_ybus` | 系統をデータで表す | $Y_{ff}=(y_s+jb/2)/\tau^2$、$S_i = V_i(\sum_j Y_{ij}V_j)^*$ |
| 02 | `02_power_flow` | 潮流計算 | ミスマッチ、ヤコビアン、二次収束 |
| 03 | `03_voltage_stability` | 潮流が解けないとき | Q 制限、P-V 曲線、電圧崩壊 |
| 04 | `04_dc_and_sensitivity` | 直流潮流と感度係数 | PTDF、LODF、補償定理 |
| 05 | `05_economic_dispatch` | 経済負荷配分 | 等増分燃料費、KKT |
| 06 | `06_dc_opf_and_lmp` | 系統制約つき経済配分 | 直流最適潮流、ノード価格、混雑レント |
| 07 | `07_unit_commitment` | 発電機起動停止計画 | 0-1 変数、最低運転停止時間 |
| 08 | `08_reserve_and_vre` | 予備力と変動性電源 | 運転予備力、下げ代、出力抑制 |
| 09 | `09_security` | セキュリティ | N-1、スクリーニング、SCED |
| 10 | `10_swing_equation` | 動揺方程式 | 慣性、$P_e = P_{max}\sin\delta$ |
| 11 | `11_equal_area` | 等面積法 | 加速面積と減速面積 |
| 12 | `12_cct` | 臨界事故除去時間 | 保護の速度要求、慣性低下問題 |
| 13 | `13_small_signal` | 定態安定度 | 固有値、同期化力係数、減衰比 |
| 14 | `14_avr_and_damping` | AVR の功罪 | 高応答 AVR が制動を壊す |
| 15 | `15_pss_design` | PSS 設計 | 位相補償、GEP(s) |
| 16 | `16_lfc_governor` | 周波数制御 | ガバナ（比例）と LFC（積分） |
| 17 | `17_multimachine` | 多機系統 | Kron 縮約、機器間動揺 |
| 18 | `18_adequacy` | アデカシー | COPT、LOLP / LOLE / EUE、モンテカルロ |
| 19 | `19_integrated` | 総合演習 | 5 テーマを 1 日の運用として通す |

科目の山場は 2 つあります。**第 14 → 15 回**では、端子電圧を保つための AVR が
動揺の制動を悪化させ（固有値が右半面に移り）、それを PSS が位相補償で取り戻す、
という流れで「なぜ制御器が必要なのか」を一本の線でつなげます。
**第 09 → 17 回**では、N-1 で拘束する枝 5-7 の開放が第 17 回の標準事故と
**同じ枝**であることを使い、同じ 1 つの事象を静的には過負荷、動的には脱調として
評価します。

全 20 回の一覧と、**回どうしの受け渡し**（前の回の出力が次の回の入力になる
仕掛け）は [docs/course_map.md](docs/course_map.md) にまとめています。

- `notebooks/` : 解答入り（教員用）
- `exercises/` : 穴埋め版（学生配布用）

どちらも `notebooks/src/*.py` から自動生成しています。

## パッケージの構造

```
src/
├── gridops/              運用・計画パート（第 00〜09, 18, 19 回）
│   ├── case.py           科目全体で共有する系統データ（層に分かれたケース）
│   ├── loader.py         ケースファイルの読み書き
│   ├── casedata/         同梱ケース（cwd に依存せず名前で引ける）
│   ├── ybus.py           母線アドミタンス行列、連結性、橋の検出
│   ├── powerflow.py      交流潮流（Newton / Gauss-Seidel / 減結合 / 直流）
│   ├── voltage.py        P-V 曲線、電圧感度、最小特異値
│   ├── dc.py             直流潮流、PTDF、LODF
│   ├── solvers.py        PuLP の唯一の窓口。双対の符号をここで正規化する
│   ├── dispatch.py       等 λ 法、ペナルティファクタ、直流最適潮流、ノード価格
│   ├── commitment.py     起動停止計画（混合整数計画）、優先順位法
│   ├── adequacy.py       容量停止確率表、LOLP / LOLE / EUE、モンテカルロ、ELCC
│   ├── security.py       N-1 スクリーニング（熱容量と電圧）、SCED
│   ├── interop.py        genstab への受け渡し（一方向）
│   └── plotting.py       教材用の作図
└── genstab/              安定度パート（第 10〜17 回）
    ├── units.py          基準値（ω_s, 50/60 Hz）
    ├── events.py         事故スケジュール（発生・除去）
    ├── network.py        SMIB のネットワーク（事故前・中・後）
    ├── machines/         古典モデル（2 次）と 1 軸モデル（3 次）
    ├── controllers/      AVR・ガバナ・PSS・状態フィードバック（すべてオプション）
    ├── system.py         発電機 + 制御器 + ネットワークの合成
    ├── simulate.py       時間領域シミュレーション
    ├── smallsignal.py    固有値解析、参加係数
    ├── linearize.py      数値線形化 → python-control への変換
    ├── eac.py            等面積法、CCT
    ├── frequency.py      孤立系の周波数制御
    ├── multimachine.py   多機系統（Kron 縮約）
    └── plotting.py       教材用の作図
```

### 設計の要点

**依存は一方向**: `gridops` → `genstab` の橋渡しは `gridops.interop` だけが
行い、import は関数の中にあります。`genstab` は `gridops` を知りません。

**層に分かれたケースデータ**: `gridops` の `casedata/wscc9.yaml` は、潮流計算の
入力（母線種別・設定電圧・需要・線路）と潮流の**解**（`solution` 層）を
はっきり分けています。解は答え合わせと初期値にだけ使います。安定度パートが
直接読む `cases/wscc9.yaml`（潮流解を所与とする形式）も同梱しており、
第 17 回で「所与の解」と「自分で解いた解」が同じ系を立ち上げることを
確かめます。

**注入を作る場所を 1 つに閉じる**: 負荷を「負の発電」として足し込む符号ミスは
最も頻出するバグです。`Case.bus_injection()` だけが注入を組み立てます。

**単位を接尾辞で区別する**: ネットワーク量は p.u.、費用と容量は MW と円です。
識別子と YAML のキーの `_mw` 接尾辞が目印で、変換は `Case.to_mw` /
`Case.to_pu` だけを通します。

**双対の符号を 1 か所で決める**: `gridops.solvers` だけが `pulp` を import
します。需給バランスを `lpSum(p) == demand` の向きで書くと双対がそのまま
限界費用になり、逆向きだと符号が反転します。規約をコードの構造で強制しています。

**制御器のプラグイン構造**: 発電機モデルと各制御器はそれぞれ自分の状態微分を
返すだけで、状態ベクトルの連結は `SMIBSystem` が行います。接続時に定常状態で
初期化されるので、接続した瞬間に系が乱れることはありません。

## 検証

```bash
conda activate pwsyseng
pytest
```

数値が正しいことを、**独立に導いた基準**と突き合わせて確認しています。
主なものは次のとおりです（詳細は各テストファイルの docstring）。

| 検証内容 | 独立な基準 |
|---|---|
| 潮流解 | Anderson & Fouad の公表解（\|V\| の差 4.7e-5、位相の差 4.9e-5 deg）|
| ヤコビアン | 解析式 vs 中心差分 |
| 電圧安定限界 | 2 母線の解析ノーズ点 vs 継続法 |
| PTDF / LODF | 数値微分 / 枝を消した直流潮流の直接再計算 |
| 経済負荷配分 | KKT の閉形式 vs 二分法 vs `scipy.optimize` の三者一致 |
| ノード価格 | 双対の値ではなく**意味**（負荷を 1 MW 増やした総費用の増分）|
| 起動停止計画 | 小規模ケースの全列挙 vs 混合整数計画（総費用で比較）|
| 容量停止確率表 | 同一容量なら `scipy.stats.binom` と厳密一致 |
| モンテカルロ | 点推定ではなく、95% 信頼区間が解析解を含むこと |
| 動揺方程式 | 既存の検証済みコードの定式化との一致 |
| 等面積法 | 解析解 vs 二分探索による数値 CCT |
| 定態安定度 | 数値固有値 vs $\omega_n = \sqrt{K_s\omega_s/2H}$ の解析式 |
| PSS | 設計した位相進みが GEP(s) の位相遅れと一致 |
| 2 パッケージの接続 | 自力潮流解から求めた内部起電力と CCT が教科書解と一致 |

## 教員向け

notebook の中身を直すときは `notebooks/src/*.py` を編集してください
（`.ipynb` は JSON で差分が読めないため、原本は percent 形式の `.py` で
持っています）。編集後に次を実行すると、解答入りと穴埋め版の両方が
生成されます。

```bash
python tools/build_notebooks.py          # すべて
python tools/build_notebooks.py 05 14    # 番号を指定
```

穴埋めは `# BEGIN SOLUTION` から `# END SOLUTION` までを削ることで作ります。
`# TODO(L1)`（式 1 行）/ `(L2)`（関数・ループ）/ `(L3)`（設計判断）の 3 層に
分けている理由と注意点は [docs/course_map.md](docs/course_map.md) を参照
してください。

数値計算がおかしいときは、ソルバの設定を探る前にまずデータを確認します。

```bash
python tools/check_case.py
```

### 作図の方針

軸ラベル・凡例・タイトルはすべて英語で書いています。日本語フォントの有無は
OS によって異なり、学生の環境で豆腐（□）になる事故が起きやすいためです。
説明は notebook の markdown セルに日本語で書いてください。

## ライセンス

MIT License です。授業や自習に自由に使ってください。改変・再配布も歓迎します。
詳細は [LICENSE](LICENSE) を参照してください。

他の授業で転用する場合や、内容について気づいた点がある場合は、
Issue や Pull Request を歓迎します。

## 外部レビュー

安定度パートは公開直後に独立したバグレビューを受け、指摘された問題のうち
実害を再現できたものを修正しています。記録は
[docs/review_notes.md](docs/review_notes.md) を参照してください。

## モデルの仮定と限界

教材として意図的に単純化している箇所があります。詳細は
[docs/model_assumptions.md](docs/model_assumptions.md) にまとめています。
主なものは次のとおりです。

- 同梱ケースの熱容量・燃料費・信頼度データは原典になく、**教材用の自作**です
- 直流近似は電圧と無効電力の情報を失います（枝 4-5 で交流の $|S|$ が
  直流の $P$ より 47.6% 大きい）
- 熱容量だけを見る N-1 スクリーニングは、電圧で失格になる事故を見落とします
- 1 軸モデルは突極性を無視（$x_q = x'_d$）しています
- 多機系統の発電機はすべて古典モデルで、負荷は定インピーダンスに変換しています

## データの出典

同梱している系統データは公知の標準ベンチマークで、出典をファイル冒頭に
明記しています。原典にない値（熱容量・燃料費・信頼度データ・時系列需要）は
教材用の自作であることも併せて記しています。

> P. M. Anderson and A. A. Fouad, *Power System Control and Stability*,
> 2nd ed., IEEE Press, 2003, Chapter 2.

第三者が編纂した研究用データは同梱せず、取得手順のみ示しています。
方針は [docs/data_provenance.md](docs/data_provenance.md) を参照してください。

## 参考文献

[docs/references.md](docs/references.md) を参照してください。

## 経緯

本リポジトリは、単体で公開していた安定度教材
[genstab](https://github.com/ShintaroNegishi/genstab) に、運用・計画パート
（gridops）を加えて 1 つの科目として統合したものです。
