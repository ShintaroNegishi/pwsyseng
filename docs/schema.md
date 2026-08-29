# ケースファイルの仕様（schema 2）

## なぜ層に分けるのか

安定度パート（genstab）の `cases/wscc9.yaml` は、母線の `(v, angle_deg)` と
発電機の `(p, q)` を **入力として** 持っています。潮流計算を対象外にしていた
パッケージとしては正しい設計でした。

しかし潮流を学生自身に解かせるには、これでは「答えを読み込んで答えを出す」
教材になってしまいます。そこで gridops のケースは、**潮流計算の入力**と
**潮流の解**を層に分けています。

```
network       母線種別・設定電圧・需要・線路（熱容量つき）  ← 潮流計算の入力
units         号機の諸元（容量・費用・信頼度・慣性）
commitment    時系列需要と予備力率
reliability   年間需要の生成パラメータ
stability     事故スケジュール（genstab へ渡す）
contingencies N-1 の想定事故
solution      潮流解  ← 答え合わせと初期値にだけ使う。入力には使わない
```

テーマごとに別スキーマは作りません。1 つのファイルに全テーマ分の層を置き、
その回で使う層だけを読みます。学生が回ごとに系統を覚え直さずに済むことを
優先しました。

層が欠けているケースを使うと、`Case.require("units")` が
「その層はどの回で必要になるか」を添えた `ValueError` を投げます。

## 単位の規約

| 量 | 単位 | 目印 |
|---|---|---|
| ネットワーク量（電圧・潮流・インピーダンス・注入・無効電力制限） | p.u. | 接尾辞なし |
| 費用量（容量・出力・ランプ率・燃料費） | MW, 円 | `_mw` 接尾辞 |

変換は `Case.to_mw()` と `Case.to_pu()` の 2 つだけを通します。

## 符号の規約

母線への**注入**を正とします。発電が正、負荷が負です。
ただし `Bus.pd` / `Bus.qd` は**負荷を正の値で**書きます
（`pd: 1.25` は 125 MW の負荷）。符号の反転は `Case.bus_injection()` が
行い、注入を組み立てる場所はそこ 1 か所だけです。

## トップレベル

```yaml
schema: 2                # 必須。2 以外は読み込みを拒否する
name: ...                # ケースの名前
base_mva: 100.0          # p.u. の基準容量
frequency_hz: 60.0       # 定格周波数
source: ...              # 系統データの出典（書誌または URL と取得日）
modifications: ...       # 原典から改変した点、自作した値
```

## `buses`

```yaml
buses:
  - {id: 1, type: slack, v_set: 1.0400, name: G1 bus}
  - {id: 5, type: pq, pd: 1.25, qd: 0.50, name: Load A}
```

| キー | 既定 | 意味 |
|---|---|---|
| `id` | 必須 | 母線番号（ファイル内で一意）|
| `type` | `pq` | `slack` / `pv` / `pq`。**`slack` はちょうど 1 つ** |
| `v_set` | 1.0 | PV / slack 母線の設定電圧 [p.u.]。PQ では初期値 |
| `pd`, `qd` | 0.0 | 負荷（正の値で書く）[p.u.] |
| `gs`, `bs` | 0.0 | 母線シャント [p.u.]。調相設備を表すのに使う |
| `v_min`, `v_max` | `voltage_limits` | 運用上の電圧の上下限 [p.u.] |
| `name` | `""` | 表示名 |

`voltage_limits: {v_min: 0.95, v_max: 1.05}` をトップレベルに置くと、
個別指定のない母線に適用されます。

**未知数の数合わせ**が母線種別で決まります。

$$ \text{未知数} = 2 n_{PQ} + n_{PV} = \text{方程式の数} = (n_{PV}+n_{PQ}) + n_{PQ} $$

`tools/check_case.py` がこの対応を表示します。

## `branches`

```yaml
branches:
  - {from: 1, to: 4, x: 0.0576, rate_a: 2.00, rate_b: 2.40, name: T1}
  - {from: 4, to: 5, r: 0.0100, x: 0.0850, b: 0.176, rate_a: 1.00, rate_b: 1.50}
```

| キー | 既定 | 意味 |
|---|---|---|
| `from`, `to` | 必須 | 両端の母線番号 |
| `r`, `x` | 0.0 / 必須 | 直列インピーダンス [p.u.] |
| `b` | 0.0 | 全充電サセプタンス [p.u.]（両端に半分ずつ配分）|
| `tap` | 1.0 | タップ比 $\tau$。**from 側**に置く（MATPOWER 規約）|
| `shift_deg` | 0.0 | 位相調整角 $\phi$ [deg] |
| `rate_a` | $\infty$ | 常時許容容量 $\|S\|$ [p.u.] |
| `rate_b` | $\infty$ | 緊急時許容容量 $\|S\|$ [p.u.]（N-1 の判定に使う）|

タップと位相調整をまとめて $\bar a = \tau e^{j\phi}$ と置き、枝の 2×2
アドミタンス行列は

$$
Y_{ff} = \frac{y_s + jb/2}{\tau^2}, \quad
Y_{ft} = -\frac{y_s}{\bar a^{*}}, \quad
Y_{tf} = -\frac{y_s}{\bar a}, \quad
Y_{tt} = y_s + \frac{jb}{2}
$$

となります。`tap=1, shift_deg=0` で素の π 型に縮退します。

**注意 1**: `rate_a` / `rate_b` は**皮相電力 $|S|$** の制限です。
直流潮流の有効電力 $P$ と比べてはいけません。WSCC 9 母線の枝 4-5 では
交流の $|S|$ が直流の $P$ より 47.6% 大きい値になります。

**注意 2**: `shift_deg != 0` があると $Y_{ft} \ne Y_{tf}$ となり、
**Ybus は非対称になります**。対称性を仮定したコードを書かないでください。

## `units`

号機です。発電所は「同一母線に並ぶ号機の集合」として `plant` で表します。
テーマごとに使うフィールドが違い、使わないものは既定値のままで構いません。

| 群 | キー |
|---|---|
| 共通 | `name`（必須）, `bus`（必須）, `plant` |
| 出力 | `p_max_mw`, `p_min_mw`, `q_min`, `q_max` |
| 費用 | `var_cost` [円/MWh], `quadratic` [円/(MW²h)], `noload_cost` [円/h], `startup_cost` [円/回] |
| 時間結合 | `min_up`, `min_down` [h], `ramp_up`, `ramp_down` [MW/h], `startup_ramp`, `shutdown_ramp` [MW] |
| 初期状態 | `u0`（1: 運転, 0: 停止）, `hours_in_state` [h] |
| 信頼度 | `forced_outage_rate`, `mttf` [h], `mttr` [h] |
| 安定度 | `h` [s], `xd_prime` [p.u.], `d` [p.u.] |

**`u0` と `hours_in_state` は最低運転／停止時間の初期条件です。**
`u0=1, hours_in_state=1, min_up=8` の号機は、計画期間の最初の 7 時間は
停止できません。この拘束を課さないと、8 時間の最低運転時間を持つ石炭機を
1 時間目に止められてしまいます。

## `commitment`

```yaml
commitment:
  peak_mw: 315.0
  reserve_rate: 0.10
  allocation: {5: 0.396825, 6: 0.285714, 8: 0.317460}
  profiles:
    summer_weekday: [24 個の正規化係数]
  vre:
    name: PV
    capacity_mw: 120.0
    profile: [24 個の設備利用率]
```

`allocation` は時系列需要を母線に配分する比です。合計が 1 になるようにします。

## `reliability`

```yaml
reliability:
  annual: {seed: 0, seasonal_amplitude: 0.18, weekend_factor: 0.88, noise_sigma: 0.03}
  rounding_mw: 5.0
  target_lole_hours: 3.0
```

8760 時間の需要系列そのものは持ちません。**第三者の実需要データを
再配布しない方針**から、seed 固定の生成関数で合成します。

## `stability`

genstab に渡す事故スケジュールです。

```yaml
stability:
  fault:
    t_fault: 1.0
    t_clear: 1.0833
    buses: [7]
    tripped_branches: [{from: 5, to: 7}]
  protection_time: 0.080
```

## `contingencies`

N-1 の想定事故です。

```yaml
contingencies:
  branches: [{from: 4, to: 5}, {from: 4, to: 6}, ...]
```

**橋（開放すると系統が分離する枝）は含めません。** WSCC 9 母線では変圧器
3 本がこれに当たります。含めてしまうと、直流潮流の「系統全体で需給が一致する」
という前提が崩れます。`gridops.ybus.bridges()` が橋を検出し、
`screen_n1` は候補から除いた事実を `SecurityReport.skipped` に残します。

## `solution`

**入力ではありません。答え合わせと初期値にだけ使います。**

```yaml
solution:
  source: "..."
  digits: 4                 # 出典の記載桁数。テストの許容差の根拠になる
  buses:
    - {id: 1, v: 1.0400, angle_deg: 0.0000}
  generation:
    - {bus: 1, p: 0.716, q: 0.270}
  checks:                   # 独立に確かめられる量。テストの期待値
    losses_pu: 0.046410
    slack_p: 0.716410
    internal_emf: [1.0566, 1.0502, 1.0170]
```

`checks` にテストの期待値をケース側に持たせているので、ケースを差し替えても
テストコードを書き換えずに済みます。

## genstab の schema 1 との関係

`cases/wscc9.yaml`（版 1、安定度パートが直接読む形式）は、母線に潮流解 `(v, angle_deg)` を
**入力として**持ちます。`gridops.load_case` は版 1 のファイルを読むと
日本語の `ValueError` を投げます。逆方向（gridops のケース + 自力の潮流解
→ genstab の `MultiMachineSystem`）は `gridops.interop.to_genstab` が
行います。**依存の向きは gridops → genstab の一方向**で、
genstab のソースには手を入れません。

## 新しいケースを作るとき

1. `schema: 2` を書く
2. `source` に出典を、`modifications` に自作した値を書く
3. `python tools/check_case.py path/to/my_case.yaml` を通す
4. 熱容量を設計するなら `python tools/design_ratings.py` を参考にする
5. `docs/data_provenance.md` の表に 1 行足す
