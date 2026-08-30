from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"write: {path}")


def replace(
    path: str,
    old: str,
    new: str,
    *,
    count: int | None = None,
    required: bool = True,
) -> int:
    text = read(path)
    found = text.count(old)
    if required and found == 0:
        raise RuntimeError(f"replacement target not found in {path}: {old[:100]!r}")
    if count is not None and found != count:
        raise RuntimeError(
            f"unexpected replacement count in {path}: expected {count}, found {found}: {old[:100]!r}"
        )
    if found:
        text = text.replace(old, new, count if count is not None else -1)
        write(path, text)
    return found


def sub(
    path: str,
    pattern: str,
    repl: str,
    *,
    count: int = 0,
    required: bool = True,
    flags: int = re.MULTILINE | re.DOTALL,
) -> int:
    text = read(path)
    updated, found = re.subn(pattern, repl, text, count=count, flags=flags)
    if required and found == 0:
        raise RuntimeError(f"regex target not found in {path}: {pattern[:120]!r}")
    if found:
        write(path, updated)
    return found


# -----------------------------------------------------------------------------
# Permanent documentation added by this review
# -----------------------------------------------------------------------------
write(
    "docs/glossary.md",
    r'''# 用語・略語集

本教材の notebook は単独で参照されることがあるため、各 notebook でも主要な
略語を初出時に定義します。本ページは科目全体で共通する用語の一覧です。

| 略語・表記 | 正式名称 | 本教材での意味 |
|---|---|---|
| p.u. | per unit（単位法） | 基準容量・基準電圧などで規格化した量 |
| WSCC | Western Systems Coordinating Council | 3 機 9 母線標準系統の名称に残る旧組織名 |
| Ybus | bus admittance matrix（母線アドミタンス行列） | 母線電圧と注入電流を結ぶ行列 |
| PTDF | Power Transfer Distribution Factor（送電電力分布係数） | 母線間取引が枝潮流へ与える感度 |
| LODF | Line Outage Distribution Factor（線路開放分布係数） | 1 枝開放後の枝潮流変化を表す感度 |
| KKT | Karush–Kuhn–Tucker | 制約付き最適化の最適性条件 |
| DC-OPF | Direct Current Optimal Power Flow（直流最適潮流） | 直流潮流近似を用いた最適潮流計算 |
| LMP | Locational Marginal Price（地点別限界価格、ノード価格） | 各母線で需要を 1 MW 増やす限界費用 |
| UC | Unit Commitment（発電機起動停止計画） | 各時刻の発電機の運転・停止を決める問題 |
| VRE | Variable Renewable Energy（変動性再生可能エネルギー） | 太陽光・風力など出力が変動する電源 |
| SCED | Security-Constrained Economic Dispatch（セキュリティ制約付き経済配分） | 想定事故後の制約を考慮する経済配分 |
| SMIB | Single-Machine Infinite-Bus（1 機無限大母線系統） | 1 台の発電機を大規模系統へ接続した等価モデル |
| AVR | Automatic Voltage Regulator（自動電圧調整器） | 励磁を操作して端子電圧を調整する制御器 |
| PSS | Power System Stabilizer（電力系統安定化装置） | AVR に補助信号を加えて動揺を減衰させる制御器 |
| LFC | Load Frequency Control（負荷周波数制御） | 周波数偏差を除去する二次調整 |
| CCT | Critical Clearing Time（臨界事故除去時間） | 同期を維持できる事故継続時間の上限 |
| COPT | Capacity Outage Probability Table（容量停止確率表） | 停止容量とその確率の分布 |
| FOR | Forced Outage Rate（強制停止率） | 号機が強制停止状態にある確率 |
| LOLP | Loss of Load Probability（供給支障確率） | ある時点で供給力が需要を下回る確率 |
| LOLE | Loss of Load Expectation（供給支障時間・日数期待値） | 対象期間中に供給不足状態となる時間または日数の期待値 |
| EUE | Expected Unserved Energy（供給支障電力量期待値） | 供給できない電力量の期待値 |
| ELCC | Effective Load Carrying Capability（等価需要負担能力） | 信頼度を維持したまま追加できる需要 |
| CBC | COIN-OR Branch-and-Cut | 本教材で用いる無償の線形・混合整数計画ソルバ |

## PV という表記について

電力系統工学では **PV 母線**が「有効電力 P と電圧振幅 V を指定する母線」を
表します。一方、太陽光発電も photovoltaic の略として PV と呼ばれます。本教材では
混同を避けるため、後者を原則として「太陽光発電」または `solar PV` と表記します。
''',
)

write(
    "docs/environment_reproducibility.md",
    r'''# 計算環境の再現性

`environment.yml` は、授業で必要なパッケージの集合と主要な Python 版を定める
**互換環境仕様**です。インストール日によって依存パッケージの版が変わり得るため、
このファイルだけで将来にわたり完全に同一のバイナリ環境が再現されるわけではありません。

授業期間中は、次の運用を推奨します。

1. 授業開始前に Windows、macOS、Linux の GitHub Actions が通ることを確認する
2. 動作確認済みのコミットまたはリリースタグを学生へ指定する
3. 問題が生じた端末では次を保存し、教員へ共有する

```bash
conda activate pwsyseng
conda list
python -c "import sys; print(sys.version)"
python -c "import numpy, scipy, pulp, control; print(numpy.__version__, scipy.__version__, pulp.__version__, control.__version__)"
```

端末固有の完全なパッケージ一覧を保存する場合は、次を使用します。

```bash
conda list --explicit > pwsyseng-explicit-spec.txt
```

この explicit spec は OS と CPU アーキテクチャに依存します。Windows 用のファイルを
macOS や Linux へそのまま適用しないでください。科目の標準手順は引き続き
`environment.yml`を使用し、explicit spec は不具合調査と同一端末群での再現に用います。
''',
)

write(
    "docs/gridops_review_notes.md",
    r'''# 運用・計画パート（gridops）のレビュー記録

2026 年 8 月、公開後の教材を、数値計算の正しさだけでなく、学生が一般則として
受け取る説明の正確さ、公開リポジトリとしての客観性、Windows 配布時の再現性という
観点から再点検しました。

## 今回修正した主な事項

- notebook 番号、科目地図、相互参照の不整合
- 慣性定数の集約について、共通容量基準への換算を明示
- Loss of Load Expectation（LOLE）を事故回数ではなく不足状態の時間・日数期待値として説明
- 島系統を生じる枝事故について、LODF が適用できない理由と教材の対象外範囲を明確化
- 混合整数計画の整数解と影価格の関係を、連続緩和と区別して説明
- Locational Marginal Price（LMP）は PTDF 定式化でも再構成できることを明記
- Automatic Voltage Regulator（AVR）の実験説明を、実際の基準電圧ステップ応答に整合
- 簡易な負荷倍率追跡と、予測子・修正子を用いる本格的な Continuation Power Flow を区別
- 予備力モデルが同期並列中の上げ余力に基づく簡易モデルであることを明記
- 24 時間起動停止計画の終端効果を明記
- 略語の初出定義、内部環境にしか存在しない参照の除去
- notebook 生成物の同期、見出し番号、解答ブロックを CI で検査

## 今後の発展項目

次の項目は、今回の説明修正とは分けて段階的に実装する課題です。

- 運用・計画パートを別実装・標準ケースと照合する独立数値監査
- 予測子・修正子と弧長パラメータを用いる Continuation Power Flow
- 起動停止計画の look-ahead、終端状態、翌日持ち越しの明示的な定式化
- 応答時間・ランプ率・送電制約を含む予備力モデル
- 系統側電圧外乱を入力とした AVR あり・なしの直接比較
- OS 別の授業年度ロックファイルとリリース運用

既知のモデル上の単純化は `docs/model_assumptions.md`、安定度パートの独立レビューは
`docs/review_notes.md`を参照してください。
''',
)

# -----------------------------------------------------------------------------
# Course-map, README, and public-facing wording
# -----------------------------------------------------------------------------
replace("docs/course_map.md", "|---|---|---|---|---|", "|---|---|---|---|", count=1)
replace(
    "docs/course_map.md",
    "> 「第 NN 回」と書いたらこの番号を指します。notebook のファイル名の先頭の数字は\n"
    "> notebook のファイル名の数字は、この表の通し番号とすべて一致します。",
    "> 「第 NN 回」と書いたらこの番号を指します。notebook のファイル名の先頭の数字も、\n"
    "> この表の通し番号と一致します。",
    count=1,
)
replace(
    "docs/course_map.md",
    "19 本に対して授業は 15 回程度です。次の 6 本は",
    "全 20 本に対して授業は 15 回程度です。次の 4 項目は",
    count=1,
)
replace(
    "docs/course_map.md",
    "# 科目の地図 — 5 テーマをどうつなぐか\n\n",
    "# 科目の地図 — 5 テーマをどうつなぐか\n\n"
    "略語と正式名称は [用語・略語集](glossary.md) にまとめています。\n\n",
    count=1,
)

sub(
    "README.md",
    r"\*\*教員・学生ともにこの `environment\.yml` 1 つで同じ環境を作ります。\*\*\n"
    r"環境が分かれると「手元では動くのに」という問題の切り分けに時間を取られるので、\n"
    r"全員が `pwsyseng` という同じ名前の環境を使う運用にしています。",
    "**教員・学生ともに `environment.yml` から `pwsyseng` 環境を作ります。**\n"
    "このファイルは必要なパッケージ集合をそろえるための互換環境仕様であり、依存パッケージの\n"
    "厳密な版はインストール日により変わり得ます。授業年度の動作確認と環境情報の保存方法は\n"
    "[docs/environment_reproducibility.md](docs/environment_reproducibility.md) を参照してください。",
    count=1,
)
replace(
    "README.md",
    "仕掛け）は [docs/course_map.md](docs/course_map.md) にまとめています。\n\n",
    "仕掛け）は [docs/course_map.md](docs/course_map.md) にまとめています。\n"
    "略語の正式名称は [docs/glossary.md](docs/glossary.md)、環境の再現性は\n"
    "[docs/environment_reproducibility.md](docs/environment_reproducibility.md) を参照してください。\n\n",
    count=1,
)

replace(
    "environment.yml",
    "# ので、全員が pwsyseng という同じ名前の環境を使う運用にしている。\n",
    "# ので、全員が pwsyseng という同じ名前の環境を使う運用にしている。\n"
    "# ただし依存パッケージの厳密な版は固定していない。動作確認済みコミットと\n"
    "# 環境情報の保存方法は docs/environment_reproducibility.md を参照すること。\n",
    count=1,
)

# -----------------------------------------------------------------------------
# Notebook numbering and module cross-references
# -----------------------------------------------------------------------------
replace("notebooks/src/18_adequacy.py", "# # 10 アデカシー", "# # 18 アデカシー", count=1)
replace("notebooks/src/19_integrated.py", "# # 11 総合演習", "# # 19 総合演習", count=1)
replace(
    "src/gridops/voltage.py",
    "第 02 回で学生は「Newton は速い」を体験する。第 11 回でその同じ Newton が",
    "第 02 回で学生は「Newton は速い」を体験する。第 03 回でその同じ Newton が",
    count=1,
)
replace(
    "src/gridops/plotting.py",
    '"""教材用の作図ヘルパ（第 00 回〜第 11 回）。',
    '"""教材用の作図ヘルパ（運用・計画パート）。',
    count=1,
)

# -----------------------------------------------------------------------------
# Abbreviations and objective wording in notebook sources
# -----------------------------------------------------------------------------
replace(
    "notebooks/src/00_setup.py",
    "# - p.u. と MW の使い分けと、今学期ずっと使う系統に触れる",
    "# - 単位法（per unit: p.u.）とメガワット（MW）の使い分けを確認し、今学期ずっと使う系統に触れる",
    count=1,
)
replace(
    "notebooks/src/00_setup.py",
    "# 計算そのものは `PuLP`（定式化を書く道具）と `CBC`（実際に解くソルバ）が行います。",
    "# 計算そのものは `PuLP`（定式化を書く道具）と COIN-OR Branch-and-Cut（CBC、実際に解くソルバ）が行います。",
    count=1,
)
replace(
    "notebooks/src/00_setup.py",
    "# 今学期はこの WSCC 3 機 9 母線系統 1 つで押し通します。",
    "# 今学期は Western Systems Coordinating Council（WSCC）3 機 9 母線系統 1 つで押し通します。",
    count=1,
)
replace(
    "notebooks/src/00_setup.py",
    "# > **識別子の末尾が `_mw` なら MW、そうでなければ p.u.**\n#\n"
    "# `p_max_mw` は MW、`Bus.pd` は p.u.、`economic_dispatch` が返す `dispatch` は「号機名 → MW」。",
    "# > **有効電力・無効電力・設備容量について、末尾が `_mw` / `_mvar` なら実単位、**\n"
    "# > **ネットワークの `pd` / `qd` / 枝潮流は、特に断らない限り p.u. とします。**\n#\n"
    "# `p_max_mw` は MW、`Bus.pd` は p.u.、`economic_dispatch` が返す `dispatch` は「号機名 → MW」。",
    count=1,
)
replace(
    "notebooks/src/00_setup.py",
    "# ついでに、混合整数計画に **双対は存在しません**（`sol.duals` は空です）。第 07 回で",
    "# ついでに、混合整数計画の整数最適解には、線形計画と同じ意味で直接解釈できる影価格を一般には\n"
    "# 対応させられません（`sol.duals` は空です）。第 07 回で",
    count=1,
)

replace(
    "notebooks/src/01_ybus.py",
    "# # 01 系統をデータで表す — p.u. と Ybus",
    "# # 01 系統をデータで表す — 単位法（per unit: p.u.）と母線アドミタンス行列（bus admittance matrix: Ybus）",
    count=1,
)
replace(
    "notebooks/src/01_ybus.py",
    "# ## 扱う系統: WSCC 3 機 9 母線系統",
    "# ## 扱う系統: Western Systems Coordinating Council（WSCC）3 機 9 母線系統",
    count=1,
)
replace(
    "notebooks/src/01_ybus.py",
    "# 単位の規約そのもの（**識別子の末尾が `_mw` なら MW、そうでなければ p.u.**、換算は",
    "# 単位の規約そのもの（**電力・容量の末尾が `_mw` / `_mvar` なら実単位、ネットワーク量は原則 p.u.**、換算は",
    count=1,
)

replace(
    "notebooks/src/04_dc_and_sensitivity.py",
    "# # 04 直流潮流と感度係数 — PTDF と LODF",
    "# # 04 直流潮流と感度係数 — 送電電力分布係数（Power Transfer Distribution Factor: PTDF）と線路開放分布係数（Line Outage Distribution Factor: LODF）",
    count=1,
)
replace(
    "notebooks/src/05_economic_dispatch.py",
    "#   上下限があるとその等号が **不等式**に化けること（KKT）を確かめる",
    "#   上下限があるとその等号が **不等式**に化けること（Karush–Kuhn–Tucker: KKT 条件）を確かめる",
    count=1,
)
replace(
    "notebooks/src/06_dc_opf_and_lmp.py",
    "# # 06 系統制約つき経済配分 — 直流最適潮流とノード価格",
    "# # 06 系統制約つき経済配分 — 直流最適潮流（Direct Current Optimal Power Flow: DC-OPF）と地点別限界価格（Locational Marginal Price: LMP）",
    count=1,
)
replace(
    "notebooks/src/08_reserve_and_vre.py",
    "# # 08 予備力と変動性電源 — 下げ代と出力抑制",
    "# # 08 予備力と変動性再生可能エネルギー（Variable Renewable Energy: VRE）— 下げ代と出力抑制",
    count=1,
)
replace(
    "notebooks/src/09_security.py",
    "# # 09 セキュリティ — N-1 スクリーニングと SCED",
    "# # 09 セキュリティ — N-1 スクリーニングとセキュリティ制約付き経済配分（Security-Constrained Economic Dispatch: SCED）",
    count=1,
)
replace(
    "notebooks/src/09_security.py",
    "# - LODF で全事故を行列-ベクトル積 1 回ずつ掃き、交流の判定と突き合わせる",
    "# - 線路開放分布係数（Line Outage Distribution Factor: LODF）で全事故を行列-ベクトル積 1 回ずつ掃き、交流の判定と突き合わせる",
    count=1,
)
replace(
    "notebooks/src/10_swing_equation.py",
    "# 1 機無限大母線系統（SMIB）では、",
    "# 1 機無限大母線系統（Single-Machine Infinite-Bus: SMIB）では、",
    count=1,
)
replace(
    "notebooks/src/12_cct.py",
    "臨界事故除去時間 (CCT)",
    "臨界事故除去時間（Critical Clearing Time: CCT）",
    count=1,
    required=False,
)
replace(
    "notebooks/src/15_pss_design.py",
    "# # 15 電力系統安定化装置 (PSS) の設計",
    "# # 15 電力系統安定化装置（Power System Stabilizer: PSS）の設計",
    count=1,
)
replace(
    "notebooks/src/15_pss_design.py",
    "# PSS の出力 $V_s$ から電気出力 $P_e$ までの伝達関数を **GEP(s)** と呼びます。",
    "# 本教材では、PSS の出力 $V_s$ から電気出力 $P_e$ までの発電機・励磁系・電力系統の\n"
    "# 経路を表す伝達関数を、便宜上 **GEP(s)** と表記します。",
    count=1,
)
replace(
    "notebooks/src/18_adequacy.py",
    "# - 容量停止確率表 (COPT) を畳み込みで組み、**2 通りの独立な方法で検算する**",
    "# - 容量停止確率表（Capacity Outage Probability Table: COPT）を畳み込みで組み、**2 通りの独立な方法で検算する**",
    count=1,
)
replace(
    "notebooks/src/18_adequacy.py",
    "# - **LOLP / LOLE / EUE** が別のものを測っていることを同じ系統の数値で見る",
    "# - 供給支障確率（Loss of Load Probability: LOLP）、供給支障時間期待値（Loss of Load Expectation: LOLE）、\n"
    "#   供給支障電力量期待値（Expected Unserved Energy: EUE）が別のものを測ることを同じ系統の数値で見る",
    count=1,
)
replace(
    "notebooks/src/19_integrated.py",
    "#   各段で新たに判明することと、そのために **前の段へ戻る判断**を体験する",
    "#   各段で新たに判明することと、そのために **前の段へ戻る判断**を体験する\n"
    "# - 略語は、セキュリティ制約付き経済配分（SCED）、臨界事故除去時間（CCT）、\n"
    "#   供給支障時間期待値（LOLE）として初出時に定義する",
    count=1,
)

# -----------------------------------------------------------------------------
# Conceptual corrections: LOLE, MILP duals, LMP, AVR
# -----------------------------------------------------------------------------
replace(
    "src/gridops/adequacy.py",
    "LOLE       LOLP の期間合計 [h/期間, d/期間]       不足の **頻度**",
    "LOLE       LOLP の期間合計 [h/期間, d/期間]       不足状態となる時間・日数の **期待値**",
    count=1,
)
replace(
    "notebooks/src/18_adequacy.py",
    "# **LOLE の線はぴくりとも動かず 0.5 時間のままなのに、EUE は直線で増えます。**\n"
    "# 不足の確率は「停止しうる号機が止まる確率 0.05」だけで決まり $F$ によりませんが、\n"
    "# 不足の**深さ**は $110-F$ MW だからです。LOLE は「何回起きるか」しか数えません。",
    "# **LOLE の線はぴくりとも動かず 0.5 時間のままなのに、EUE は直線で増えます。**\n"
    "# 不足の確率は「停止しうる号機が止まる確率 0.05」だけで決まり $F$ によりませんが、\n"
    "# 不足の**深さ**は $110-F$ MW だからです。LOLE は不足状態にある時間の期待値を数えますが、\n"
    "# その時間に何 MW 足りないかは区別しません。",
    count=1,
)
replace(
    "notebooks/src/18_adequacy.py",
    "# - LOLP は確率、LOLE は不足の**頻度**、EUE は不足の**深さ**。LOLE が同じで EUE が",
    "# - LOLP は時点の確率、LOLE は不足状態となる時間・日数の期待値、EUE は不足の**深さ**。LOLE が同じで EUE が",
    count=1,
)

sub(
    "src/gridops/solvers.py",
    r"これは実装の手抜きではない。整数変数を含む問題の最適値は右辺について\n"
    r"区分的に一定な階段関数であり、微分（＝双対）がそもそも存在しない。\n"
    r"CBC は分枝限定の最後の緩和問題の双対を返してくるが、その値は探索の\n"
    r"経路に依存し、限界費用としての意味を持たない。",
    "これは実装の手抜きではない。混合整数計画の整数最適解には、線形計画と同じ意味で\n"
    "一意に解釈できる影価格を一般には定義できない。連続緩和には双対変数があるが、\n"
    "それを元の整数問題の限界費用としてそのまま解釈してはならない。CBC が探索中の\n"
    "緩和問題について返す双対も、整数最適解の影価格ではない。",
    count=1,
)
replace(
    "docs/solver_notes.md",
    "## 混合整数計画に双対はない\n\n"
    "`unit_commitment` は混合整数計画なので、**双対変数を返しません**。\n"
    "これは実装の都合ではなく、混合整数計画に真の双対が存在しないという\n"
    "理論的な事実です。時間別の限界費用が要るときは、得られた入切を固定して\n"
    "線形計画に落とし直す 2 段階が必要で、`gridops.commitment.marginal_prices`\n"
    "がそれを行います。",
    "## 混合整数計画の整数解に影価格を直接対応させない\n\n"
    "`unit_commitment` は混合整数計画なので、整数最適解に線形計画と同じ意味の\n"
    "双対変数を返しません。連続緩和には双対がありますが、それを元の整数問題の\n"
    "限界費用としてそのまま解釈することはできません。時間別の条件付き限界費用が\n"
    "必要なときは、得られた入切を固定して線形計画に落とし直します。\n"
    "`gridops.commitment.marginal_prices` がこの 2 段階を実行します。",
    count=1,
)
replace(
    "src/gridops/commitment.py",
    "混合整数計画の双対は取れない（:mod:`gridops.solvers` の Notes を参照）。",
    "混合整数計画の整数最適解には、線形計画と同じ意味の影価格を直接対応させない\n"
    "（:mod:`gridops.solvers` の Notes を参照）。",
    count=1,
)
replace(
    "notebooks/src/07_unit_commitment.py",
    "# - 混合整数計画に真の双対が無いことを理解し、限界費用を 2 段階で取る",
    "# - 混合整数計画の整数解に線形計画と同じ影価格を直接対応させず、限界費用を 2 段階で取る",
    count=1,
)
replace(
    "notebooks/src/06_dc_opf_and_lmp.py",
    "# 系統全体で 1 本にまとめると双対が 1 つしか出ず、**母線ごとの価格がそもそも定義できません**。\n"
    "# 母線の数だけ等式を書くから、母線の数だけ価格が出ます。",
    "# 母線別の需給等式で定式化すると、各等式の双対として LMP を直接読み取れます。\n"
    "# 一方、系統全体の需給バランスを 1 本だけ置く PTDF 定式化でも、系統限界費用と\n"
    "# 線路容量制約の双対から母線別 LMP を再構成できます。本教材は前者を採用します。",
    count=1,
)
replace(
    "notebooks/src/14_avr_and_damping.py",
    "# ## 4. AVR の「功」— 電圧を保つ\n#\n"
    "# 無限大母線の電圧が下がったとき、AVR があると端子電圧をどれだけ\n"
    "# 保てるかを見ます。ここでは基準電圧に微小なステップを与えて\n"
    "# 応答の速さを比べます。",
    "# ## 4. AVR の「功」— 基準電圧へ追従する\n#\n"
    "# ここでは AVR の基準電圧 $V_{ref}$ に微小なステップを与え、端子電圧が指令へ\n"
    "# 追従することを確認します。これは系統側の無限大母線電圧を変える外乱試験ではなく、\n"
    "# 基準値追従特性の確認です。系統側電圧外乱に対する比較は発展課題とします。",
    count=1,
)

# -----------------------------------------------------------------------------
# Inertia-base clarification
# -----------------------------------------------------------------------------
sub(
    "notebooks/src/08_reserve_and_vre.py",
    r"# 昼に火力を止めるということは、\*\*回っている鉄の塊を減らす\*\*ということです。\n"
    r"# 同期機の慣性定数は台数分だけ足し算になるので（`gridops\.interop\.aggregate_plants`\n"
    r"# が安定度へ渡すときに行う集約と同じ計算です）、起動している号機の \$H\$ を\n"
    r"# 足し上げれば、その時刻の系統慣性が出ます。\n#\n"
    r"# \$\$ H\^\{sys\}_t = \\sum_i H_i \\, u_\{it\} \\quad \[\\mathrm\{s\}\] \$\$",
    "# 昼に火力を止めるということは、**回っている回転体を減らす**ということです。\n"
    "# 一般に、各発電機の銘板容量基準の慣性定数をそのまま足すことはできず、共通の\n"
    "# 系統基準容量へ換算して $H_{eq}=\\sum_i H_i S_i/S_{base}$ とします。本教材の\n"
    "# `Unit.h` は、あらかじめ 100 MVA 共通基準へ換算した各号機の慣性寄与分です。\n"
    "# したがって、運転中の号機について次のように加算できます。\n#\n"
    "# $$ H^{sys}_t = \\sum_i \\bar H_i \\, u_{it} \\quad [\\mathrm{s\\ on\\ 100\\ MVA\\ base}] $$",
    count=1,
)
replace(
    "src/gridops/interop.py",
    "慣性は「足せる量」（回転体の運動エネルギーの和）、過渡リアクタンスは\n"
    "「並列につながる枝」なので逆数の和になる。",
    "一般には慣性定数を各機の容量基準のまま足してはならない。本ケースの\n"
    "``Unit.h`` は 100 MVA 共通基準へ換算済みの慣性寄与分として定義しているため、\n"
    "その寄与分を加算できる。過渡リアクタンスは並列につながる枝なので逆数の和になる。",
    count=1,
)
replace(
    "src/gridops/interop.py",
    "慣性定数 H は加算、過渡リアクタンス x'd は並列合成、P と Q は合計。",
    "共通系統基準へ換算済みの慣性寄与 H は加算、過渡リアクタンス x'd は並列合成、P と Q は合計。",
    count=1,
)
replace(
    "src/gridops/interop.py",
    "``H``         慣性定数 [s]（加算）",
    "``H``         100 MVA 共通基準上の慣性寄与 [s]（加算）",
    count=1,
)
for yaml_path in ("cases/wscc9.yaml", "src/gridops/casedata/wscc9.yaml"):
    replace(
        yaml_path,
        "# 慣性定数は加算、過渡リアクタンスは並列合成で原典の 3 機に厳密に戻る\n",
        "# 各号機の h は 100 MVA 共通基準へ換算済みの慣性寄与分として定義している。\n"
        "# この寄与分は加算でき、過渡リアクタンスは並列合成で原典の 3 機に戻る。\n",
        count=1,
    )
sub(
    "docs/model_assumptions.md",
    r"\| 発電機を 3 台から 7 号機に分解した \| 起動停止の意思決定を意味のあるものにし、アデカシーの状態数を確保するためです。慣性定数の合計と過渡リアクタンスの並列合成は原典の 3 機に厳密に戻ります",
    "| 発電機を 3 台から 7 号機に分解した | 起動停止の意思決定を意味のあるものにし、アデカシーの状態数を確保するためです。各号機の `h` は 100 MVA 共通基準へ換算済みの慣性寄与分であり、その加算と過渡リアクタンスの並列合成は原典の 3 機に厳密に戻ります",
    count=1,
)

# -----------------------------------------------------------------------------
# Islanding: precise wording and default reporting of omitted bridge outages
# -----------------------------------------------------------------------------
replace(
    "notebooks/src/09_security.py",
    "# 厳密にゼロ＝両端の間で送る電力の 100% がその枝を通る）見た同じ 3 本で、**事故後潮流という概念\n"
    "# そのものが成り立ちません**。**ただし「除外」は「健全」ではありません。** なお本教材の N-1 は",
    "# 厳密にゼロ＝両端の間で送る電力の 100% がその枝を通る）見た同じ 3 本です。橋の開放後は\n"
    "# 系統が島に分かれるため、連結系統を前提とする LODF は適用できません。島ごとに需給を再構成し、\n"
    "# 基準母線・周波数変動・負荷遮断を扱えば潮流自体は定義できますが、本教材の範囲外です。\n"
    "# **「除外」は「健全」ではありません。** なお本教材の N-1 は",
    count=1,
)
replace(
    "src/gridops/security.py",
    "事故後潮流という概念自体が成り立たないので\n候補から外すが、",
    "連結系統を前提とする LODF を適用できないので候補から外すが、",
    count=1,
)
replace(
    "src/gridops/security.py",
    "    bridge_set = set(bridges(case))\n\n    skipped: list[tuple[tuple[int, int], str]] = []",
    "    bridge_set = set(bridges(case))\n\n"
    "    # ケース側で橋を候補から除いていても、既定解析では『未評価』として記録する。\n"
    "    # 除外した事実を残さないと、『候補に無かった』と『評価して健全だった』を区別できない。\n"
    "    skipped: list[tuple[tuple[int, int], str]] = [\n"
    "        (\n"
    "            key,\n"
    "            \"橋（唯一の連絡路）なので開放後は系統が島に分かれる。\"\n"
    "            \"連結系統を前提とする LODF は適用できず、本教材は島ごとの\"\n"
    "            \"基準母線・需給再配分・周波数変動・負荷遮断を扱わないため未評価とする。\",\n"
    "        )\n"
    "        for key in sorted(bridge_set)\n"
    "        if contingencies is None and key not in candidates\n"
    "    ]",
    count=1,
)
sub(
    "src/gridops/security.py",
    r'"橋（この枝が唯一の連絡路）なので、開放すると系統が島に"\n\s*"分かれ、事故後潮流が定義できない。gridops\.ybus\.bridges\(\) "\n\s*"が独立に同じ枝を返す。除外は健全という意',
    '"橋（この枝が唯一の連絡路）なので、開放すると系統が島に"\n'
    '                    "分かれる。連結系統を前提とする LODF は適用できず、"\n'
    '                    "本教材は島ごとの需給再配分を扱わない。gridops.ybus.bridges() "\n'
    '                    "が独立に同じ枝を返す。除外は健全という意',
    count=1,
)

# -----------------------------------------------------------------------------
# Simplified voltage-margin tracking, reserve scope, and UC terminal effects
# -----------------------------------------------------------------------------
replace(
    "notebooks/src/03_voltage_stability.py",
    "# ## 2. 継続法は解析解に一致するか",
    "# ## 2. 負荷倍率追跡（簡易法）は解析解に一致するか",
    count=1,
)
replace(
    "notebooks/src/03_voltage_stability.py",
    "# 多母線では閉じた式が書けないので、負荷倍率を上げながら潮流を解き **解けなくなった点**を\n"
    "# 限界と呼ぶしかありません（継続法）。しかしこの手続きは「収束しない = 解が無い」と決めつけ、",
    "# 多母線では閉じた式が書けないため、本教材では前回の解を初期値にして負荷倍率を上げ、\n"
    "# 解ける倍率と解けない倍率を二分探索で挟みます。これは教育用の負荷倍率追跡であり、\n"
    "# 予測子・修正子と弧長パラメータでノーズ点を越える本格的な Continuation Power Flow ではありません。\n"
    "# また、この手続きだけでは「収束しない = 解が無い」と決めつける危険があり、",
    count=1,
)
replace(
    "src/gridops/voltage.py",
    "継続法（P-V 曲線の追跡）は「収束しなくなった点」を限界と呼ぶ手続きであり、",
    "本モジュールの負荷倍率追跡は「収束しなくなった点」を限界の下側推定とする簡易手続きであり、",
    count=1,
)
replace(
    "src/gridops/voltage.py",
    "そのままでは **手続きが自分の答えを定義してしまっている**。答え合わせの",
    "予測子・修正子と弧長パラメータを用いる本格的な Continuation Power Flow とは異なる。\nそのままでは **手続きが自分の答えを定義してしまっている**。答え合わせの",
    count=1,
)
replace(
    "docs/model_assumptions.md",
    "| 継続法はノーズ点を挟み撃ちするところまで |",
    "| 負荷倍率追跡は前回解によるウォームスタートと二分探索でノーズ点を下側から推定するところまで（本格的な Continuation Power Flow ではない） |",
    count=1,
)

replace(
    "notebooks/src/08_reserve_and_vre.py",
    "# 起動に何時間もかかる容量を予備力に数えない、というのがこの式の要点です。\n#\n",
    "# 起動に何時間もかかる容量を予備力に数えない、というのがこの式の要点です。\n#\n"
    "# ただし、これは **同期並列中の上げ余力に基づく簡易な予備力容量モデル**です。\n"
    "# 所定時間内の応答、ランプ率、最大単一事故、送電制約、一次・二次・待機予備力の区別は\n"
    "# この制約だけでは表していません。実制度上の調整力商品と同一視しないでください。\n#\n",
    count=1,
)
replace(
    "src/gridops/commitment.py",
    "という書き方は、需給が :math:`\\sum_i p_{it} = D_t` で閉じているときだけ",
    "という書き方は、需給が :math:`\\sum_i p_{it} = D_t` で閉じているときだけ",
    count=1,
)
replace(
    "src/gridops/commitment.py",
    "「いま出していない容量」に対する要求である。\n\n最低運転停止時間の初期条件",
    "「いま出していない容量」に対する要求である。\n\n"
    "ただし本モデルは同期並列中の上げ余力を容量として数える簡易表現であり、\n"
    "応答時間、ランプ率、最大単一事故、送電制約、一次・二次・待機予備力の\n"
    "区別を同時には表さない。制度上の調整力商品と同一視しないこと。\n\n"
    "最低運転停止時間の初期条件",
    count=1,
)
replace(
    "notebooks/src/07_unit_commitment.py",
    "# / `remaining_min_down()` が返します。\n\n# %%",
    "# / `remaining_min_down()` が返します。\n#\n"
    "# **計画期間の終端にも注意が必要です。** 本教材は 24 時間の外側を評価しないため、\n"
    "# 最終時刻付近の起動・停止が翌日の最低運転停止時間や起動費を負担しない終端効果があります。\n"
    "# 実務では翌日を含む look-ahead、終端状態、または 48 時間計算の中央 24 時間を用います。\n\n# %%",
    count=1,
)
replace(
    "src/gridops/commitment.py",
    "区分線形費用\n------------",
    "計画期間の終端\n----------------\n"
    "本実装は与えられた期間の外側を最適化しない。したがって最終時刻付近の\n"
    "起動・停止は、翌日の最低運転停止時間や起動費を負担しない **終端効果**を\n"
    "持つ。授業ではこの限界を明示し、実務的な評価では翌日を含む look-ahead、\n"
    "終端状態の指定、または長い期間を解いて中央部分だけを評価すること。\n\n"
    "区分線形費用\n------------",
    count=1,
)

# -----------------------------------------------------------------------------
# Public repository: remove internal-only references and improve Windows help
# -----------------------------------------------------------------------------
replace(
    "docs/references.md",
    "    年度によって行数が 8760 に満たないことがあるので、読み込み後に\n"
    "    必ず長さを確認してください。研究室の `textbook/` に取得済みの\n"
    "    ファイルがありますが、これも再配布しないでください。",
    "    年度によって行数が 8760 に満たないことがあるので、読み込み後に\n"
    "    必ず長さを確認してください。取得したファイルの利用条件を確認し、\n"
    "    本リポジトリへ再配布しないでください。",
    count=1,
)
sub(
    "docs/references.md",
    r"### 研究室の関連コード\n\n.*?\n## 安定度パート（genstab）",
    "### 発展課題への接続\n\n"
    "授業範囲を超える場合は、公開された標準ケースと一次資料を出発点にしてください。\n"
    "起動停止計画、最適潮流、電源構成最適化へ拡張するときも、データの出典・\n"
    "ライセンス・単位系・基準容量を明記し、本教材の結果と直接比較できる最小ケースを\n"
    "先に用意することを推奨します。\n\n"
    "## 安定度パート（genstab）",
    count=1,
)
replace(
    "docs/references.md",
    "  - `python-control` の使い方。研究室の `制御系設計論/` にある notebook 群が\n"
    "    この本に対応しており、学生が既に慣れている場合はそちらから入るとよいです。",
    "  - `python-control` の使い方。制御工学を未履修の場合は、伝達関数・状態空間表現・\n"
    "    周波数応答の章を先に学ぶと、本教材の第 14〜16 回へ入りやすくなります。",
    count=1,
)
replace(
    "docs/data_provenance.md",
    "| 研究室の `PWSYS` にある日本 10 エリアのユニットデータ | **同梱しない** | 実在の発電所名を含む第三者編纂データです |",
    "| 実在発電所名を含む第三者編纂のユニットデータ | **同梱しない** | 出典・利用条件・匿名化の確認が必要です |",
    count=1,
)
replace(
    "docs/data_provenance.md",
    "| IEEE 14 / 30 / 118 母線系統 | **同梱しない（発展課題）** | 研究室の `OPF/` にファイルがありますが出所が明確でありません。使う場合は MATPOWER（BSD-3-Clause）から起こし直し、ライセンスを明記する手順を発展課題として提示します |",
    "| IEEE 14 / 30 / 118 母線系統 | **同梱しない（発展課題）** | MATPOWER など出典とライセンスが明確な配布元から取得し、変換手順とライセンスを明記します |",
    count=1,
)
replace(
    "docs/solver_notes.md",
    "| **PuLP + CBC** | 採用。定式化が数式に近い形で書けて可読性が高く、conda-forge に Windows / macOS / Linux のビルド済みバイナリがあります。研究室の `energy-mix/` でも使用実績があります |",
    "| **PuLP + CBC** | 採用。定式化が数式に近い形で書けて可読性が高く、conda-forge に Windows / macOS / Linux のビルド済みバイナリがあります |",
    count=1,
)
replace(
    "src/gridops/solvers.py",
    "    \"  2. cbc が入っているか。`which cbc` で何も出なければ\\n\"",
    "    \"  2. cbc が入っているか。`python -c \\\"import shutil; print(shutil.which('cbc'))\\\"`\\n\"",
    count=1,
)
replace(
    "src/gridops/solvers.py",
    "研究室の ``genstab`` 環境は後者である。",
    "conda-forge から環境を作った場合は後者になる。",
    count=1,
)
replace(
    "docs/solver_notes.md",
    "conda activate pwsyseng\nconda install -c conda-forge coin-or-cbc\npython -c \"import pulp; print(pulp.listSolvers(onlyAvailable=True))\"",
    "conda activate pwsyseng\nconda install -c conda-forge coin-or-cbc\npython -c \"import shutil; print(shutil.which('cbc'))\"\npython -c \"import pulp; print(pulp.listSolvers(onlyAvailable=True))\"",
    count=1,
)
replace(
    "docs/review_notes.md",
    "この記録は安定度パート（`src/genstab`、旧・単体リポジトリ genstab）に\n"
    "関するものです。運用・計画パート（`src/gridops`）のレビュー記録は\n"
    "追記予定です。",
    "この記録は安定度パート（`src/genstab`、旧・単体リポジトリ genstab）に\n"
    "関するものです。運用・計画パート（`src/gridops`）のレビュー記録は\n"
    "[gridops_review_notes.md](gridops_review_notes.md) を参照してください。",
    count=1,
)

# -----------------------------------------------------------------------------
# Notebook build validation (complete replacement)
# -----------------------------------------------------------------------------
write(
    "tools/build_notebooks.py",
    r'''#!/usr/bin/env python
"""notebook のソース (.py) から .ipynb を生成する。

教材の原本は jupytext と同じ percent 形式の ``notebooks/src/*.py`` で持ち、
解答入り notebook と学生用の穴埋め版を機械的に生成する。生成時には、
見出し番号、解答ブロック、nbformat の妥当性も検査する。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "notebooks" / "src"
NOTEBOOK_DIR = ROOT / "notebooks"
EXERCISE_DIR = ROOT / "exercises"

SOLUTION_BEGIN = "# BEGIN SOLUTION"
SOLUTION_END = "# END SOLUTION"


def parse_percent_format(text: str) -> list[tuple[str, str]]:
    """percent 形式の文字列を ``(セル種別, 内容)`` のリストに分解する。"""
    cells: list[tuple[str, str]] = []
    kind = "code"
    buffer: list[str] = []

    def flush() -> None:
        if buffer and "".join(buffer).strip():
            cells.append((kind, "".join(buffer).rstrip("\n")))
        buffer.clear()

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in stripped else "code"
        elif kind == "markdown":
            if line.startswith("# "):
                buffer.append(line[2:])
            elif stripped == "#":
                buffer.append("\n")
            else:
                buffer.append(line)
        else:
            buffer.append(line)
    flush()
    return cells


def validate_solution_markers(source: str, path: Path) -> None:
    """解答ブロックの閉じ忘れ、余分な終了、入れ子を拒否する。"""
    inside = False
    begin_line = -1
    for line_number, line in enumerate(source.splitlines(), start=1):
        marker = line.strip()
        if marker == SOLUTION_BEGIN:
            if inside:
                raise ValueError(
                    f"{path}:{line_number}: {SOLUTION_BEGIN} が入れ子になっている"
                )
            inside = True
            begin_line = line_number
        elif marker == SOLUTION_END:
            if not inside:
                raise ValueError(
                    f"{path}:{line_number}: 対応する {SOLUTION_BEGIN} がない"
                )
            inside = False
    if inside:
        raise ValueError(
            f"{path}:{begin_line}: {SOLUTION_BEGIN} が {SOLUTION_END} で閉じていない"
        )


def validate_heading(source_path: Path, cells: list[tuple[str, str]]) -> None:
    """ファイル名の番号と最初の Markdown 見出し番号を一致させる。"""
    expected = source_path.stem.split("_", 1)[0]
    first_markdown = next((body for kind, body in cells if kind == "markdown"), "")
    match = re.search(r"^#\s+(\d{2})(?:\s|$)", first_markdown, flags=re.MULTILINE)
    if match is None:
        raise ValueError(f"{source_path}: 最初の Markdown セルに '# NN' 見出しがない")
    actual = match.group(1)
    if actual != expected:
        raise ValueError(
            f"{source_path}: ファイル番号 {expected} と見出し番号 {actual} が一致しない"
        )


def strip_solutions(source: str, path: Path) -> str:
    """解答部分を削除し、同じインデントの ``...`` に置き換える。"""
    validate_solution_markers(source, path)
    output: list[str] = []
    inside = False
    indent = ""

    for line in source.splitlines():
        marker = line.strip()
        if marker == SOLUTION_BEGIN:
            inside = True
            indent = line[: len(line) - len(line.lstrip())]
            continue
        if marker == SOLUTION_END:
            inside = False
            output.append(f"{indent}...  # ここを埋めること")
            continue
        if not inside:
            output.append(line)
    return "\n".join(output)


def build(source_path: Path, *, with_solutions: bool) -> nbformat.NotebookNode:
    """1 つのソースから notebook を組み立て、構造を検証する。"""
    text = source_path.read_text(encoding="utf-8")
    validate_solution_markers(text, source_path)
    cells = parse_percent_format(text)
    validate_heading(source_path, cells)

    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (pwsyseng)",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata["language_info"] = {"name": "python"}

    for kind, content in cells:
        if kind == "markdown":
            notebook.cells.append(nbformat.v4.new_markdown_cell(content))
        else:
            body = content if with_solutions else strip_solutions(content, source_path)
            if body.strip():
                notebook.cells.append(nbformat.v4.new_code_cell(body))

    nbformat.validate(notebook)
    return notebook


def main(argv: list[str]) -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    EXERCISE_DIR.mkdir(parents=True, exist_ok=True)

    sources = sorted(SOURCE_DIR.glob("*.py"))
    if argv:
        sources = [p for p in sources if any(p.name.startswith(a) for a in argv)]
    if not sources:
        print(f"変換対象が見つからない: {SOURCE_DIR}", file=sys.stderr)
        return 1

    for source in sources:
        name = source.stem + ".ipynb"
        solution = build(source, with_solutions=True)
        nbformat.write(solution, NOTEBOOK_DIR / name)

        exercise = build(source, with_solutions=False)
        nbformat.write(exercise, EXERCISE_DIR / name)

        has_blanks = SOLUTION_BEGIN in source.read_text(encoding="utf-8")
        note = "（穴埋めあり）" if has_blanks else "（穴埋めなし）"
        print(f"  {source.name} -> notebooks/{name}, exercises/{name} {note}")

    print(f"\n{len(sources)} 件の notebook を生成し、構造を検証した。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
''',
)

# -----------------------------------------------------------------------------
# CI (complete replacement)
# -----------------------------------------------------------------------------
write(
    ".github/workflows/tests.yml",
    r'''# 3 つの OS で環境とテストを確認し、主要 notebook は Windows でも実行する。
name: tests

env:
  PYTHONUTF8: "1"

on:
  push:
    branches: [main]
  pull_request:
  workflow_dispatch:

jobs:
  test:
    name: テスト (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    defaults:
      run:
        shell: bash -el {0}
    steps:
      - uses: actions/checkout@v4

      - name: conda 環境を作る
        uses: conda-incubator/setup-miniconda@v3
        with:
          environment-file: environment.yml
          activate-environment: pwsyseng
          miniforge-version: latest
          auto-activate-base: false

      - name: pwsyseng をインストール
        run: pip install -e .

      - name: 依存パッケージの確認
        run: |
          python -c "import sys; print('Python  ', sys.version.split()[0])"
          python -c "import numpy, scipy, control; print('control ', control.__version__)"
          python -c "import slycot; print('slycot  ', slycot.__version__)"
          python -c "import pulp; print('pulp    ', pulp.__version__)"
          python -c "import genstab; print('genstab ', genstab.__version__)"
          python -c "import gridops; print('gridops ', gridops.__version__)"

      - name: CBC が使えることを確認する
        run: |
          python - <<'PY'
          import shutil
          import pulp
          print("cbc executable:", shutil.which("cbc"))
          available = pulp.listSolvers(onlyAvailable=True)
          print("使えるソルバ:", available)
          if not ({"COIN_CMD", "PULP_CBC_CMD"} & set(available)):
              raise SystemExit(
                  "CBC が見つからない。environment.yml の coin-or-cbc を確認すること。"
              )
          PY

      - name: 同梱ケースの整合性を確認する
        run: python tools/check_case.py

      - name: テスト
        run: pytest -q

      - name: doctest
        run: |
          pytest --doctest-modules -q \
            src/genstab/__init__.py \
            src/genstab/linearize.py \
            src/genstab/smallsignal.py \
            src/genstab/machines/classical.py \
            src/gridops/__init__.py \
            src/gridops/case.py

  notebooks:
    name: notebook 実行 (${{ matrix.os }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, windows-latest]
    defaults:
      run:
        shell: bash -el {0}
    steps:
      - uses: actions/checkout@v4

      - name: conda 環境を作る
        uses: conda-incubator/setup-miniconda@v3
        with:
          environment-file: environment.yml
          activate-environment: pwsyseng
          miniforge-version: latest
          auto-activate-base: false

      - name: pwsyseng をインストール
        run: pip install -e .

      - name: notebook を再生成し、コミット済み生成物との一致を確認する
        run: |
          python tools/build_notebooks.py
          git diff --exit-code -- notebooks exercises

      - name: notebook を実行する
        env:
          MPLBACKEND: Agg
          GRIDOPS_FAST: "1"
        run: |
          outdir="${RUNNER_TEMP}/nbexec"
          mkdir -p "$outdir"
          if [[ "$RUNNER_OS" == "Windows" ]]; then
            notebooks=(
              notebooks/00_setup.ipynb
              notebooks/03_voltage_stability.ipynb
              notebooks/07_unit_commitment.ipynb
              notebooks/14_avr_and_damping.ipynb
              notebooks/15_pss_design.ipynb
              notebooks/18_adequacy.ipynb
              notebooks/19_integrated.ipynb
            )
          else
            notebooks=(notebooks/*.ipynb)
          fi
          executed=0
          broken=0
          for nb in "${notebooks[@]}"; do
            name=$(basename "$nb")
            log="${outdir}/${name}.log"
            executed=$((executed + 1))
            if jupyter nbconvert --to notebook --execute --output-dir="$outdir" "$nb" > "$log" 2>&1; then
              echo "OK      $name"
            else
              echo "FAILED  $name"
              tail -60 "$log"
              broken=$((broken + 1))
            fi
          done
          echo "実行した notebook: ${executed} 件 / 失敗: ${broken} 件"
          test "$executed" -gt 0
          test "$broken" -eq 0

  notebooks-full:
    name: notebook 完全実行（手動）
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    defaults:
      run:
        shell: bash -el {0}
    steps:
      - uses: actions/checkout@v4
      - name: conda 環境を作る
        uses: conda-incubator/setup-miniconda@v3
        with:
          environment-file: environment.yml
          activate-environment: pwsyseng
          miniforge-version: latest
          auto-activate-base: false
      - name: インストールと完全実行
        env:
          MPLBACKEND: Agg
        run: |
          pip install -e .
          python tools/build_notebooks.py
          git diff --exit-code -- notebooks exercises
          outdir="${RUNNER_TEMP}/nbexec-full"
          mkdir -p "$outdir"
          for nb in notebooks/*.ipynb; do
            jupyter nbconvert --to notebook --execute --output-dir="$outdir" "$nb"
          done
''',
)

# -----------------------------------------------------------------------------
# Consistency tests
# -----------------------------------------------------------------------------
write(
    "tests/test_course_consistency.py",
    r'''"""教材の番号・公開表現・既定の解析範囲の整合性を検査する。"""

from __future__ import annotations

import re
from pathlib import Path

from gridops import load_case
from gridops.security import screen_n1

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_source_heading_matches_filename() -> None:
    for source in sorted((ROOT / "notebooks" / "src").glob("*.py")):
        expected = source.stem.split("_", 1)[0]
        text = source.read_text(encoding="utf-8")
        match = re.search(r"^# # (\d{2})(?:\s|$)", text, flags=re.MULTILINE)
        assert match is not None, source
        assert match.group(1) == expected, source


def test_public_docs_do_not_reference_private_workspace_paths() -> None:
    paths = [
        ROOT / "docs" / "references.md",
        ROOT / "docs" / "data_provenance.md",
        ROOT / "docs" / "solver_notes.md",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    forbidden = (
        "研究室の `textbook/`",
        "`PWSYS/PWSYS/CPSDAM.py`",
        "`banditUC/`",
        "`inverseUC/`",
        "研究室の `OPF/`",
        "研究室の `energy-mix/`",
        "研究室の `制御系設計論/`",
    )
    for token in forbidden:
        assert token not in text


def test_default_n1_report_records_bridge_outages_as_unassessed() -> None:
    report = screen_n1(load_case("wscc9"), method="lodf", check_voltage=False)
    assert [key for key, _reason in report.skipped] == [(1, 4), (2, 7), (3, 9)]
    assert all("未評価" in reason or "適用でき" in reason for _key, reason in report.skipped)
''',
)

# -----------------------------------------------------------------------------
# Publishing checklist and review links
# -----------------------------------------------------------------------------
replace(
    "docs/publishing_checklist.md",
    "python tools/build_notebooks.py    # notebook を最新の src から再生成\n",
    "python tools/build_notebooks.py    # notebook を最新の src から再生成・構造検査\n"
    "git diff --exit-code -- notebooks exercises  # 生成物のコミット漏れがないこと\n",
    count=1,
)
replace(
    "docs/publishing_checklist.md",
    "CI は `src` から生成した notebook を実行するので、コミットし忘れても CI は\n"
    "通ってしまいます。学生が受け取るのはコミットされた `.ipynb` です。",
    "CI は再生成後に `git diff --exit-code` を実行し、コミット済みの `.ipynb` が\n"
    "原本と一致しない場合に失敗します。学生が受け取る生成物まで同期していることを確認できます。",
    count=1,
)

print("All course-material review fixes were applied.")
