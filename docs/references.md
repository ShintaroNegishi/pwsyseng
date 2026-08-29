# 参考文献

## 運用・計画パート（gridops）

### 教科書

- A. J. Wood, B. F. Wollenberg, and G. B. Sheblé, *Power Generation,
  Operation, and Control*, 3rd ed., Wiley, 2013.
  - 経済負荷配分（等増分燃料費の原理、ペナルティファクタ）、起動停止計画
    （優先順位法・動的計画法・ラグランジュ緩和）の標準的な教科書です。
    本教材の第 05〜08 回はこの本の構成に沿っています。
- J. D. Glover, M. S. Sarma, and T. J. Overbye, *Power System Analysis
  and Design*, 6th ed., Cengage, 2017.
  - 潮流計算の導入。母線分類とヤコビアンの組み立てが丁寧です。
- 電気学会編『電力系統工学』オーム社
- 新田目倖造『電力系統技術計算の基礎』電気書院
  - 潮流計算と系統計算の日本語での定番です。
- 関根泰次『電力系統解析理論』電気書院

### 潮流計算

- B. Stott and O. Alsaç, "Fast Decoupled Load Flow," *IEEE Trans. Power
  Apparatus and Systems*, vol. PAS-93, no. 3, pp. 859–869, 1974.
  - 減結合法の原典。高電圧系統で $\partial P/\partial |V|$ と
    $\partial Q/\partial\theta$ が小さいことが根拠であることを、
    本教材では `jacobian_blocks` で実際に測って確かめます。
- V. Ajjarapu and C. Christy, "The Continuation Power Flow: A Tool for
  Steady State Voltage Stability Analysis," *IEEE Trans. Power Systems*,
  vol. 7, no. 1, pp. 416–423, 1992.
  - 継続法。P-V 曲線のノーズ点を越えて追跡する方法です。

### 感度係数とセキュリティ

- A. J. Wood et al. 上掲書 第 11 章（Power System Security）
  - PTDF / LODF、性能指数によるコンティンジェンシーのランキング、
    そのマスキングの問題まで扱っています。
- T. Güler, G. Gross, and M. Liu, "Generalized Line Outage Distribution
  Factors," *IEEE Trans. Power Systems*, vol. 22, no. 2, pp. 879–881, 2007.
  - LODF を補償定理から導く形。分母がゼロになる場合（枝が橋である場合）の
    扱いも述べられています。

### 起動停止計画

- M. Carrión and J. M. Arroyo, "A Computationally Efficient
  Mixed-Integer Linear Formulation for the Thermal Unit Commitment
  Problem," *IEEE Trans. Power Systems*, vol. 21, no. 3, pp. 1371–1378, 2006.
- D. Rajan and S. Takriti, "Minimum Up/Down Polytopes of the Unit
  Commitment Problem with Start-Up Costs," IBM Research Report RC23628, 2005.
  - 最低運転時間・最低停止時間の tight な定式化（窓和の形）。
    本教材の `unit_commitment` はこの形を使っています。
- S. A. Kazarlis, A. G. Bakirtzis, and V. Petridis, "A Genetic Algorithm
  Solution to the Unit Commitment Problem," *IEEE Trans. Power Systems*,
  vol. 11, no. 1, pp. 83–92, 1996.
  - 10 機系統のベンチマーク。本教材には同梱していませんが、
    発展課題として自分で YAML に起こす題材になります。

### 価格とノード価格

- F. C. Schweppe, M. C. Caramanis, R. D. Tabors, and R. E. Bohn,
  *Spot Pricing of Electricity*, Kluwer, 1988.
  - ノード価格 (LMP) の原典。混雑が価格を母線ごとに割ることの理論的な基礎です。

### 信頼度（アデカシー）

- R. Billinton and R. N. Allan, *Reliability Evaluation of Power Systems*,
  2nd ed., Plenum Press, 1996.
  - 容量停止確率表 (COPT) の畳み込み、LOLP / LOLE / EUE の定義、
    モンテカルロ法。本教材の第 10 回はこの本に従っています。
- IEEE PES Task Force, "Capacity Value of Wind Power," *IEEE Trans.
  Power Systems*, vol. 26, no. 2, pp. 564–572, 2011.
  - 等価容量価値 (ELCC) の考え方。

### 制度

- 電力広域的運営推進機関（OCCTO）「供給計画の取りまとめ」
  - 日本の供給信頼度基準（EUE 基準）についての一次情報です。
- 資源エネルギー庁「容量市場について」

### 実データの入手先（同梱していません）

再配布しない方針のため、取得手順のみ記します。発展課題としてどうぞ。

- 東京電力パワーグリッド「エリア需給実績データ」
  <https://www.tepco.co.jp/forecast/html/area_data-j.html>
  - 1 時間ごとのエリア需要。CSV の文字コードは `cp932` です。
    年度によって行数が 8760 に満たないことがあるので、読み込み後に
    必ず長さを確認してください。研究室の `textbook/` に取得済みの
    ファイルがありますが、これも再配布しないでください。
- 日本卸電力取引所 (JEPX)「取引情報」
  <https://www.jepx.jp/electricpower/market-data/spot/>
- MATPOWER（IEEE 標準系統のケースファイル、BSD-3-Clause）
  <https://matpower.org/>
  - IEEE 14 / 30 / 118 母線を自分で YAML に起こす発展課題に使えます。
    ライセンス表記を忘れないこと。

### 研究室の関連コード

授業の範囲を超えた話題に興味がある学生向けです。

- `PWSYS/PWSYS/CPSDAM.py` — 日本 10 エリア規模の需給・予備力計画
- `banditUC/`, `inverseUC/` — 起動停止計画を研究として扱ったもの
- `OPF/OPF.py` — 直流最適潮流（Gurobi）
- `energy-mix/` — 電源構成の最適化（PuLP）

## 安定度パート（genstab）

### 教科書

系統安定性の標準的な教科書です。本パッケージの定式化はこれらに従っています。

- P. Kundur, *Power System Stability and Control*, McGraw-Hill, 1994.
  - 動揺方程式の標準形、Heffron-Phillips モデル、PSS の位相補償設計。
    本パッケージの単位系（速度偏差を p.u. で持つ形）はこの本に従っています。
- P. M. Anderson and A. A. Fouad, *Power System Control and Stability*,
  2nd ed., IEEE Press, 2003.
  - 等面積法、多機系統の Kron 縮約、WSCC 3 機 9 母線系統のデータ。
- J. Machowski, J. Bialek, and J. Bumby, *Power System Dynamics:
  Stability and Control*, 2nd ed., Wiley, 2008.
- 電気学会編『電力系統工学』オーム社
- 新田目倖造『電力系統技術計算の基礎』電気書院

### 制御工学（Python）

- 南裕樹『Pythonによる制御工学入門』オーム社
  - `python-control` の使い方。研究室の `制御系設計論/` にある notebook 群が
    この本に対応しており、学生が既に慣れている場合はそちらから入るとよいです。
- python-control ドキュメント: https://python-control.readthedocs.io/

### AVR と PSS

- F. P. deMello and C. Concordia, "Concepts of Synchronous Machine Stability
  as Affected by Excitation Control," *IEEE Trans. Power Apparatus and Systems*,
  vol. PAS-88, no. 4, pp. 316–329, 1969.
  - 高応答 AVR が動揺モードの制動を悪化させることを示した古典的論文。
    本教材の第 14 回の内容そのものです。
- E. V. Larsen and D. A. Swann, "Applying Power System Stabilizers,
  Part I–III," *IEEE Trans. Power Apparatus and Systems*, vol. PAS-100,
  no. 6, pp. 3017–3046, 1981.
  - PSS の位相補償設計の実務的な手引き。第 15 回で使う GEP(s) の考え方は
    この論文に基づいています。

### 標準モデル

- IEEE Std 421.5, *IEEE Recommended Practice for Excitation System Models
  for Power System Stability Studies*.
  - `SimpleExciter` は ST1A の、`IEEEType1Exciter` は Type-1 の簡略形です。

### 慣性低下問題

第 12 回と第 16 回で触れる、再生可能エネルギー大量導入に伴う系統慣性の低下について。

- P. Tielens and D. Van Hertem, "The relevance of inertia in power systems,"
  *Renewable and Sustainable Energy Reviews*, vol. 55, pp. 999–1009, 2016.
