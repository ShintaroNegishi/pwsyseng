# 用語・略語集

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
