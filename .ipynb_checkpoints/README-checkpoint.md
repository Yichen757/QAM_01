# Momentum Portfolio Backtesting (1929–2024)
本專案旨在重建並驗證美股市場中「動能投資策略（Momentum）」的績效與穩健性。透過整合 CRSP 股票資料、Fama-French 因子與官方 decile 統計資料，進行完整的因子建構、分組、報酬計算與驗證分析.

## 專案結構
MomentumTrade/
├── data/
│   └── raw/              # 原始資料（pkl、csv、txt）
├── notebooks/            # 主 Jupyter Notebook
├── output/               # 分析結果與輸出檔案
├── src/                  # 各功能模組 (data_process, portfolio, evaluation)
├── README.md             # 專案說明文件
└── requirements.txt      # 套件安裝需求（如需）

## 專案目標與流程
1. 資料清理與整合
讀取 CRSP 股票資料，過濾出普通股、主要交易所股票。

處理遺漏值與非數字報酬欄位，計算市值與 Lag 市值（避免 Look-Ahead Bias）。

2. 動能排序與分組
計算每支股票的 12–2 個月累積報酬作為動能因子。

根據 NYSE-only（KRF）與全市場（DM）基準進行 decile 分組。

3. 報酬計算與 WML 構建
使用 Lag 市值進行每月加權報酬計算。

計算 WML（Winner Minus Loser）策略報酬。

4. 回測驗證與統計分析
與 Daniel & Moskowitz、Kenneth R. French 官方 decile 檔案比對。

檢查年化報酬、標準差、Sharpe Ratio、偏態等統計量。

5.  視覺化與討論
繪製 1929–2024 及 2010–2024 年間的 WML 累積對數報酬。

## 輸出範例
Step 1: Loaded and cleaned CRSP stock data
        Year  Month  PERMNO  ...  Ret  Ranking_Ret
0       1930      1  10006.0  ... 0.04    0.284
...

Step 5: Validated KRF portfolio performance against benchmark data
         Decile     Mean     Std    Sharpe     Skew   Corr_w_KRF
0       WML       0.012    0.059    0.203     -0.4     0.91
...

## 環境需求
* Python 3.8+

* pandas

* numpy

* matplotlib

## 資料來源
* CRSP(https://wrds-www.wharton.upenn.edu/pages/data/crsp/)

* Fama-French Data Library(https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

* Daniel & Moskowitz's Decile Data(http://web.mit.edu/adamk/www/data.htm)