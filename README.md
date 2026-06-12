# QuantBacktester

## Overview
An algorithmic stock market backtesting platform that compares four strategies: 
buy and hold, a rule-based strategy using EMA, RSI, and ADX indicators, a 
Logistic Regression model, and a Random Forest model. ML models train on 10 
years of historical data prior to the analysis window using time-series 
cross-validation to prevent lookahead bias. The platform models realistic 
trading conditions including transaction costs (0.1% per trade) and slippage. 
Walk-forward validation is used to optimize rule-based parameters on a training 
period before testing on unseen data. Results include Sharpe ratio, maximum 
drawdown, win rate, trade count, and equity curves across any ticker and date 
range.

## Key Findings

Tested across AAPL, TSLA, AMZN, MSFT, NVDA, JNJ, COIN, and PLTR from 2019-2025:

**Logistic Regression** performed best on stocks with strong, gradual momentum 
(MSFT, V) — beating buy and hold on both return and Sharpe when trained over 
a long historical window. On MSFT 2020-2024, LR returned 2.82x with a Sharpe 
of 1.32 vs buy and hold's 0.86.

**Random Forest** was tuned with `max_depth=5`, `min_samples_leaf=5`, and 
`predict_proba` confidence threshold of 0.55 to reduce overtrading. This 
reduced trade count from 400+ to under 150 on most stocks and improved Sharpe 
significantly. RF performed best on high-volatility stocks like TSLA and NVDA, 
often performing close to buy and hold. Without these constraints, RF bled 
returns through excessive transaction costs.

**Rule-Based EMA strategy** consistently limited max drawdown by 30-40% 
compared to buy and hold in bear years but underperformed in bull runs. 
Failed to capture parabolic moves (TSLA 2020, NVDA 2023).

**Transaction costs kill high-frequency ML strategies on volatile stocks.** 
RF on TSLA without probability thresholding made 400+ trades over 4 years, 
losing most of its edge to 0.1% per-trade costs.

**All strategies failed on extremely high-volatility, unpredictable stocks** 
like TSLA during parabolic years — the signal trained on historical data 
cannot anticipate regime-breaking moves.

**Engineering note:** Identified and fixed a critical data mutation bug where 
`run_backtest` modified the input dataframe in place, corrupting all subsequent 
strategy evaluations in the same run. Fixed by adding `.copy()` before 
processing.

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run dashboard.py`
4. Enter any valid Yahoo Finance ticker and date range

## Tech Stack
- Python, Streamlit, scikit-learn, yfinance, pandas
- C++ execution engine via pybind11 (in progress)