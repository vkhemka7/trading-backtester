# QuantBacktester

## Overview
An algorithmic stock market backtesting platform that compares four strategies: 
buy and hold, a rule-based strategy using EMA, RSI, and ADX indicators, a 
Logistic Regression model, and a Random Forest model. ML models train on 10 
years of historical data prior to the analysis window using time-series 
cross-validation to prevent lookahead bias. The platform models realistic 
trading conditions including transaction costs (0.1% per trade). Walk-forward 
validation is used to optimize rule-based parameters on a training period before 
testing on unseen data. Results include Sharpe ratio, maximum drawdown, win rate, 
trade count, and equity curves across any ticker and date range.

## Key Findings

Tested across AAPL, TSLA, AMZN, MSFT, NVDA, JNJ, LMT, KO, COIN, and PLTR from 2019-2025.

**Logistic Regression** performed best on stocks with strong, gradual, sustained 
momentum — MSFT, V. Its linear decision boundary suits consistent directional 
trends. On MSFT 2020-2024, LR returned 2.82x with a Sharpe of 1.32 vs buy and 
hold's 0.86. LR consistently failed to catch price spikes and performed poorly 
on erratic, sentiment-driven stocks like TSLA where its linear boundary cannot 
adapt to non-directional price action.

**Random Forest** was tuned with `max_depth=5`, `min_samples_leaf=5`, and 
`predict_proba` threshold of 0.50. Without these constraints RF made 400+ trades 
on volatile stocks and bled returns entirely through transaction costs. RF was the 
only model that partially detected price spikes but with low confidence scores — 
insufficient to act on at the 0.50 threshold. RF outperformed on NVDA — 6.47x 
return vs 8.29x buy and hold while cutting max drawdown from -66% to -39%.

**Rule-Based EMA strategy** consistently limited max drawdown by 30-40% in bear 
years but underperformed in bull runs. The lagging nature of EMA crossover means 
entries happen after most of the move has already occurred.

**Transaction costs** are the single biggest killer of ML strategy returns on 
volatile stocks. RF on TSLA without probability thresholding made 400+ trades 
over 4 years, losing most of its edge entirely to 0.1% per-trade costs.

**All strategies failed on erratic high-volatility stocks** like TSLA during 
parabolic years. No model trained on historical patterns can anticipate 
regime-breaking, sentiment-driven price action.

**Core finding:** No single strategy dominates across all stock types. The 
platform correctly surfaces this — the value is not in the strategies themselves 
but in the honest evaluation framework that reveals when and why each approach 
breaks down.

**Engineering note:** Identified and fixed a critical data mutation bug where 
`run_backtest` modified the input dataframe in place, corrupting all subsequent 
strategy evaluations in the same run. Fixed by adding `.copy()` before processing.

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run dashboard.py`
4. Enter any valid Yahoo Finance ticker and date range

## Tech Stack
- Python, Streamlit, scikit-learn, yfinance, pandas
- C++ execution engine via pybind11 (in progress)