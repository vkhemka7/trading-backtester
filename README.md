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

## Key Finding
The active strategies are conservative and do not beat buy and hold on raw 
returns over the long run, but consistently outperform on risk-adjusted metrics 
— Sharpe ratio and maximum drawdown. Tested across AAPL, TSLA, AMZN, MSFT, 
and NVDA from 2019-2025, the strategies limited maximum drawdown by 30-40% 
compared to buy and hold during bear years (2022 especially) at the cost of 
underperforming during parabolic bull runs. The platform correctly identifies 
that active strategies add value in high-volatility regimes and lose value in 
strong directional trends — a finding consistent across all five tickers tested.

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run dashboard.py`
4. Enter any valid Yahoo Finance ticker and date range

## Tech Stack
- Python, Streamlit, scikit-learn, yfinance, pandas
- C++ execution engine via pybind11 (in progress)