from strategies.rule_based import generate_signal
from strategies.ml_based import generate_ml_signals
from backtester import run_backtest
from indicators import calculate_indicators
from metrics import sharpe, max_drawdown, total_returns, win_rate,trade_count
from optimizer import walk_forward
from data_loader import get_data
import yfinance as yf
import pandas as pd

def compare_strategies(stock, start, end):
    warmup_start = str(int(start[:4]) - 20) + start[4:]
    full_data = get_data(stock, warmup_start, end)
    if full_data is None or len(full_data) < 252:
        return None
    full_data = calculate_indicators(full_data)
    
    ml_signal, rml_signal = generate_ml_signals(full_data)
    
    analysis_data = full_data[full_data.index >= start].copy()
    
    ml_signal = ml_signal[ml_signal.index >= start]
    rml_signal = rml_signal[rml_signal.index >= start]

    rule_signal = generate_signal(analysis_data, ema_short=20, ema_long=50, rsi_thresh=75, adx_thresh=25)
    
    rule_result = run_backtest(analysis_data, rule_signal)
    ml_result = run_backtest(analysis_data, ml_signal)
    rml_result = run_backtest(analysis_data, rml_signal)
    
    rule_return = total_returns(rule_result['Cumulative Strategy Returns'])
    rule_sharpe = sharpe(rule_result['Strategy Returns'])
    rule_dd = max_drawdown(rule_result['Cumulative Strategy Returns'])
    rule_wr= win_rate(rule_result['Strategy Returns'])
    rule_tc= trade_count(rule_signal)

    ml_return = total_returns(ml_result['Cumulative Strategy Returns'])
    ml_sharpe = sharpe(ml_result['Strategy Returns'])
    ml_dd = max_drawdown(ml_result['Cumulative Strategy Returns'])
    ml_wr= win_rate(ml_result['Strategy Returns'])
    ml_tc= trade_count(ml_signal)

    rml_return = total_returns(rml_result['Cumulative Strategy Returns'])
    rml_sharpe = sharpe(rml_result['Strategy Returns'])
    rml_dd = max_drawdown(rml_result['Cumulative Strategy Returns'])
    rml_wr= win_rate(rml_result['Strategy Returns'])
    rml_tc= trade_count(rml_signal)



    bah_return = analysis_data['Cumulative Returns'].iloc[-1] / analysis_data['Cumulative Returns'].iloc[0]
    bah_sharpe = sharpe(analysis_data['Returns'])
    bah_dd = max_drawdown(analysis_data['Cumulative Returns'])
    bah_wr= win_rate(analysis_data['Returns'])

    results = {
        'Strategy': ['Buy & Hold', 'Rule-Based', 'Logistic Reg', 'Random Forest'],
        'Return': [bah_return, rule_return, ml_return, rml_return],
        'Sharpe': [bah_sharpe, rule_sharpe, ml_sharpe, rml_sharpe],
        'Max Drawdown': [bah_dd, rule_dd, ml_dd, rml_dd],'Win Rate': [bah_wr, rule_wr, ml_wr, rml_wr],
        'Trade Count':[None, rule_tc, ml_tc, rml_tc]
    }
    
    equity_curves = pd.DataFrame({
        'Buy & Hold': analysis_data['Cumulative Returns'] / analysis_data['Cumulative Returns'].iloc[0],
        'Rule-Based': rule_result['Cumulative Strategy Returns'],
        'Logistic Reg': ml_result['Cumulative Strategy Returns'],
        'Random Forest': rml_result['Cumulative Strategy Returns']
    })
    
    return pd.DataFrame(results), equity_curves, analysis_data['Close']
    
    
if __name__ == "__main__":
    compare_strategies("AMZN", "2020-01-01", "2025-01-01")    
    


    



