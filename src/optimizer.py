from data_loader import get_data
from indicators import calculate_indicators
from backtester import run_backtest
from strategies.rule_based import generate_signal
import pandas as pd

def optimize(stock, start, end, ema_short_range, ema_long_range, rsi_range, adx_range):
    results = []
    data = get_data(stock, start ,end)
    data = calculate_indicators(data)
    for ema_short in ema_short_range:
        for ema_long in ema_long_range:
            for rsi_thresh in rsi_range:
                for adx_thresh in adx_range:
                    signal= generate_signal(data, ema_short, ema_long, rsi_thresh, adx_thresh)
                    total_return, sharpe, max_dd = run_backtest(data, signal, transaction_cost=0.001)
                    results.append({'ema_short': ema_short,'ema_long': ema_long,'rsi_thresh': rsi_thresh,'adx_thresh': adx_thresh,'total_return': total_return,'sharpe': sharpe,'max_drawdown': max_dd})
                
    return pd.DataFrame(results)

def walk_forward(stock, train_start, train_end, test_start, test_end, ema_short_range, ema_long_range, rsi_range, adx_range):

    train_results= optimize(stock, train_start, train_end, ema_short_range, ema_long_range, rsi_range, adx_range)
    best= train_results.loc[train_results['sharpe'].idxmax()]

    test_data = get_data(stock, test_start, test_end)
    test_data = calculate_indicators(test_data)
    signal = generate_signal(test_data, int(best['ema_short']), int(best['ema_long']), int(best['rsi_thresh']), int(best['adx_thresh']))
    total_return, sharpe, max_dd = run_backtest(test_data, signal)
    return {
        'best_params': best,
        'test_return': total_return,
        'test_sharpe': sharpe,
        'test_max_dd': max_dd
    }
