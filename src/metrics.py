
def sharpe(returns):
    sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
    return sharpe

def max_drawdown(returns):
    rolling_max = returns.cummax()
    drawdown = (returns - rolling_max) / rolling_max
    max_dd = drawdown.min()
    return max_dd

def total_returns(returns):
    total_return = returns.iloc[-1]
    return total_return

def win_rate(returns):
    return (returns > 0).sum() / (returns != 0).sum()

def trade_count(signal):
    return ((signal.diff() !=0).sum())