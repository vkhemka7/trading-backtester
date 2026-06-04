import numpy as np

def generate_signal(data, ema_short, ema_long, rsi_thresh, adx_thresh):
    #data['Signal'] = np.where(((data[f'SMA_{sma_short}'] > data[f'SMA_{sma_long}']) & (data['RSI'] < rsi_thresh) & (data['ADX'] > adx_thresh)), 1, 0)
    data['Signal'] = np.where(((data[f'EMA_{ema_short}'] > data[f'EMA_{ema_long}']) & (data['RSI'] < rsi_thresh) & (data['ADX'] > adx_thresh)), 1, 0)
    return data['Signal']