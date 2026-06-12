import numpy as np

def run_backtest(data, signal, transaction_cost=0.001):

    data = data.copy().dropna()

    #strategy brain
    data['Signal'] = signal
    data['Position']= data['Signal'].diff()

    data['Strategy Returns'] = data['Returns'] * data['Signal'].shift(1)
    data['Strategy Returns'] = data['Strategy Returns'] - (transaction_cost * (data['Position'] != 0))
    data['Cumulative Strategy Returns']=(1+data['Strategy Returns']).cumprod()  

    return data

   
    
   