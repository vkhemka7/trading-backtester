import numpy as np
import pandas as pd

def calculate_indicators(data):
    #getting SMA columns
    data['SMA_10'] = data['Close'].rolling(window=10).mean()    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['EMA_30'] = data['Close'].ewm(span=30, adjust=False).mean()
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
    data['EMA_100'] = data['Close'].ewm(span=100, adjust=False).mean()
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()


    #getting daily and cumulative returns
    data['Returns']= data['Close'].pct_change()
    data['Cumulative Returns']=(1+data['Returns']).cumprod()

    #calculating RSIm
    data['daily change']= data['Close']-data['Close'].shift(1)
    data['gains']=data['daily change'].clip(lower=0)
    data['loss']= data['daily change'].clip(upper=0).abs()
    data['RSI']= 100-(100/(1+(data['gains'].ewm(span=14, adjust= False).mean()/data['loss'].ewm(span=14, adjust= False).mean())))

    #calculating adx
    data['upmove']= data['High']-data['High'].shift(1)
    data['downmove']= data['Low'].shift(1)-data['Low']
    data['PDM']=np.where((((data)['upmove']>data['downmove']) & (data['upmove']>0)),data['upmove'],0)
    data['NDM']=np.where(((data['upmove']<data['downmove']) & (data['downmove']>0)),data['downmove'],0)
    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift(1)).abs()
    low_close = (data["Low"] - data["Close"].shift(1)).abs()
    data["TR"] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['PDI']= (100*data['PDM'].ewm(span=14, adjust=False).mean())/(data['TR']).ewm(span=14, adjust=False).mean()
    data['NDI']= (100*data['NDM'].ewm(span=14, adjust=False).mean())/(data['TR']).ewm(span=14, adjust=False).mean()
    data['DX']=((data['PDI']-data['NDI'])/(data['PDI']+data['NDI'])).abs()*100
    data['ADX']=data['DX'].ewm(span=14, adjust=False).mean()
    
    return data