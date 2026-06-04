import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

pd.set_option('display.max_rows', None)

def get_data(stock,start,end):
    data = yf.download(stock, start=start, end=end)
    return data


def calculate_indicators(data):
    #getting SMA columns
    data['SMA_10'] = data['Close'].rolling(window=10).mean()    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_30'] = data['Close'].rolling(window=30).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_100'] = data['Close'].rolling(window=100).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

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

    

def backtester(data, sma_short=20, sma_long=50, rsi_thresh=75):

    data = data.dropna()

    #strategy brain
    data['Signal'] = np.where(((data[f'SMA_{sma_short}'] > data[f'SMA_{sma_long}']) & (data['RSI'] < rsi_thresh) & (data['ADX'] > 25)), 1, 0)
    data['Position']= data['Signal'].diff()

    data['Strategy Returns'] = data['Returns'] * data['Signal'].shift(1)
    cost = 0.001
    data['Strategy Returns'] = data['Strategy Returns'] - (cost * (data['Position'] != 0))
    data['Cumulative Strategy Returns']=(1+data['Strategy Returns']).cumprod()  

    #plotting
    #data['Cumulative Returns'].plot(label='cmu returns')
    #data['Cumulative Strategy Returns'].plot(label='cmu strategy returns')
    #print(data[['Returns', 'Signal', 'Strategy Returns', 'Cumulative Strategy Returns', 'RSI', 'ADX']].to_string()) 

    rolling_max = data['Cumulative Strategy Returns'].cummax()
    drawdown = (data['Cumulative Strategy Returns'] - rolling_max) / rolling_max

    sharpe = (data['Strategy Returns'].mean() / data['Strategy Returns'].std()) * (252 ** 0.5)
    total_return = data['Cumulative Strategy Returns'].iloc[-1]
    max_dd = drawdown.min()
    return total_return, sharpe, max_dd

def optimize(stock, start, end, sma_short_range, sma_long_range, rsi_range):
    results = []
    data = get_data(stock, start, end)
    data = calculate_indicators(data)
    for sma_short in sma_short_range:
        for sma_long in sma_long_range:
            for rsi_thresh in rsi_range:
                total_return, sharpe, max_dd = backtester(data.copy(), sma_short, sma_long, rsi_thresh)
                results.append({'sma_short': sma_short,'sma_long': sma_long,'rsi_thresh': rsi_thresh,'total_return': total_return,'sharpe': sharpe,'max_drawdown': max_dd})
                
    return pd.DataFrame(results)

def walk_forward(stock, train_start, train_end, test_start, test_end, sma_short_range, sma_long_range, rsi_range):

    train_results= optimize(stock, train_start, train_end, sma_short_range, sma_long_range, rsi_range)
    best= train_results.loc[train_results['sharpe'].idxmax()]

    test_data = get_data(stock, test_start, test_end)
    test_data = calculate_indicators(test_data)
    total_return, sharpe, max_dd = backtester(test_data, int(best['sma_short']), int(best['sma_long']), int(best['rsi_thresh']))

    return {
        'best_params': best,
        'test_return': total_return,
        'test_sharpe': sharpe,
        'test_max_dd': max_dd
    }


def ml_strategy(data):
    data['Target']= np.where((data['Returns'].shift(-1)>0),1,0)

    data['SMA_Ratio'] = data['SMA_20'] / data['SMA_50']

    features=['SMA_Ratio', 'RSI', 'ADX']
    X= data[features].dropna()
    Y= data['Target'].loc[X.index]

    tscv = TimeSeriesSplit(n_splits=5)

    all_predictions=[]
    all_rf_predictions=[]

    for train_index, test_index in tscv.split(X):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        Y_train = Y.iloc[train_index]
        Y_test = Y.iloc[test_index]

        model= LogisticRegression(class_weight='balanced')
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        all_predictions.append((X_test.index, predictions))

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, Y_train)
        rf_predictions = rf_model.predict(X_test)
        all_rf_predictions.append((X_test.index, rf_predictions))


    data['ML_Signal'] = 0
    for index, preds in all_predictions:
        data.loc[index, 'ML_Signal'] = preds

    data['RML_Signal'] = 0
    for index, preds in all_rf_predictions:
        data.loc[index, 'RML_Signal'] = preds
 
    data['ML_Strategy Returns'] = data['Returns'] * data['ML_Signal'].shift(1)
    data['Cumulative ML Returns'] = (1 + data['ML_Strategy Returns']).cumprod()

    data['RML_Strategy Returns'] = data['Returns'] * data['RML_Signal'].shift(1)
    data['Cumulative RML Returns'] = (1 + data['RML_Strategy Returns']).cumprod()

    return data

    

def performance_report(data):
    ###print("Random Forest Strategy Return:", data['Cumulative RML Returns'].iloc[-1])   print("ML Strategy Return:", data['Cumulative ML Returns'].iloc[-1])  print("Buy and Hold Return:", data['Cumulative Returns'].iloc[-1])
    ###

    ml_sharpe = (data['ML_Strategy Returns'].mean() / data['ML_Strategy Returns'].std()) * (252 ** 0.5)
    #print("ML sharpe", sharpe)
    rml_sharpe = (data['RML_Strategy Returns'].mean() / data['RML_Strategy Returns'].std()) * (252 ** 0.5)
    #print("RML sharpe", sharpe)

    rolling_max = data['Cumulative ML Returns'].cummax()
    drawdown = (data['Cumulative ML Returns'] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    #print("imML max drawdown", max_dd)

    r_rolling_max = data['Cumulative RML Returns'].cummax()
    r_drawdown = (data['Cumulative RML Returns'] - r_rolling_max) / r_rolling_max
    r_max_dd = r_drawdown.min()
    #print("RML max drawdown", max_dd)

    #print("ML number of trades", (data['ML_Signal'].diff() != 0).sum() )

    #print("RML number of trades", (data['RML_Signal'].diff() != 0).sum() )

    return {'rml return': data['Cumulative RML Returns'].iloc[-1], 'ml return': data['Cumulative ML Returns'].iloc[-1], 'normal return':data['Cumulative Returns'].iloc[-1], 'rml sharpe': rml_sharpe, 'ml_sharpe': ml_sharpe, 'ml_drawdown': max_dd, 'rml_drawdown': r_max_dd, 'ml_trades':(data['ML_Signal'].diff() != 0).sum(), 'rml_trades':(data['RML_Signal'].diff() != 0).sum()}

def annual_analysis(stock, start, end):
    results=[]
    for j in stock:
        for i in range(start, end):
            data= get_data(j, str(i)+"-01-01", str(i+1)+"-01-01")
            if data.empty:
                continue
            data= calculate_indicators(data)
            arr= ml_strategy(data)

            metrics = performance_report(arr)
            metrics['year'] = i
            metrics['ticker']=j
            results.append(metrics)
    return pd.DataFrame(results)
   

if __name__ == "__main__":
    results = walk_forward("AAPL", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31",
                           sma_short_range=[10,20,30],
                           sma_long_range=[50,100,200],
                           rsi_range=[65,70,75,80])
    print(results)

    data = get_data("AAPL", "2019-01-01", "2025-01-01")
    data = calculate_indicators(data)
    arr = ml_strategy(data)
    arr= performance_report(arr)

    print(annual_analysis(["VEDL."], 2019, 2025).to_string())