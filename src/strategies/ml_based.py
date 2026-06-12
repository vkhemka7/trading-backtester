from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def generate_ml_signals(data):
    data['Target']= np.where((data['Returns'].shift(-1)>0),1,0)

    data['SMA_Ratio'] = data['SMA_20'] / data['SMA_50']

    data['EMA_Ratio'] = data['EMA_20'] / data['EMA_50']





    features=['EMA_Ratio', 'RSI', 'ADX']
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

        rf_model = RandomForestClassifier(n_estimators=50, random_state=42, min_samples_leaf=5, max_depth=5)
        rf_model.fit(X_train, Y_train)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        rf_predictions = (rf_proba > 0.5).astype(int)
        all_rf_predictions.append((X_test.index, rf_predictions))


    data['ML_Signal'] = 0
    for index, preds in all_predictions:
        data.loc[index, 'ML_Signal'] = preds

    data['RML_Signal'] = 0
    for index, preds in all_rf_predictions:
        data.loc[index, 'RML_Signal'] = preds
 
   
    return data['ML_Signal'], data['RML_Signal']