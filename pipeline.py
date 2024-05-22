import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from ta.volume import money_flow_index, ease_of_movement
from ta.volatility import average_true_range, ulcer_index
from ta.trend import adx, trix
from ta.momentum import rsi, stoch
from FinancialMachineLearning.labeling.labeling import *
from FinancialMachineLearning.features.volatility import daily_volatility

def psycological_index(data, window = 12):
    up_days = data['Close'].diff() > 0
    sentiment = up_days.rolling(window = window).sum() / window
    return sentiment

def get_exog_features(start_date, end_date) -> pd.DataFrame:
    features = ['^FVX', '^TYX', '^VIX', 'JPY=X', '^GSPC', 'GC=F']
    exog = yf.download(
        features,
        start = start_date,
        end = end_date
    )['Adj Close']
    exog[['^FVX', '^TYX', 'JPY=X', '^GSPC', 'GC=F']] = exog[['^FVX', '^TYX', 'JPY=X', '^GSPC', 'GC=F']].pct_change()
    exog['^VIX'] = exog['^VIX'] * 0.01

    return exog
def pipeline(
        ticker,
        start_date,
        end_date,
        upper_barrier,
        lower_barrier,
        vert_barrier,
        min_ret
) -> tuple:

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date
    )

    data['money_flow_index'] = money_flow_index(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        volume=data['Volume'],
        window=14
    ) * 0.01  # normalizing
    data['ease_of_movement'] = ease_of_movement(
        high=data['High'],
        low=data['Low'],
        volume=data['Volume'],
        window=14
    ) * 0.01
    data['average_true_range'] = average_true_range(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=14
    ).pct_change()
    data['ulcer_index'] = ulcer_index(
        close=data['Close'],
        window=14
    ) * 0.01
    data['adx'] = adx(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=14
    ) * 0.01
    data['trix'] = trix(
        close=data['Close'],
        window=14
    ) * 0.01
    data['rsi'] = rsi(
        close=data['Close'],
        window=14
    ) * 0.01
    data['stoch'] = stoch(
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        window=14
    ) * 0.01
    data['psycological_index'] = psycological_index(
        data, window=14
    )

    data['ret'] = data['Close'].pct_change()
    data['std'] = data['Close'].pct_change().rolling(window=20).std()
    data['skew'] = data['Close'].pct_change().rolling(window=20).skew()
    data['kurt'] = data['Close'].pct_change().rolling(window=20).kurt()

    exog = get_exog_features(start_date, end_date)

    data = pd.concat(
        [data, exog], axis=1
    )

    ### Labeling

    vertical_barrier = add_vertical_barrier(
        data.index,
        data['Close'],
        num_days=vert_barrier  # expariation limit
    )
    vertical_barrier.head()

    volatility = daily_volatility(
        data['Close'],
        lookback=60  # moving average span
    )

    triple_barrier_events = get_events(
        close=data['Close'],
        t_events=data.index[2:],
        pt_sl=[upper_barrier, lower_barrier],  # profit taking 2, stopping loss 1
        target=volatility,  # dynamic threshold
        min_ret=min_ret,  # minimum position return
        num_threads=1,  # number of multi-thread
        vertical_barrier_times=vertical_barrier,  # add vertical barrier
        side_prediction=None  # betting side prediction (primary model)
    )
    triple_barrier_events.head()

    labels = meta_labeling(
        triple_barrier_events,
        data['Close']
    )

    triple_barrier_events['side'] = labels['bin']

    meta_labels = meta_labeling(
        triple_barrier_events,  # with side labels
        data['Close']
    )

    data['side'] = triple_barrier_events['side'].copy()
    data['label'] = meta_labels['bin'].copy()

    data.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis=1, inplace=True)
    data.dropna(inplace=True)

    matrix = data[data['side'] != 0]
    X = matrix.drop(['side', 'label'], axis=1)
    y = matrix['label']

    X_train, X_test = X.loc[:'2019'], X.loc['2020':]
    y_train, y_test = y.loc[:'2019'], y.loc['2020':]

    return X_train, X_test, y_train, y_test