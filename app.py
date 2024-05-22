import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')
from pipeline import *

start_date = '2000-01-01'
end_date = pd.to_datetime('today')
@st.cache_data
def load_data(ticker):
    data = yf.download(
        ticker,
        start = start_date,
        end = end_date
    )
    data.reset_index(inplace=True)
    return data

st.set_page_config(layout="wide")

st.title("Meta Trading")

tickers = {
    'Apple' : 'AAPL',
    'Micro Soft' : 'MSFT',
    'Tesla' : 'TSLA',
    'Google' : 'GOOGL',
    'IBM' : 'IBM',
    'Crude oil ETF' : 'USO',
    'Gold ETF' : 'GLD',
    'USA Tresury Bond ETF' : 'BND',
    'KODEX ETF' : '069500.KS',
    'SPY ETF' : 'SPY',
    'QQQ ETF' : 'QQQ'
}
stg = ['Triple barrier', 'Moving Average Strategy','RSI Strategy']

col1, col2 = st.columns([3, 2])

with col2:
    ticker = st.selectbox('Pick up your security :', tickers.keys())
    strategy = st.selectbox('Pick up your strategy :', stg)

    data = load_data(tickers[ticker])

    upper_barrier = st.number_input('Upper barrier', min_value = 0.0, max_value = 5.0, step = 0.25, value = 2.0)
    lower_barrier = st.number_input('Lower barrier', min_value = 0.0, max_value = 5.0, step = 0.25, value = 1.0)
    vert_barrier = st.number_input('Vertical barrier', min_value = 0, max_value = 120, step = 1, value = 7)
    min_ret = st.number_input('Minimum return rate', min_value = 0.0, max_value = 5.0, step = 0.25, value = 3.0)

    if strategy == 'Triple barrier' :

        X_train, X_test, y_train, y_test = pipeline(
            tickers[ticker],
            start_date,
            end_date,
            upper_barrier,
            lower_barrier,
            vert_barrier,
            min_ret * 0.01
        )

        # Machine Learning

        forest = RandomForestClassifier(
            criterion='entropy',
            class_weight='balanced_subsample',
            random_state=42,
            n_estimators=1000,
            max_features=8,
            oob_score=True
        )

        fit = forest.fit(X = X_train, y = y_train)

        y_prob = forest.predict_proba(X_test)[:, 1]
        y_pred = forest.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        st.header("Machine Learning Signal")
        st.write("Trading Signal from Machine Learning Model")

        signal_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,  # 0에서 100 사이의 값으로 변환
            title={'text': f"Date: {pd.to_datetime('today')}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                }
            )
        )

        st.plotly_chart(signal_fig, use_container_width=True)

data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA60'] = data['Close'].rolling(window=60).mean()

# 심리선 계산 (14일 기준)
def calculate_psy(data, window=14):
    diff = data['Close'].diff()
    up_days = (diff > 0).astype(int)
    psy = up_days.rolling(window=window).sum() / window * 100
    return psy

data['PSY'] = calculate_psy(data)

recent_data = data.iloc[-60:]
price_range = [recent_data['Low'].min(), recent_data['High'].max()]
volume_range = [0, recent_data['Volume'].max() * 2]

################ Visualization ################

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights = [0.6, 0.2, 0.2], vertical_spacing = 0.03)


fig.add_trace(go.Candlestick(x=data['Date'],
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Candlestick'), row=1, col=1)
fig.add_trace(go.Scatter(x=data['Date'],
                         y=data['MA20'],
                         mode='lines',
                         name='MA20',
                         line=dict(color='orange')), row=1, col=1)
fig.add_trace(go.Scatter(x=data['Date'],
                         y=data['MA60'],
                         mode='lines',
                         name='MA60',
                         line=dict(color='darkorange')), row=1, col=1)
fig.add_trace(go.Bar(x=data['Date'],
                     y=data['Volume'],
                     name='Volume',
                     marker=dict(color='lightblue')), row=2, col=1)
fig.add_trace(go.Scatter(x=data['Date'],
                         y=data['PSY'],
                         mode='lines',
                         name='PSY',
                         line=dict(color='purple')), row=3, col=1)
fig.update_layout(title=f'{ticker} Candle Chart',
                  yaxis_title='Price',
                  xaxis_title='Date',
                  showlegend = True,
                  height = 800,
                  autosize = True,
                  xaxis_rangeslider_visible = False)
fig.update_xaxes(range=[recent_data['Date'].iloc[0], recent_data['Date'].iloc[-1]], row=1, col=1)
fig.update_xaxes(range=[recent_data['Date'].iloc[0], recent_data['Date'].iloc[-1]], row=2, col=1)
fig.update_xaxes(range=[recent_data['Date'].iloc[0], recent_data['Date'].iloc[-1]], row=3, col=1)
fig.update_yaxes(range=price_range, row=1, col=1)
fig.update_yaxes(range=volume_range, row=2, col=1)
fig.update_yaxes(fixedrange=False, row=1, col=1)

with col1:
    st.plotly_chart(fig, use_container_width=True)

