import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
import numpy as np
from datetime import datetime
from plotly.subplots import make_subplots
import networkx as nx
from hmmlearn.hmm import GaussianHMM
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter

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
    'Amazon' : 'AMZN',
    'NVIDIA' : 'NVDA',
    'Exxon Mobile' : 'XOM',
    'Netflix' : 'NFLX',
    'META' : 'META',
    'Costco' : 'COST',
    'Coca-cola' : 'KO',
    'Samsung' : '005930.KS',
    'Hyundai' : '005380.KS',
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
    min_ret = st.number_input('Minimum return rate', min_value = 0.0, max_value = 5.0, step = 0.25, value = 3.0)

    if strategy == 'Triple barrier' :

        upper_barrier = st.number_input('Upper barrier', min_value=0.0, max_value=5.0, step=0.25, value=2.0)
        lower_barrier = st.number_input('Lower barrier', min_value=0.0, max_value=5.0, step=0.25, value=1.0)
        vert_barrier = st.number_input('Vertical barrier', min_value=0, max_value=120, step=1, value=7)

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
        signal = 'Buy' if y_prob[-1] > 0.5 else 'Sell'
        st.write(signal)

        signal_fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = y_prob[-1] * 100,
            title={'text': f"Date: {pd.to_datetime('today')}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                }
            )
        )

        signal_fig.update_layout(
            height=300,
            width=400
        )

        st.plotly_chart(signal_fig, use_container_width=True)

    elif strategy == 'Moving Average Strategy' :
        short_window = st.number_input('Short Window', min_value=1, max_value=60, step=1, value=5)
        long_window = st.number_input('Long Window', min_value=1, max_value=252, step=1, value=20)
        X_train, X_test, y_train, y_test = pipeline_moving_average(
            tickers[ticker],
            start_date,
            end_date,
            short_window,
            long_window,
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

        fit = forest.fit(X=X_train, y=y_train)

        y_prob = forest.predict_proba(X_test)[:, 1]
        y_pred = forest.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        accuracy = accuracy_score(y_test, y_pred)

        st.header("Machine Learning Signal")
        signal = 'Buy' if y_prob[-1] > 0.5 else 'Sell'
        st.write(signal)

        signal_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=y_prob[-1] * 100,
            title={'text': f"Date: {pd.to_datetime('today')}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
            }
        )
        )

        signal_fig.update_layout(
            height=300,
            width=400
        )

        st.plotly_chart(signal_fig, use_container_width=True)

    hmm = yf.download(ticker, start=start_date, end=end_date)
    features = ['^VIX', tickers[ticker]]
    exog = yf.download(
        features,
        start=start_date,
        end=end_date
    )['Adj Close']
    log_return = exog.pct_change(fill_method=None)
    log_return.dropna(inplace=True)
    X_train = log_return[:'2019']
    X_test = log_return['2020':]
    n_states = 3

    hmm_model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )
    hmm_model.fit(X_train.values)
    hmm_pred_prob = pd.DataFrame(
        hmm_model.predict_proba(X_test.values),
        index=X_test.index,
        columns=[f'State_{i}' for i in range(n_states)]
    )

    bull_probability = round(hmm_pred_prob.iloc[-1]['State_0'] * 100, 2)
    neutral_probability = round(hmm_pred_prob.iloc[-1]['State_1'] * 100, 2)
    bear_probability = round(hmm_pred_prob.iloc[-1]['State_2'] * 100, 2)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=['Probability'],
            x=[bull_probability],
            name='Bull',
            orientation='h',
            marker=dict(color='#00CC96', line=dict(width=0.5)),
            text=[f"{bull_probability}%"],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=20, color='white')
        )
    )

    fig.add_trace(
        go.Bar(
            y=['Probability'],
            x=[neutral_probability],
            name='Neutral',
            orientation='h',
            marker=dict(color='#F4D44D', line=dict(width=0.5)),
            text=[f"{neutral_probability}%"],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=20, color='black')
        )
    )

    fig.add_trace(
        go.Bar(
            y=['Probability'],
            x=[bear_probability],
            name='Bear',
            orientation='h',
            marker=dict(color='#EF553B', line=dict(width=0.5)),
            text=[f"{bear_probability}%"],
            textposition='inside',
            insidetextanchor='middle',
            textfont=dict(size=20, color='white')
        )
    )

    if bull_probability > bear_probability and bull_probability > neutral_probability:
        mrkt_sent = 'BULL'
    elif bear_probability > bull_probability and bear_probability > neutral_probability:
        mrkt_sent = 'BEAR'
    else:
        mrkt_sent = 'NEUTRAL'

    fig.update_layout(
        title_text=mrkt_sent,
        barmode='stack',
        xaxis=dict(
            range=[0, 100],
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        height=120,
        width=800,
        plot_bgcolor='white'
    )

    st.plotly_chart(fig, use_container_width=True)

data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA60'] = data['Close'].rolling(window=60).mean()

def calculate_psy(data, window=14):
    diff = data['Close'].diff()
    up_days = (diff > 0).astype(int)
    psy = up_days.rolling(window=window).sum() / window * 100
    return psy
def calculate_bollinger_bands(data, window=20, num_std=2):
    data['MA20'] = data['Close'].rolling(window=window).mean()
    data['BB_up'] = data['MA20'] + num_std * data['Close'].rolling(window=window).std()
    data['BB_dn'] = data['MA20'] - num_std * data['Close'].rolling(window=window).std()
    return data

data['PSY'] = calculate_psy(data)
data = calculate_bollinger_bands(data)

recent_data = data.iloc[-60:]
price_range = [recent_data['Low'].min(), recent_data['High'].max()]
volume_range = [0, recent_data['Volume'].max() * 2]

################ Visualization ################

# Create the figure
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03)

# Add candlestick
fig.add_trace(go.Candlestick(x=data['Date'],
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='Candlestick'), row=1, col=1)
# Add moving averages
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
# Add Bollinger Bands
fig.add_trace(go.Scatter(x=data['Date'],
                         y=data['BB_up'],
                         mode='lines',
                         name='BB_upper',
                         line=dict(color='lightgrey')), row=1, col=1)
fig.add_trace(go.Scatter(x=data['Date'],
                         y=data['BB_dn'],
                         mode='lines',
                         name='BB_lower',
                         line=dict(color='lightgrey'),
                         fill='tonexty', fillcolor='rgba(211, 211, 211, 0.2)'), row=1, col=1)  # Fill color

# Add volume
fig.add_trace(go.Bar(x=data['Date'],
                     y=data['Volume'],
                     name='Volume',
                     marker=dict(color='lightblue')), row=2, col=1)
# Add PSY
fig.add_trace(go.Scatter(x=data['Date'],
                         y=data['PSY'],
                         mode='lines',
                         name='PSY',
                         line=dict(color='purple')), row=3, col=1)

# Update layout
fig.update_layout(title=f'{ticker} Candle Chart',
                  yaxis_title='Price',
                  xaxis_title='Date',
                  showlegend=True,
                  height=800,
                  autosize=True,
                  xaxis_rangeslider_visible=False)
fig.update_xaxes(range=[recent_data['Date'].iloc[0], recent_data['Date'].iloc[-1]], row=1, col=1)
fig.update_xaxes(range=[recent_data['Date'].iloc[0], recent_data['Date'].iloc[-1]], row=2, col=1)
fig.update_xaxes(range=[recent_data['Date'].iloc[0], recent_data['Date'].iloc[-1]], row=3, col=1)
fig.update_yaxes(range=price_range, row=1, col=1)
fig.update_yaxes(range=volume_range, row=2, col=1)
fig.update_yaxes(fixedrange=False, row=1, col=1)


with col1:
    st.plotly_chart(fig, use_container_width=True)

st.subheader('Additional Information')

col1, col2 = st.columns([1, 1])

with col1 :

    nx_returns = pd.read_parquet('data/SP500_returns.parquet')
    nx_stock = yf.download(tickers[ticker], start = '2020-01-01', end = '2024-06-01')['Adj Close'].pct_change().dropna()

    corr = nx_returns.corrwith(nx_stock)

    threshold = 0.5
    related_stocks = corr[corr > threshold].index.to_list()

    related_returns = nx_returns[related_stocks]
    corr_matrix = related_returns.corr()

    distance_matrix = 1 - corr_matrix

    G = nx.Graph()
    for i in range(len(distance_matrix.columns)):
        for j in range(i+1, len(distance_matrix.columns)):
            G.add_edge(distance_matrix.columns[i], distance_matrix.columns[j], weight=distance_matrix.iloc[i, j])

    mst = nx.minimum_spanning_tree(G)
    pos = nx.spring_layout(mst)
    pos[tickers[ticker]] = np.array([0, 0])

    edge_x = []
    edge_y = []
    for edge in mst.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    for node in mst.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f'{tickers[ticker]} related Securities',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        width=600,
                        height=500,
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text=f"Network between {tickers[ticker]} and S&P 500",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    st.plotly_chart(fig, use_container_width=True)

with col2 :
    def get_news_rss(query=tickers[ticker]):
        url = f'https://news.google.com/rss/search?q={query}'
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'xml')
        headlines = soup.find_all('title')
        news_texts = [headline.get_text() for headline in headlines[2:]]  # Skip the first two entries as they are not relevant
        return news_texts

    # Preprocess the news texts and remove stopwords
    def preprocess_text(texts, stopwords):
        text = ' '.join(texts)
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower()
        text_words = text.split()
        text_words = [word for word in text_words if word not in stopwords]
        processed_text = ' '.join(text_words)
        return processed_text, Counter(text_words)


    stopwords = set(
        ['yahoo', 'finance', ticker, tickers[ticker], 'yahoo finance', 'stock', 'stocks', 'market']
    )

    news_texts = get_news_rss(query=tickers[ticker])

    if not news_texts:
        st.write("Unable to fetch news. Please try a different method.")
    else:
        news_texts_processed, word_freq = preprocess_text(news_texts, stopwords)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(news_texts_processed)
        words = wordcloud.words_
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        word_x = []
        word_y = []
        word_freq_list = []
        word_size = []
        word_colors = []
        for word, freq in words.items():
            word_x.append(np.random.uniform(0, 1))
            word_y.append(np.random.uniform(0, 1))
            word_freq_list.append(word)
            word_size.append(freq * 100)
            word_colors.append(np.random.choice(colors))

        fig = go.Figure(data=[go.Scatter(
            x=word_x,
            y=word_y,
            text=word_freq_list,
            mode='text',
            textfont=dict(
                size=word_size,
                color=word_colors
            )
        )])

        fig.update_layout(
            title='Up to date Keywords',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=600,
            height=500,
        )

        # Display the word cloud in Streamlit
        st.plotly_chart(fig, use_container_width=True)