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
from konlpy.tag import Okt
import urllib.request
import ssl
import json
import FinanceDataReader as fdr
import ta

############################# 패키지 및 모듈 ######################################


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

### 밑에 한국어 사전에도 추가해주세용!!! --> 뉴스크롤링용
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

col1, col2 = st.columns([5, 2])

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


####################### 주식차트와 종목, 전략설정 아래부분 ###########################

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

### 전략사전 
with col2:


    st.title('전략사전: 주식 투자 기술적 분석 전략')

    # 전략 목록
    strategies = {
        "Triple Barrier Strategy": {
            "description": """
            트리플 배리어 전략은 자산 가격이 특정 조건에 도달했을 때 매매를 결정하는 방법이다. 이름처럼 3가지 장벽이 있으며 다음과 같다.\n
            상한선 (Upper Barrier): 주가가 일정 비율 이상 상승했을 때\n
            하한선 (Lower Barrier): 주가가 일정 비율 이상 하락했을 때\n 
            시간 제한 (Time Barrier): 일정 시간이 지났을 때\n
            예시\n
            애플 주식을 100달러에 샀다고 가정해봅시다.\n
            상한성 20%, 하한선 20%, 시간제한 7일이라고 하였을 떄\n
            상한선: 애플 주식이 120달러가 되면 (20% 상승) 주식을 팝니다.\n
            하한선: 애플 주식이 80달러가 되면 (20% 하락) 주식을 팝니다.\n
            시간 제한: 주식을 산 후 7일이 지나면 주식을 팝니다.\n
            이 전략은 위험 관리를 강화하고, 기대 수익을 극대화하는 데 도움을 줍니다.
            """,
            "image": "images/0*XrMZ6tBERWex91jN.webp"
        },
        "Moving Average Strategy": {
            "description": """
            Moving Average 전략은 단순 이동 평균(SMA) 또는 지수 이동 평균(EMA)을 사용하여 추세의 방향을 판단합니다.
            이 전략은 추세를 따르는 트레이딩에 매우 유용합니다.
            """,
            "image": "images/639aaf3c2d96d116ed0818215f94f337.jpg"
        },
        "RSI Strategy": {
            "description": """
            RSI 지표는 주식이 과매수 상태인지 과매도 상태인지를 알려줍니다:\n
            과매수 상태 (Overbought): RSI 값이 70 이상일 때, 주식이 너무 많이 올라서 곧 떨어질 가능성이 높습니다.\n
            과매도 상태 (Oversold): RSI 값이 30 이하일 때, 주식이 너무 많이 떨어져서 곧 오를 가능성이 높습니다.
            RSI 전략은 주식의 매수와 매도 시점을 다음과 같이 결정합니다:\n
            매수 시점: RSI 값이 30 이하로 떨어질 때, 주식이 과매도 상태라고 판단하여 주식을 삽니다.\n
            매도 시점: RSI 값이 70 이상으로 올라갈 때, 주식이 과매수 상태라고 판단하여 주식을 팝니다.\n
            예시\n
            애플 주식의 RSI 값을 계산해본다고 가정해봅시다.\n
            RSI 값이 30 이하로 떨어짐: 주식이 너무 많이 떨어져서 이제 곧 오를 가능성이 높다고 판단하여 주식을 매수합니다.\n
            RSI 값이 70 이상으로 올라감: 주식이 너무 많이 올라서 이제 곧 떨어질 가능성이 높다고 판단하여 주식을 매도합니다.
            """,
            "image": "images/68a2b4d351474b06ad3f8162c7d85f3f.png"
        }
    }

    # 사용자 입력을 통한 전략 선택
    selected_strategy = st.selectbox("전략을 선택하세요:", list(strategies.keys()))

    # 선택된 전략 정보 출력
    st.header(selected_strategy)
    st.write(strategies[selected_strategy]["description"])
    st.image(strategies[selected_strategy]["image"])


with col2 :

    ## 네이버증권 주요뉴스 크롤러 + 뉴스 제목과 본문의 키워드까지 출력!

    # 특정 페이지의 뉴스를 가져오는 함수
    def get_news_from_page(page_url):
        response = requests.get(page_url)  # 페이지 URL로 GET 요청
        response.raise_for_status()  # 요청이 성공하지 않으면 예외 발생
        
        soup = BeautifulSoup(response.text, 'html.parser')  # 페이지 HTML을 파싱
        news_section = soup.find('ul', class_='newsList')  # 뉴스 리스트 섹션을 찾음
        if news_section is None:
            print(f"No news section found on {page_url}")
            return []
        
        news_items = news_section.find_all('li')  # 뉴스 아이템들을 모두 찾음
        
        news_list = []
        for item in news_items:
            title_tag = item.find('dd', class_='articleSubject').find('a')  # 각 뉴스 아이템의 제목 태그를 찾음
            if title_tag is None:
                print(f"No title tag found in item: {item}")
                continue
            
            title = title_tag.get_text(strip=True)  # 제목 텍스트를 가져옴
            if not title:
                print(f"Empty title found in item: {item}")
                continue

            link = "https://finance.naver.com" + title_tag.get('href', '')
            if not link:
                print(f"No link found in item: {item}")
                continue

            summary_tag = item.find('dd', class_='articleSummary')
            summary = summary_tag.get_text(strip=True) if summary_tag else ""
            
            news_list.append({'title': title, 'link': link, 'summary': summary})  # 뉴스 리스트에 추가
        
        return news_list  # 뉴스 리스트를 반환

    # 마지막 페이지 번호를 가져오는 함수
    def get_last_page_number(date):
        url = f"https://finance.naver.com/news/mainnews.naver?date={date}"
        response = requests.get(url)  # 메인 뉴스 페이지로 GET 요청
        response.raise_for_status()  # 요청이 성공하지 않으면 예외 발생
        
        soup = BeautifulSoup(response.text, 'html.parser')  # 페이지 HTML을 파싱
        page_nav = soup.find('table', class_='Nnavi')  # 페이지 내비게이션 섹션을 찾음

        if page_nav is None:
            return 1  # 페이지 내비게이션이 없으면 첫 페이지로 간주
        
        page_links = page_nav.find_all('a', href=True)  # 페이지 링크들을 모두 찾음
        page_numbers = []
        
        for link in page_links:
            href = link['href']
            if 'page=' in href:
                try:
                    page_number = int(href.split('page=')[-1])  # 페이지 번호를 추출
                    page_numbers.append(page_number)
                except ValueError:
                    continue
        
        if not page_numbers:
            return 1  # 페이지 번호가 없으면 첫 페이지로 간주

        return max(page_numbers)  # 가장 큰 페이지 번호를 반환

    # 모든 뉴스를 가져오는 함수
    def get_all_news(date):
        base_url = f"https://finance.naver.com/news/mainnews.naver?date={date}&page="
        last_page_number = get_last_page_number(date)  # 마지막 페이지 번호를 가져옴
        
        all_news = []
        for page in range(1, last_page_number + 1):  # 1번 페이지부터 마지막 페이지까지 반복
            page_url = base_url + str(page)  # 각 페이지 URL 생성
            news_list = get_news_from_page(page_url)  # 해당 페이지의 뉴스 목록 가져오기
            all_news.extend(news_list)  # 전체 뉴스 목록에 추가
        
        return all_news  # 모든 뉴스 목록 반환


    # 불용어 처리
    stopwords = ['.', '(', ')', ',', "'", '%', '-', 'X', ').', '×','의','자','에','안','번','호','을','이','다','만','로','가','를',
                '경제', '증시', '투자', '증권', '증시', '주가', '최고', '발표', '올해', '지난해', '국내', '최근', '최대', '종목', '현지', '가격', '가운데', '시간',
                '관련', '거래', '결제', '소식', '대해', '하루', '오전', '오후', '다시', '이후', '이전',
                '이데일리', '헤럴드경제', '연합뉴스', '파이낸셜뉴스', '매일경제', '머니투데이',
                "and", "are", "is", "the", "a", "an", "in", "on", "of"
                ]

    # 사용자 정의 복합 명사 리스트
    custom_compound_nouns = ['인공지능', '자연어처리', '딥러닝', '머신러닝', '데이터사이언스']

    # 텍스트에서 복합 명사를 인식하고 결합
    def combine_compound_nouns(text, compound_nouns):
        for noun in compound_nouns:
            text = text.replace(noun, noun.replace(" ", "_"))
        return text

    # 자주 등장하는 키워드를 분석하는 함수
    def analyze_keywords(news_list):

        okt = Okt()
        text = " ".join([item['title'] + " " + item['summary'] for item in news_list])
        
        # 한국어 명사 추출
        korean_words = okt.nouns(text)
        # 영어 단어 추출
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)
        # 한국어 명사와 영어 단어 합치기
        all_words = korean_words + english_words
        
        # 불용어 제거 및 필터링
        filtered_words = [word for word in all_words if word not in stopwords and len(word) > 1 and not word.isdigit()]

        # 단어 빈도 계산
        counter = Counter(filtered_words)
        common_words = counter.most_common(50)  # 상위 50개의 자주 등장하는 단어 추출

        return common_words

    date = datetime.today().strftime('%Y-%m-%d')
    daily_all_news = get_all_news(date)  # 모든 뉴스 가져오기
    news_df = pd.DataFrame(daily_all_news)  # 뉴스 데이터를 DataFrame으로 저장
    
    # 자주 등장하는 키워드 분석
    keywords = analyze_keywords(daily_all_news)

    # 단어 구름 생성을 위해 딕셔너리 형태로 변환
    word_freq_dict = dict(keywords)

    # word_freq_dict 내용 출력 (디버깅을 위해)
    if not word_freq_dict:
        print("word_freq_dict가 비어 있어서 워드 클라우드를 생성할 수 없습니다.")
    else:
        print("word_freq_dict:", word_freq_dict)

    # WordCloud 객체 생성
    if word_freq_dict:
        wordcloud = WordCloud(font_path='/Library/Fonts/AppleGothic.ttf',  # 한글 폰트 경로
                            width=800, 
                            height=400, 
                            background_color='white').generate_from_frequencies(word_freq_dict)
        
        # WordCloud 이미지를 numpy 배열로 변환
        wordcloud_image = wordcloud.to_array()

        # Plotly를 사용하여 이미지 시각화
        fig = go.Figure(go.Image(z=wordcloud_image),
                        layout=go.Layout(
                            title='Today Market Keywords',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            width=600,
                            height=400,
                            margin_l=0,
                            margin_r=0,
                            margin_b=0,
                            margin_t=50,
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False))
                        )

        st.plotly_chart(fig, use_container_width=True)

### 
with col2 :

    ## 네이버 API --> 따로 발급받아서 써야 함
    client_id = "43LJLHp46cLkGVClgpdA"
    client_secret = "gOd6F3Mme5"


    def getRequestUrl(url):
        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id", client_id)
        req.add_header("X-Naver-Client-Secret", client_secret)

        context = ssl._create_unverified_context()

        try:
            response = urllib.request.urlopen(req, context=context)
            if response.getcode() == 200:
                print("[%s] Url Request Success" % datetime.now())
                return response.read().decode('utf-8')
        except Exception as e:
            print(e)
            print("[%s] Error for URL : %s" % (datetime.now(), url))
            return None


    def getNaverSearch(node, srcText, start, display):
        base = "https://openapi.naver.com/v1/search"
        node = "/%s.json" % node
        parameters = "?query=%s&start=%s&display=%s" % (urllib.parse.quote(srcText), start, display)

        url = base + node + parameters
        responseDecode = getRequestUrl(url)  

        if responseDecode is None:
            return None
        else:
            return json.loads(responseDecode)


    def getPostData(post, jsonResult, cnt):
        title = post['title']
        description = post['description']
        org_link = post['originallink']
        link = post['link']

        pDate = datetime.strptime(post['pubDate'], '%a, %d %b %Y %H:%M:%S +0900')
        pDate = pDate.strftime('%Y-%m-%d %H:%M:%S')

        jsonResult.append({'cnt': cnt, 'title': title, 'description': description,
                        'org_link': org_link, 'link': link, 'pDate': pDate})
        return

    translation_dict = {
        'Apple': '애플',
        'Micro Soft': '마이크로소프트',
        'Tesla': '테슬라',
        'Google': '구글',
        'IBM': 'IBM',
        'Amazon': '아마존',
        'NVIDIA': '엔비디아',
        'Exxon Mobile': '엑슨모빌',
        'Netflix': '넷플릭스',
        'META': '메타',
        'Costco': '코스트코',
        'Coca-Cola': '코카콜라',
        'Samsung': '삼성전자',
        'Hyundai': '현대자동차',
        'Crude oil ETF': '원유',
        'Gold ETF': '금',
        'USA Treasury Bond ETF': '미국국채',
        'KODEX ETF': '코스피',
        'SPY ETF': 'S&P500',
        'QQQ ETF': '나스닥'
    }
   
    def Crawling_json():
        node = 'news'  
        srcText = translation_dict[ticker]
        cnt = 0
        jsonResult = []

        jsonResponse = getNaverSearch(node, srcText, 1, 100)  
        if jsonResponse is None:
            print("No response received. Exiting.")
            return

        total = jsonResponse.get('total', 0)

        while ((jsonResponse != None) and (jsonResponse['display'] != 0)):
            for post in jsonResponse['items']:
                cnt += 1
                getPostData(post, jsonResult, cnt)  

            start = jsonResponse['start'] + jsonResponse['display']
            jsonResponse = getNaverSearch(node, srcText, start, 100) 

        print('전체 검색 : %d 건' % total)

        ## 크롤링한 데이터의 json파일
        jsonFile = json.dumps(jsonResult, indent=4, sort_keys=True, ensure_ascii=False)

        print("가져온 데이터 : %d 건" % (cnt))
        #print('%s_naver_%s.json SAVED' % (srcText, node))

        return jsonFile, srcText, node, jsonResult

    # 뉴스기사 json형태로 저장
    def Save_json(jsonFile, srcText, node, jsonResult):
        with open('%s_naver_%s.json' % (srcText, node), 'w', encoding='utf8') as outfile:
            jsonFile = json.dumps(jsonResult, indent=4, sort_keys=True, ensure_ascii=False)
            outfile.write(jsonFile)

    # DataFrame 상태에서 wordkeyword까지
    def DataFrame_list(jsonFile):
        # Load JSON data -> Convert to DataFrame
        data = json.loads(jsonFile)
        search_df = pd.DataFrame(data)

        # 리스트화
        news_list = search_df[['title', 'link', 'description']].to_dict(orient='records')

        return news_list

    # 불용어 처리
    stopwords = ['.', '(', ')', ',', "'", '%', '-', 'X', ').', '×', 'quot', '<b>', '</b>',
                '의','자','에','안','번','호','을','이','다','만','로','가','를',
                '경제', '증시', '투자', '증권', '증시', '주가', '최고', '발표', '올해', '지난해', '국내', '최근', '최대', '종목', '현지', '가격', '가운데', '시간',
                '관련', '거래', '결제', '소식', '대해', '하루', '오전', '오후', '다시', '이후', '이전',
                '이데일리', '헤럴드경제', '연합뉴스', '파이낸셜뉴스', '매일경제', '머니투데이',
                'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 
                'any', 'are', 'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 
                'below', 'between', 'both', 'but', 'by', 'can\'t', 'cannot', 'could', 'couldn\'t', 
                'did', 'didn\'t', 'do', 'does', 'doesn\'t', 'doing', 'don\'t', 'down', 'during', 
                'each', 'few', 'for', 'from', 'further', 'had', 'hadn\'t', 'has', 'hasn\'t', 'have', 
                'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 'her', 'here', 'here\'s', 
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 'i\'d', 'i\'ll', 
                'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 'itself', 
                'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 'of', 
                'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 
                'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 
                'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 
                'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 
                'they\'ll', 'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 
                'until', 'up', 'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 
                'were', 'weren\'t', 'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 
                'while', 'who', 'who\'s', 'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 
                'you', 'you\'d', 'you\'ll', 'you\'re', 'you\'ve', 'your', 'yours', 'yourself'
                ]

    # 사용자 정의 복합 명사 리스트
    custom_compound_nouns = ['인공지능', '자연어처리', '딥러닝', '머신러닝', '데이터사이언스']

    # 자주 등장하는 키워드를 분석하는 함수
    def analyze_keywords(news_list):

        okt = Okt()
        text = " ".join([item['title'] + " " + item['description'] for item in news_list])
        
        # 한국어 명사 추출
        korean_words = okt.nouns(text)
        # 영어 단어 추출
        english_words = re.findall(r'\b[a-zA-Z]+\b', text)
        # 한국어 명사와 영어 단어 합치기
        all_words = korean_words + english_words
        
        # 불용어 제거 및 필터링
        filtered_words = [word for word in all_words if word not in stopwords and len(word) > 1 and not word.isdigit()]

        # 단어 빈도 계산
        counter = Counter(filtered_words)
        common_words = counter.most_common(50)  # 상위 50개의 자주 등장하는 단어 추출

        return common_words

    def wordcloud_display(keywords):
        # 단어 구름 생성을 위해 딕셔너리 형태로 변환
        word_freq_dict = dict(keywords)

        # WordCloud 객체 생성
        wordcloud = WordCloud(font_path='/Library/Fonts/AppleGothic.ttf',  # 한글 폰트 경로
                            width=800, 
                            height=400, 
                            background_color='white').generate_from_frequencies(word_freq_dict)
        
        # WordCloud 이미지를 numpy 배열로 변환
        wordcloud_image = wordcloud.to_array()

        # Plotly를 사용하여 이미지 시각화
        fig = go.Figure(go.Image(z=wordcloud_image),
                        layout=go.Layout(
                            title=f'Today {ticker} Keywords',
                            titlefont_size=16,
                            showlegend=False,
                            hovermode='closest',
                            width=600,
                            height=400,
                            margin_l=0,
                            margin_r=0,
                            margin_b=0,
                            margin_t=50,
                            xaxis=dict(visible=False),
                            yaxis=dict(visible=False))
                        )

        st.plotly_chart(fig, use_container_width=True)

    
    jsonFile, srcText, node, jsonResult = Crawling_json()
    
    #Save_json(jsonFile, srcText, node, jsonResult)

    news_list = DataFrame_list(jsonFile)
    keywords = analyze_keywords(news_list)

    wordcloud_display(keywords) 

with col1:

    ## CD(91일)
    def get_CD_interest_rate(api_key, start_date, end_date):
        
        url = 'http://ecos.bok.or.kr/api/StatisticSearch/' + api_key + '/json/kr/1/100/[STAT_CODE]/D/YYYYMMDD/YYYYMMDD/[SUB_CODE]'
        
        url = url.replace("[STAT_CODE]", "817Y002")
        url = url.replace("YYYYMMDD/YYYYMMDD", start_date + "/" + end_date)
        url = url.replace("[SUB_CODE]", "010502000")

        response = requests.get(url)
        result = response.json()

        list_total_count=(int)(result['StatisticSearch']['list_total_count'])
        list_count=(int)(list_total_count/100) + 1

        rows=[]
        for i in range(0,list_count):
            
            start = str(i * 100 + 1)
            end = str((i + 1) * 100)
            
            url = "http://ecos.bok.or.kr/api/StatisticSearch/" + api_key + "/json/kr/" + start + "/" + end + "/" + "[STAT_CODE]/D/YYYYMMDD/YYYYMMDD/[SUB_CODE]"
        
            url = url.replace("[STAT_CODE]", "817Y002")
            url = url.replace("YYYYMMDD/YYYYMMDD", start_date + "/" + end_date)
            url = url.replace("[SUB_CODE]", "010502000")

            response = requests.get(url)
            data = response.json()
            
            # API로부터 받은 데이터를 파싱하여 기준금리를 추출
            rows += data['StatisticSearch']['row']
        
        CD_rates = pd.DataFrame(rows)

        return CD_rates

    ## 국고채3년
    def get_Korea_interest_rate(api_key, start_date, end_date):
        
        url = 'http://ecos.bok.or.kr/api/StatisticSearch/' + api_key + '/json/kr/1/100/[STAT_CODE]/D/YYYYMMDD/YYYYMMDD/[SUB_CODE]'
        
        url = url.replace("[STAT_CODE]", "817Y002")
        url = url.replace("YYYYMMDD/YYYYMMDD", start_date + "/" + end_date)
        url = url.replace("[SUB_CODE]", "010200000")

        response = requests.get(url)
        result = response.json()

        list_total_count=(int)(result['StatisticSearch']['list_total_count'])
        list_count=(int)(list_total_count/100) + 1

        rows=[]
        for i in range(0,list_count):
            
            start = str(i * 100 + 1)
            end = str((i + 1) * 100)
            
            url = "http://ecos.bok.or.kr/api/StatisticSearch/" + api_key + "/json/kr/" + start + "/" + end + "/" + "[STAT_CODE]/D/YYYYMMDD/YYYYMMDD/[SUB_CODE]"
        
            url = url.replace("[STAT_CODE]", "817Y002")
            url = url.replace("YYYYMMDD/YYYYMMDD", start_date + "/" + end_date)
            url = url.replace("[SUB_CODE]", "010200000")

            response = requests.get(url)
            data = response.json()
            
            rows += data['StatisticSearch']['row']
        
        Korea_rates = pd.DataFrame(rows)

        return Korea_rates

    ## 회사채3년(AA-)
    def get_Company_interest_rate(api_key, start_date, end_date):
        
        url = 'http://ecos.bok.or.kr/api/StatisticSearch/' + api_key + '/json/kr/1/100/[STAT_CODE]/D/YYYYMMDD/YYYYMMDD/[SUB_CODE]'
        
        url = url.replace("[STAT_CODE]", "817Y002")
        url = url.replace("YYYYMMDD/YYYYMMDD", start_date + "/" + end_date)
        url = url.replace("[SUB_CODE]", "010300000")

        response = requests.get(url)
        result = response.json()

        list_total_count=(int)(result['StatisticSearch']['list_total_count'])
        list_count=(int)(list_total_count/100) + 1

        rows=[]
        for i in range(0,list_count):
            
            start = str(i * 100 + 1)
            end = str((i + 1) * 100)
            
            url = "http://ecos.bok.or.kr/api/StatisticSearch/" + api_key + "/json/kr/" + start + "/" + end + "/" + "[STAT_CODE]/D/YYYYMMDD/YYYYMMDD/[SUB_CODE]"
        
            url = url.replace("[STAT_CODE]", "817Y002")
            url = url.replace("YYYYMMDD/YYYYMMDD", start_date + "/" + end_date)
            url = url.replace("[SUB_CODE]", "010300000")

            response = requests.get(url)
            data = response.json()
            
            rows += data['StatisticSearch']['row']
        
        Company_rates = pd.DataFrame(rows)

        return Company_rates


    ### 보안유의!!
    api_key = 'C7GJE1C09ADYZNF4MH30'

    # 금리
    rate_start_date = "20240101"
    rate_end_date = "20240531"
    CD_rates = get_CD_interest_rate(api_key, rate_start_date, rate_end_date)
    Korea_rates = get_Korea_interest_rate(api_key, rate_start_date, rate_end_date)
    Company_rates = get_Company_interest_rate(api_key, rate_start_date, rate_end_date)

    ## 금리
    labels = [CD_rates.iloc[0,3], Korea_rates.iloc[0,3], Company_rates.iloc[0,3]]
    values = [float(CD_rates.iloc[-1,-1]), float(Korea_rates.iloc[-1,-1]), float(Company_rates.iloc[-1,-1])]

    # 조회한 마지막날짜를 출력하도록 설계 --> datetime.today로 오늘날짜를 출력하도록 할수도 있음
    dates = [rate_end_date[0:4] + '-' + rate_end_date[4:6] + '-' + rate_end_date[6:]] * 3
    changes = [(float(CD_rates.iloc[-1,-1]) - float(CD_rates.iloc[-2,-1]))/float(CD_rates.iloc[-2,-1]),
                (float(Korea_rates.iloc[-1,-1]) - float(Korea_rates.iloc[-2,-1]))/float(Korea_rates.iloc[-2,-1]),
                (float(Company_rates.iloc[-1,-1]) - float(Company_rates.iloc[-2,-1]))/float(Company_rates.iloc[-2,-1])]

    # 플롯 생성
    fig = go.Figure()

    # 각 지표를 별도의 텍스트 박스로 추가
    for i, (label, value, date, change) in enumerate(zip(labels, values, dates, changes)):
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=value,
            number={'suffix': " "},
            delta={'reference': value, 'relative': True, 'valueformat': '.2%'},
            title={'text': f"{label}<br><span style='font-size:0.8em;color:gray'>{date} / 변동: {change}</span>"},
            domain={'row': i // 2, 'column': i % 2}
        ))

    # 그리드 레이아웃 설정
    fig.update_layout(
        grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
        height=600,
        margin=dict(l=20, r=20, t=30, b=30)
    )

    st.plotly_chart(fig, use_container_width=True)





with col1:
    ### KOSPI 100개기업 시총순위별로 사각형크기 나누고 주가와 당일 등락률표시

    # 크롤링
    kospi100 = fdr.StockListing('KOSPI')

    # 시가총액으로 정렬하고 상위 100개 선택
    kospi100 = kospi100.sort_values(by='Marcap', ascending=False).head(100)

    # 섹터 열 추가, 업종명만 엑셀파일 참고라 코스피에 대규모상장만 안하면 교체필요 없음
    sector_df = pd.read_excel("20240514_업종분류표.xlsx")

    sector_df.rename(columns={'종목코드':'Code'}, inplace=True)
    kospi100_df = pd.merge(kospi100, sector_df, on='Code')
    kospi100_df = pd.concat([kospi100_df.iloc[:,1:17], kospi100_df.iloc[:,19:20]],axis=1)
    kospi100_df.rename(columns={'업종명':'Sector'}, inplace=True)

    # 시가총액을 숫자형으로 변환
    kospi100_df['Marcap'] = kospi100_df['Marcap'].astype(float)

    # 함수 정의: 음수는 그대로, 양수는 +와 % 추가
    kospi100_df['ChagesRatio'] = np.where(kospi100_df['ChagesRatio']<0,
                                    kospi100_df['ChagesRatio'].astype(str) + "%",
                                    "+" + kospi100_df['ChagesRatio'].astype(str) + "%")

    # 섹터별로 데이터를 그룹화
    grouped = kospi100_df.groupby('Sector').sum()

    fig = px.treemap(kospi100_df, path=['Sector', 'Name', 'ChagesRatio'], values='Marcap',
                    color='Sector', hover_data=['Marcap'],
                    color_continuous_scale='Viridis', title="KOSPI Market Capitalization")
    st.plotly_chart(fig, use_container_width=True)

