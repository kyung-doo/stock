import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import math
import json
import random
import tensorflow as tf
import plotly.graph_objs as go

# ✅ 랜덤 시드 고정
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ✅ 최근 거래일 찾기
def get_last_trading_day():
    today = datetime.date.today()
    tmp = yf.download("^KS11", start=today - datetime.timedelta(days=10), end=today,
                      progress=False, auto_adjust=False)
    return tmp.index[-1].date()

end = get_last_trading_day()
start = end - datetime.timedelta(days=365*3)
kospi = yf.download("^KS11", start=start, end=end, progress=False, auto_adjust=False)

# 종가만 Series로 추출
df = kospi[['Close']].squeeze()   # ✅ 항상 Series로 변환

# 정규화
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(df.values.reshape(-1,1))

# 시계열 데이터셋 생성 (7일 예측)
def create_dataset(data, time_step=20, future_days=7):
    X, y = [], []
    for i in range(len(data)-time_step-future_days):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step : i+time_step+future_days, 0])
    return np.array(X), np.array(y)

time_step = 20
future_days = 7
X, y = create_dataset(scaled, time_step, future_days)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 학습/테스트 분리
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM 모델 정의
model = Sequential()
model.add(tf.keras.Input(shape=(time_step,1)))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(future_days))
model.compile(loss='mean_squared_error', optimizer='adam')

# 학습
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 예측
test_pred = model.predict(X_test, verbose=0)
test_pred = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test)

# RMSE
rmse = math.sqrt(mean_squared_error(y_test_inv, test_pred))

# 오늘 종가 + 향후 7 거래일 예측
todayClose = float(df.iloc[-1].item())
last_data = scaled[-time_step:]
last_data = last_data.reshape(1, time_step, 1)
next_pred = model.predict(last_data, verbose=0)
weekClose = scaler.inverse_transform(next_pred)[0].tolist()

# 최근 3개월 실제 vs 예측 그래프
three_months = df.index[-65:]
actual_recent = df.iloc[-65:]
pred_recent = test_pred[-65:, 0]

trace_actual = go.Scatter(
    x=[d.strftime("%Y년 %m월 %d일") for d in actual_recent.index],  # ✅ 한글 날짜
    y=actual_recent.tolist(),
    mode='lines',
    name='실제 KOSPI',
    line=dict(color='blue')
)

trace_pred = go.Scatter(
    x=[d.strftime("%Y년 %m월 %d일") for d in three_months],        # ✅ 한글 날짜
    y=pred_recent.tolist(),
    mode='lines',
    name='예측 KOSPI',
    line=dict(color='red', dash='dash')
)

layout = go.Layout(
    title=dict(text="최근 3개월 KOSPI 실제 vs 예측"),
    xaxis=dict(title=dict(text="날짜")),
    yaxis=dict(title=dict(text="지수")),
    template="plotly_white",
    font=dict(family="Malgun Gothic", size=12)  # ✅ 한글 폰트 지정
)

fig = go.Figure(data=[trace_actual, trace_pred], layout=layout)

# ✅ 안전 변환 함수
def safe_convert(obj):
    import numpy as np
    import pandas as pd
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, dict):
        return {k: safe_convert(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [safe_convert(v) for v in obj]
    return obj

plot_data = safe_convert(fig.to_plotly_json())

# ✅ JSON 출력
print(json.dumps({
    "rmse": round(rmse, 2),
    "todayClose": round(todayClose, 2),
    "weekClose": [round(x, 2) for x in weekClose],
    "chart": plot_data
}, ensure_ascii=False))
