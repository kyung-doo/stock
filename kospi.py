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

# ✅ 랜덤 시드 고정 (재현성 확보)
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ✅ 최근 거래일 + 하루 뒤로 end 설정
def get_last_trading_day():
    today = datetime.date.today()
    # 최근 10일치 데이터 받아서 마지막 거래일 찾기
    tmp = yf.download("^KS11", start=today - datetime.timedelta(days=10), end=today)
    return tmp.index[-1].date()

end = get_last_trading_day()
start = end - datetime.timedelta(days=365*3)  # 최근 3년치 데이터
kospi = yf.download("^KS11", start=start, end=end)

# 2) 종가만 사용
df = kospi[['Close']].copy()

# 3) 데이터 정규화
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(df)

# 4) 시계열 데이터셋 생성 함수
def create_dataset(data, time_step=20):
    X, y = [], []
    for i in range(len(data)-time_step-1):
        X.append(data[i:(i+time_step), 0])
        y.append(data[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 20
X, y = create_dataset(scaled, time_step)

# 5) LSTM 입력 형태로 변환
X = X.reshape(X.shape[0], X.shape[1], 1)

# 6) 학습/테스트 분리
train_size = int(len(X)*0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 7) LSTM 모델 정의
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(time_step,1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 8) 학습 (로그 제거)
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 9) 예측
test_pred = model.predict(X_test, verbose=0)

# 10) 역정규화
test_pred = scaler.inverse_transform(test_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# 11) 평가 (RMSE)
rmse = math.sqrt(mean_squared_error(y_test_inv, test_pred))

# 12) 오늘 종가와 내일 종가 예측
todayClose = float(df['Close'].iloc[-1])  # 최근 거래일 종가
last_data = scaled[-time_step:]
last_data = last_data.reshape(1, time_step, 1)
next_pred = model.predict(last_data, verbose=0)
tomorrowClose = float(scaler.inverse_transform(next_pred)[0][0])  # 다음 거래일 예측 종가



# ✅ JSON 출력 (지수 포함)
print(json.dumps({
    "rmse": round(rmse, 2),
    "todayClose": round(todayClose, 2),
    "tomorrowClose": round(tomorrowClose, 2)
}, ensure_ascii=False))
