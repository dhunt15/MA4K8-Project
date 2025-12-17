import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import talib
import yfinance as yf
from sklearn.preprocessing import StandardScaler

btc = yf.Ticker("BTC-USD")

df = btc.history(period = "max")
df = df.reset_index()

df = df.drop(columns=['Dividends', 'Stock Splits'], errors='ignore')

df = df.rename(columns={
    "Date": "date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

df['change_ptc'] = df['close'].pct_change() * 100

numeric_cols = ['open','high','low','close','volume','change_ptc']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

open_ = df['open']
high  = df['high']
low   = df['low']
close = df['close']

features = {}

# === Trend and Moving Averages ===
features["DEMA"] = talib.DEMA(close, timeperiod=20)
features["EMA"] = talib.EMA(close, timeperiod=20)
features["HT_TRENDLINE"] = talib.HT_TRENDLINE(close)
features["KAMA"] = talib.KAMA(close, timeperiod=20)
features["MIDPOINT"] = talib.MIDPOINT(close, timeperiod=14)
features["MIDPRICE"] = talib.MIDPRICE(high, low, timeperiod=14)
features["SAR"] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
features["SAREXT"] = talib.SAREXT(high, low)
features["SMA3"] = talib.SMA(close, timeperiod=3)
features["SMA5"] = talib.SMA(close, timeperiod=5)
features["SMA10"] = talib.SMA(close, timeperiod=10)
features["SMA20"] = talib.SMA(close, timeperiod=20)
features["T3"] = talib.T3(close, timeperiod=5, vfactor=0.7)
features["TEMA"] = talib.TEMA(close, timeperiod=20)
features["TRIMA"] = talib.TRIMA(close, timeperiod=20)
features["WMA"] = talib.WMA(close, timeperiod=20)

# === Bollinger Bands ===
upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
features["BBAND_upper"] = upper
features["BBAND_middle"] = middle
features["BBAND_lower"] = lower
features["BBAND_width"] = (upper - lower) / middle
features["BBAND_upper_signal"] = (close > upper).astype(int)
features["BBAND_lower_signal"] = (close < lower).astype(int)

# === Momentum Indicators ===
features["ADX14"] = talib.ADX(high, low, close, timeperiod=14)
features["ADX20"] = talib.ADX(high, low, close, timeperiod=20)
features["ADXR"] = talib.ADXR(high, low, close, timeperiod=14)
features["AROONOSC"] = talib.AROONOSC(high, low, timeperiod=14)
features["BOP"] = talib.BOP(open_, high, low, close)
features["CCI3"] = talib.CCI(high, low, close, timeperiod=3)
features["CCI5"] = talib.CCI(high, low, close, timeperiod=5)
features["CCI10"] = talib.CCI(high, low, close, timeperiod=10)
features["CCI14"] = talib.CCI(high, low, close, timeperiod=14)
features["CMO"] = talib.CMO(close, timeperiod=14)
features["DX"] = talib.DX(high, low, close, timeperiod=14)

macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
features["MACD"] = macd
features["MACDSIGNAL"] = macdsignal
features["MACDHIST"] = macdhist

features["MINUS_DI"] = talib.MINUS_DI(high, low, close, timeperiod=14)
features["PLUS_DI"] = talib.PLUS_DI(high, low, close, timeperiod=14)
features["MINUS_DM"] = talib.MINUS_DM(high, low, timeperiod=14)
features["PLUS_DM"] = talib.PLUS_DM(high, low, timeperiod=14)

features["MOM1"] = talib.MOM(close, timeperiod=1)
features["MOM3"] = talib.MOM(close, timeperiod=3)
features["MOM5"] = talib.MOM(close, timeperiod=5)
features["MOM10"] = talib.MOM(close, timeperiod=10)
features["APO"] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
features["PPO"] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
features["ROC"] = talib.ROC(close, timeperiod=10)
features["ROCP"] = talib.ROCP(close, timeperiod=10)
features["ROCR"] = talib.ROCR(close, timeperiod=10)
features["ROCR100"] = talib.ROCR100(close, timeperiod=10)
features["RSI5"] = talib.RSI(close, timeperiod=5)
features["RSI10"] = talib.RSI(close, timeperiod=10)
features["RSI14"] = talib.RSI(close, timeperiod=14)

slowk, slowd = talib.STOCH(high, low, close)
features["SLOWK"] = slowk
features["SLOWD"] = slowd
fastk, fastd = talib.STOCHF(high, low, close)
features["FASTK"] = fastk
features["FASTD"] = fastd
features["TRIX"] = talib.TRIX(close, timeperiod=30)
features["ULTOSC"] = talib.ULTOSC(high, low, close)
features["WILLR"] = talib.WILLR(high, low, close)

features["ATR"] = talib.ATR(high, low, close)
features["NATR"] = talib.NATR(high, low, close)
features["TRANGE"] = talib.TRANGE(high, low, close)

features["HT_DCPERIOD"] = talib.HT_DCPERIOD(close)
features["HT_DCPHASE"] = talib.HT_DCPHASE(close)
features["HT_TRENDMODE"] = talib.HT_TRENDMODE(close)

candlestick_functions = {name: getattr(talib, name) for name in dir(talib) if name.startswith("CDL")}
for name, func in candlestick_functions.items():
    try:
        features[name] = func(open_, high, low, close)
    except Exception as e:
        print(f"Skipping {name}: {e}")

features_df = pd.DataFrame(features, index=df.index)

df = pd.concat([df, features_df], axis=1).copy()

intervals = [
    (-100, -11), (-11, -9), (-9, -7), (-7, -5), (-5, -3),
    (-3, -1), (-1, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2),
    (-0.2, 0.2),
    (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1),
    (1, 3), (3, 5), (5, 7), (7, 9), (9, 11), (11, np.inf)
]
labels = list(range(-10, 11))

def label_return(r):
    for (low, high), label in zip(intervals, labels):
        if low <= r < high:
            return label
    return 0

df['theta'] = df['change_ptc'].apply(label_return)

#Scale features

cols_to_scale = [
    'open','high','low','close','volume',
    'DEMA','EMA', 'HT_TRENDLINE','KAMA','MIDPOINT','MIDPRICE', 'SAR', 'SAREXT', 'SMA3',
    'SMA5','SMA10','SMA20',
    'T3','TEMA','TRIMA','WMA','BBAND_upper','BBAND_middle','BBAND_lower','BBAND_width',
    'BBAND_upper_signal','BBAND_lower_signal','ADX14','ADX20','ADXR','AROONOSC',
    'BOP','CCI3','CCI5','CCI10','CCI14','CMO','DX','MACD','MACDSIGNAL','MACDHIST',
    'MINUS_DI','PLUS_DI','MINUS_DM','PLUS_DM','MOM1','MOM3','MOM5','MOM10','APO',
    'PPO','ROC','ROCP','ROCR','ROCR100','TRIX','ATR','TRANGE'
]

scaler = StandardScaler()
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

#Drop rows with missing features

feature_cols = [col for col in df.columns if col != 'date']
df = df.dropna(subset=feature_cols).reset_index(drop=True)

#print to csv
df.to_csv("btc_features_talib.csv", index=False, encoding="utf-8-sig")
