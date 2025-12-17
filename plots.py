import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import acf

# -----------------------------
# 1️⃣ Read CSV
# -----------------------------
df = pd.read_csv("btc_features_talib.csv", encoding="utf-8-sig")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# -----------------------------
# 3️⃣ Time series plots
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['close_original'], color='blue')
plt.title('BTC Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.savefig("btc_price_timeseries.png")
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(df['date'], df['change_ptc'], color='green')
plt.title('BTC Daily Returns')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.grid(True)
plt.savefig("btc_returns_timeseries.png")
plt.close()

# -----------------------------
# 4️⃣ Histogram of returns
# -----------------------------
plt.figure(figsize=(8,5))
plt.hist(df['change_ptc'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of BTC Returns')
plt.xlabel('Daily Return (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig("btc_returns_histogram.png")
plt.close()

# -----------------------------
# 5️⃣ Autocorrelation of returns
# -----------------------------

autocorr = acf(df['change_ptc'], nlags = 50, fft = False)
lags = range(1, len(autocorr))
autocorr = autocorr[1:]

plt.figure(figsize=(10,5))
plt.stem(lags, autocorr, linefmt='skyblue', markerfmt='o', basefmt='k-')
plt.axhline(0, color='black', linewidth=1) 
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of BTC Returns')
plt.savefig("btc_returns_autocorr.png")
plt.close()

# -----------------------------
# 6️⃣ Feature correlation matrix
# -----------------------------
features_to_corr = df.columns[1:73]
corr = df[features_to_corr].corr()

plt.figure(figsize=(20,18))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.xticks(rotation=90, fontsize=6)  # rotate x labels vertically, make font smaller
plt.yticks(rotation=0, fontsize=6)   # keep y labels horizontal, small font
plt.title('Feature Correlation Matrix')
plt.savefig("btc_feature_correlation.png")
plt.close()

print("All exploratory figures saved as PNGs in the current directory")
