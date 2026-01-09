import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import matplotlib.dates as mdates

# Load splits
train_df = pd.read_csv("train.csv", encoding="utf-8-sig")
val_df   = pd.read_csv("validation.csv", encoding="utf-8-sig")
test_df  = pd.read_csv("test.csv", encoding="utf-8-sig")

# Feature columns
feature_cols = [
    col for col in train_df.columns
    if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 'change_ptc', 'theta', 'close_original', 'recursive_volatility']
]

X_train = train_df[feature_cols]
X_val   = val_df[feature_cols]
X_test  = test_df[feature_cols]

y_train = train_df['change_ptc'] / 100.0   # convert to decimal returns
y_val   = val_df['change_ptc'] / 100.0
y_test  = test_df['change_ptc'] / 100.0
sigma2_val = val_df['recursive_volatility']

def loss(realized_return, a_t, a_prev, sigma2_t, lam=0.1, c=0.001):
    log_term = -np.log(1 + a_t * realized_return)
    risk_term = lam * (a_t ** 2) * sigma2_t
    trans_term = c * np.abs(a_t - a_prev)
    return log_term + risk_term + trans_term

#Naive baseline 

def always_buy(n):
    return np.ones(n)

def always_hold(n):
    return np.zeros(n)

#Last return baseline

def last_return_rule(returns):
    actions = np.zeros(len(returns))
    for t in range(1, len(returns)):
        actions[t] = np.sign(returns.iloc[t-1])
    return actions

#Linear Regression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_val)
actions_lr = np.clip(np.sign(y_pred_lr), 0, 1)

#Logistic Regression

logit = Pipeline([
    ("scaler", StandardScaler()),
    ("logit", LogisticRegression(
        max_iter=5000,
        solver="lbfgs"
    ))
])
logit.fit(X_train, (y_train > 0).astype(int))

proba_up = logit.predict_proba(X_val)[:, 1]
actions_logit = np.where(proba_up > 0.5, 1.0, 0.0)

#Backtest

def backtest(actions, returns, sigma2, capital=1_000, lam=0.1, c=0.001):
    wealth = np.zeros(len(returns))
    losses = np.zeros(len(returns))

    wealth[0] = capital
    a_prev = 0.0

    for t in range(len(returns)):
        a_t = actions[t]
        r_t = returns.iloc[t]

        losses[t] = loss(r_t, a_t, a_prev, sigma2[t], lam, c)
        wealth[t] = wealth[t-1] * (1 + a_t * r_t - c * abs(a_t - a_prev)) if t > 0 else capital

        a_prev = a_t

    return wealth, losses

#Evaluation metrics

def evaluate(wealth, actions, returns):
    daily_returns = np.diff(wealth) / wealth[:-1]

    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns)
        * np.sqrt(252)
        if np.std(daily_returns) > 0 else 0.0
    )

    cum_max = np.maximum.accumulate(wealth)
    drawdown = np.max((cum_max - wealth) / cum_max)

    hit_rate = np.mean(
        np.sign(actions[:-1]) == np.sign(returns.iloc[1:])
    )

    return {
        "Final Wealth": wealth[-1],
        "Sharpe": sharpe,
        "Max Drawdown": drawdown,
        "Hit Rate": hit_rate
    }

#Run baselines 

strategies = {
    "Always Buy": always_buy(len(y_val)),
    "Always Hold": always_hold(len(y_val)),
    "Last Return": last_return_rule(y_val),
    "Linear Regression": actions_lr,
    "Logistic Regression": actions_logit
}

results = {}

for name, actions in strategies.items():
    wealth, losses = backtest(actions, y_val, sigma2_val)
    results[name] = evaluate(wealth, actions, y_val)

#Results

val_df['date'] = pd.to_datetime(val_df['date'], errors='coerce')
val_df = val_df.dropna(subset=['date'])

results_df = pd.DataFrame(results).T
print(results_df)

plt.figure(figsize=(12, 8))

for name, actions in strategies.items():
    wealth, _ = backtest(actions, y_val, sigma2_val)
    plt.plot(val_df['date'], wealth, label=name)

plt.legend()
plt.title("Cumulative Wealth – Baseline Strategies")
plt.xlabel("Date")
plt.ylabel("Wealth ($)")
plt.grid(alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)  
plt.tight_layout()
plt.savefig("baseline_wealth.png")
plt.close()

