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
full_df  = pd.concat([train_df, val_df]).reset_index(drop=True)

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
full_df["change_ptc"] = full_df["change_ptc"]/100.0
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
        actions[t] = 1.0 if returns.iloc[t-1] > 0 else 0.0
    return actions

#Linear Regression

pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_val)
actions_lr = np.where(y_pred > 0, 1.0, 0.0)

#Logistic Regression

y_train_binary = (y_train > 0).astype(int)
y_val_binary   = (y_val > 0).astype(int)

logit_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('logit', LogisticRegression(max_iter=5000, solver='lbfgs'))
])

logit_pipeline.fit(X_train, y_train_binary)

proba_val = logit_pipeline.predict_proba(X_val)[:, 1]  # probability of up move

actions_logit = np.where(proba_val > 0.5, 1.0, 0.0)


#Backtest

def backtest_baseline(actions, returns, capital=1_000, c=0.001):

    wealth = np.zeros(len(returns))
    wealth[0] = capital

    a_prev = 0.0

    for t in range(1, len(returns)):
        a_t = actions[t]
        r_t = returns.iloc[t]

        # portfolio update with transaction costs
        wealth[t] = wealth[t-1] * (1 + a_t * r_t - c * abs(a_t - a_prev))

        a_prev = a_t

    return wealth

#Evaluate

def evaluate_baseline(wealth, actions, returns):
    daily_returns = np.diff(wealth) / wealth[:-1]

    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.25)
        if np.std(daily_returns) > 0 else 0.0
    )

    cum_max = np.maximum.accumulate(wealth)
    max_drawdown = np.max((cum_max - wealth) / cum_max)

    hit_rate = np.mean(
        np.sign(actions[:-1]) == np.sign(returns.iloc[1:])
    )

    return {
        "Final Wealth": wealth[-1],
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate
    }

#Run baselines 

returns = val_df["change_ptc"] / 100.0

strategies = {
    "Always Buy": always_buy(len(returns)),
    "Always Hold": always_hold(len(returns)),
    "Last Return": last_return_rule(returns),
    "Linear Regression": actions_lr,
    "Logistic Regression": actions_logit
}

results = {}

for name, actions in strategies.items():
    wealth = backtest_baseline(actions, returns)
    results[name] = evaluate_baseline(wealth, actions, returns)

baseline_df = pd.DataFrame(results).T
print(baseline_df)
