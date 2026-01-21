import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Load your data ---
df_train = pd.read_csv("train.csv", encoding="utf-8-sig")
df_val   = pd.read_csv("validation.csv", encoding="utf-8-sig")

feature_cols = [col for col in df_train.columns 
                if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 
                'change_ptc', 'theta']]
volatility = df_val['recursive_volatility']

#Combine theta bins
def combine_bins(theta_series):
    new_theta = theta_series.copy()
    new_theta[(theta_series >= -10) & (theta_series <= -7)] = -4
    new_theta[theta_series == -6] = -3
    new_theta[theta_series == -5] = -2
    new_theta[(theta_series >= -4) & (theta_series <= -1)] = -1
    new_theta[(theta_series >= 1)  & (theta_series <= 4)]  = 1
    new_theta[theta_series == 5] = 2
    new_theta[theta_series == 6] = 3
    new_theta[(theta_series >= 7)  & (theta_series <= 10)] = 4

    return new_theta.astype(int)

X_train = df_train[feature_cols]
y_train = combine_bins(df_train['theta']) + 4 # categorical classes
X_val   = df_val[feature_cols]
y_val   = combine_bins(df_val['theta']) + 4

#theta_bins = np.array([-100, -11, -9, -7, -5, -3, -1, -0.8, -0.6, -0.4, -0.2, 
#                       0.2, 0.4, 0.6, 0.8, 1.0, 3.0, 5.0, 7.0, 9.0, 11, 100]) / 100.0

theta_bins = np.array([-100, -5, -3, -1, -0.2, 0.2, 1, 3, 5, 100]) / 100

theta_midpoints = np.array([-10, -4, -2, -0.6, 0, 0.6, 2, 4, 10])/100  # for plug-in decision rule

# --- Train Random Forest ---
rf = RandomForestClassifier(
    criterion = "entropy",
    n_estimators=50,
#    max_depth=8,
#    class_weight = "balanced",
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

# --- Predict probability distribution ---
P_theta_given_X_raw = rf.predict_proba(X_val)  # shape: (n_samples, n_classes)
P_theta_given_X = np.zeros((len(X_val), len(theta_midpoints)))
for i, cls in enumerate(rf.classes_):
    P_theta_given_X[:, cls] = P_theta_given_X_raw[:, i]

# P_theta_given_X.shape should be (n_samples, n_classes)
print("Shape:", P_theta_given_X.shape)
print("First row probabilities:", P_theta_given_X[0])
print("Sum of first row:", P_theta_given_X[0].sum())

y_pred_class = rf.predict(X_val)
print("Predicted classes (first 20):", y_pred_class[:20] - 4)
print("Unique classes predicted:", np.unique(y_pred_class) - 4)

import matplotlib.pyplot as plt

#importances = rf.feature_importances_
#plt.barh(feature_cols, importances)
#plt.title("Random Forest Feature Importances")
#plt.show()

#print(y_train.value_counts())


# --- Decision Rule ---
actions_space = np.arange(0.00, 1.01, 0.01)  # portfolio fraction 0%-100%
lam = 0.1
c   = 0.001
a_prev = 0.0
gamma = 0.001

actions = []
for t, p_row in enumerate(P_theta_given_X):
    # Expected next-day return
#    E_theta = np.sum(p_row * theta_midpoints)

    # Simple reactionary decision rule with penalties
    best_a = 0.0
    min_loss = np.inf
    sigma_t = volatility.iloc[t]

    for a in actions_space:
        # Expected loss: negative expected growth + volatility penalty + transaction cost
        # Reactionary: penalize opposite positions
        loss = (-0.1 * np.log(1+a * theta_midpoints) - 0.9 *(a*theta_midpoints)) * p_row
        loss += lam * (a**2) * (sigma_t**2) + gamma * (a - a_prev)**2
        total_loss = np.sum(loss)
        if total_loss < min_loss:
            min_loss = total_loss
            best_a = a

    actions.append(best_a)
    a_prev = best_a

print("Actions:", actions[:20])
# --- Backtest / Evaluate ---
#Backtest

def backtest(actions, returns, capital=1_000, c=0.001):

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

def evaluate(wealth, actions, returns):
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

returns = df_val["change_ptc"] / 100.0

strategies = {
    "Random Forest": actions
}

results = {}

for name, actions in strategies.items():
    wealth = backtest(actions, returns)
    results[name] = evaluate(wealth, actions, returns)

randomforest_df = pd.DataFrame(results).T
print(randomforest_df)

#Plot baselines

df_val['date'] = pd.to_datetime(df_val['date'], errors='coerce')
df_val = df_val.dropna(subset=['date'])

plt.figure(figsize=(12, 8))

for name, actions in strategies.items():
    wealth = backtest(actions, returns)
    plt.plot(df_val['date'], wealth, label=name)

plt.plot(df_val['date'], df_val['close_original']/df_val['close_original'][0] * 1000, label = "Bitcoin price")
plt.title("Cumulative Wealth – Random Forest Strategy")
plt.xlabel("Date")
plt.ylabel("Wealth ($)")
plt.legend()
plt.grid(alpha=0.3)

# Format x-axis to show months/years nicely
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("RF.png")
plt.close()
