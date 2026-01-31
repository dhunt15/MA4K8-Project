import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import ParameterGrid

# --- Load your data ---
df_train = pd.read_csv("train.csv", encoding="utf-8-sig")
df_val   = pd.read_csv("validation.csv", encoding="utf-8-sig")
df_test = pd.read_csv("test.csv", encoding = "utf-8-sig")

feature_cols = [col for col in df_train.columns 
                if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 
                'change_ptc', 'theta']]
volatility = df_val['recursive_volatility']
volatility_test = df_test['recursive_volatility']

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
X_test  = df_test[feature_cols]
y_test  = combine_bins(df_test['theta']) + 4

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
        loss += lam * (a**2) * sigma_t + gamma * (a - a_prev)**2
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

#Walk-Forward Cross Validation

def compute_actions(P_theta_given_X, theta_midpoints, volatility,
                    lam=0.1, gamma=0.001, alpha=0.1):

    actions_space = np.linspace(0.2, 1.0, 101)
    actions = []
    a_prev = 0.0

    for t, p_row in enumerate(P_theta_given_X):
        sigma_t = volatility.iloc[t]
        min_loss = np.inf
        best_a = 0.0

        for a in actions_space:
            expected_loss = np.sum(
                (-alpha * np.log(1 + a * theta_midpoints)
                 - (1 - alpha) * a * theta_midpoints) * p_row
            )

            total_loss = (expected_loss + lam * a**2 * sigma_t + gamma * (a - a_prev) ** 2
            )

            if total_loss < min_loss:
                min_loss = total_loss
                best_a = a

        actions.append(best_a)
        a_prev = best_a

    return np.array(actions)

def walk_forward_cv_random_forest(
    df,
    feature_cols,
    theta_midpoints,
    train_size=500,
    val_size=50,
    step=50,
    rf_params=None
):
    results = []

    if rf_params is None:
        rf_params = dict(
            n_estimators=100,
            criterion = 'entropy',
            max_depth=None,
            random_state=42,
            n_jobs=-1
        )

    n = len(df)

    for start in range(0, n - train_size - val_size + 1, step):
        train_idx = slice(start, start + train_size)
        val_idx   = slice(start + train_size, start + train_size + val_size)

        df_train = df.iloc[train_idx]
        df_val   = df.iloc[val_idx]

        X_train = df_train[feature_cols]
        y_train = combine_bins(df_train['theta']) + 4
        X_val   = df_val[feature_cols]
        y_val   = combine_bins(df_val['theta']) + 4
        volatility_val = df_val["recursive_volatility"]

        # --- Train RF ---
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)

        # --- Predict probabilities ---
        P_raw = rf.predict_proba(X_val)
        P = np.zeros((len(X_val), len(theta_midpoints)))

        for i, cls in enumerate(rf.classes_):
            P[:, cls] = P_raw[:, i]

        # --- Decision rule ---
        actions = compute_actions(
            P,
            theta_midpoints,
            volatility_val
        )

        # --- Backtest ---
        returns = df_val["change_ptc"] / 100.0
        wealth = backtest(actions, returns)
        metrics = evaluate(wealth, actions, y_val)

        metrics["train_start"] = df_train["date"].iloc[0]
        metrics["train_end"]   = df_train["date"].iloc[-1]
        metrics["val_start"]   = df_val["date"].iloc[0]
        metrics["val_end"]     = df_val["date"].iloc[-1]

        results.append(metrics)

    return pd.DataFrame(results)

full_df = pd.concat([df_train, df_val]).reset_index(drop=True)

cv_results = walk_forward_cv_random_forest(
    full_df,
    feature_cols,
    theta_midpoints,
    train_size=500,
    val_size=50,
    step=50
)

print(cv_results)
print(cv_results[["Final Wealth", "Sharpe"]].describe())

print(rf.feature_importances_)
P_theta_given_X.mean(axis=0)
np.unique(actions)

#rf_grid = {
#    "n_estimators": [50, 100, 200],
#    "max_depth": [5, 8, None],
#    "max_features": ["sqrt", "log2", 0.8]
#}

#rf_grid1 = {
#    "n_estimators": [100],
#    "max_depth": [5],
#   "max_features": [0.8],
#    "min_samples_split": [2,5,10],
#    "min_samples_leaf": [1,2,5]
#}

#cv_results_list = []

#for params in ParameterGrid(rf_grid):
#    cv_res = walk_forward_cv_random_forest(
#        full_df,
#        feature_cols,
#        theta_midpoints,
#        train_size=500,
#        val_size=50,
#        step=50,
#        rf_params=params
#    )
    # Add the hyperparameters to the results
#    cv_res["n_estimators"] = params["n_estimators"]
#    cv_res["max_depth"] = params["max_depth"]
#    cv_res["max_features"] = params["max_features"]
    
#    cv_results_list.append(cv_res)
#    print(f"Parameters {params} done")
#for params in ParameterGrid(rf_grid1):
#    cv_res = walk_forward_cv_random_forest(
#        full_df,
#        feature_cols,
#        theta_midpoints,
#        train_size=500,
#        val_size=50,
#        step=50,
#        rf_params=params
#    )
#    # Add the hyperparameters to the results
#    cv_res["min_samples_split"] = params["min_samples_split"]
#    cv_res["min_samples_leaf"] = params["min_samples_leaf"]

#    cv_results_list.append(cv_res)
#    print(f"Parameters {params} done")


# Combine all CV results
#cv_results_all = pd.concat(cv_results_list, ignore_index=True)

# Average Sharpe for each hyperparameter combination
#best_idx_Sharpe = cv_results_all.groupby(
#    ["n_estimators", "max_depth", "max_features"])["Sharpe"].mean().idxmax()
#best_n_estimators, best_max_depth, best_max_features = best_idx_Sharpe

#best_idx_Sharpe = (
#    cv_results_all.groupby(["min_samples_split", "min_samples_leaf"])["Sharpe"]
#    .mean()).idxmax()
#best_min_samples_split, best_min_samples_leaf = best_idx_Sharpe

#Average Wealth for each hyperparameter combination
#best_idx_w = cv_results_all.groupby(
#    ["n_estimators", "max_depth", "max_features"])["Final Wealth"].mean().idxmax()
#best_n_estimators_w, best_max_depth_w, best_max_features_w = best_idx_w

#best_idx_w =  (
#    cv_results_all.groupby(["min_samples_split", "min_samples_leaf"])["Final Wealth"]
#    .mean()).idxmax()
#best_min_samples_split_w, best_min_samples_leaf_w = best_idx_w

#print("Best RF hyperparameters for Sharpe from CV:")
#print("n_estimators:", best_n_estimators)
#print("max_depth:", best_max_depth)
#print("max_features", best_max_features)
#print("min_samples_split", best_min_samples_split)
#print("min_samples_leaf", best_min_samples_leaf)

#print("Best RF hyperparameters for Final Wealth from CV:")
#print("n_estimators:", best_n_estimators_w)
#print("max_depth:", best_max_depth_w)
#print("max_features", best_max_features_w)
#print("min_samples_split", best_min_samples_split_w)
#print("min_samples_leaf", best_min_samples_leaf_w)

#rf_cv =  RandomForestClassifier( n_estimators = best_n_estimators, 
#    max_depth = best_max_depth, max_features = best_max_features, 
#    criterion = "entropy", n_jobs = -1, random_state = 42)

rf_cv =  RandomForestClassifier( n_estimators = 100,
    max_depth = 5, max_features = 0.8, min_samples_split = 5,
    min_samples_leaf = 1, criterion = "entropy", 
    n_jobs = -1, random_state = 42)


#rf_w =  RandomForestClassifier( n_estimators = best_n_estimators_w,
#    max_depth = best_max_depth_w, max_features = best_max_features_w,
#    criterion = "entropy", n_jobs = -1, random_state = 42)

rf_w =  RandomForestClassifier( n_estimators = 200,
    max_depth = 5, max_features = 0.8, min_samples_split = 5,
    min_samples_leaf = 5, criterion = "entropy",
    n_jobs = -1, random_state = 42)

from sklearn.calibration import CalibratedClassifierCV

rf_cv_cal = CalibratedClassifierCV(rf_cv, method = "sigmoid", cv=3)
rf_w_cal = CalibratedClassifierCV(rf_w, method = "sigmoid", cv=3)


rf_cv_cal.fit(X_train, y_train)
rf_w_cal.fit(X_train, y_train)

# --- Predict probability distribution ---
P_theta_given_X_raw_cv = rf_cv_cal.predict_proba(X_val)  # shape: (n_samples, n_classes)
P_theta_given_X_cv = np.zeros((len(X_val), len(theta_midpoints)))

P_theta_given_X_raw_w = rf_w_cal.predict_proba(X_val)  # shape: (n_samples, n_classes)
P_theta_given_X_w = np.zeros((len(X_val), len(theta_midpoints)))

for i, cls in enumerate(rf_cv_cal.classes_):
    P_theta_given_X_cv[:, cls] = P_theta_given_X_raw_cv[:, i]

for i, cls in enumerate(rf_w_cal.classes_):
    P_theta_given_X_w[:, cls] = P_theta_given_X_raw_w[:, i]

hyperparameters = {
    "alpha": [0.1, 0.15, 0.2, 0.25],
    "lam": [0.05, 0.1, 0.15, 0.2],
    "gamma": [0.0001, 0.0005, 0.001]
}

results_hyp = []

for params in ParameterGrid(hyperparameters):
    alpha = params["alpha"]
    lam = params["lam"]
    gamma = params["gamma"]

    actions_cv = compute_actions(P_theta_given_X_cv, theta_midpoints, volatility,
                        lam, gamma, alpha)
    actions_w =  compute_actions(P_theta_given_X_w, theta_midpoints, volatility,
                        lam, gamma, alpha)

    returns = df_val["change_ptc"] / 100.0

    wealth_cv = backtest(actions_cv, returns)
    wealth_w  = backtest(actions_w, returns)

    metrics_cv = evaluate(wealth_cv, actions_cv, returns)
    metrics_w  = evaluate(wealth_w, actions_w, returns)

    turnover_cv = np.mean(np.abs(np.diff(actions_cv)))
    turnover_w  = np.mean(np.abs(np.diff(actions_w)))

    results_hyp.append({
        "alpha": alpha,
        "lam": lam,
        "gamma": gamma,

        "Sharpe_cv": metrics_cv["Sharpe"],
        "MaxDD_cv": metrics_cv["Max Drawdown"],
        "FinalWealth_cv": metrics_cv["Final Wealth"],
        "Turnover_cv": turnover_cv,

        "Sharpe_w": metrics_w["Sharpe"],
        "FinalWealth_w": metrics_w["Final Wealth"],
    })

    print(f"Done alpha = {alpha}, lambda = {lam}, gamma = {gamma}")

df_hyp = pd.DataFrame(results_hyp)

best_sharpe_row = df_hyp.loc[df_hyp["Sharpe_cv"].idxmax()]

print("Best parameters for Sharpe:")
print(f"alpha  = {best_sharpe_row['alpha']}")
print(f"lambda = {best_sharpe_row['lam']}")
print(f"gamma  = {best_sharpe_row['gamma']}")
print(f"Sharpe = {best_sharpe_row['Sharpe_cv']:.4f}")
print(f"Final Wealth = {best_sharpe_row['FinalWealth_cv']:.4f}")

best_wealth_row = df_hyp.loc[df_hyp["FinalWealth_w"].idxmax()]

print("\nBest parameters for Final Wealth:")
print(f"alpha        = {best_wealth_row['alpha']}")
print(f"lambda       = {best_wealth_row['lam']}")
print(f"gamma        = {best_wealth_row['gamma']}")
print(f"Sharpe = {best_wealth_row['Sharpe_w']:.4f}")
print(f"Final Wealth = {best_wealth_row['FinalWealth_w']:.4f}")


#Run forests

returns = df_val["change_ptc"] / 100.0

strategies = {
    "Before CV": actions,
    "After CV (Sharpe)": compute_actions(P_theta_given_X_cv, theta_midpoints, volatility,
                    lam=best_sharpe_row['lam'], gamma=best_sharpe_row['gamma'], 
                    alpha=best_sharpe_row['alpha']
                    ),
    "After CV (Final Wealth)": compute_actions(P_theta_given_X_w, theta_midpoints, volatility,
                    lam=best_wealth_row['lam'], gamma=best_wealth_row['gamma'], 
                    alpha=best_wealth_row['alpha']
                    )
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

plt.plot(df_val['date'], df_val['close_original']/df_val['close_original'][0] * 1000, label = "Bitcoin price", lw=2)
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

#Run on test data
returns_test = df_test['change_ptc'] / 100

P_theta_given_X_test = rf_cv_cal.predict_proba(X_test)
P_theta_given_X = np.zeros((len(X_test), len(theta_midpoints)))
for i, cls in enumerate(rf_cv_cal.classes_):
    P_theta_given_X[:, cls] = P_theta_given_X_test[:,i]

actions_test_sharpe = compute_actions(
    P_theta_given_X, theta_midpoints, volatility_test,
    lam=best_sharpe_row['lam'], alpha=best_sharpe_row['alpha']
)

wealth_test_sharpe = backtest(actions_test_sharpe, returns_test)

results_test = pd.DataFrame({"Sharpe Strategy": evaluate(
    wealth_test_sharpe, actions_test_sharpe, returns_test)}).T
print(results_test)

df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')
df_test = df_test.dropna(subset=['date'])

plt.figure(figsize=(12,8))
plt.plot(df_test['date'], wealth_test_sharpe, label="Sharpe Strategy")
plt.plot(df_test['date'], df_test['close_original']/df_test['close_original'][0]*1000, label="Buy & Hold")
plt.xlabel("Date")
plt.ylabel("Wealth")
plt.title("Test Set Backtest")
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("RF test.png")
plt.close()
