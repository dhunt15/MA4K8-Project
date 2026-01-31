import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV

# --- Load your data ---
df_train = pd.read_csv("train.csv", encoding="utf-8-sig")
df_val   = pd.read_csv("validation.csv", encoding="utf-8-sig")
df_test = pd.read_csv("test.csv", encoding = "utf-8-sig")

feature_cols = [col for col in df_train.columns 
                if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 
                'change_ptc', 'theta']]
volatility_val = df_val['recursive_volatility']
volatility_test = df_test['recursive_volatility']
returns_val = df_val['change_ptc']/100
returns_test = df_test['change_ptc']/100


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

theta_bins = np.array([-100, -5, -3, -1, -0.2, 0.2, 1, 3, 5, 100]) / 100

theta_midpoints = np.array([-15, -4, -2, -0.6, 0, 0.6, 2, 4, 15])/100  # for plug-in decision rule

#Cross-validation

from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

def evaluate_prediction(P, y_true):
    """
    P: (n_samples, n_classes) predicted probabilities
    y_true: integer class labels
    """
    metrics = {}

    metrics["LogLoss"] = log_loss(y_true, P, labels = np.arange(len(theta_midpoints)))

    # Expected return error (regression-style)
    y_true_ret = theta_midpoints[y_true]
    y_pred_ret = P @ theta_midpoints
    metrics["MSE"] = np.mean((y_pred_ret - y_true_ret) ** 2)

    #Get rid of theta0 bin
    y_true_sign = np.sign(y_true_ret)
    mask = y_true_sign != 0
    if mask.sum() == 0:
        metrics["Directional_Acc"] = np.nan
        metrics["ROC_AUC"] = np.nan
        return metrics

    # Directional accuracy
    metrics["Directional_Acc"] = np.mean(
        np.sign(y_pred_ret[mask]) == np.sign(y_true_ret[mask])
    )

    #ROC AUC
    p_up = P[:, theta_midpoints > 0].sum(axis = 1)
    y_up = (y_true_sign[mask] > 0).astype(int)
    metrics["ROC_AUC"] = roc_auc_score(y_up, p_up[mask])

    return metrics


def walk_forward_cv_rf_predictive(
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
            criterion="entropy",
            random_state=42,
            n_jobs=1
        )

    n = len(df)

    for start in range(0, n - train_size - val_size + 1, step):
        train_idx = slice(start, start + train_size)
        val_idx   = slice(start + train_size, start + train_size + val_size)

        df_train = df.iloc[train_idx]
        df_val   = df.iloc[val_idx]

        X_train = df_train[feature_cols]
        y_train = combine_bins(df_train["theta"]) + 4
        X_val   = df_val[feature_cols]
        y_val   = combine_bins(df_val["theta"]) + 4

        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)

        P_raw = rf.predict_proba(X_val)
        P = np.zeros((len(X_val), len(theta_midpoints)))

        for i, cls in enumerate(rf.classes_):
            P[:, cls] = P_raw[:, i]

        pred_metrics = evaluate_prediction(P, y_val)

        results.append(pred_metrics)

    return pd.DataFrame(results)

def evaluate_rf_params(params, full_df, feature_cols, theta_midpoints,
                       train_size=500, val_size=50, step=50):
    """Train and evaluate RF for a single parameter combination."""
    cv_res = walk_forward_cv_rf_predictive(
        full_df,
        feature_cols,
        theta_midpoints,
        train_size=train_size,
        val_size=val_size,
        step=step,
        rf_params=params
    )
    return {
        **params,
        "LogLoss_median": cv_res["LogLoss"].median(),
        "MSE_median": cv_res["MSE"].median(),
        "DirectionalAcc_mean": cv_res["Directional_Acc"].mean(),
        "ROC_AUC_mean": cv_res["ROC_AUC"].mean()
    }

def tune_rf_parallel(rf_grid, full_df, feature_cols, theta_midpoints,
                     train_size=500, val_size=50, step=50, n_jobs=-1):
    # Generate all parameter combinations
    param_list = list(ParameterGrid(rf_grid))

    # Run in parallel
    rf_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_rf_params)(params, full_df, feature_cols, theta_midpoints,
                                    train_size, val_size, step)
        for params in param_list
    )

    # Convert to DataFrame
    rf_df = pd.DataFrame(rf_results)

    # Select best row by LogLoss then MSE
    best_row = rf_df.sort_values(by=["ROC_AUC_mean", "DirectionalAcc_mean"], 
        ascending=[False, False]).iloc[0]

    max_depth_val = best_row["max_depth"]
    max_depth = None if pd.isna(max_depth_val) else int(max_depth_val)

    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": max_depth,
        "max_features": best_row["max_features"],
        "criterion": best_row["criterion"],
        "min_samples_split": int(best_row["min_samples_split"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"])
    }

    print(f"Best RF parameters:")
    print(f"n_estimators = {best_params['n_estimators']}, "
          f"max_depth = {best_params['max_depth']}, "
          f"max_features = {best_params['max_features']}, "
          f"criterion = {best_params['criterion']}, "
          f"min_samples_split = {best_params['min_samples_split']}, "
          f"min_samples_leaf = {best_params['min_samples_leaf']}"
        )
    
    return best_params, rf_df


rf_grid = {
    "n_estimators": int(100),
    "max_depth": None,
    "max_features":0.8, 
    "criterion": "entropy",
    "min_samples_split": 5,
    "min_samples_leaf": 2
}

full_df = pd.concat([df_train, df_test])
#best_rf_params, rf_df = tune_rf_parallel(rf_grid, full_df, feature_cols, theta_midpoints, 
#    train_size = 500, val_size = 50, step = 50, n_jobs=12)

best_rf_params = rf_grid

#Random forest model
rf = RandomForestClassifier(**best_rf_params,random_state = 42, n_jobs = 1)
rf.fit(X_train, y_train)

#Calibrated random forest model

rf_cal = CalibratedClassifierCV(rf, method = 'isotonic', cv=5, n_jobs = -1)
rf_cal.fit(X_train, y_train)

P_val = rf.predict_proba(X_val)
P_val_cal = rf_cal.predict_proba(X_val)
P = np.zeros((len(X_val), len(theta_midpoints)))
P_cal = np.zeros((len(X_val), len(theta_midpoints)))

for i, cls in enumerate(rf.classes_):
    P[:, cls] = P_val[:, i]
    P_cal[:, cls] = P_val_cal[:,i]

print("Validation metrics: ", evaluate_prediction(P, y_val) )
print("Calibrated validation metrics: ", evaluate_prediction(P_cal, y_val))

#Find best alpha, lambda gamma

def compute_actions(P_theta_given_X, theta_midpoints, volatility,
                    lam=0.1, gamma=0.001, alpha=0.5,
                    signal_scale=50.0, delta = 0.1, a0=0.5):
    actions = []
    a_prev = a0
    a_grid = np.linspace(0.0, 1.0, 101)  # active adjustment

    for t, p_row in enumerate(P_theta_given_X):
        sigma = volatility.iloc[t]
        # Bayesian expected return
#        mu = np.dot(p_row, theta_midpoints)
        utility = (
            alpha * np.log(1 + np.outer(a_grid, theta_midpoints)) +
            (1 - alpha) * np.outer(a_grid, theta_midpoints)
        )

        EU = utility @ p_row

        # Loss over action grid
        loss = (
#            - signal_scale * mu * (a_grid - a0)
            - signal_scale * EU
            + lam * sigma * (a_grid - a0) ** 2 
            + gamma * (a_grid - a_prev) ** 2 - delta * np.log(1e-4 + a_grid)
        )

        best_a = a_grid[np.argmin(loss)]
        actions.append(best_a)
        a_prev = best_a

    return np.array(actions)


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

def evaluate(wealth, actions, returns):
    daily_returns = np.diff(wealth) / wealth[:-1]

    sharpe = (
        np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.25)
        if np.std(daily_returns) > 0 else 0.0
    )

    cum_max = np.maximum.accumulate(wealth)
    max_drawdown = np.max((cum_max - wealth) / cum_max)
    
    da = np.diff(actions)
    r = returns.iloc[1:].values
    hit_rate = np.mean(da*r>0)

    buy_hit = np.mean((da>0) & (r>0))
    sell_hit = np.mean((da<0) & (r<0))

    return {
        "Final Wealth": wealth[-1],
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate,
        "Buy Hit": buy_hit,
        "Sell Hit": sell_hit
    }


alpha_list  = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
lambda_list = [0.1, 0.5, 1.0, 2]
gamma_list  = [0.0, 0.001, 0.01, 0.1]
scale_list  = [1, 5,10, 20]
delta_list = [0.1, 0.2, 0.3]


best_metrics = None
best_params  = None

best_cal_metrics = None
best_cal_params = None

results = []

def evaluate_decision_params(params, P_val, P_val_cal, theta_midpoints,
    volatility_val, returns_val):
    
    alpha, lam, gamma, scale, delta = params

    # Uncalibrated
    actions_uncal = compute_actions(
        P_val, theta_midpoints, volatility_val,
        alpha=alpha, lam=lam, gamma=gamma, signal_scale=scale, delta = delta
    )
    wealth_uncal = backtest(actions_uncal, returns_val)
    metrics_uncal = evaluate(wealth_uncal, actions_uncal, returns_val)

    # Calibrated
    actions_cal = compute_actions(
        P_val_cal, theta_midpoints, volatility_val,
        alpha=alpha, lam=lam, gamma=gamma, signal_scale=scale, delta = delta
    )
    wealth_cal = backtest(actions_cal, returns_val)
    metrics_cal = evaluate(wealth_cal, actions_cal, returns_val)

    return {
        "alpha": alpha,
        "lam": lam,
        "gamma": gamma,
        "signal_scale": scale,
        "delta": delta,

        "Sharpe_uncal": metrics_uncal["Sharpe"],
        "Sharpe_cal": metrics_cal["Sharpe"],

        "FinalWealth_uncal": metrics_uncal["Final Wealth"],
        "FinalWealth_cal": metrics_cal["Final Wealth"]
    }

from itertools import product

param_grid = list(product(alpha_list, lambda_list, gamma_list, scale_list, delta_list))

results = Parallel(n_jobs=12, backend="loky")(
    delayed(evaluate_decision_params)(
        params,
        P_val,
        P_val_cal,
        theta_midpoints,
        volatility_val,
        returns_val
    )
    for params in param_grid
)

results_df = pd.DataFrame(results)

best_uncal = results_df.sort_values("Sharpe_uncal", ascending=False).iloc[0]
best_cal   = results_df.sort_values("Sharpe_cal", ascending=False).iloc[0]

best_uncal_params = best_uncal[["alpha","lam","gamma","signal_scale", "delta"]].to_dict()
best_cal_params   = best_cal[["alpha","lam","gamma","signal_scale", "delta"]].to_dict()
print("Best uncalibrated params: ", best_uncal_params)
print("Best calibrated params: ", best_cal_params)

# Indices of positive-return bins
pos_idx = np.where(theta_midpoints > 0)[0]

# Uncalibrated probability of positive return
p_pos = P[:, pos_idx].sum(axis=1)

# Calibrated probability of positive return
p_pos_cal = P_cal[:, pos_idx].sum(axis=1)

# True outcome
y_pos = (theta_midpoints[y_val] > 0).astype(int)

from sklearn.calibration import calibration_curve

def plot_reliability(p, y, label, n_bins=10):
    frac_pos, mean_pred = calibration_curve(
        y, p, n_bins=n_bins, strategy="quantile"
    )
    plt.plot(mean_pred, frac_pos, marker='o', label=label)

plt.figure(figsize=(6,6))

plot_reliability(p_pos, y_pos, label="Uncalibrated RF")
plot_reliability(p_pos_cal, y_pos, label="Calibrated RF")

# Perfect calibration line
plt.plot([0,1], [0,1], linestyle="--", color="black", alpha=0.7)

plt.xlabel("Predicted probability of positive return")
plt.ylabel("Empirical frequency")
plt.title("Reliability Diagram (Positive Return)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Reliability diagram")
plt.close()

from sklearn.metrics import brier_score_loss

brier_uncal = brier_score_loss(y_pos, p_pos)
brier_cal   = brier_score_loss(y_pos, p_pos_cal)

print(f"Brier score (uncalibrated): {brier_uncal:.4f}")
print(f"Brier score (calibrated):   {brier_cal:.4f}")


#Trade

actions = compute_actions(P, theta_midpoints, volatility_val,
                          lam = best_uncal_params["lam"], gamma = best_uncal_params["gamma"],
                          alpha = best_uncal_params["alpha"],
                          signal_scale = best_uncal_params["signal_scale"],
                          delta = best_uncal_params["delta"])


actions_cal = compute_actions(P_cal, theta_midpoints, volatility_val,
                          lam = best_cal_params["lam"], gamma = best_cal_params["gamma"], 
                          alpha = best_cal_params["alpha"], 
                          signal_scale = best_cal_params["signal_scale"],
                          delta = best_cal_params["delta"])

wealth = backtest(actions, df_val["change_ptc"] / 100)
wealth_cal = backtest(actions_cal, df_val["change_ptc"]/100)
metrics = evaluate(wealth, actions, df_val["change_ptc"] / 100)
metrics_cal = evaluate(wealth_cal, actions_cal, df_val["change_ptc"]/100)
print("Uncalibrated validation metrics: ", metrics)
print("Calibrated validation metrics: ", metrics_cal)

#Plot model

df_val['date'] = pd.to_datetime(df_val['date'], errors='coerce')
df_val = df_val.dropna(subset=['date'])

plt.figure(figsize=(12, 8))

plt.plot(df_val['date'], wealth, label = "Uncalibrated model performance", color = "blue")
plt.plot(df_val['date'], wealth_cal, label = "Calibrated model performance", color = "green")
plt.plot(df_val['date'], df_val['close_original']/df_val['close_original'][0] * 1000,
    label = "Bitcoin price", lw=2, color = "red")
plt.title("Cumulative Wealth – Random Forest Strategy")
plt.xlabel("Date")
plt.ylabel("Wealth ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("RF2.png")
plt.close()

#Run on test data

# --- Step 1: Predict probabilities on test set ---
P_test_raw = rf.predict_proba(X_test)
P_test = np.zeros((len(X_test), len(theta_midpoints)))

for i, cls in enumerate(rf.classes_):
    P_test[:, cls] = P_test_raw[:, i]

P_test_cal_raw = rf_cal.predict_proba(X_test)
P_test_cal = np.zeros((len(X_test), len(theta_midpoints)))

for i, cls in enumerate(rf_cal.classes_):
    P_test_cal[:, cls] = P_test_cal_raw[:, i]

# --- Step 2: Compute actions using tuned alpha, lambda, gamma ---
actions_test = compute_actions(
    P_test,
    theta_midpoints,
    volatility_test,
    alpha=best_uncal_params["alpha"],
    lam=best_uncal_params["lam"],
    gamma=best_uncal_params["gamma"]
)

actions_test_cal = compute_actions(
    P_test_cal,
    theta_midpoints,
    volatility_test,
    alpha=best_cal_params["alpha"],
    lam=best_cal_params["lam"],
    gamma=best_cal_params["gamma"],
    signal_scale=best_cal_params["signal_scale"]
)

# --- Step 3: Backtest the strategy ---
returns_test = df_test["change_ptc"] / 100
wealth_test = backtest(actions_test, returns_test)
wealth_test_cal = backtest(actions_test_cal, returns_test)

# --- Step 4: Evaluate strategy ---
metrics_test = evaluate(wealth_test, actions_test, returns_test)
metrics_test_cal = evaluate(wealth_test_cal, actions_test_cal, returns_test)
print("Test set trading performance:", metrics_test)
print("Calibrated test metrics:", metrics_test_cal)

# --- Step 5: Plot cumulative wealth ---
df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')
df_test = df_test.dropna(subset=['date'])

plt.figure(figsize=(12,6))
plt.plot(df_test['date'], wealth_test, label="Uncalibrated strategy", color='blue')
plt.plot(df_test["date"], wealth_test_cal, label="Calibrated Strategy", color="green")
plt.plot(df_test['date'], df_test['close_original']/df_test['close_original'][0] * 1000,
    label = "Bitcoin price", lw=2, color = "red")
plt.xlabel("Date")
plt.ylabel("Wealth")
plt.title("Test Set Strategy Cumulative Wealth")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("RF2_test")
plt.close()

# --- Optional: plot actions vs returns ---
plt.figure(figsize=(12,4))
plt.plot(df_test['date'], actions_test, label="Uncalibrated Actions", color='orange')
plt.plot(df_test["date"], actions_test_cal, label="Calibrated Actions", color="purple")
plt.plot(df_test['date'], returns_test, label="Daily Return", color='green', alpha=0.5)
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Test Set Actions vs Returns")
plt.xticks(rotation=45)
plt.grid(True)
plt.legend()
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("RF2_actionsvreturns")
plt.close()

