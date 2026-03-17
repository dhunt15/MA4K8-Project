import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from xgboost import XGBClassifier

# --- Load your data ---
df_train = pd.read_csv("train.csv", encoding="utf-8-sig")
df_val   = pd.read_csv("validation.csv", encoding="utf-8-sig")
df_test = pd.read_csv("test.csv", encoding = "utf-8-sig")

#df_entire = pd.concat([df_train, df_val, df_test])
#df_train = df_entire.iloc[1478:2209] # 1/1/2019 - 1/1/2021
#df_test = df_entire.iloc[3308:3674]

feature_cols = [col for col in df_train.columns 
                if col not in ['date', 'open', 'high', 'low', 'close', 'volume', 
                'change_ptc', 'theta']]

trend_ma_features = ["SMA3", "SMA5", "SMA10", "SMA20", "EMA", "DEMA", "TEMA", "TRIMA", "WMA",
    "T3", "KAMA","MIDPOINT", "MIDPRICE","HT_TRENDLINE"]

bollinger_features = ["BBAND_upper", "BBAND_middle", "BBAND_lower",
    "BBAND_width","BBAND_upper_signal", "BBAND_lower_signal"]

momentum_return_features = ["MOM1", "MOM3", "MOM5", "MOM10","ROC", "ROCP", "ROCR", "ROCR100",
    "APO", "PPO","MACD", "MACDSIGNAL", "MACDHIST","TRIX"]

oscillator_features = ["RSI5", "RSI10", "RSI14","WILLR", "ULTOSC","CMO",
    "CCI3", "CCI5", "CCI10", "CCI14","BOP","FASTK", "FASTD","SLOWK", "SLOWD","AROONOSC"]

directional_features = ["ADX14", "ADX20", "ADXR","DX","PLUS_DI", "MINUS_DI",
    "PLUS_DM", "MINUS_DM"]

volatility_features = ["ATR", "NATR", "TRANGE","SAR", "SAREXT"]

hilbert_features = ["HT_DCPERIOD","HT_DCPHASE","HT_TRENDMODE"]

candlestick_features = [col for col in df_train.columns if col.startswith("CDL")]

#feature_cols = [col for col in feature_cols if col not in candlestick_features]

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


def walk_forward_cv_xgb_predictive(df, feature_cols, theta_midpoints, train_size=500,
    val_size=50, step=50, xgb_params=None, early_stopping_rounds=25):
    results = []
    K = len(theta_midpoints)

    if xgb_params is None:
        xgb_params = dict(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1.0,
            gamma=0.0, tree_method="hist", objective="multi:softprob", num_class=K,
            eval_metric="mlogloss", random_state=42, n_jobs=1)

    n = len(df)

    for start in range(0, n - train_size - val_size + 1, step):
        train_idx = slice(start, start + train_size)
        val_idx   = slice(start + train_size, start + train_size + val_size)

        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]

        X_train = df_train[feature_cols]
        y_train = combine_bins(df_train["theta"]) + 4
        X_val   = df_val[feature_cols]
        y_val   = combine_bins(df_val["theta"]) + 4

        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, eval_set = [(X_val, y_val)], verbose = False) 

        P = model.predict_proba(X_val)
        if P.shape[1] != K:
            raise ValueError(f"Expected {K} classes but got {P.shape}")

        pred_metrics = evaluate_prediction(P, y_val)
        results.append(pred_metrics)

    return pd.DataFrame(results)

def evaluate_xgb_params(params, full_df, feature_cols, theta_midpoints, train_size=500,
    val_size=50, step=50):
    cv_res = walk_forward_cv_xgb_predictive(full_df, feature_cols, theta_midpoints,
        train_size=train_size, val_size=val_size, step=step, xgb_params=params)
    return {
        **params,
        "LogLoss_median": cv_res["LogLoss"].median(),
        "MSE_median": cv_res["MSE"].median(),
        "DirectionalAcc_mean": cv_res["Directional_Acc"].mean(),
        "ROC_AUC_mean": cv_res["ROC_AUC"].mean()
    }

def tune_xgb_parallel(xgb_grid, full_df, feature_cols, theta_midpoints, train_size=500,
    val_size=50, step=50, n_jobs=-1):
    
    param_list = list(ParameterGrid(xgb_grid))

    xgb_results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_xgb_params)(
            params, full_df, feature_cols, theta_midpoints, train_size, val_size, step
        )
        for params in param_list
    )

    xgb_df = pd.DataFrame(xgb_results)

    # Pick best by ROC AUC then Directional Acc (same as your RF selection)
    best_row = xgb_df.sort_values(by=["ROC_AUC_mean", "DirectionalAcc_mean"],
        ascending=[False, False]).iloc[0]

    # Return the best params as a plain dict
    best_params = {k: best_row[k] for k in xgb_grid.keys()}

    print("Best XGB parameters:")
    print(best_params)

    return best_params, xgb_df

def tilt_posterior_to_match_p_pos(P, pos_idx, p_pos_cal, eps=1e-12):
    """
    P: (n, K) uncalibrated multiclass posterior from RF (rows sum to 1)
    p_pos_cal: (n,) calibrated P(theta > 0)
    Returns: P_tilt where sum over pos_idx ~= p_pos_cal for each row.
    """
    P = P.copy()

    p_pos_uncal = P[:, pos_idx].sum(axis=1)
    p_pos_uncal = np.clip(p_pos_uncal, eps, 1 - eps)
    p_pos_cal   = np.clip(p_pos_cal,   eps, 1 - eps)

    s_pos = p_pos_cal / p_pos_uncal
    s_neg = (1 - p_pos_cal) / (1 - p_pos_uncal)

    P_tilt = P * s_neg[:, None]
    P_tilt[:, pos_idx] = P[:, pos_idx] * s_pos[:, None]

    # numerical cleanup
    P_tilt = np.clip(P_tilt, 0.0, 1.0)
    P_tilt /= P_tilt.sum(axis=1, keepdims=True)

    return P_tilt

def to_python_scalars(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out

K = len(theta_midpoints)

xgb_grid = {
    "n_estimators": [800],
    "learning_rate": [0.05],
    "max_depth": [4],
    "subsample": [0.8],
    "colsample_bytree": [0.6],
    "min_child_weight": [1.0],
    "gamma": [0.5],
    "reg_lambda": [1.0],
    "reg_alpha": [0.0],

    "tree_method": ["hist"],
    "objective": ["multi:softprob"],
    "num_class": [K],
    "eval_metric": ["mlogloss"],
    "random_state": [42],
    "n_jobs": [1],
    "early_stopping_rounds": [25]
}

best_params, xgb_df = tune_xgb_parallel(xgb_grid, df_train, feature_cols, theta_midpoints,
    train_size=500, val_size=50, step=50, n_jobs=12)

best_params = to_python_scalars(best_params)

model = XGBClassifier(**best_params)

# optional early stopping using your held-out validation set
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],verbose=False)

P_val = model.predict_proba(X_val)  # shape (n_val, K)

# --- isotonic calibrate P(theta > 0) ---
pos_idx = np.where(theta_midpoints > 0)[0]
p_pos_val = P_val[:, pos_idx].sum(axis=1)
y_pos_val = (theta_midpoints[y_val] > 0).astype(int)

from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_pos_val, y_pos_val)
p_pos_val_cal = iso.transform(p_pos_val)

P_val_cal = tilt_posterior_to_match_p_pos(P_val, pos_idx, p_pos_val_cal)

print("Validation metrics (uncal):", evaluate_prediction(P_val, y_val))
print("Validation metrics (cal):  ", evaluate_prediction(P_val_cal, y_val))

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

alpha_list  = [0.2]
lambda_list = [0.1]
gamma_list  = [0.00]
scale_list  = [5]
delta_list = [0.03]


best_metrics = None
best_params  = None

best_cal_metrics = None
best_cal_params = None

results = []

from sklearn.model_selection import TimeSeriesSplit

def sharpe_from_wealth(wealth):
    daily_returns = np.diff(wealth) / wealth[:-1]
    if daily_returns.size == 0 or np.std(daily_returns) == 0:
        return 0.0
    return np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.25)

def max_drawdown_from_wealth(wealth):
    cum_max = np.maximum.accumulate(wealth)
    return np.max((cum_max - wealth) / cum_max)

def walkforward_policy_score(params, P_uncal, P_cal, volatility, returns, n_splits=5,
    min_train_size=200, a0=0.5, dd_penalty=0.3, capital=1_000):

    alpha, lam, gamma, scale, delta = params
    tscv = TimeSeriesSplit(n_splits=n_splits)

    sharpe_uncal_folds, sharpe_cal_folds = [], []
    mdd_uncal_folds,    mdd_cal_folds    = [], []
    score_uncal_folds,  score_cal_folds  = [], []

    n = len(returns)

    for train_idx, test_idx in tscv.split(np.arange(n)):
        if len(train_idx) < min_train_size:
            continue

        test_end = test_idx[-1] + 1  # exclusive

        # Use the fold's test returns
        returns_test = returns.iloc[test_idx].reset_index(drop=True)

        # --- Uncalibrated ---
        actions_prefix = compute_actions(
            P_uncal[:test_end],
            theta_midpoints,
            volatility.iloc[:test_end],
            alpha=alpha, lam=lam, gamma=gamma,
            signal_scale=scale, delta=delta, a0=a0
        )
        actions_test = pd.Series(actions_prefix[test_idx]).reset_index(drop=True).values

        wealth_test = backtest(actions_test, returns_test, capital=capital)
        sh = sharpe_from_wealth(wealth_test)
        dd = max_drawdown_from_wealth(wealth_test)

        sharpe_uncal_folds.append(sh)
        mdd_uncal_folds.append(dd)
        score_uncal_folds.append(sh - dd_penalty * dd)

        # --- Calibrated / Tilted ---
        actions_prefix_cal = compute_actions(
            P_cal[:test_end],
            theta_midpoints,
            volatility.iloc[:test_end],
            alpha=alpha, lam=lam, gamma=gamma,
            signal_scale=scale, delta=delta, a0=a0
        )
        actions_test_cal = pd.Series(actions_prefix_cal[test_idx]).reset_index(drop=True).values

        wealth_test_cal = backtest(actions_test_cal, returns_test, capital=capital)
        sh_c = sharpe_from_wealth(wealth_test_cal)
        dd_c = max_drawdown_from_wealth(wealth_test_cal)
        sharpe_cal_folds.append(sh_c)
        mdd_cal_folds.append(dd_c)
        score_cal_folds.append(sh_c - dd_penalty * dd_c)

    # If too few folds, return very poor score to avoid selecting it
    if len(score_uncal_folds) < max(2, n_splits // 2):
        return (-np.inf, -np.inf,
                sharpe_uncal_folds, sharpe_cal_folds,
                mdd_uncal_folds,    mdd_cal_folds)

    return (np.median(score_uncal_folds),
            np.median(score_cal_folds),
            sharpe_uncal_folds,
            sharpe_cal_folds,
            mdd_uncal_folds,
            mdd_cal_folds)

def eval_one(params, P_uncal, P_cal, volatility, returns):
    score_u, score_c, sharpe_u, sharpe_c, mdd_u, mdd_c = walkforward_policy_score(
        params, P_uncal=P_uncal, P_cal=P_cal, volatility=volatility, returns=returns,
        n_splits=5, min_train_size=200, a0=0.5, dd_penalty=0.3, capital=1_000)

    alpha, lam, gamma, scale, delta = params
    return {
        "alpha": alpha,
        "lam": lam,
        "gamma": gamma,
        "signal_scale": scale,
        "delta": delta,

        "Score_uncal_median": score_u,
        "Score_cal_median": score_c,

        "Sharpe_uncal_median": float(np.median(sharpe_u)) if len(sharpe_u) else np.nan,
        "Sharpe_cal_median":   float(np.median(sharpe_c)) if len(sharpe_c) else np.nan,
        "MaxDD_uncal_median":  float(np.median(mdd_u))    if len(mdd_u)    else np.nan,
        "MaxDD_cal_median":    float(np.median(mdd_c))    if len(mdd_c)    else np.nan,
    }

from itertools import product

param_grid = list(product(alpha_list, lambda_list, gamma_list, scale_list, delta_list))

results = Parallel(n_jobs=12, backend="loky")(
    delayed(eval_one)(p, P_val, P_val_cal, volatility_val, returns_val)
    for p in param_grid
)

results_df = pd.DataFrame(results)

best_uncal = results_df.sort_values("Score_uncal_median", ascending=False).iloc[0]
best_cal   = results_df.sort_values("Score_cal_median", ascending=False).iloc[0]

best_uncal_params = best_uncal[["alpha","lam","gamma","signal_scale", "delta"]].to_dict()
best_cal_params   = best_cal[["alpha","lam","gamma","signal_scale", "delta"]].to_dict()
print("Best uncalibrated params (WF median): ", best_uncal_params)
print("Best calibrated params: (WF median)", best_cal_params)

pos_idx = np.where(theta_midpoints > 0)[0]

p_pos = p_pos_val
p_pos_cal = p_pos_val_cal
y_pos = y_pos_val

from sklearn.calibration import calibration_curve

def plot_reliability(p, y, label, n_bins=10):
    frac_pos, mean_pred = calibration_curve(
        y, p, n_bins=n_bins, strategy="uniform"
    )
    plt.plot(mean_pred, frac_pos, marker='o', label=label)

plt.figure(figsize=(6,6))

plot_reliability(p_pos, y_pos, label="Uncalibrated XGB")
plot_reliability(p_pos_cal, y_pos, label="Calibrated XGB")

# Perfect calibration line
plt.plot([0,1], [0,1], linestyle="--", color="black", alpha=0.7)

plt.xlabel("Predicted probability of positive return")
plt.ylabel("Empirical frequency")
plt.title("Reliability Diagram (Positive Return)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("XGB Reliability diagram.png")
plt.close()

plt.figure(figsize=(6,4))
plt.hist(p_pos, bins=20, alpha=0.5, label="Uncalibrated", density=True)
plt.hist(p_pos_cal, bins=20, alpha=0.5, label="Calibrated", density=True)
plt.xlabel("Predicted probability")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("XGB Histogram.png")
plt.close()


from sklearn.metrics import brier_score_loss

brier_uncal = brier_score_loss(y_pos, p_pos)
brier_cal   = brier_score_loss(y_pos, p_pos_cal)

print(f"Brier score (uncalibrated): {brier_uncal:.4f}")
print(f"Brier score (calibrated):   {brier_cal:.4f}")


#Trade

actions = compute_actions(P_val, theta_midpoints, volatility_val,
                          lam = best_uncal_params["lam"], gamma = best_uncal_params["gamma"],
                          alpha = best_uncal_params["alpha"],
                          signal_scale = best_uncal_params["signal_scale"],
                          delta = best_uncal_params["delta"])

print("Mean uncalibrated exposure:", np.mean(actions))

actions_cal = compute_actions(P_val_cal, theta_midpoints, volatility_val,
                          lam = best_uncal_params["lam"], gamma = best_uncal_params["gamma"], 
                          alpha = best_uncal_params["alpha"], 
                          signal_scale = best_uncal_params["signal_scale"],
                          delta = best_uncal_params["delta"])

print("Mean calibrated exposure:", np.mean(actions_cal))

wealth = backtest(actions, df_val["change_ptc"] / 100)
wealth_cal = backtest(actions_cal, df_val["change_ptc"]/100)
metrics = evaluate(wealth, actions, df_val["change_ptc"] / 100)
metrics_cal = evaluate(wealth_cal, actions_cal, df_val["change_ptc"]/100)
print("Uncalibrated validation metrics: ", metrics)
print("Calibrated validation metrics: ", metrics_cal)

val_results_xgb = {
    "wealth": wealth_cal.tolist(),
    "dates": df_val["date"].astype(str).tolist()
}

wealth_val_5050 = df_val['close_original']/df_val['close_original'][0] * 500 + 500

val_5050 = {
    "wealth": wealth_val_5050.tolist(),
    "dates": df_val["date"].astype(str).tolist()
}

import json
with open("val_xgb_results.json", "w") as f:
    json.dump(val_results_xgb, f)

with open("val_5050.json", "w") as f:
    json.dump(val_5050, f)


#Plot model

df_val['date'] = pd.to_datetime(df_val['date'], errors='coerce')
df_val = df_val.dropna(subset=['date'])

plt.figure(figsize=(12, 8))

plt.plot(df_val['date'], wealth, label = "Uncalibrated model performance", color = "blue")
plt.plot(df_val['date'], wealth_cal, label = "Calibrated model performance", color = "green")
plt.plot(df_val['date'], df_val['close_original']/df_val['close_original'][0] * 1000,
    label = "Bitcoin price", color = "red")
plt.plot(df_val['date'], df_val['close_original']/df_val['close_original'][0] * 500 + 500, 
    label = "50/50 strategy", color = "orange")
plt.title("Cumulative Wealth – Random Forest Strategy")
plt.xlabel("Date")
plt.ylabel("Wealth ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("XGB.png")
plt.close()

#Run on test data
# --- Step 1: Predict multiclass probabilities on test set (uncalibrated RF) ---
P_test = model.predict_proba(X_test)

if P_test.shape[1] != K:
    raise ValueError(f"Expected {K} classes but got {P_test.shape}")

# --- Step 1.5: Posterior tilt on TEST using calibrated p_pos ---
pos_idx = np.where(theta_midpoints > 0)[0]
p_pos_test = P_test[:, pos_idx].sum(axis=1)
p_pos_test_cal = iso.transform(p_pos_test)
P_test_tilt = tilt_posterior_to_match_p_pos(P_test, pos_idx, p_pos_test_cal)

# --- Step 2: Compute actions using tuned params ---
actions_test = compute_actions(
    P_test_tilt,
    theta_midpoints,
    volatility_test,
    alpha=best_uncal_params["alpha"],
    lam=best_uncal_params["lam"],
    gamma=best_uncal_params["gamma"],
    signal_scale=best_uncal_params["signal_scale"],
    delta=best_uncal_params["delta"]
)

print("Mean calibrated test exposure:", np.mean(actions_test))

def pnl_loss_series(actions, returns, c=0.001, a0=0.5, eps=1e-12):
    a = np.asarray(actions, dtype=float)
    r = np.asarray(returns, dtype=float)

    T = len(r)
    loss = np.zeros(T, dtype=float)

    a_prev = a0
    for t in range(T):
        gross = 1.0 + a[t] * r[t] - c * abs(a[t] - a_prev)
        # keep log safe
        gross = max(gross, eps)
        loss[t] = -np.log(gross)
        a_prev = a[t]

    return loss

def empirical_risk_from_pnl(actions, returns, c=0.001, a0=0.0):
    loss = pnl_loss_series(actions, returns, c=c, a0=a0)
    return float(loss.mean()), loss

def oracle_actions_pnl(returns, c=0.001, a0=0.0, grid=np.linspace(0.0, 1.0, 101), eps=1e-12):
    r = np.asarray(returns, dtype=float)
    T = len(r)
    a_star = np.zeros(T, dtype=float)

    a_prev = a0
    for t in range(T):
        gross = 1.0 + grid * r[t] - c * np.abs(grid - a_prev)
        gross = np.maximum(gross, eps)
        loss = -np.log(gross)

        best_a = grid[np.argmin(loss)]
        a_star[t] = best_a
        a_prev = best_a

    return a_star

returns_test = (df_test["change_ptc"] / 100).values
R_hat, loss_hat = empirical_risk_from_pnl(actions_test, returns_test, c=0.001, a0=0.5)

actions_oracle = oracle_actions_pnl(returns_test, c=0.001, a0=0.5)
R_oracle, loss_oracle = empirical_risk_from_pnl(actions_oracle, returns_test, c=0.001, a0=0.5)

regret_series = loss_hat - loss_oracle
regret = float(regret_series.mean())

print("Empirical P&L risk (test):", R_hat)
print("Oracle P&L risk (test):", R_oracle)
print("Regret vs oracle (test):", regret)

def block_bootstrap_means(series, block_len=10, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(series)
    T = len(x)
    starts = np.arange(0, T - block_len + 1)
    n_blocks = int(np.ceil(T / block_len))

    out = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate([x[s:s+block_len] for s in chosen])[:T]
        out[b] = sample.mean()
    return out

def percentile_ci(samples, alpha=0.05):
    lo = np.percentile(samples, 100*alpha/2)
    hi = np.percentile(samples, 100*(1-alpha/2))
    return float(lo), float(hi)

boot_R = block_bootstrap_means(loss_hat, block_len=10)
boot_reg = block_bootstrap_means(regret_series, block_len=10)

print("Risk CI:", percentile_ci(boot_R))
print("Regret CI:", percentile_ci(boot_reg))

# p-value for regret > 0 (policy worse than oracle)
pval = float(np.mean(boot_reg > 0))
print("p-value (regret > 0):", pval)

def newey_west_se_mean(x, L=None):
    x = np.asarray(x, dtype=float)
    T = len(x)
    mu = x.mean()
    u = x - mu

    # Lag selection (common rule-of-thumb)
    if L is None:
        L = int(np.floor(4 * (T / 100.0)**(2/9)))  # Andrews-style rule

    # HAC variance estimate of sqrt(T)*(mean - true_mean)
    gamma0 = np.mean(u * u)
    S = gamma0

    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1)          # Bartlett kernel
        gamma_l = np.mean(u[l:] * u[:-l])
        S += 2.0 * w * gamma_l

    # Var(mean) = S / T
    se = np.sqrt(S / T)
    return float(se), int(L)

# risk (policy)
R_hat = float(np.mean(loss_hat))
se_R, L_R = newey_west_se_mean(loss_hat, L=None)
t_R = R_hat / se_R if se_R > 0 else np.nan

print("NW risk mean:", R_hat, "SE:", se_R, "L:", L_R, "t:", t_R)

# regret
regret = float(np.mean(regret_series))
se_reg, L_reg = newey_west_se_mean(regret_series, L=None)
t_reg = regret / se_reg if se_reg > 0 else np.nan

print("NW regret mean:", regret, "SE:", se_reg, "L:", L_reg, "t:", t_reg)

z = 1.96
ci_R = (R_hat - z * se_R, R_hat + z * se_R)
ci_reg = (regret - z * se_reg, regret + z * se_reg)

print("NW 95% CI risk:", ci_R)
print("NW 95% CI regret:", ci_reg)

import math

def norm_sf(z):
    # survival function for N(0,1): P(Z > z)
    return 0.5 * math.erfc(z / math.sqrt(2))

# H1: regret > 0
pval_regret = norm_sf(t_reg)
print("NW one-sided p-value (regret > 0):", pval_regret)

from statsmodels.graphics.tsaplots import plot_acf

plot_acf(loss_hat, lags=30)
plt.title("ACF of per-period log P&L loss")
plt.savefig("XGB ACF.png")
plt.close()

# Stress test
#Sensitivity to c 

c_list = [0.0, 0.0005, 0.001, 0.002, 0.005]
results_tc = []
returns_test_pd = df_test["change_ptc"]/100

for c in c_list:
    R_hat, _ = empirical_risk_from_pnl(actions_test, returns_test, c=c)
    R_or, _  = empirical_risk_from_pnl(actions_oracle, returns_test, c=c)
    regret = R_hat - R_or
    wealth = backtest(actions_test, returns_test_pd, capital=1_000, c=c)
    results_tc.append({"c": c,"Risk": R_hat, "OracleRisk": R_or, "Regret": regret, "Final Wealth": wealth[-1]})

df_tc = pd.DataFrame(results_tc)
print(df_tc)

#Sensitivity to horizon

def forward_compound_return(r, h):
    r = pd.Series(r).reset_index(drop=True)
    return (1 + r).rolling(window=h).apply(np.prod, raw=True).shift(-h) - 1

def add_horizon_target(df_in, h):
    df = df_in.copy()

    # daily return in decimals
    df["r1"] = df["change_ptc"] / 100.0

    df[f"r_fwd_{h}"] = ((1 + df["r1"]).rolling(window=h)
        .apply(np.prod, raw=True).shift(-h) - 1)

    # drop rows where forward return is undefined
    df = df.dropna(subset=[f"r_fwd_{h}"]).reset_index(drop=True)

    return df

def make_labels_from_forward(df_h, h):
    r = df_h[f"r_fwd_{h}"].to_numpy()
    y = np.digitize(r, theta_bins[1:-1], right = True)
    return y.astype(int)

def full_probs(rf, X, n_classes=9):
    P_raw = rf.predict_proba(X)
    P = np.zeros((len(X), n_classes))
    for i, cls in enumerate(rf.classes_):
        P[:, cls] = P_raw[:, i]
    return P
'''

def run_one_horizon(h,
                   df_train, df_val, df_test,feature_cols, theta_midpoints,
                   params, action_params,c=0.001):

    # 1) Build horizon datasets (copies)
    tr = add_horizon_target(df_train, h)
    va = add_horizon_target(df_val,   h)
    te = add_horizon_target(df_test,  h)

    # 2) Labels
    y_tr = make_labels_from_forward(tr, h)
    y_va = make_labels_from_forward(va, h)
    y_te = make_labels_from_forward(te, h)

    # 3) Features
    X_tr = tr[feature_cols]
    X_va = va[feature_cols]
    X_te = te[feature_cols]

    # 4) Train RF (no leakage)
    model = XGBClassifier(**best_params, random_state = 42, n_jobs = 1)
    model.fit(X_tr, y_tr)

    # 5) Predict probs on val/test (map to full 9-class matrix)
    n_classes = len(theta_midpoints)
    P_va = full_probs(rf, X_va, n_classes=len(theta_midpoints))
    P_te = full_probs(rf, X_te, n_classes=len(theta_midpoints))

    pos_idx = np.where(theta_midpoints > 0)[0]
    p_pos_test = P_te[:, pos_idx].sum(axis=1)
    p_pos_test_cal = iso.transform(p_pos_test)
    P_te = tilt_posterior_to_match_p_pos(P_te, pos_idx, p_pos_test_cal)


    # 6) Horizon volatility scaling (Interpretation B)
    # If recursive_volatility is daily std dev, scale by sqrt(h)
    #sigma_va = np.sqrt(h) * va["recursive_volatility"]
    sigma_te = np.sqrt(h) * te["recursive_volatility"]

    # 7) Compute actions
    #actions_va = compute_actions(P_va, theta_midpoints, sigma_va, **action_params)
    actions_te = compute_actions(P_te, theta_midpoints, sigma_te, **action_params)

    # 8) Evaluate P&L risk using h-day forward returns (realized payoff)
    #r_va = va[f"r_fwd_{h}"].values
    r_te = te[f"r_fwd_{h}"].values

    #R_va, _ = empirical_risk_from_pnl(actions_va, r_va, c=c, a0=0.0)
    R_te, loss_te = empirical_risk_from_pnl(actions_te, r_te, c=c, a0=0.0) 

    # 9) Oracle benchmark on test (optional but strong)
    actions_or = oracle_actions_pnl(r_te, c=c, a0=0.0)
    R_or, loss_or = empirical_risk_from_pnl(actions_or, r_te, c=c, a0=0.0)
    regret = R_te - R_or

    return {
        "h": h,
  #      "T_val": len(r_va),
        "T_test": len(r_te),
   #     "Risk_per_day_val": R_va/h,
        "Risk_per_day_test": R_te/h,
        "OracleRisk_per_day_test": R_or/h,
        "Regret_per_day_test": regret/h,
        "MeanExposure_test": float(np.mean(actions_te))
    }

#h_list = [1, 2, 5, 10]
h_list = [1]

action_params = dict(alpha=best_uncal_params["alpha"], lam=best_uncal_params["lam"],
    delta=best_uncal_params["delta"], signal_scale = best_uncal_params["signal_scale"])

def run_horizons_parallel(h_list, df_train, df_val, df_test, feature_cols, theta_midpoints,
                          best_params, action_params,c=0.001, method="compound", n_jobs=12):
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_one_horizon)(h, df_train, df_val, df_test, feature_cols, theta_midpoints,
            best_params, action_params,c)
        for h in h_list
    )
    return pd.DataFrame(results).sort_values("h").reset_index(drop=True)

if __name__ == "__main__":
#    h_list = [1, 2, 5, 10]
    h_list=[1]
    action_params = dict(alpha=best_uncal_params["alpha"], lam=best_uncal_params["lam"],
        delta=best_uncal_params["delta"], signal_scale = best_uncal_params["signal_scale"])

    df_h = run_horizons_parallel(h_list,df_train=df_train, df_val=df_val, df_test=df_test,
        feature_cols=feature_cols, theta_midpoints=theta_midpoints, best_params = best_params,
        action_params=action_params, c=0.001, n_jobs=12)

    print(df_h.to_string(index=False))
'''
# --- Step 3: Backtest the strategy ---
returns_test = df_test["change_ptc"] / 100
wealth_test = backtest(actions_test, returns_test)

test_results_xgb = {
    "wealth": wealth_test.tolist(),
    "dates": df_test["date"].astype(str).tolist()
}

import json
with open("test_xgb_results.json", "w") as f:
    json.dump(test_results_xgb, f)

# --- Step 4: Evaluate strategy ---
metrics_test = evaluate(wealth_test, actions_test, returns_test)
print("Test set trading performance:", metrics_test)

# --- Step 5: Plot cumulative wealth ---
df_test['date'] = pd.to_datetime(df_test['date'], errors='coerce')
df_test = df_test.dropna(subset=['date'])

plt.figure(figsize=(12,6))
plt.plot(df_test['date'], wealth_test, label="Calibrated strategy", color='blue')
plt.plot(df_test['date'], df_test['close_original']/df_test['close_original'][0] * 1000,
    label = "Bitcoin price", color = "red")
plt.plot(df_test['date'], df_test['close_original']/df_test['close_original'][0] * 500 + 500,
    label = "50/50 strategy",color = "orange")
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
plt.savefig("XGB_test")
plt.close()

# --- Optional: plot actions vs returns ---
plt.figure(figsize=(12,4))
plt.plot(df_test['date'], actions_test, label="Calibrated Actions", color='orange')
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
plt.savefig("XGB_actionsvreturns")
plt.close()

#Drawdown plot
def drawdown(wealth):
    cum_max = np.maximum.accumulate(wealth)
    return (wealth - cum_max) / cum_max

wealth5050 = df_test['close_original']/df_test['close_original'][0] * 500 + 500
wealth5050 = wealth5050.to_numpy()
print("Final wealth 50/50:", wealth5050[-1])

test_5050 = {
    "wealth": wealth5050.tolist(),
    "dates": df_test["date"].astype(str).tolist()
}

with open("test_5050.json", "w") as f:
    json.dump(test_5050, f)

wealth = wealth_test
dd = drawdown(wealth)
dd5050 = drawdown(wealth5050)

plt.figure(figsize=(12,4))
plt.plot(df_test["date"], dd, label = "Calibrated strategy")
plt.plot(df_test["date"], dd5050, label = "50/50 strategy")
plt.title("Drawdown")
plt.ylabel("Drawdown")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("XGB_drawdown.png")
plt.close()

#Annualised returns
log_rets = np.diff(np.log(wealth))
ann_return = log_rets.mean() * 365.25
ann_vol = log_rets.std(ddof=1) * np.sqrt(365.25)

print("Annualised return:", ann_return)
print("Annualise volatility:", ann_vol)
print("Annualised Sharpe:", ann_return/ann_vol)

log_rets = np.diff(np.log(wealth5050))
ann_return = log_rets.mean() * 365.25
ann_vol = log_rets.std(ddof=1) * np.sqrt(365.25)

print("50/50 Annualised return:", ann_return)
print("50/50 Annualise volatility:", ann_vol)
print("50/50 Annualised Sharpe:", ann_return/ann_vol)


#Hit rates
def hit_rates(actions, returns, a0=0.5):
    buy = actions[:-1] > a0
    sell = actions[:-1] < a0
    r = returns[1:]

    buy_hit = np.mean((r > 0)[buy]) if buy.any() else np.nan
    sell_hit = np.mean((r < 0)[sell]) if sell.any() else np.nan

    hit_rate = np.mean(
        ((actions[:-1] > a0) & (r > 0)) |
        ((actions[:-1] < a0) & (r < 0))
    )

    return hit_rate, buy_hit, sell_hit

hit_rate, buy_hit, sell_hit = hit_rates(actions_test, returns_test)
print("Hit Rate:", hit_rate)
print("Buy_hit:", buy_hit)
print("Sell_hit:", sell_hit)

y_true = np.sign(theta_midpoints[y_test])
y_pred = np.sign(P_test @ theta_midpoints)
mask = y_true != 0

from sklearn.metrics import precision_score, recall_score
precision = precision_score(y_true[mask] > 0, y_pred[mask] > 0)
recall = recall_score(y_true[mask] > 0, y_pred[mask] > 0)

print("Precision:", precision)
print("Recall:", recall)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(
    y_true[mask] > 0,
    y_pred[mask] > 0
)

print("Confusion matrix:", cm)

#50/50 Baseline metrics

actions5050 = np.ones(len(returns_test))/2
metrics5050 = evaluate(wealth5050, actions5050, returns_test)
print("50/50 metrics:", metrics5050)

R_hat5050, loss_hat5050 = empirical_risk_from_pnl(actions5050, returns_test, c=0.001, a0=0.5)

regret_series = loss_hat5050 - loss_oracle
regret5050 = float(regret_series.mean())

print("Empirical P&L risk (50/50):", R_hat5050)
print("Oracle P&L risk (50/50):", R_oracle)
print("Regret vs oracle (50/50):", regret5050)

import statsmodels.api as sm

def compute_model_returns(actions, returns, c=0.001):
    returns = np.asarray(returns)
    actions = np.asarray(actions)

    r_model = np.zeros(len(returns))
    a_prev = 0.0

    for t in range(len(returns)):
        a_t = actions[t]
        r_t = returns[t]

        r_model[t] = a_t * r_t - c * abs(a_t - a_prev)
        a_prev = a_t

    return r_model

r_model = compute_model_returns(actions_test, returns_test)
r_market = returns_test

print("Mean model return:", r_model.mean())
print("Mean BTC return:", r_market.mean())
print("Mean exposure:", actions_test.mean())

X = sm.add_constant(r_market)
model = sm.OLS(r_model, X).fit(cov_type='HAC', cov_kwds={'maxlags':6})

print(model.summary())

import numpy as np

T = len(r_model)
trading_days = 365  # crypto

# Final wealth from simple returns
final_wealth = np.prod(1 + r_model)

# CAGR
years = T / trading_days
cagr = final_wealth**(1/years) - 1

print("CAGR:", cagr)
ann_vol = np.std(r_model, ddof=1) * np.sqrt(trading_days)
print("Annualised volatility (simple):", ann_vol)
ann_return = np.mean(r_model) * trading_days
sharpe_simple = ann_return / ann_vol

print("Annualised return (mean-based):", ann_return)
print("Sharpe (simple returns):", sharpe_simple)

