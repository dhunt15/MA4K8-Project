import json
import pandas as pd
import matplotlib.pyplot as plt

# Load RF
with open("rf_val.json") as f:
    rf_val = json.load(f)

with open("rf_test.json") as f:
    rf_test = json.load(f)

# Load XGB
with open("val_xgb_results.json") as f:
    xgb_val = json.load(f)

with open("test_xgb_results.json") as f:
    xgb_test = json.load(f)

#Load 50/50

with open("test_5050.json") as f:
    test_5050 = json.load(f)

with open("val_5050.json") as f:
    val_5050 = json.load(f)


# Convert to DataFrame

def build_df(data, wealth_key="wealth"):
    return pd.DataFrame({
        "date": pd.to_datetime(data["dates"]),
        "wealth": data[wealth_key]
    })

# ---------- VALIDATION ----------

df_rf_val   = build_df(rf_val)
df_xgb_val  = build_df(xgb_val)
df_5050_val = build_df(val_5050)

df_val = df_rf_val.merge(
    df_xgb_val, on="date", suffixes=("_rf", "_xgb")
).merge(
    df_5050_val, on="date"
)

df_val.rename(columns={"wealth": "wealth_5050"}, inplace=True)

plt.figure(figsize=(12,6))
plt.plot(df_val["date"], df_val["wealth_rf"],   label="Random Forest", color = "blue")
plt.plot(df_val["date"], df_val["wealth_xgb"],  label="XGBoost", color = "orange")
plt.plot(df_val["date"], df_val["wealth_5050"], label="50/50", color = "gray")

plt.title("Validation Curve Comparison", fontsize=18)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Wealth", fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Validation.png")
plt.close()


# ---------- TEST ----------

df_rf_test   = build_df(rf_test)
df_xgb_test  = build_df(xgb_test)
df_5050_test = build_df(test_5050)

df_test = df_rf_test.merge(
    df_xgb_test, on="date", suffixes=("_rf", "_xgb")
).merge(
    df_5050_test, on="date"
)

df_test.rename(columns={"wealth": "wealth_5050"}, inplace=True)

plt.figure(figsize=(12,6))
plt.plot(df_test["date"], df_test["wealth_rf"],   label="Random Forest", color = "blue")
plt.plot(df_test["date"], df_test["wealth_xgb"],  label="XGBoost", color = "orange")
plt.plot(df_test["date"], df_test["wealth_5050"], label="50/50", color = "gray")

plt.title("Test Curve Comparison", fontsize=18)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Wealth", fontsize=14)
plt.legend(fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Test.png")
plt.close()
