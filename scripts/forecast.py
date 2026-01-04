import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_RAW = os.path.join(ROOT, "data", "raw", "bengaluru _metro.csv")
DATA_CLEAN = os.path.join(ROOT, "data", "processed", "clean_bengaluru_metro.csv")
DATA_FEATURES = os.path.join(ROOT, "data", "processed", "features_bengaluru_metro.csv")
REPORTS_EDA_DIR = os.path.join(ROOT, "reports", "eda")
REPORTS_METRICS_DIR = os.path.join(ROOT, "reports", "metrics")
MODELS_DIR = os.path.join(ROOT, "models", "forecasting")

AVG_FARE = 30.0
SC_DISCOUNT = 0.10
PASS_1D_PRICE = 70
PASS_3D_PRICE = 200
PASS_5D_PRICE = 300
FORECAST_DAYS = 14

def preprocess():
    df = pd.read_csv(DATA_RAW)
    df["Record Date"] = pd.to_datetime(df["Record Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Record Date"])  # drop invalid dates
    df = df.sort_values("Record Date")


    cols = {
        "Total Smart Cards": "smart",
        "Total Tokens": "tokens",
        "Total QR": "qr",
        "Total NCMC": "ncmc",
        "Group Ticket": "group",
        "One Day Pass": "pass1",
        "Three Day Pass": "pass3",
        "Five Day Pass": "pass5",
    }
    df = df.rename(columns=cols)
    for c in cols.values():
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    
    df["Total_Flow"] = df["smart"] + df["tokens"] + df["qr"] + df["ncmc"] + df["group"] + df["pass1"] + df["pass3"] + df["pass5"]

    
    smart_revenue = df["smart"] * (AVG_FARE * (1.0 - SC_DISCOUNT))
    token_revenue = df["tokens"] * AVG_FARE
    qr_revenue = df["qr"] * AVG_FARE
    ncmc_revenue = df["ncmc"] * AVG_FARE
    group_revenue = df["group"] * AVG_FARE
    pass_revenue = df["pass1"] * PASS_1D_PRICE + df["pass3"] * PASS_3D_PRICE + df["pass5"] * PASS_5D_PRICE
    df["Total_Revenue"] = smart_revenue + token_revenue + qr_revenue + ncmc_revenue + group_revenue + pass_revenue

    df.to_csv(DATA_CLEAN, index=False)
    print(f"[INFO] Cleaned dataset saved to: {DATA_CLEAN}")
    return df


def forecast(df, target_col, out_csv, out_plot):
    ds_y = df[["Record Date", target_col]].rename(columns={"Record Date": "ds", target_col: "y"})
    ds_y = ds_y.dropna()  # ensure no NaN

    m = Prophet()
    m.fit(ds_y)
    future = m.make_future_dataframe(periods=FORECAST_DAYS, freq="D")
    forecast = m.predict(future)
    forecast.to_csv(out_csv, index=False)

    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 5))
    plt.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="tab:purple")
    plt.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="tab:purple", alpha=0.2)
    plt.title(f"{target_col} Forecast")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"[INFO] Forecast saved: {out_csv}, plot: {out_plot}")

def main():
    os.makedirs(REPORTS_EDA_DIR, exist_ok=True)
    os.makedirs(REPORTS_METRICS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = preprocess()

    forecast(df, "Total_Flow",
             os.path.join(MODELS_DIR, "forecast_flow_prophet.csv"),
             os.path.join(REPORTS_EDA_DIR, "forecast_flow.png"))

    forecast(df, "Total_Revenue",
             os.path.join(MODELS_DIR, "forecast_revenue_prophet.csv"),
             os.path.join(REPORTS_EDA_DIR, "forecast_revenue.png"))

if __name__ == "__main__":
    main()
