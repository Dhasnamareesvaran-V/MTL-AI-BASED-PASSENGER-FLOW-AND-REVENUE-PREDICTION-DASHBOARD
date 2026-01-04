import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEAN = os.path.join(ROOT, "data", "processed", "clean_bengaluru_metro.csv")
REPORTS_EDA_DIR = os.path.join(ROOT, "reports", "eda")
REPORTS_METRICS_DIR = os.path.join(ROOT, "reports", "metrics")

def ensure_dirs():
    os.makedirs(REPORTS_EDA_DIR, exist_ok=True)
    os.makedirs(REPORTS_METRICS_DIR, exist_ok=True)

def main():
    ensure_dirs()
    df = pd.read_csv(DATA_CLEAN, parse_dates=["Record Date"])
    df = df.dropna(subset=["Record Date", "Total_Flow", "Total_Revenue"])

    
    summary = df[["Total_Flow", "Total_Revenue"]].describe()
    summary.to_string(open(os.path.join(REPORTS_METRICS_DIR, "eda_summary.txt"), "w"))

    
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["Total_Flow"], bins=30, ax=ax[0], color="tab:blue")
    ax[0].set_title("Passenger Flow Distribution")
    sns.histplot(df["Total_Revenue"], bins=30, ax=ax[1], color="tab:green")
    ax[1].set_title("Revenue Distribution")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_EDA_DIR, "flow_revenue_distribution.png"))
    plt.close(fig)

    
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(df["Record Date"], df["Total_Flow"], color="tab:blue")
    ax[0].set_title("Daily Passenger Flow")
    ax[1].plot(df["Record Date"], df["Total_Revenue"], color="tab:green")
    ax[1].set_title("Daily Revenue")
    plt.tight_layout()
    fig.savefig(os.path.join(REPORTS_EDA_DIR, "flow_revenue_timeseries.png"))
    plt.close(fig)

    print("[INFO] EDA completed. Outputs saved.")

if __name__ == "__main__":
    main()
