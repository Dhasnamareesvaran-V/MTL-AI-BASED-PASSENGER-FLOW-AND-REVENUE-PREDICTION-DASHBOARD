import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CLEAN = os.path.join(ROOT, "data", "processed", "clean_bengaluru_metro.csv")
FEATURES_OUT = os.path.join(ROOT, "data", "processed", "features_bengaluru_metro.csv")

def main():

    df = pd.read_csv(DATA_CLEAN, parse_dates=["Record Date"])
    df = df.dropna(subset=["Record Date", "Total_Flow", "Total_Revenue"])
    df = df.sort_values("Record Date")

    
    df["day_of_week"] = df["Record Date"].dt.day_name()
    df["is_weekend"] = df["Record Date"].dt.weekday >= 5
    df["month"] = df["Record Date"].dt.month
    df["year"] = df["Record Date"].dt.year

    df["flow_lag1"] = df["Total_Flow"].shift(1)
    df["flow_lag7"] = df["Total_Flow"].shift(7)
    df["revenue_lag1"] = df["Total_Revenue"].shift(1)
    df["revenue_lag7"] = df["Total_Revenue"].shift(7)

    
    df["flow_roll7"] = df["Total_Flow"].rolling(7).mean()
    df["flow_roll14"] = df["Total_Flow"].rolling(14).mean()
    df["revenue_roll7"] = df["Total_Revenue"].rolling(7).mean()
    df["revenue_roll14"] = df["Total_Revenue"].rolling(14).mean()

    
    df.to_csv(FEATURES_OUT, index=False)
    print(f"[INFO] Features dataset saved to: {FEATURES_OUT}")

if __name__ == "__main__":
    main()
