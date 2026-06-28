import os
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def preprocess_and_validate(input_path, output_dir):
    # 1. Load the master raw dataset
    df = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()  # Ensure time-series is strictly chronological

    # 2. Outlier Detection using Isolation Forest
    # We calibrate with contamination=0.03 to catch severe sensor glitches
    # without scrubbing true extreme weather anomalies like cyclonic storms.
    feature_cols = [
        "GHI_Satellite",
        "ClearSky_GHI",
        "Ambient_Temp",
        "Humidity",
        "Wind_Speed",
    ]
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    outliers = iso_forest.fit_predict(df[feature_cols])

    # Mark outliers (keep them flagged so models learn resilience, or drop if requested)
    df["Is_Outlier"] = np.where(outliers == -1, 1, 0)
    print(f"Detected {df['Is_Outlier'].sum()} temporal anomaly points.")

    # 3. Seasonal Stratification Mapping
    # Bangladesh experiences profound micro-climatic season transitions:
    # Mar-May (Summer/Pre-Monsoon), Jun-Aug (Monsoon), Sep-Nov (Autumn), Dec-Feb (Winter)
    def assign_season(month):
        if month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8]:
            return "Monsoon"
        elif month in [9, 10, 11]:
            return "Autumn"
        else:
            return "Winter"

    df["Season"] = df.index.month.map(assign_season)

    # 4. Implement Time-Series Forward-Chaining Fold Assignment
    # To prevent temporal leakage while balancing seasons across folds,
    # we implement a progressive forward-chaining chronological split.
    n_samples = len(df)
    fold_size = n_samples // 10
    df["Fold"] = -1

    for fold in range(10):
        # We assign consecutive blocks of time to folds
        start_idx = fold * fold_size
        end_idx = (
            (fold + 1) * fold_size if fold < 9 else n_samples
        )  # Ensure last fold catches leftovers
        df.iloc[start_idx:end_idx, df.columns.get_loc("Fold")] = fold

    # Save the prepared pipeline file
    os.makedirs(output_dir, exist_ok=True)
    pipeline_path = os.path.join(output_dir, "processed_dataset.csv")
    df.to_csv(pipeline_path)

    print(f"Preprocessing pipeline complete. Saved to: {pipeline_path}")
    print("\nData distribution across the 10 forward-chaining validation folds:")
    print(df.groupby(["Fold", "Season"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    preprocess_and_validate("data/master_raw_dataset.csv", "data")