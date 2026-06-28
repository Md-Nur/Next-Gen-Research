import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    # Avoid division by zero in MAPE if any target is perfectly 0
    mape = mean_absolute_percentage_error(y_true + 1e-5, y_pred + 1e-5) * 100
    return r2, mape


def run_baseline_pipeline(data_path, target_subsystem):
    """target_subsystem: 'solar' or 'biogas'"""
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    # Define features and targets based on subsystem selection
    features = [
        "GHI_Satellite",
        "ClearSky_GHI",
        "Ambient_Temp",
        "Humidity",
        "Wind_Speed",
    ]

    if target_subsystem == "solar":
        targets = {"PlantA": "PlantA_E_solar", "PlantB": "PlantB_E_solar"}
    else:
        targets = {"PlantA": "PlantA_E_biogas", "PlantB": "PlantB_E_biogas"}

    # Define our baseline dictionary
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42),
        "ANN": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }

    # Tracking matrix for results
    results = {
        model_name: {"PlantA_R2": [], "PlantA_MAPE": [], "PlantB_R2": [], "PlantB_MAPE": []}
        for model_name in models
    }

    print(f"\n==========================================")
    print(f" Executing Baselines for: {target_subsystem.upper()} SUBSYSTEM")
    print(f"==========================================")

    # 10-Fold Forward Chaining Training Loop
    # In time-series validation, Fold 'k' uses Folds 0 to k-1 for training, and Fold k for validation.
    for val_fold in range(1, 10):
        train_df = df[df["Fold"] < val_fold]
        val_df = df[df["Fold"] == val_fold]

        X_train, X_val = train_df[features], val_df[features]

        # Fit Scaler on training split ONLY to protect against data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for model_name, model in models.items():
            # ---- Process Plant A ----
            y_train_A = train_df[targets["PlantA"]]
            y_val_A = val_df[targets["PlantA"]]

            # Input scaled features for linear/ANN models, raw for trees
            X_tr = (
                X_train_scaled
                if model_name in ["LinearRegression", "ANN"]
                else X_train
            )
            X_va = (
                X_val_scaled if model_name in ["LinearRegression", "ANN"] else X_val
            )

            model.fit(X_tr, y_train_A)
            preds_A = model.predict(X_va)
            r2_A, mape_A = evaluate_model(y_val_A, preds_A)

            results[model_name]["PlantA_R2"].append(r2_A)
            results[model_name]["PlantA_MAPE"].append(mape_A)

            # ---- Process Plant B ----
            y_train_B = train_df[targets["PlantB"]]
            y_val_B = val_df[targets["PlantB"]]

            model.fit(X_tr, y_train_B)
            preds_B = model.predict(X_va)
            r2_B, mape_B = evaluate_model(y_val_B, preds_B)

            results[model_name]["PlantB_R2"].append(r2_B)
            results[model_name]["PlantB_MAPE"].append(mape_B)

    # Print Average Metrics across all folds for the writing team
    for model_name in models:
        print(f"\n📌 Model: {model_name}")
        print(
            f"  Plant A -> Mean R²: {np.mean(results[model_name]['PlantA_R2']):.4f} | Mean MAPE: {np.mean(results[model_name]['PlantA_MAPE']):.2f}%"
        )
        print(
            f"  Plant B -> Mean R²: {np.mean(results[model_name]['PlantB_R2']):.4f} | Mean MAPE: {np.mean(results[model_name]['PlantB_MAPE']):.2f}%"
        )


if __name__ == "__main__":
    # Run the evaluations for both solar and biogas configurations
    run_baseline_pipeline("data/processed_dataset.csv", "solar")
    run_baseline_pipeline("data/processed_dataset.csv", "biogas")