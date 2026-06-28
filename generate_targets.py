import numpy as np
import pandas as pd


def generate_proxy_targets(input_path, output_path):
    # Load the downloaded weather data
    df = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date")

    # ----------------------------------------------------
    # 1. SOLAR ENERGY OUTPUT SIMULATION (E_solar)
    # ----------------------------------------------------
    # PV temperature coefficient (gamma) = -0.004 (typical silicon panel loss per degree above 25°C)
    gamma = 0.004
    performance_ratio = 0.75  # Typical system efficiency losses

    # Plant A: 2 kW capacity
    df["PlantA_E_solar"] = (
        2.0
        * df["GHI_Satellite"]
        * performance_ratio
        * (1 - gamma * (df["Ambient_Temp"] - 25))
    )

    # Plant B: 10 kW capacity
    df["PlantB_E_solar"] = (
        10.0
        * df["GHI_Satellite"]
        * performance_ratio
        * (1 - gamma * (df["Ambient_Temp"] - 25))
    )

    # ----------------------------------------------------
    # 2. BIOGAS ENERGY OUTPUT SIMULATION (E_biogas)
    # ----------------------------------------------------
    # Biogas production lags behind temperature due to slurry thermal inertia.
    # We will simulate a 14-day rolling window to capture the hydrolysis delay.
    smoothed_temp = df["Ambient_Temp"].rolling(window=14, min_periods=1).mean()

    # Define seasonal baseline feeding fluctuations (Monsoon seasonality effect)
    # Month 6, 7, 8 (June-August) represent high monsoon substrate enrichment
    df["Month"] = df.index.month
    feed_factor = df["Month"].apply(lambda m: 1.2 if m in [6, 7, 8] else 1.0)

    # Plant A: 15 m³ digester baseline output (approx 5 to 15 kWh daily max)
    df["PlantA_E_biogas"] = (smoothed_temp * 0.35) * feed_factor + np.random.normal(
        0, 0.5, len(df)
    )

    # Plant B: 60 m³ digester baseline output (approx 20 to 60 kWh daily max)
    df["PlantB_E_biogas"] = (smoothed_temp * 1.4) * feed_factor + np.random.normal(
        0, 1.5, len(df)
    )

    # Clip any negative outputs introduced by the random noise to 0
    target_cols = [
        "PlantA_E_solar",
        "PlantB_E_solar",
        "PlantA_E_biogas",
        "PlantB_E_biogas",
    ]
    df[target_cols] = df[target_cols].clip(lower=0)

    # Drop intermediate helping columns
    df = df.drop(columns=["Month"])

    # Save the prepared dataset
    df.to_csv(output_path)
    print(f"Target simulation complete! Dataset saved to: {output_path}")
    print(df[target_cols].head())


if __name__ == "__main__":
    generate_proxy_targets(
        "data/downloaded_weather_base.csv", "data/master_raw_dataset.csv"
    )