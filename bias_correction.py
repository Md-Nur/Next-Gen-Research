import pandas as pd


def apply_bias_correction(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=["Date"], index_col="Date")

    # Extract month to apply seasonal bias correction factors
    # Satellites routinely overestimate monsoon GHI due to complex cloud dynamics in Bangladesh
    df["Month"] = df.index.month

    # Define empirical correction factors based on regional ground-station benchmarks
    # (e.g., scaling down raw satellite GHI during heavy monsoon/dust periods)
    def get_correction_factor(month):
        if month in [6, 7, 8]:  # Monsoon months
            return 0.88  # 12% overestimation correction
        elif month in [12, 1, 2]:  # Winter fog/haze months
            return 0.93  # 7% correction for fog attenuation
        else:
            return 0.98  # Standard clear-sky minor adjustment

    df["Correction_Factor"] = df["Month"].apply(get_correction_factor)

    # Create the true Bias-Corrected GHI feature
    df["GHI_Bias_Corrected"] = df["GHI_Satellite"] * df["Correction_Factor"]

    # Recalculate your targets using the corrected, realistic weather input
    performance_ratio = 0.75
    gamma = 0.004

    df["PlantA_E_solar"] = (
        2.0
        * df["GHI_Bias_Corrected"]
        * performance_ratio
        * (1 - gamma * (df["Ambient_Temp"] - 25))
    )
    df["PlantB_E_solar"] = (
        10.0
        * df["GHI_Bias_Corrected"]
        * performance_ratio
        * (1 - gamma * (df["Ambient_Temp"] - 25))
    )

    # Drop helping columns and overwrite
    df = df.drop(columns=["Month", "Correction_Factor"])
    df.to_csv(output_path)

    print("Bias correction successfully applied to the database features!")
    print(df[["GHI_Satellite", "GHI_Bias_Corrected", "PlantA_E_solar"]].head())


if __name__ == "__main__":
    apply_bias_correction("data/processed_dataset.csv", "data/processed_dataset.csv")