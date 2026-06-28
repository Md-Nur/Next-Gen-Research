import os
import pandas as pd
import requests

# 1. Configuration parameters
# Lat/Lon for center of Narail District, Bangladesh
LATITUDE = 23.1667
LONGITUDE = 89.5000

# Target Time Window: 2-Year Historical Frame (Jan 2023 - Dec 2024)
START_DATE = "20230101"
END_DATE = "20241231"

# Parameters mapping to system attributes:
# ALLSKY_SFC_SW_DWN = Global Horizontal Irradiance (GHI) - Core input for Solar PV prediction
# CLRSKY_SFC_SW_DWN = Clear Sky Solar Irradiance - Essential for computing cloud attenuation
# T2M               = Ambient Temperature at 2m - Governing factor for PV thermal efficiency and digester heating
# RH2M              = Relative Humidity at 2m
# WS10M             = Wind Speed at 10m - Controls ambient convective cooling
PARAMS = ["ALLSKY_SFC_SW_DWN", "CLRSKY_SFC_SW_DWN", "T2M", "RH2M", "WS10M"]


def fetch_nasa_power_data(lat, lon, start, end, parameters):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={','.join(parameters)}&"
        f"community=RE&"  # Renewable Energy community profile
        f"longitude={lon}&"
        f"latitude={lat}&"
        f"start={start}&"
        f"end={end}&"
        f"format=JSON"
    )

    print(f"Connecting to NASA API for Coordinates: Lat {lat}, Lon {lon}...")
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        json_data = response.json()
        # Parse parameter data dictionary
        raw_records = json_data["properties"]["parameter"]

        # Convert nested dictionaries into standard DataFrame
        df = pd.DataFrame(raw_records)
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df.index.name = "Date"

        # Relabel parameters into highly descriptive engineering headers
        df = df.rename(
            columns={
                "ALLSKY_SFC_SW_DWN": "GHI_Satellite",
                "CLRSKY_SFC_SW_DWN": "ClearSky_GHI",
                "T2M": "Ambient_Temp",
                "RH2M": "Humidity",
                "WS10M": "Wind_Speed",
            }
        )
        return df
    else:
        raise Exception(
            f"API Connection Failed. Status: {response.status_code}, Msg: {response.text}"
        )


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    try:
        master_weather_df = fetch_nasa_power_data(
            LATITUDE, LONGITUDE, START_DATE, END_DATE, PARAMS
        )
        master_weather_df.to_csv("data/downloaded_weather_base.csv")
        print("\nInitialization Complete!")
        print(f"Successfully saved {len(master_weather_df)} records.")
        print(master_weather_df.head())
    except Exception as e:
        print(f"Execution Error: {e}")