"""
Script to merge generation data from multiple sources into a single DataFrame.
"""

import pandas as pd
from fetch_load import get_total_load
from fetch_prices import get_energy_prices
from fetch_generation_per_type import get_generation_all_types
from fetch_weather import fetch_weather

def merge_all_sources(start_date, end_date, country_code="10YNL----------L"):
    """
    Merge generation, load, prices, and weather data into a single DataFrame.
    
    Parameters:
        start_date (str): Start date in format YYYYMMDDHH00.
        end_date (str): End date in format YYYYMMDDHH00.
        country_code (str): ENTSO-E country code (default is Netherlands).
    
    Returns:
        pd.DataFrame: Merged DataFrame with all data sources.
    """
    # Fetch data from all sources
    df_load = get_total_load(start_date, end_date, country_code)
    df_prices = get_energy_prices(start_date, end_date, country_code)
    df_generation = get_generation_all_types(start_date, end_date, country_code)
    df_weather = fetch_weather(52.4931, 5.4264, start_date, end_date)  # Coordinates for the Amsterdam

    # Merge all DataFrames on datetime
    dfs = []
    for df in [df_weather, df_generation, df_load, df_prices]:
        if df is not None and not df.empty:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            dfs.append(df)

    # Concatenate along columns based on datetime index
    df_concat = pd.concat(dfs, axis=1).reset_index()

    return df_concat

if __name__ == "__main__":
    # Example usage
    start_date = "202406250000"
    end_date = "202406260000"
    
    merged_data = merge_all_sources(start_date, end_date)
    print(merged_data.head())
    print(f"Total rows: {len(merged_data)}")
    print(f"Columns: {merged_data.columns.tolist()}")