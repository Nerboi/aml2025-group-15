"""
Fetch hourly weather data from Open-Meteo API and return as DataFrame.
This script retrieves weather data for a specified latitude and longitude.
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime, timezone

# Setup Open-Meteo client with caching and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather(lat, lon, start_date, end_date):
    """
    Fetch hourly weather data from Open-Meteo aligned with ENTSO-E date format.
    
    ENTSO-E date format example: '202406250000'
    """
    # Convert to ISO date (YYYY-MM-DD) for Open-Meteo API
    start_dt = datetime.strptime(start_date, "%Y%m%d%H%M")
    end_dt = datetime.strptime(end_date, "%Y%m%d%H%M")

    iso_start = start_dt.strftime("%Y-%m-%d")
    iso_end = end_dt.strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 52.52,
        "longitude": 13.41,
        "start_date": iso_start,
        "end_date": iso_end,
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,direct_radiation"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    hourly = response.Hourly()
    hourly_data = {
        "datetime": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(),
        "wind_speed_180m": hourly.Variables(1).ValuesAsNumpy(),
        "wind_speed_120m": hourly.Variables(2).ValuesAsNumpy(),
        "direct_radiation": hourly.Variables(3).ValuesAsNumpy()
    }

    df = pd.DataFrame(hourly_data)
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[(df["datetime"] >= iso_start) & (df["datetime"] < iso_end)].reset_index(drop=True)

    return df

if __name__ == "__main__":
    start_date = "202406250000"
    end_date = "202406260000"
    df_weather = get_weather(52.4931, 5.4264, start_date, end_date)
    print(df_weather.head())
