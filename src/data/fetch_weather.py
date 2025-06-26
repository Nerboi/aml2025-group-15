"""
Fetch hourly weather data from Open-Meteo API and return as DataFrame.
This script retrieves weather data for a specified latitude and longitude.
"""

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup Open-Meteo client with caching and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def fetch_weather(lat, lon):
    """
    Fetch hourly weather data from Open-Meteo and return as DataFrame with city name.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_180m,wind_speed_120m,direct_radiation",
    }

    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]

    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()} s")

    # Process hourly data
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_180m = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_120m = hourly.Variables(2).ValuesAsNumpy()
    hourly_direct_radiation = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_2m": hourly_temperature_2m,
        "wind_speed_180m": hourly_wind_speed_180m,
        "wind_speed_120m": hourly_wind_speed_120m,
        "direct_radiation": hourly_direct_radiation
    }

    df = pd.DataFrame(hourly_data)

    return df

if __name__ == "__main__":
    df_weather = fetch_weather(52.4931, 5.4264)
    print(df_weather.head())
