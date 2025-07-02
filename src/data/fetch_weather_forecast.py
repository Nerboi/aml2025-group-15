import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup Open-Meteo client with caching and retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_historical_forecast(lat, lon, start_date, end_date):
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,   # YYYY-MM-DD
        "end_date": end_date,       # YYYY-MM-DD
        "hourly": "temperature_2m,wind_speed_10m,wind_speed_100m,direct_radiation",
        "timezone": "UTC"
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
        "wind_speed_180m": hourly.Variables(1).ValuesAsNumpy(), # change this to wind_speed_10m and wind_speed_100m
        "wind_speed_120m": hourly.Variables(2).ValuesAsNumpy(),
        "direct_radiation": hourly.Variables(3).ValuesAsNumpy()
    }

    df = pd.DataFrame(hourly_data)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    return df

if __name__ == "__main__":
    start_date = "2025-06-30"
    end_date = "2025-07-02"
    df_weather = get_historical_forecast(52.4931, 5.4264, start_date, end_date)
    print(df_weather.tail())
