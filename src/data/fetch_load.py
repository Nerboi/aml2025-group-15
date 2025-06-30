"""
Fetch Actual Load from ENTSO-E and return as DataFrame.
"""

import http.client
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
if ENTSOE_API_KEY is None:
    print("ENTSOE_API_KEY not found. Please set it in your .env file.")

def get_total_load(start_date, end_date, country_code="10YNL----------L"):
    """
    Fetch total load data from ENTSO-E for a given date range and country code.
    """
    conn = http.client.HTTPSConnection("web-api.tp.entsoe.eu")

    params = (
        f"securityToken={ENTSOE_API_KEY}"
        f"&documentType=A65"
        f"&processType=A16"
        f"&outBiddingZone_Domain={country_code}"
        f"&periodStart={start_date}"
        f"&periodEnd={end_date}"
    )

    conn.request("GET", f"/api?{params}")
    res = conn.getresponse()
    xml_data = res.read().decode()

    root = ET.fromstring(xml_data)
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

    gen_data = []

    for timeseries in root.findall(".//ns:TimeSeries", ns):
        period = timeseries.find(".//ns:Period", ns)
        time_interval = period.find(".//ns:timeInterval", ns)
        start = time_interval.find(".//ns:start", ns).text
        resolution = period.find(".//ns:resolution", ns).text

        dt_start = datetime.strptime(start, "%Y-%m-%dT%H:%MZ")

        for point in period.findall(".//ns:Point", ns):
            position = int(point.find(".//ns:position", ns).text)
            quantity = float(point.find(".//ns:quantity", ns).text)
            timestamp = dt_start + pd.to_timedelta(position - 1, unit="h")

            gen_data.append({
                "datetime": timestamp,
                "quantity_MW": quantity,
            })

    df = pd.DataFrame(gen_data)
    # Enforce start and end date boundaries
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.groupby("datetime", as_index=False)[f"quantity_MW"].sum()

    start_dt = datetime.strptime(start_date, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)].reset_index(drop=True)

    return df

if __name__ == "__main__":
    start_date = "202406250000"
    end_date = "202406260000"
    df_generation = get_total_load(start_date, end_date)
    print(df_generation.head())
