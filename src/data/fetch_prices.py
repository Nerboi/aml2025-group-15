"""
Fetch Energy prices from ENTSO-E and return as DataFrame.
This script retrieves energy prices for a specified date range and country code.
"""

import http.client
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
if ENTSOE_API_KEY is None:
    print("ENTSOE_API_KEY not found. Please set it in your .env file.")

def get_energy_prices(start_date, end_date, country_code="10YNL----------L"):
    conn = http.client.HTTPSConnection("web-api.tp.entsoe.eu")

    params = (
        f"securityToken={ENTSOE_API_KEY}"
        f"&documentType=A44"
        f"&in_Domain={country_code}"
        f"&out_Domain={country_code}"
        f"&periodStart={start_date}"
        f"&periodEnd={end_date}"
    )

    conn.request("GET", f"/api?{params}")

    res = conn.getresponse()
    xml_data = res.read().decode()

    root = ET.fromstring(xml_data)

    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}

    prices = []
    for timeseries in root.findall(".//ns:TimeSeries", ns):
        period = timeseries.find(".//ns:Period", ns)
        time_interval = period.find(".//ns:timeInterval", ns)
        start = time_interval.find(".//ns:start", ns).text
        resolution = period.find(".//ns:resolution", ns).text

        dt_start = datetime.strptime(start, "%Y-%m-%dT%H:%MZ")

        for i, point in enumerate(period.findall(".//ns:Point", ns)):
            price = float(point.find(".//ns:price.amount", ns).text)
            timestamp = dt_start + pd.to_timedelta(i, unit="h")

            prices.append({"datetime": timestamp, "price_EUR_MWh": price})

    df = pd.DataFrame(prices)
    return df

if __name__ == "__main__":
    # Format: YYYYMMDDHH00
    start_date = "202406250000"
    end_date = "202406260000"
    df_prices = get_energy_prices(start_date, end_date)
    print(df_prices.head())


