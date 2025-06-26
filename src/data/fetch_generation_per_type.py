"""
Fetch Actual Generation per Production Type from ENTSO-E and return as DataFrame.
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

def get_generation_per_type(start_date, end_date, psr_type, country_code="10YNL----------L"):
    """
    Fetch Actual Generation per Production Type from ENTSO-E and return as DataFrame.
    If hourly=True, data is aggregated to hourly intervals.
    """
    conn = http.client.HTTPSConnection("web-api.tp.entsoe.eu")

    params = (
        f"securityToken={ENTSOE_API_KEY}"
        f"&documentType=A75"
        f"&processType=A16"
        f"&in_Domain={country_code}"
        f"&periodStart={start_date}"
        f"&periodEnd={end_date}"
        f"&psrType={psr_type}"
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
        psr_type = timeseries.find(".//ns:MktPSRType/ns:psrType", ns).text

        for point in period.findall(".//ns:Point", ns):
            position = int(point.find(".//ns:position", ns).text)
            quantity = float(point.find(".//ns:quantity", ns).text)
            timestamp = dt_start + pd.to_timedelta(position - 1, unit="h")

            gen_data.append({
                "datetime": timestamp,
                "quantity_MW": quantity,
                "psrType": psr_type
            })

    df = pd.DataFrame(gen_data)

    return df

if __name__ == "__main__":
    start_date = "202406250000"
    end_date = "202406260000"
    df_generation = get_generation_per_type(start_date, end_date, psr_type="B16")
    print(df_generation.head())
