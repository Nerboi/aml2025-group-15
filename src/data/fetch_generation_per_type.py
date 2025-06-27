"""
Fetch Actual Generation per Production Type from ENTSO-E and return as DataFrame.
"""

import http.client
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

PSRTYPE_LABELS = {
    "B01": "biomass",
    "B02": "fossil_brown_coal_lignite",
    "B03": "fossil_coal_derived_gas",
    "B04": "fossil_gas",
    "B05": "fossil_hard_coal",
    "B06": "fossil_oil",
    "B07": "fossil_oil_shale",
    "B08": "fossil_peat",
    "B09": "geothermal",
    "B10": "hydro_pumped_storage",
    "B11": "hydro_run_of_river",
    "B12": "hydro_water_reservoir",
    "B13": "marine",
    "B14": "nuclear",
    "B15": "other_renewable",
    "B16": "solar",
    "B17": "waste",
    "B18": "wind_offshore",
    "B19": "wind_onshore",
    "B20": "other",
    "B25": "energy_storage"
}

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

    # Get data from API
    conn.request("GET", f"/api?{params}")
    res = conn.getresponse()
    xml_data = res.read().decode()

    root = ET.fromstring(xml_data)
    ns = {"ns": "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"}

    label = PSRTYPE_LABELS.get(psr_type)
    gen_data = []

    # Parse from XML to DataFrame
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
                f"quantity_{label}": quantity,
            })

    df = pd.DataFrame(gen_data)

    if df is None or df.empty:
        return df
    
    # Enforce start and end date boundaries
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.groupby("datetime", as_index=False)[f"quantity_{label}"].sum() # Group by date and hour

    start_dt = datetime.strptime(start_date, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)

    df = df[(df["datetime"] >= start_dt) & (df["datetime"] < end_dt)]

    return df


def get_generation_all_types(start_date, end_date, country_code="10YNL----------L"):
    """
    Fetch actual generation for all available production types, return as single merged DataFrame.
    Merges on hourly timestamps, columns labeled by quantity_<production_type>.
    """
    dfs = []

    for psr_type, label in PSRTYPE_LABELS.items():
        df = get_generation_per_type(start_date, end_date, psr_type=psr_type, country_code=country_code)

        if df is None or df.empty:
            print(f"No data for {label} ({psr_type}), skipping.")
            continue

        df = df.set_index("datetime")
        dfs.append(df)

    if not dfs:
        print("No generation data returned for any type.")
        return pd.DataFrame()

    # Concatenate on columns, timestamps aligned
    merged_df = pd.concat(dfs, axis=1).reset_index().sort_values("datetime")

    return merged_df

if __name__ == "__main__":
    start_date = "202406250000"
    end_date = "202406260000"
    df_generation = get_generation_per_type(start_date, end_date, psr_type="B16")
    print(df_generation.head())
    df_all_gen = get_generation_all_types(start_date, end_date)
    print(df_all_gen.head())
