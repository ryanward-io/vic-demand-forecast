"""Download and cache the inputs: AEMO half-hourly demand for Victoria and Melbourne temperature.

All timestamps are naive NEM time (AEST, UTC+10 all year, no daylight saving), which is what AEMO publishes.
A half-hour is labelled by its end, as AEMO does: 00:30 covers 00:00-00:30.
"""

from __future__ import annotations

import io
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd

REGION = "VIC1"
# AEMO's aggregated price and demand files, one CSV per region per month (updated daily for the current month).
AEMO_URL = "https://www.aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{ym}_{region}.csv"
# AEMO blocks requests without a browser-like user agent.
HEADERS = {"User-Agent": "Mozilla/5.0 (vic-demand-forecast; https://ryanward.dev)"}

# Melbourne CBD. Open-Meteo: free, no key, CC BY 4.0. "Australia/Brisbane" is UTC+10 with no DST, i.e. NEM time.
LAT, LON = -37.81, 144.96
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NEM_TZ = "Australia/Brisbane"
# The reanalysis archive runs about 5 days behind; anything more recent comes from the forecast API.
ARCHIVE_LAG_DAYS = 10

FIRST_MONTH = "2019-01"


def _months(start: str, end: date) -> list[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(start, end.strftime("%Y-%m"), freq="M")]


def fetch_demand_month(ym: str, region: str = REGION) -> pd.Series:
    """One month of demand in MW, averaged to half-hours (files since Oct 2021 are 5-minute)."""
    r = httpx.get(AEMO_URL.format(ym=ym, region=region), headers=HEADERS, follow_redirects=True, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["SETTLEMENTDATE"], date_format="%Y/%m/%d %H:%M:%S")
    s = df.set_index("SETTLEMENTDATE")["TOTALDEMAND"].astype(float)
    # Period-ending labels: the 5-minute intervals ending 00:05..00:30 make up the half-hour ending 00:30.
    per_half_hour = pd.Timedelta("30min") // s.index.to_series().diff().min()
    halves = s.resample("30min", closed="right", label="right")
    s = halves.mean()[halves.count() == per_half_hour]  # drop a half-hour that isn't complete yet
    s.index.name = "time"
    return s.rename("demand")


def load_demand(cache: Path, today: date | None = None) -> pd.Series:
    """All demand from FIRST_MONTH to now. Finished months are cached forever; the current month is always refetched."""
    today = today or date.today()
    current = today.strftime("%Y%m")
    cache.mkdir(parents=True, exist_ok=True)
    parts = []
    for ym in _months(FIRST_MONTH, today):
        path = cache / f"demand_{ym}.parquet"
        if path.exists() and ym != current:
            parts.append(pd.read_parquet(path)["demand"])
            continue
        s = fetch_demand_month(ym)
        if ym != current:
            s.to_frame().to_parquet(path)
        parts.append(s)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")].dropna()


def _open_meteo(url: str, params: dict) -> pd.Series:
    r = httpx.get(url, params={"latitude": LAT, "longitude": LON, "hourly": "temperature_2m", "timezone": NEM_TZ, **params},
                  timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    return pd.Series(h["temperature_2m"], index=pd.to_datetime(h["time"]), name="temp", dtype=float)


def load_temperature(cache: Path, today: date | None = None) -> pd.Series:
    """Hourly temperature (deg C) from FIRST_MONTH to two days ahead, interpolated to half-hours.

    Past values are reanalysis (what actually happened); the last ~10 days and the future come from the forecast API.
    Note for backtests: using actual temperatures flatters any model that uses them, since a real forecast only has a weather forecast.
    """
    today = today or date.today()
    cache.mkdir(parents=True, exist_ok=True)
    split = today - timedelta(days=ARCHIVE_LAG_DAYS)
    parts = []
    # Archive in yearly chunks; years that have fully passed the archive lag are cached.
    for year in range(int(FIRST_MONTH[:4]), split.year + 1):
        path = cache / f"temp_{year}.parquet"
        end = min(date(year, 12, 31), split)
        if path.exists() and end == date(year, 12, 31):
            parts.append(pd.read_parquet(path)["temp"])
            continue
        s = _open_meteo(ARCHIVE_URL, {"start_date": f"{year}-01-01", "end_date": end.isoformat()})
        if end == date(year, 12, 31):
            s.to_frame().to_parquet(path)
        parts.append(s)
    recent = _open_meteo(FORECAST_URL, {"past_days": ARCHIVE_LAG_DAYS + 2, "forecast_days": 3})
    s = pd.concat(parts + [recent[recent.index.date > split]]).sort_index()
    s = s[~s.index.duplicated(keep="last")].dropna()
    return s.resample("30min").interpolate()
