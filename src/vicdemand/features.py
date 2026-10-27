"""Features for the machine-learning models: one row per (origin, target half-hour), using only what is known at the origin.

Built for many origins at once (training) or one (a live forecast) by the same code, so training and forecasting can't drift apart.
"""

from __future__ import annotations

from functools import cache

import holidays
import numpy as np
import pandas as pd

STEP = pd.Timedelta("30min")
DAY = 48


@cache
def _vic_holidays(years: tuple[int, ...]) -> frozenset:
    return frozenset(holidays.AU(subdiv="VIC", years=years))


def holiday_flags(times: pd.DatetimeIndex) -> np.ndarray:
    """1 where the half-hour falls on a Victorian public holiday (period-ending: 00:00 belongs to the day before)."""
    days = (times - STEP).normalize()
    hols = _vic_holidays(tuple(range(days.min().year, days.max().year + 1)))
    return np.array([d.date() in hols for d in days])


def make(demand: pd.Series, weather: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int) -> pd.DataFrame:
    """Rows indexed by (origin, time). `weather` must cover the target times (with a forecast, for live use)."""
    steps = np.arange(1, horizon + 1)
    origin = np.repeat(origins.to_numpy(), horizon)
    step = np.tile(steps, len(origins))
    t = pd.DatetimeIndex(origin + step * STEP)
    day = (t - STEP).normalize()  # the calendar day each half-hour belongs to

    def lag(delta) -> np.ndarray:
        return demand.reindex(t - delta).to_numpy()

    # Latest known value for this half-hour of the day: yesterday's for "today", the day before's for "tomorrow".
    days_back = np.ceil(step / DAY).astype(int)
    week = 7 * DAY * STEP
    daily = weather.resample("D", closed="right", label="left")  # period-ending half-hours, grouped by their calendar day
    day_max, day_mean = daily["temp"].max(), daily["temp"].mean()
    known_level = demand.rolling(DAY).mean()  # mean of the 24 h up to each time

    X = pd.DataFrame({
        "step": step,
        "slot": ((t - STEP).hour * 2 + (t - STEP).minute // 30).to_numpy(),
        "dow": day.dayofweek.to_numpy(),
        "holiday": holiday_flags(t).astype(int),
        "doy_sin": np.sin(2 * np.pi * day.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * day.dayofyear / 365.25),
        "temp": weather["temp"].reindex(t).to_numpy(),
        "solar": weather["solar"].reindex(t).to_numpy(),
        "day_max_temp": day_max.reindex(day).to_numpy(),
        "day_mean_temp": day_mean.reindex(day).to_numpy(),
        "prev_day_max_temp": day_max.reindex(day - pd.Timedelta(days=1)).to_numpy(),  # heat builds up over hot spells
        "lag_recent_day": demand.reindex(t - days_back * DAY * STEP).to_numpy(),
        "temp_recent_day": weather["temp"].reindex(t - days_back * DAY * STEP).to_numpy(),
        "lag_week": lag(week),
        "temp_week": weather["temp"].reindex(t - week).to_numpy(),
        "mean_4w": np.nanmean(np.stack([lag(k * week) for k in range(1, 5)]), axis=0),
        "level_24h": known_level.reindex(pd.DatetimeIndex(origin)).to_numpy(),
        "year": day.year.to_numpy(),  # slow drift: more rooftop solar, electrification
    }, index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(origin), t], names=["origin", "time"]))
    return X
