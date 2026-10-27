"""Forecasting models behind one interface, so the backtest and the daily job treat them all the same.

A model gets everything known at the forecast origin and returns one value per half-hour for the next `horizon` steps:

    forecast(demand, temp, origin, horizon) -> pd.Series indexed by target time

`demand` is cut off at the origin by the caller, so a model can't peek. `temp` runs past the origin (it stands in
for a weather forecast); models that use it must only read the target times.

These are the benchmarks from the forecasting review post: anything cleverer has to beat them to be worth it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

STEP = pd.Timedelta("30min")
DAY = 48  # half-hours
WEEK = 7 * DAY


class Model(Protocol):
    name: str
    label: str

    def forecast(self, demand: pd.Series, temp: pd.Series, origin: pd.Timestamp, horizon: int) -> pd.Series: ...


def target_index(origin: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(origin + STEP, periods=horizon, freq=STEP, name="time")


@dataclass
class SeasonalNaive:
    """Repeat the last observed season: lag 48 is "same as yesterday", lag 336 "same as this day last week"."""

    name: str
    label: str
    lag: int

    def forecast(self, demand, temp, origin, horizon):
        targets = target_index(origin, horizon)
        last_season = demand.loc[:origin].iloc[-self.lag:].to_numpy()
        return pd.Series(np.resize(last_season, horizon), index=targets, name=self.name)


@dataclass
class WeeklyMean:
    """Average of the same half-hour on the same weekday over the last `weeks` weeks: smooths out one-off days."""

    name: str
    label: str
    weeks: int = 4

    def forecast(self, demand, temp, origin, horizon):
        targets = target_index(origin, horizon)
        past = demand.loc[:origin].iloc[-self.weeks * WEEK:].to_numpy().reshape(self.weeks, WEEK).mean(axis=0)
        return pd.Series(np.resize(past, horizon), index=targets, name=self.name)


MODELS: dict[str, Model] = {m.name: m for m in [
    SeasonalNaive("naive_day", "Same as yesterday", DAY),
    SeasonalNaive("naive_week", "Same day last week", WEEK),
    WeeklyMean("mean_4w", "Average of last 4 weeks"),
]}
