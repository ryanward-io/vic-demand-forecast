"""Forecasting models behind one interface, so the backtest and the daily job treat them all the same.

A model gets everything known at the forecast origin and returns one value per half-hour for the next `horizon` steps:

    forecast(demand, weather, origin, horizon) -> pd.Series indexed by target time

`demand` is cut off at the origin by the caller, so a model can't peek. `weather` (columns temp, solar) is what actually
happened up to the origin and a weather *forecast* after it. Models that learn from history also have
`fit(demand, weather)`, called on data before the test window.

The ladder, simplest first (from the 2024 forecasting review post: anything cleverer has to beat the benchmarks):
benchmarks -> MSTL (classical decomposition + ETS) -> LightGBM with calendar and weather -> an LSTM trained from scratch
(lstm.py) -> Chronos-2 (pretrained foundation model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from . import features

STEP = pd.Timedelta("30min")
DAY = 48  # half-hours
WEEK = 7 * DAY


class Model(Protocol):
    name: str
    label: str

    def forecast(self, demand: pd.Series, weather: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.Series: ...


def target_index(origin: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(origin + STEP, periods=horizon, freq=STEP, name="time")


@dataclass
class SeasonalNaive:
    """Repeat the last observed season: lag 48 is "same as yesterday", lag 336 "same as this day last week"."""

    name: str
    label: str
    lag: int

    def forecast(self, demand, weather, origin, horizon):
        last_season = demand.loc[:origin].iloc[-self.lag:].to_numpy()
        return pd.Series(np.resize(last_season, horizon), index=target_index(origin, horizon), name=self.name)


@dataclass
class WeeklyMean:
    """Average of the same half-hour on the same weekday over the last `weeks` weeks: smooths out one-off days."""

    name: str
    label: str
    weeks: int = 4

    def forecast(self, demand, weather, origin, horizon):
        past = demand.loc[:origin].iloc[-self.weeks * WEEK:].to_numpy().reshape(self.weeks, WEEK).mean(axis=0)
        return pd.Series(np.resize(past, horizon), index=target_index(origin, horizon), name=self.name)


@dataclass
class MSTL:
    """Classical statistics: split the last 8 weeks into trend + daily + weekly cycles (MSTL), extend the trend with ETS.
    Uses demand only, no weather. The Python cousin of what the fpp book (and my R-forecasting repo) teaches."""

    name: str = "mstl"
    label: str = "MSTL + ETS"
    weeks: int = 8

    def forecast(self, demand, weather, origin, horizon):
        from statsforecast.models import MSTL as _MSTL, AutoETS

        y = demand.loc[:origin].iloc[-self.weeks * WEEK:].to_numpy()
        model = _MSTL(season_length=[DAY, WEEK], trend_forecaster=AutoETS(model="ZZN"))
        mean = model.forecast(y=y, h=horizon)["mean"]
        return pd.Series(mean, index=target_index(origin, horizon), name=self.name)


@dataclass
class LightGBM:
    """Gradient-boosted trees on calendar, weather and recent-demand features (see features.py): the M5 winner's toolkit.

    One "direct" model for all 96 lead times (the step is a feature). It predicts the gap from the 4-week average for that
    half-hour, so it only has to learn what's different about this day (weather, holidays), not the level itself.
    """

    name: str = "lightgbm"
    label: str = "LightGBM (calendar + weather)"
    params: dict = field(default_factory=lambda: {
        "objective": "l1", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0,
    })
    rounds: int = 600
    booster: object = None

    def fit(self, demand, weather):
        import lightgbm as lgb

        # Every midnight with a full 4 weeks of history behind it and 2 days of actuals after it.
        origins = pd.date_range(demand.index[0].normalize() + pd.Timedelta(days=29), demand.index[-1] - pd.Timedelta(days=2), freq="D")
        X = features.make(demand, weather, origins, 2 * DAY)
        y = demand.reindex(X.index.get_level_values("time")).to_numpy() - X["mean_4w"].to_numpy()
        ok = ~np.isnan(y)
        self.booster = lgb.train(self.params, lgb.Dataset(X[ok], y[ok]), num_boost_round=self.rounds)
        return self

    def forecast(self, demand, weather, origin, horizon):
        X = features.make(demand, weather, pd.DatetimeIndex([origin]), horizon)
        pred = self.booster.predict(X) + X["mean_4w"].to_numpy()
        return pd.Series(pred, index=target_index(origin, horizon), name=self.name)


@dataclass
class Chronos2:
    """Amazon's Chronos-2 (Oct 2025): a transformer pretrained on a large mix of real and synthetic time series that
    forecasts series it has never seen ("zero-shot"), with no training on ours. It reads past demand alongside past
    and forecast weather and holidays as covariates. It forecasts a distribution: we use the median as the forecast and
    keep the 10% and 90% quantiles (an 80% range) in the Series' `attrs` as "low" and "high"."""

    name: str = "chronos2"
    label: str = "Chronos-2 (foundation model)"
    model_id: str = "amazon/chronos-2"
    context: int = 12 * WEEK  # 12 weeks of half-hours
    pipeline: object = None

    def _load(self):
        if self.pipeline is None:
            import torch
            from chronos import Chronos2Pipeline

            torch.set_num_threads(max(1, torch.get_num_threads()))
            self.pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map="cpu")
        return self.pipeline

    def inputs(self, demand, weather, origin, horizon) -> dict:
        past = demand.loc[:origin].iloc[-self.context:]
        future = target_index(origin, horizon)
        cov = lambda idx: {"temp": weather["temp"].reindex(idx).to_numpy(), "solar": weather["solar"].reindex(idx).to_numpy(),
                           "holiday": features.holiday_flags(idx).astype(float)}
        return {"target": past.to_numpy(), "past_covariates": cov(past.index), "future_covariates": cov(future)}

    def forecast(self, demand, weather, origin, horizon):
        return self.forecast_many([(demand, weather, origin)], horizon)[0]

    def forecast_many(self, cases, horizon):
        """Batch several origins in one call (much faster for backtests). cases: [(demand, weather, origin)]."""
        pipe = self._load()
        quantiles, _ = pipe.predict_quantiles([self.inputs(d, w, o, horizon) for d, w, o in cases], prediction_length=horizon,
                                              quantile_levels=[0.1, 0.5, 0.9])
        out = []
        for q, (_, _, o) in zip(quantiles, cases):
            f = pd.Series(q[0, :, 1].numpy(), index=target_index(o, horizon), name=self.name)
            f.attrs = {"low": q[0, :, 0].numpy(), "high": q[0, :, 2].numpy()}
            out.append(f)
        return out


@dataclass
class Timestral:
    """Mistral's Timestral (Oct 2026): another zero-shot foundation model, given exactly the same inputs as Chronos-2
    (12 weeks of demand; temperature, solar and holidays known over the history and the forecast) so the two compare
    like for like. It outputs 21 quantiles; we keep the median and the 10-90% range, as for Chronos-2.

    Weights: Modified MIT licence, (c) 2026 Mistral AI. Free unless your company's or employer's revenue is above
    US$20M a month; see https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE."""

    name: str = "timestral"
    label: str = "Timestral (foundation model)"
    model_id: str = "mistralai/Timestral-209M"
    context: int = 12 * WEEK
    forecaster: object = None

    def _load(self):
        if self.forecaster is None:
            from timestral import TimestralForecaster

            self.forecaster = TimestralForecaster.from_pretrained(self.model_id)
        return self.forecaster

    def _covariates(self, weather, idx) -> np.ndarray:
        return np.stack([weather["temp"].reindex(idx).to_numpy(), weather["solar"].reindex(idx).to_numpy(),
                         features.holiday_flags(idx).astype(float)])

    def forecast(self, demand, weather, origin, horizon):
        return self.forecast_many([(demand, weather, origin)], horizon)[0]

    def forecast_many(self, cases, horizon):
        """Batch several origins in one call. cases: [(demand, weather, origin)]."""
        import torch

        context, known = [], []
        for d, w, o in cases:
            past = d.loc[:o].iloc[-self.context:]
            context.append(past.to_numpy(dtype=float))
            known.append(self._covariates(w, past.index.append(target_index(o, horizon))))
        with torch.inference_mode():
            f = self._load().forecast(np.stack(context)[:, None, :], prediction_length=horizon, known_covariates=np.stack(known))
        median, low, high = (np.asarray(t[:, 0].float().cpu()) for t in (f.median, f.quantile(0.1), f.quantile(0.9)))
        out = []
        for i, (_, _, o) in enumerate(cases):
            s = pd.Series(median[i], index=target_index(o, horizon), name=self.name)
            s.attrs = {"low": low[i], "high": high[i]}
            out.append(s)
        return out


BENCHMARKS = [
    SeasonalNaive("naive_day", "Same as yesterday", DAY),
    SeasonalNaive("naive_week", "Same day last week", WEEK),
    WeeklyMean("mean_4w", "Average of last 4 weeks"),
]


def all_models(cache: Path = Path("data/cache")) -> dict[str, Model]:
    from .aemo import AEMOForecast
    from .lstm import LSTM

    return {m.name: m for m in [*BENCHMARKS, MSTL(), LightGBM(), LSTM(), Chronos2(), Timestral(), AEMOForecast(cache=cache / "aemo")]}


# Averages of other models' forecasts. Models that make different kinds of mistakes often beat each of them alone.
ENSEMBLES = {"ensemble": ("Average of LightGBM and Chronos-2", ["lightgbm", "chronos2"])}


def ensemble(rows: pd.DataFrame, name: str, members: list[str]) -> pd.DataFrame:
    """Rows with columns model, origin, time, forecast (+ anything else) -> the members' mean as model `name`."""
    keys = [c for c in ("origin", "time", "step", "actual", "issued") if c in rows.columns]
    mean = rows[rows["model"].isin(members)].groupby(keys, dropna=False)["forecast"].mean().reset_index()
    return mean.assign(model=name)
