"""Rolling-origin backtest (the "tsCV" from the forecasting review post).

Every day in the test window we stand at midnight, pretend we know nothing after it, forecast the next two days,
and later compare with what happened. Scores are split by lead time: "today" is steps 1-48, "tomorrow" 49-96.

Two things keep it honest:
- models that learn are trained once, on data that ends before the test window starts;
- after the origin, models see the weather *forecast* that existed at the time, never the weather that happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import DAY, Model, target_index

HORIZON = 2 * DAY


def weather_at(origin: pd.Timestamp, actual: pd.DataFrame, forecasts: dict[int, pd.DataFrame], horizon: int = HORIZON) -> pd.DataFrame:
    """Weather as a forecaster standing at `origin` saw it: actuals up to the origin, then the 1-day-old forecast for
    today and the 2-day-old one for tomorrow."""
    targets = target_index(origin, horizon)
    ahead = pd.concat([forecasts[1].reindex(targets[:DAY]), forecasts[2].reindex(targets[DAY:])])
    return pd.concat([actual.loc[:origin], ahead])


def run(models: list[Model], demand: pd.Series, weather: pd.DataFrame, start: str, end: str,
        weather_forecasts: dict[int, pd.DataFrame] | None = None, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per (model, origin, target): the forecast and the actual.

    Without `weather_forecasts` the models see actual weather after the origin (only for tests and quick checks)."""
    train_end = pd.Timestamp(start) - pd.Timedelta("30min")
    for m in models:
        if hasattr(m, "fit"):
            m.fit(demand.loc[:train_end], weather.loc[:train_end])

    origins = [o for o in pd.date_range(start, end, freq="D") if o in demand.index]
    cases = [(demand.loc[:o], weather_at(o, weather, weather_forecasts, horizon) if weather_forecasts else weather, o) for o in origins]
    rows = []
    for m in models:
        if hasattr(m, "forecast_many"):
            preds = [f for i in range(0, len(cases), 32) for f in m.forecast_many(cases[i:i + 32], horizon)]
        else:
            preds = [m.forecast(d, w, o, horizon) for d, w, o in cases]
        for (_, _, o), f in zip(cases, preds):
            rows.append(pd.DataFrame({"model": m.name, "origin": o, "time": f.index, "step": np.arange(1, horizon + 1),
                                      "forecast": f.to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    out["actual"] = demand.reindex(out["time"]).to_numpy()
    return out.dropna(subset=["actual"])


def lead(step: pd.Series) -> pd.Series:
    return np.where(step <= DAY, "today", "tomorrow")


def score(results: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE in MW, MAPE in %, per model and lead time. Lower is better."""
    r = results.assign(lead=lead(results["step"]), err=results["forecast"] - results["actual"])
    g = r.groupby(["model", "lead"])
    return pd.DataFrame({
        "mae": g["err"].apply(lambda e: e.abs().mean()),
        "rmse": g["err"].apply(lambda e: np.sqrt((e ** 2).mean())),
        "mape": g.apply(lambda d: (d["err"].abs() / d["actual"]).mean() * 100, include_groups=False),
        "days": g["origin"].nunique(),
    }).round(2)
