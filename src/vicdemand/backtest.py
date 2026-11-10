"""Rolling-origin backtest (the "tsCV" from the forecasting review post).

Every day in the test window we stand at midnight, pretend we know nothing after it, forecast the next two days,
and later compare with what happened. Scores are split by lead time: "today" is steps 1-48, "tomorrow" 49-96.

Two things keep it honest:
- models that learn are retrained monthly, each time on data up to that day only;
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
    origins = [o for o in pd.date_range(start, end, freq="D") if o in demand.index]
    cases = [(demand.loc[:o], weather_at(o, weather, weather_forecasts, horizon) if weather_forecasts else weather, o) for o in origins]
    rows = []
    for m in models:
        if hasattr(m, "fit"):
            # Retrain at the start of each month on everything known by then: the live job retrains weekly, and a model
            # frozen for a whole year would be judged on a handicap it never has in practice.
            preds, fitted_for = [], None
            for d, w, o in cases:
                if o.to_period("M") != fitted_for:
                    fitted_for = o.to_period("M")
                    m.fit(demand.loc[:o], weather.loc[:o])
                preds.append(m.forecast(d, w, o, horizon))
        elif hasattr(m, "forecast_many"):
            preds = [f for i in range(0, len(cases), 32) for f in m.forecast_many(cases[i:i + 32], horizon)]
        else:
            preds = [m.forecast(d, w, o, horizon) for d, w, o in cases]
        for (_, _, o), f in zip(cases, preds):
            rows.append(pd.DataFrame({"model": m.name, "origin": o, "time": f.index, "step": np.arange(1, horizon + 1),
                                      "forecast": f.to_numpy(), **{k: f.attrs[k] for k in ("low", "high") if k in f.attrs}}))
    out = pd.concat(rows, ignore_index=True)
    out["actual"] = demand.reindex(out["time"]).to_numpy()
    return out.dropna(subset=["actual"])


def lead(step: pd.Series) -> pd.Series:
    return np.where(step <= DAY, "today", "tomorrow")


def coverage(results: pd.DataFrame) -> pd.DataFrame:
    """For models with a range (columns low/high): how often the actual fell inside it, and how wide it was, by lead time.
    A well-calibrated 80% range contains the actual about 80% of the time."""
    r = results.dropna(subset=["low", "high"]).assign(lead=lambda d: lead(d["step"]))
    inside = (r["actual"] >= r["low"]) & (r["actual"] <= r["high"])
    g = r.assign(inside=inside, width=r["high"] - r["low"]).groupby(["model", "lead"])
    return pd.DataFrame({"inside_pct": g["inside"].mean() * 100, "width_mw": g["width"].mean()}).round(1)


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
