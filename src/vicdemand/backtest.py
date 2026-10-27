"""Rolling-origin backtest (the "tsCV" from the forecasting review post).

Every day in the test window we stand at midnight, pretend we know nothing after it, forecast the next two days,
and later compare with what happened. Scores are split by lead time: "today" is steps 1-48, "tomorrow" 49-96.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import DAY, Model

HORIZON = 2 * DAY


def run(models: list[Model], demand: pd.Series, temp: pd.Series, start: str, end: str, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per (model, origin, target): the forecast and the actual."""
    origins = pd.date_range(start, end, freq="D")
    rows = []
    for origin in origins:
        known = demand.loc[:origin]
        if known.index[-1] != origin:
            continue  # data gap at the origin
        for m in models:
            f = m.forecast(known, temp, origin, horizon)
            rows.append(pd.DataFrame({"model": m.name, "origin": origin, "time": f.index, "step": np.arange(1, horizon + 1),
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
