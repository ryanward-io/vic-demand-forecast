"""The daily job: refresh data, issue a forecast from every model, log it, and score earlier forecasts against what happened.

Everything lives under one data directory (a Modal Volume in production, ./data locally):

    cache/            downloaded AEMO and Open-Meteo files
    forecast_log.parquet   every forecast ever issued (the live track record; never rewritten)
    latest.json       what the web page shows
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import backtest, data
from .models import DAY, MODELS

HISTORY_DAYS = 7  # actuals shown on the chart before the forecast starts


def _points(s: pd.Series) -> list:
    return [[t.strftime("%Y-%m-%dT%H:%M"), None if pd.isna(v) else round(float(v), 1)] for t, v in s.items()]


def run_daily(root: Path) -> dict:
    demand = data.load_demand(root / "cache")
    temp = data.load_temperature(root / "cache")
    # Stand at the last midnight we have complete data for.
    origin = demand.index[demand.index.normalize() == demand.index][-1]
    known = demand.loc[:origin]

    issued = datetime.now(timezone.utc).isoformat(timespec="seconds")
    forecasts = {name: m.forecast(known, temp, origin, backtest.HORIZON) for name, m in MODELS.items()}

    log_path = root / "forecast_log.parquet"
    new = pd.concat([pd.DataFrame({"model": name, "origin": origin, "time": f.index, "step": range(1, len(f) + 1),
                                   "forecast": f.to_numpy(), "issued": issued}) for name, f in forecasts.items()])
    log = pd.concat([pd.read_parquet(log_path), new]) if log_path.exists() else new
    # Rerunning on the same day replaces that day's forecast rather than double counting it.
    log = log.drop_duplicates(["model", "origin", "step"], keep="last")
    log.to_parquet(log_path)

    scored = log.assign(actual=demand.reindex(log["time"]).to_numpy()).dropna(subset=["actual"])
    live = backtest.score(scored).reset_index().to_dict("records") if len(scored) else []

    start = origin - pd.Timedelta(days=HISTORY_DAYS)
    end = forecasts[next(iter(forecasts))].index[-1]
    latest = {
        "issued": issued,
        "origin": origin.strftime("%Y-%m-%dT%H:%M"),
        "region": data.REGION,
        "steps_per_day": DAY,
        "models": [{"name": name, "label": m.label} for name, m in MODELS.items()],
        "actual": _points(demand.loc[start:]),
        "temp": _points(temp.loc[start:end]),
        "forecasts": {name: _points(f) for name, f in forecasts.items()},
        "live_score": live,
    }
    (root / "latest.json").write_text(json.dumps(latest))
    return latest
