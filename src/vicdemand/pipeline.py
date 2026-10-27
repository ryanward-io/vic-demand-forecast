"""The daily job: refresh data, issue a forecast from every model, log it, and score earlier forecasts against what happened.

Everything lives under one data directory (a Modal Volume in production, ./data locally):

    cache/                 downloaded AEMO and Open-Meteo files
    fitted/                trained models (LightGBM, LSTM), refitted once a week
    forecast_log.parquet   every forecast ever issued (the live track record; never rewritten)
    latest.json            what the web page shows
"""

from __future__ import annotations

import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import backtest, data
from .models import DAY, ENSEMBLES, Model, all_models, ensemble

HISTORY_DAYS = 7  # actuals shown on the chart before the forecast starts
REFIT_DAYS = 7  # retrain learned models weekly: a day's extra data changes little, and training is the slow part


def _points(s: pd.Series) -> list:
    return [[t.strftime("%Y-%m-%dT%H:%M"), None if pd.isna(v) else round(float(v), 1)] for t, v in s.items()]


def _fitted(m: Model, demand: pd.Series, weather: pd.DataFrame, folder: Path) -> Model:
    """A trained copy of `m`: loaded from disk if trained within REFIT_DAYS, otherwise trained now on all history and saved."""
    path = folder / f"{m.name}.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < REFIT_DAYS * 86400:
        return pickle.loads(path.read_bytes())
    folder.mkdir(parents=True, exist_ok=True)
    m.fit(demand, weather.loc[:demand.index[-1]])
    path.write_bytes(pickle.dumps(m))
    return m


def run_daily(root: Path, models: dict[str, Model] | None = None) -> dict:
    models = models or all_models()
    demand = data.load_demand(root / "cache")
    weather = data.load_weather(root / "cache")
    # Stand at the last midnight we have complete data for.
    origin = demand.index[demand.index.normalize() == demand.index][-1]
    known = demand.loc[:origin]
    models = {name: _fitted(m, known, weather, root / "fitted") if hasattr(m, "fit") else m for name, m in models.items()}

    issued = datetime.now(timezone.utc).isoformat(timespec="seconds")
    forecasts = {name: m.forecast(known, weather, origin, backtest.HORIZON) for name, m in models.items()}

    new = pd.concat([pd.DataFrame({"model": name, "origin": origin, "time": f.index, "step": range(1, len(f) + 1),
                                   "forecast": f.to_numpy(), "issued": issued}) for name, f in forecasts.items()])
    labels = {name: m.label for name, m in models.items()}
    for name, (label, members) in ENSEMBLES.items():
        if set(members) <= set(forecasts):
            e = ensemble(new, name, members)
            new = pd.concat([new, e])
            forecasts[name] = e.set_index("time")["forecast"]
            labels[name] = label

    log_path = root / "forecast_log.parquet"
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
        "models": [{"name": name, "label": label} for name, label in labels.items()],
        "actual": _points(demand.loc[start:]),
        "temp": _points(weather["temp"].loc[start:end]),
        "forecasts": {name: _points(f) for name, f in forecasts.items()},
        "live_score": live,
    }
    (root / "latest.json").write_text(json.dumps(latest))
    return latest
