"""The daily job: refresh data, issue a forecast from every model, log it, and score earlier forecasts against what happened.

Everything lives under one data directory (a Modal Volume in production, ./data locally):

    cache/                 downloaded AEMO and Open-Meteo files
    fitted/                trained models (LightGBM, LSTM), refitted once a week
    forecast_log.parquet   every forecast ever issued (the live track record: rows are only ever added, never replaced)
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
# The nightly run issues about an hour after its midnight origin. A forecast issued much later (a manual rerun, or a night
# when AEMO's file lagged and the origin fell back a day) has seen more of the day it forecasts, so it isn't scored.
MAX_LATE = pd.Timedelta(hours=3)


def _late(log: pd.DataFrame) -> pd.Series:
    issued = pd.to_datetime(log["issued"], utc=True).dt.tz_convert(data.NEM_TZ).dt.tz_localize(None)
    return issued - pd.to_datetime(log["origin"]) > MAX_LATE


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
    models = models or all_models(root / "cache")
    demand = data.load_demand(root / "cache")
    weather = data.load_weather(root / "cache")
    # Stand at the last midnight we have complete data for.
    origin = demand.index[demand.index.normalize() == demand.index][-1]
    known = demand.loc[:origin]
    models = {name: _fitted(m, known, weather, root / "fitted") if hasattr(m, "fit") else m for name, m in models.items()}

    issued = datetime.now(timezone.utc).isoformat(timespec="seconds")
    forecasts = {}
    for name, m in models.items():
        # One model failing (e.g. NEMWEB down for AEMO's forecast) shouldn't cost the others their forecast.
        try:
            forecasts[name] = m.forecast(known, weather, origin, backtest.HORIZON)
        except Exception as e:  # noqa: BLE001 - logged and skipped; tomorrow's run tries again
            print(f"{name}: no forecast today ({type(e).__name__}: {e})")
    if not forecasts:
        raise RuntimeError("every model failed")

    new = pd.concat([pd.DataFrame({"model": name, "origin": origin, "time": f.index, "step": range(1, len(f) + 1),
                                   "forecast": f.to_numpy(), "issued": issued}) for name, f in forecasts.items()])
    labels = {name: models[name].label for name in forecasts}
    for name, (label, members) in ENSEMBLES.items():
        if set(members) <= set(forecasts):
            e = ensemble(new, name, members)
            new = pd.concat([new, e])
            forecasts[name] = e.set_index("time")["forecast"]
            labels[name] = label

    log_path = root / "forecast_log.parquet"
    log = pd.concat([pd.read_parquet(log_path), new]) if log_path.exists() else new
    # Rerunning on the same day keeps the forecast that was logged first: the track record is what was said at the time,
    # not a later, better-informed redo.
    log = log.drop_duplicates(["model", "origin", "step"], keep="first")
    log.to_parquet(log_path)
    # Show what was logged for this origin, so the chart and the track record never disagree after a rerun. A range is
    # only shown when it belongs to the logged forecast (this run's), since ranges aren't logged.
    today = log[log["origin"] == origin]
    ranges = {name: f for name, f in forecasts.items() if "low" in f.attrs and (today.loc[today["model"] == name, "issued"] == issued).all()}
    forecasts = {name: today[today["model"] == name].set_index("time")["forecast"].rename(name) for name in forecasts}

    on_time = log[~_late(log)]
    scored = on_time.assign(actual=demand.reindex(on_time["time"]).to_numpy()).dropna(subset=["actual"])
    live = backtest.score(scored).reset_index().to_dict("records") if len(scored) else []

    start = origin - pd.Timedelta(days=HISTORY_DAYS)
    end = forecasts[next(iter(forecasts))].index[-1]
    latest = {
        "issued": today["issued"].min(),  # when the forecasts shown were first made (a rerun doesn't move this)
        "late": bool(_late(today).any()),  # made well after the origin, so left out of the track record
        "origin": origin.strftime("%Y-%m-%dT%H:%M"),
        "region": data.REGION,
        "steps_per_day": DAY,
        "models": [{"name": name, "label": label} for name, label in labels.items()],
        "actual": _points(demand.loc[start:]),
        "temp": _points(weather["temp"].loc[start:end]),
        "forecasts": {name: _points(f) for name, f in forecasts.items()},
        # Forecast ranges (e.g. Chronos-2's 10-90% quantiles): {model: [[time, low, high], ...]}
        "ranges": {name: [[t.strftime("%Y-%m-%dT%H:%M"), round(float(lo), 1), round(float(hi), 1)]
                          for t, lo, hi in zip(f.index, f.attrs["low"], f.attrs["high"])]
                   for name, f in ranges.items()},
        "live_score": live,
        # Each model's live miss as a % of "same day last week", on the same half-hours (see backtest.relative).
        "live_relative": backtest.relative(scored) if len(scored) else {},
    }
    (root / "latest.json").write_text(json.dumps(latest))
    return latest
