import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vicdemand import backtest, pipeline
from vicdemand.api import create_app
from vicdemand.lstm import LSTM
from vicdemand.models import BENCHMARKS, DAY, MSTL, WEEK, LightGBM


def cheap_models():
    """Everything except Chronos-2 (a 500 MB download), with the learned models shrunk so tests stay fast."""
    return {m.name: m for m in [*BENCHMARKS, MSTL(), LightGBM(rounds=20), LSTM(epochs=1, every_hours=24)]}


@pytest.fixture
def series():
    idx = pd.date_range("2026-01-01 00:30", periods=10 * WEEK, freq="30min")
    rng = np.random.default_rng(0)
    slot = np.arange(len(idx)) % DAY
    demand = pd.Series(5000 + 800 * np.sin(2 * np.pi * slot / DAY) + rng.normal(0, 50, len(idx)), index=idx, name="demand")
    widx = pd.date_range(idx[0], periods=len(idx) + 3 * DAY, freq="30min")
    weather = pd.DataFrame({"temp": 20 + 5 * np.sin(2 * np.pi * np.arange(len(widx)) / DAY), "solar": 100.0}, index=widx)
    return demand, weather


def test_seasonal_naive_repeats_last_season(series):
    demand, weather = series
    origin = pd.Timestamp("2026-02-01 00:00")
    f = BENCHMARKS[0].forecast(demand.loc[:origin], weather, origin, 2 * DAY)
    assert f.index[0] == origin + pd.Timedelta("30min")
    yesterday = demand.loc[:origin].iloc[-DAY:].to_numpy()
    assert (f.to_numpy() == np.tile(yesterday, 2)).all()


def test_models_never_read_past_the_origin(series):
    demand, weather = series
    origin = pd.Timestamp("2026-02-20 00:00")
    poisoned = demand.copy()
    poisoned.loc[origin + pd.Timedelta("30min"):] = 1e9
    for m in cheap_models().values():
        if hasattr(m, "fit"):
            m.fit(demand.loc[:origin], weather.loc[:origin])
        f = m.forecast(poisoned.loc[:origin], weather, origin, 2 * DAY)
        assert f.notna().all() and (f < 1e6).all(), m.name


def test_backtest_uses_forecast_weather_not_actuals(series):
    _, weather = series
    origin = pd.Timestamp("2026-02-01 00:00")
    fc = {d: weather + 100 * d for d in (1, 2)}  # make each source recognisable
    w = backtest.weather_at(origin, weather, fc)
    assert (w.loc[:origin, "temp"] < 100).all()
    assert (w["temp"].iloc[-2 * DAY:-DAY] > 100).all() and (w["temp"].iloc[-DAY:] > 200).all()


def test_backtest_scores_by_lead(series):
    demand, weather = series
    r = backtest.run(BENCHMARKS, demand, weather, "2026-02-01", "2026-02-07")
    s = backtest.score(r)
    assert set(s.index.get_level_values("lead")) == {"today", "tomorrow"}
    assert (s["days"] == 7).all()


def test_daily_job_and_api(series, tmp_path, monkeypatch):
    demand, weather = series
    monkeypatch.setattr(pipeline.data, "load_demand", lambda cache: demand)
    monkeypatch.setattr(pipeline.data, "load_weather", lambda cache: weather)
    monkeypatch.setattr(pipeline, "ENSEMBLES", {"ens": ("Average", ["lightgbm", "lstm"])})
    latest = pipeline.run_daily(tmp_path, cheap_models())
    assert (tmp_path / "fitted" / "lightgbm.pkl").exists()
    pipeline.run_daily(tmp_path, cheap_models())  # rerunning the same day must not duplicate the log
    n_models = len(cheap_models()) + 1
    assert len(pd.read_parquet(tmp_path / "forecast_log.parquet")) == n_models * 2 * DAY
    assert len(latest["forecasts"]) == n_models and all(len(f) == 2 * DAY for f in latest["forecasts"].values())

    client = TestClient(create_app(tmp_path))
    body = client.get("/forecast", headers={"Origin": "https://ryanward.dev"})
    assert body.status_code == 200 and body.json()["origin"] == latest["origin"]
    assert body.headers["access-control-allow-origin"] == "https://ryanward.dev"
    assert "access-control-allow-origin" not in client.get("/forecast", headers={"Origin": "https://evil.example"}).headers
