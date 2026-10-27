import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from vicdemand import backtest, pipeline
from vicdemand.api import create_app
from vicdemand.models import DAY, MODELS, WEEK


@pytest.fixture
def series():
    idx = pd.date_range("2026-01-01 00:30", periods=10 * WEEK, freq="30min")
    demand = pd.Series(np.arange(len(idx), dtype=float), index=idx, name="demand")
    temp = pd.Series(20.0, index=pd.date_range(idx[0], periods=len(idx) + 3 * DAY, freq="30min"), name="temp")
    return demand, temp


def test_seasonal_naive_repeats_last_season(series):
    demand, temp = series
    origin = pd.Timestamp("2026-02-01 00:00")
    f = MODELS["naive_day"].forecast(demand.loc[:origin], temp, origin, 2 * DAY)
    assert f.index[0] == origin + pd.Timedelta("30min")
    yesterday = demand.loc[:origin].iloc[-DAY:].to_numpy()
    assert (f.to_numpy() == np.tile(yesterday, 2)).all()


def test_models_never_read_past_the_origin(series):
    demand, temp = series
    origin = pd.Timestamp("2026-02-01 00:00")
    poisoned = demand.copy()
    poisoned.loc[origin + pd.Timedelta("30min"):] = 1e9
    for m in MODELS.values():
        assert (m.forecast(poisoned.loc[:origin], temp, origin, 2 * DAY) < 1e9).all(), m.name


def test_backtest_scores_by_lead(series):
    demand, temp = series
    r = backtest.run(list(MODELS.values()), demand, temp, "2026-02-01", "2026-02-07")
    s = backtest.score(r)
    assert set(s.index.get_level_values("lead")) == {"today", "tomorrow"}
    assert (s["days"] == 7).all()


def test_daily_job_and_api(series, tmp_path, monkeypatch):
    demand, temp = series
    monkeypatch.setattr(pipeline.data, "load_demand", lambda cache: demand)
    monkeypatch.setattr(pipeline.data, "load_temperature", lambda cache: temp)
    latest = pipeline.run_daily(tmp_path)
    pipeline.run_daily(tmp_path)  # rerunning the same day must not duplicate the log
    assert len(pd.read_parquet(tmp_path / "forecast_log.parquet")) == len(MODELS) * 2 * DAY
    assert all(len(f) == 2 * DAY for f in latest["forecasts"].values())

    client = TestClient(create_app(tmp_path))
    body = client.get("/forecast", headers={"Origin": "https://ryanward.dev"})
    assert body.status_code == 200 and body.json()["origin"] == latest["origin"]
    assert body.headers["access-control-allow-origin"] == "https://ryanward.dev"
    assert "access-control-allow-origin" not in client.get("/forecast", headers={"Origin": "https://evil.example"}).headers
