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
    log = pd.read_parquet(tmp_path / "forecast_log.parquet")
    again = pipeline.run_daily(tmp_path, cheap_models())  # rerunning the same day must not duplicate or replace the log
    n_models = len(cheap_models()) + 1
    assert len(pd.read_parquet(tmp_path / "forecast_log.parquet")) == n_models * 2 * DAY
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "forecast_log.parquet"), log)
    assert again["issued"] == latest["issued"] and again["forecasts"] == latest["forecasts"]
    assert len(latest["forecasts"]) == n_models and all(len(f) == 2 * DAY for f in latest["forecasts"].values())

    client = TestClient(create_app(tmp_path))
    body = client.get("/forecast", headers={"Origin": "https://ryanward.dev"})
    assert body.status_code == 200 and body.json()["origin"] == latest["origin"]
    assert body.headers["access-control-allow-origin"] == "https://ryanward.dev"
    assert "access-control-allow-origin" not in client.get("/forecast", headers={"Origin": "https://evil.example"}).headers
    assert "max-age" in body.headers["cache-control"]
    assert client.get("/openapi.json").status_code == 404


def test_api_refreshes_at_most_every_few_minutes(tmp_path):
    (tmp_path / "latest.json").write_text('{"origin": "a"}')
    calls = []
    client = TestClient(create_app(tmp_path, refresh=lambda: calls.append(1)))
    for _ in range(5):
        assert client.get("/forecast").json() == {"origin": "a"}
    assert len(calls) == 1


def test_api_survives_a_failed_refresh(tmp_path):
    (tmp_path / "latest.json").write_text('{"origin": "a"}')
    def boom():
        raise RuntimeError("volume busy")
    assert TestClient(create_app(tmp_path, refresh=boom)).get("/forecast").status_code == 200


def test_late_forecasts_are_flagged():
    log = pd.DataFrame({"origin": pd.to_datetime(["2026-11-12 00:00", "2026-11-12 00:00"]),
                        "issued": ["2026-11-11T15:00:23+00:00", "2026-11-12T01:00:00+00:00"]})  # 01:00 and 11:00 NEM time
    assert list(pipeline._late(log)) == [False, True]


def _results(errors: dict[str, list[float]]) -> pd.DataFrame:
    """One row per (model, day): forecast = actual + error, all at step 1."""
    rows = [{"model": m, "origin": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i), "step": 1, "actual": 1000.0,
             "forecast": 1000.0 + e} for m, es in errors.items() for i, e in enumerate(es) if e is not None]
    return pd.DataFrame(rows)


def test_relative_pairs_rows_with_the_base():
    # "b" only has the first two days, where the base missed by 100; on all four days the base averages 250.
    r = _results({"naive_week": [100, 100, 400, 400], "a": [50, 50, 200, 200], "b": [50, 50, None, None]})
    rel = backtest.relative(r)
    assert rel["a"] == 50 and rel["b"] == 50 and rel["naive_week"] == 100


def test_compare_reports_difference_and_ties():
    rng = np.random.default_rng(1)
    base = rng.uniform(100, 300, 200)
    clear = backtest.compare(_results({"a": list(base - 50), "b": list(base)}), "a", "b")
    assert clear["mw"] == 50 and not clear["tie"] and clear["a_better_pct"] == 100
    noise = rng.normal(0, 30, 200)
    tie = backtest.compare(_results({"a": list(base + noise), "b": list(base)}), "a", "b")
    assert tie["tie"] and tie["lo"] < 0 < tie["hi"]


def test_aemo_csv_parser_reads_one_table():
    from vicdemand.aemo import _table

    text = "\n".join([
        "C,NEMP.WORLD,PREDISPATCHIS,AEMO,PUBLIC,2026/10/27,23:31:51",
        "I,PREDISPATCH,CASE_SOLUTION,1,PREDISPATCHSEQNO,RUNNO",
        "D,PREDISPATCH,CASE_SOLUTION,1,2026102740,1",
        "I,PREDISPATCH,REGION_SOLUTION,9,REGIONID,INTERVENTION,TOTALDEMAND,DATETIME",
        'D,PREDISPATCH,REGION_SOLUTION,9,VIC1,0,5227.51,"2026/10/28 00:30:00"',
        'D,PREDISPATCH,REGION_SOLUTION,9,NSW1,0,7000,"2026/10/28 00:30:00"',
        "C,END OF REPORT,7",
    ])
    df = _table(text, "REGION_SOLUTION")
    assert list(df["REGIONID"]) == ["VIC1", "NSW1"]
    assert df.loc[0, "DATETIME"] == "2026/10/28 00:30:00" and float(df.loc[0, "TOTALDEMAND"]) == 5227.51


def test_timestral_forecasts_a_range_without_peeking(series):
    """The real model at its smallest size (21 MB), with a short context to match the 10-week test series."""
    from vicdemand.models import Timestral

    demand, weather = series
    origin = pd.Timestamp("2026-02-01 00:00")
    m = Timestral(model_id="mistralai/Timestral-5M", context=2 * WEEK)
    f = m.forecast(demand, weather, origin, 2 * DAY)
    assert len(f) == 2 * DAY and f.index[0] == origin + pd.Timedelta("30min")
    assert (f.attrs["low"] <= f.to_numpy() + 1e-6).all() and (f.to_numpy() <= f.attrs["high"] + 1e-6).all()
    assert abs(f.mean() - 5000) < 500  # it picked up the level of the series
    tampered = demand.copy()
    tampered.loc[origin + pd.Timedelta("30min"):] = 0
    np.testing.assert_allclose(m.forecast(tampered, weather, origin, 2 * DAY).to_numpy(), f.to_numpy())
