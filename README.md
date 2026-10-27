# Victorian electricity demand forecast

Half-hourly forecasts of Victoria's electricity demand (AEMO region VIC1) for today and tomorrow, issued every night,
logged, and scored against what actually happened. Every model is also backtested over the past year with a
rolling origin, so the page leads with how well each one would have done.

- **Page:** static, `index.html` + `assets/` (GitHub Pages).
- **API:** https://ryanward-io--vic-demand-forecast.us-east.modal.run/forecast (Modal; only https://ryanward.dev may read it from a browser).

## How it works

| Piece | File | What it does |
| --- | --- | --- |
| Data | `src/vicdemand/data.py` | AEMO monthly price-and-demand CSVs (5-minute since 2021, averaged to half-hours) and Open-Meteo temperature for Melbourne, cached as parquet |
| Models | `src/vicdemand/models.py` | One interface, `forecast(demand, temp, origin, horizon)`. Benchmarks so far: same as yesterday, same day last week, average of the last 4 weeks |
| Backtest | `src/vicdemand/backtest.py`, `scripts/backtest.py` | Every day in the window: stand at midnight, forecast 96 half-hours, score MAE / RMSE / MAPE split into "today" and "tomorrow" |
| Daily job | `src/vicdemand/pipeline.py` | Refresh data, forecast with every model, append to the forecast log, rescore the live track record, write `latest.json` |
| API | `src/vicdemand/api.py` | `GET /forecast` returns `latest.json` |
| Infrastructure | `modal_app.py` | Image, Volume, 01:00 AEST cron, web endpoint and their cost limits |

## Running it

```sh
uv sync
uv run pytest
uv run python scripts/backtest.py           # writes results/backtest.json
uv run python dev.py                        # daily job + local API on :8002
python3 -m http.server 8011                 # then open http://localhost:8011/?api=http://localhost:8002

uv run modal deploy modal_app.py            # deploy (starts the nightly schedule)
uv run modal run modal_app.py::daily        # run the nightly job now
```

## Data

- Demand: [AEMO aggregated price and demand data](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data). Check AEMO's copyright permissions before republishing raw data; this repo only publishes derived forecasts and scores.
- Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0): ERA5 reanalysis for history, forecast API for the last ~10 days and the next 2.

Times are NEM time (AEST, UTC+10 all year). Half-hours are labelled by their end, as AEMO does.
