# Victorian electricity demand forecast

**Live:** https://ryanward.dev/vic-demand-forecast/

Every night nine models forecast Victoria's half-hourly electricity demand (AEMO region VIC1) for the next 48 hours:
three simple benchmarks, MSTL + ETS, LightGBM, an LSTM, two zero-shot foundation models (Chronos-2 and Timestral) and an
average of LightGBM and Chronos-2. Each forecast is logged and scored against what happened and against AEMO's own
forecast. A year-long rolling-origin backtest shows how each would have done, not just what it says today.

**Headline (backtest, Oct 2025 to Sep 2026):** AEMO's forecast is off by 4.0% today and 4.4% tomorrow. The best of mine, the
LightGBM + Chronos-2 average, is off by 4.8% and 5.6%; the simple benchmarks by 11 to 14%. Chronos-2, never trained on this
data, ties with LightGBM trained on seven years of it.

- **Page:** static, `index.html` + `assets/` (GitHub Pages; `_config.yml` keeps the Python code off the site).
- **API:** https://ryanward-io--vic-demand-forecast.us-east.modal.run/forecast (Modal; only https://ryanward.dev may read it from a browser).

## How it works

| Piece | File | What it does |
| --- | --- | --- |
| Data | `src/vicdemand/data.py` | AEMO monthly price-and-demand CSVs (5-minute since 2021, averaged to half-hours) and Open-Meteo temperature for Melbourne, cached as parquet |
| Models | `src/vicdemand/models.py`, `lstm.py`, `features.py` | One interface, `forecast(demand, weather, origin, horizon)`: three benchmarks, MSTL + ETS, LightGBM, an LSTM, Chronos-2 and Timestral (zero-shot foundation models, each with an 80% range), a LightGBM + Chronos-2 average, and AEMO's own forecast (`aemo.py`) as the reference |
| Backtest | `src/vicdemand/backtest.py`, `scripts/backtest.py` | Every day in the window: stand at midnight, forecast 96 half-hours, score MAE / RMSE / MAPE split into "today" and "tomorrow". Learned models retrain monthly on data up to that day; after the origin, models see the archived weather *forecast*, not what happened |
| Daily job | `src/vicdemand/pipeline.py` | Refresh data, retrain learned models weekly, forecast with every model, append to the forecast log, rescore the live track record, write `latest.json` |
| API | `src/vicdemand/api.py` | `GET /forecast` returns `latest.json` |
| Infrastructure | `modal_app.py` | Image, Volume, 01:00 AEST cron, web endpoint and their cost limits |

## Running it

```sh
uv sync
uv run pytest
uv run python scripts/backtest.py --start 2025-10-01 --end 2026-09-28   # the published backtest -> results/backtest.json
uv run python dev.py                        # daily job + local API on :8002
python3 -m http.server 8011                 # then open http://localhost:8011/?api=http://localhost:8002

uv run modal deploy modal_app.py            # deploy (starts the nightly schedule)
uv run modal run modal_app.py::daily        # run the nightly job now (a rerun never replaces a forecast already logged)
```

## Data, licences and attribution

| Source | Used for | Terms |
| --- | --- | --- |
| [AEMO aggregated price and demand data](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/aggregated-data), VIC1 | Actual demand | © AEMO. [Copyright permissions](https://www.aemo.com.au/privacy-and-legal-notices/copyright-permissions): any purpose, with accurate attribution to AEMO as author |
| [NEMWEB](https://nemweb.com.au/Reports/) pre-dispatch (PredispatchIS) and ST PASA reports | AEMO's own forecast, the reference | As above. This project is independent and not endorsed by AEMO |
| [Open-Meteo](https://open-meteo.com/) historical, forecast and previous-runs APIs | Temperature and solar radiation for Melbourne | [CC BY 4.0](https://open-meteo.com/en/license); the free API is for non-commercial use (this site has no ads or subscriptions) and < 10,000 calls a day (I make a handful). ERA5 reanalysis contains modified Copernicus Climate Change Service information |
| [Chronos-2](https://huggingface.co/amazon/chronos-2) (Amazon) | Foundation model | Apache 2.0. Paper: [Chronos-2: From Univariate to Universal Forecasting](https://arxiv.org/abs/2510.15821) |
| [Timestral-209M](https://huggingface.co/mistralai/Timestral-209M) (Mistral AI) | Foundation model | Weights: [Modified MIT](https://huggingface.co/mistralai/Timestral-209M/blob/main/LICENSE), attribution "2026 - Mistral AI"; no rights for companies (or employees of companies) with revenue above US$20M a month. Inference code [`mistralai/timestral`](https://github.com/mistralai/timestral): Apache 2.0. Weights are downloaded at build time, never committed |
| Python libraries (pandas, NumPy, LightGBM, statsforecast, PyTorch, Transformers, holidays, FastAPI, ...) | Everything else | MIT / BSD / Apache 2.0 |

The code is MIT. The repo publishes only derived results (scores in `results/`); the live API republishes a week of AEMO demand and AEMO's
forecast alongside mine, with attribution on the page. Forecasts are for interest only, not for operational or trading use.

Times are NEM time (AEST, UTC+10 all year). Half-hours are labelled by their end, as AEMO does.
