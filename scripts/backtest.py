"""Rolling-origin backtest over a test window; writes results/backtest.json for the page.

Each model's forecasts are saved to data/backtest/<model>.parquet, so slow models can be run on their own and the
summary rebuilt from whatever has been run:

    uv run python scripts/backtest.py                       # all models, last full year before this month
    uv run python scripts/backtest.py --models lightgbm     # rerun one model, keep the others' saved results
    uv run python scripts/backtest.py --models              # just rebuild the summary

The published results use a fixed window; to reproduce them exactly (the default window moves with today's date):

    uv run python scripts/backtest.py --start 2025-10-01 --end 2026-09-28
"""

import argparse
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from vicdemand import backtest, data
from vicdemand.models import ENSEMBLES, all_models, ensemble

MODELS = all_models()
p = argparse.ArgumentParser()
first_of_month = pd.Timestamp(date.today().replace(day=1))
p.add_argument("--start", default=str((first_of_month - pd.DateOffset(years=1)).date()))
p.add_argument("--end", default=str((first_of_month - pd.Timedelta(days=3)).date()))
p.add_argument("--models", nargs="*", default=list(MODELS))
args = p.parse_args()

cache, runs = Path("data/cache"), Path("data/backtest")
runs.mkdir(parents=True, exist_ok=True)
demand, weather = data.load_demand(cache), data.load_weather(cache)
forecasts = data.load_weather_forecasts(cache)
for name in args.models:
    r = backtest.run([MODELS[name]], demand, weather, args.start, args.end, forecasts)
    r.to_parquet(runs / f"{name}.parquet")
    print(backtest.score(r).to_string(header=False))

results = pd.concat([pd.read_parquet(runs / f"{m}.parquet") for m in MODELS if (runs / f"{m}.parquet").exists()])
results = results[(results["origin"] >= args.start) & (results["origin"] <= args.end)]
for name, (label, members) in ENSEMBLES.items():
    if set(members) <= set(results["model"]):
        results = pd.concat([results, ensemble(results, name, members)])
table = backtest.score(results)
print(table.to_string())
ranges = backtest.coverage(results) if "low" in results else pd.DataFrame()
print(ranges.to_string())

# The comparisons the page quotes, so its numbers come from here rather than being typed in by hand.
# A second average (LightGBM + Timestral) answers "would swapping Timestral into the average help?"; it isn't published as a model.
alt = ensemble(results, "ensemble_timestral", ["lightgbm", "timestral"]) if {"lightgbm", "timestral"} <= set(results["model"]) else None
pool = pd.concat([results, alt]) if alt is not None else results
pairs = [("ensemble", "lightgbm"), ("ensemble", "chronos2"), ("aemo", "ensemble"), ("chronos2", "lightgbm"),
         ("timestral", "chronos2"), ("lightgbm", "timestral"), ("ensemble_timestral", "ensemble"), ("lightgbm", "lstm")]
comparisons = [backtest.compare(pool, a, b) for a, b in pairs if {a, b} <= set(pool["model"])]
for c in comparisons:
    print(c)
monthly = (results.assign(err=(results["forecast"] - results["actual"]).abs(), month=results["origin"].dt.strftime("%Y-%m"))
           .groupby(["model", "month"])["err"].mean().round(1).reset_index().rename(columns={"err": "mae"}))
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

labels = {m.name: m.label for m in MODELS.values()} | {n: label for n, (label, _) in ENSEMBLES.items()}
present = [m for m in labels if m in set(results["model"])]
out = Path("results")
out.mkdir(exist_ok=True)
(out / "backtest.json").write_text(json.dumps({
    "start": args.start, "end": args.end, "weather": "archived forecasts (Open-Meteo previous runs)",
    "run": {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": commit,
            "command": f"uv run python scripts/backtest.py --start {args.start} --end {args.end}"},
    "models": [{"name": m, "label": labels[m]} for m in present],
    "scores": table.reset_index().to_dict("records"),
    "relative": backtest.relative(results),
    "comparisons": comparisons,
    "ranges": ranges.reset_index().to_dict("records"),
    "monthly": monthly.to_dict("records"),
}, indent=1))
