"""Rolling-origin backtest of every model over a test window; writes results/backtest.json for the page.

    uv run python scripts/backtest.py                       # last full year before this month
    uv run python scripts/backtest.py --start 2024-01-01 --end 2024-12-31
"""

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd

from vicdemand import backtest, data
from vicdemand.models import MODELS

p = argparse.ArgumentParser()
first_of_month = pd.Timestamp(date.today().replace(day=1))
p.add_argument("--start", default=str((first_of_month - pd.DateOffset(years=1)).date()))
p.add_argument("--end", default=str((first_of_month - pd.Timedelta(days=3)).date()))
p.add_argument("--models", nargs="*", default=list(MODELS))
args = p.parse_args()

cache = Path("data/cache")
demand, temp = data.load_demand(cache), data.load_temperature(cache)
results = backtest.run([MODELS[m] for m in args.models], demand, temp, args.start, args.end)
table = backtest.score(results)
print(table.to_string())

out = Path("results")
out.mkdir(exist_ok=True)
(out / "backtest.json").write_text(json.dumps({
    "start": args.start, "end": args.end,
    "models": [{"name": m, "label": MODELS[m].label} for m in args.models],
    "scores": table.reset_index().to_dict("records"),
}, indent=1))
