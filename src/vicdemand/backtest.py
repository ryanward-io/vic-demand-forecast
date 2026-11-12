"""Rolling-origin backtest (the "tsCV" from the forecasting review post).

Every day in the test window we stand at midnight, pretend we know nothing after it, forecast the next two days,
and later compare with what happened. Scores are split by lead time: "today" is steps 1-48, "tomorrow" 49-96.

Two things keep it honest:
- models that learn are retrained monthly, each time on data up to that day only;
- after the origin, models see the weather *forecast* that existed at the time, never the weather that happened.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import DAY, Model, target_index

HORIZON = 2 * DAY


def weather_at(origin: pd.Timestamp, actual: pd.DataFrame, forecasts: dict[int, pd.DataFrame], horizon: int = HORIZON) -> pd.DataFrame:
    """Weather as a forecaster standing at `origin` saw it: actuals up to the origin, then the 1-day-old forecast for
    today and the 2-day-old one for tomorrow."""
    targets = target_index(origin, horizon)
    ahead = pd.concat([forecasts[1].reindex(targets[:DAY]), forecasts[2].reindex(targets[DAY:])])
    return pd.concat([actual.loc[:origin], ahead])


def run(models: list[Model], demand: pd.Series, weather: pd.DataFrame, start: str, end: str,
        weather_forecasts: dict[int, pd.DataFrame] | None = None, horizon: int = HORIZON) -> pd.DataFrame:
    """One row per (model, origin, target): the forecast and the actual.

    Without `weather_forecasts` the models see actual weather after the origin (only for tests and quick checks)."""
    origins = [o for o in pd.date_range(start, end, freq="D") if o in demand.index]
    cases = [(demand.loc[:o], weather_at(o, weather, weather_forecasts, horizon) if weather_forecasts else weather, o) for o in origins]
    rows = []
    for m in models:
        if hasattr(m, "fit"):
            # Retrain at the start of each month on everything known by then: the live job retrains weekly, and a model
            # frozen for a whole year would be judged on a handicap it never has in practice.
            preds, fitted_for = [], None
            for d, w, o in cases:
                if o.to_period("M") != fitted_for:
                    fitted_for = o.to_period("M")
                    m.fit(demand.loc[:o], weather.loc[:o])
                preds.append(m.forecast(d, w, o, horizon))
        elif hasattr(m, "forecast_many"):
            preds = [f for i in range(0, len(cases), 32) for f in m.forecast_many(cases[i:i + 32], horizon)]
        else:
            preds = [m.forecast(d, w, o, horizon) for d, w, o in cases]
        for (_, _, o), f in zip(cases, preds):
            rows.append(pd.DataFrame({"model": m.name, "origin": o, "time": f.index, "step": np.arange(1, horizon + 1),
                                      "forecast": f.to_numpy(), **{k: f.attrs[k] for k in ("low", "high") if k in f.attrs}}))
    out = pd.concat(rows, ignore_index=True)
    out["actual"] = demand.reindex(out["time"]).to_numpy()
    return out.dropna(subset=["actual"])


def lead(step: pd.Series) -> pd.Series:
    return np.where(step <= DAY, "today", "tomorrow")


def coverage(results: pd.DataFrame) -> pd.DataFrame:
    """For models with a range (columns low/high): how often the actual fell inside it, and how wide it was, by lead time.
    A well-calibrated 80% range contains the actual about 80% of the time."""
    r = results.dropna(subset=["low", "high"]).assign(lead=lambda d: lead(d["step"]))
    inside = (r["actual"] >= r["low"]) & (r["actual"] <= r["high"])
    g = r.assign(inside=inside, width=r["high"] - r["low"]).groupby(["model", "lead"])
    return pd.DataFrame({"inside_pct": g["inside"].mean() * 100, "width_mw": g["width"].mean()}).round(1)


def score(results: pd.DataFrame) -> pd.DataFrame:
    """MAE and RMSE in MW, MAPE in %, per model and lead time. Lower is better."""
    r = results.assign(lead=lead(results["step"]), err=results["forecast"] - results["actual"])
    g = r.groupby(["model", "lead"])
    return pd.DataFrame({
        "mae": g["err"].apply(lambda e: e.abs().mean()),
        "rmse": g["err"].apply(lambda e: np.sqrt((e ** 2).mean())),
        "mape": g.apply(lambda d: (d["err"].abs() / d["actual"]).mean() * 100, include_groups=False),
        "days": g["origin"].nunique(),
        "n": g["err"].count(),  # half-hours scored: lower than days * 48 when a source was missing some
    }).round(2)


def relative(results: pd.DataFrame, base: str = "naive_week") -> dict[str, float]:
    """Each model's total miss as a percentage of `base`'s, over the two days, on exactly the half-hours both forecast.
    50 means it missed by half as much as the lazy guess. Pairing the rows keeps a model with gaps (or few live days)
    from being compared with the base on a different set of days."""
    err = results.assign(err=(results["forecast"] - results["actual"]).abs()).pivot_table(
        index=["origin", "step"], columns="model", values="err")
    if base not in err:
        return {}
    out = {}
    for m in err.columns:
        both = err[[base] if m == base else [m, base]].dropna()
        if len(both):
            out[m] = round(float(100 * both[m].sum() / both[base].sum()), 1)
    return out


def compare(results: pd.DataFrame, a: str, b: str, n_boot: int = 4000, seed: int = 0) -> dict:
    """How much smaller `a`'s average miss is than `b`'s (MW, over the two days), with a 95% paired bootstrap interval
    over origins, and the share of days `a` was closer. A positive `mw` means `a` is better. The interval resamples whole
    days, because errors within a day are strongly correlated."""
    per_day = (results.assign(err=(results["forecast"] - results["actual"]).abs())
               .pivot_table(index="origin", columns="model", values="err", aggfunc="mean")[[a, b]].dropna())
    diff = (per_day[b] - per_day[a]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"a": a, "b": b, "mw": round(float(diff.mean()), 1), "lo": round(float(lo), 1), "hi": round(float(hi), 1),
            "a_better_pct": round(float((diff > 0).mean() * 100), 0), "days": int(len(diff)), "tie": bool(lo <= 0 <= hi)}
