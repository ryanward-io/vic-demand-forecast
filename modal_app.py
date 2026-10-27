"""Infrastructure for the Victorian demand forecast on Modal.

This file *is* the infrastructure: Modal reads it and creates the image, storage, schedule and web endpoint it describes.

    uv run modal deploy modal_app.py          # create or update the live app (the schedule starts on deploy)
    uv run modal run modal_app.py::daily      # run the daily job once, now (e.g. right after the first deploy)

The forecasting code lives in src/vicdemand and knows nothing about Modal.
"""

from pathlib import Path

import modal

app = modal.App("vic-demand-forecast")

# Versions pinned to match uv.lock, so production runs what the tests ran.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("pandas==3.1.0", "pyarrow==26.0.0", "numpy==2.5.4", "httpx==0.28.1", "fastapi[standard]==0.162.0")
    .add_local_python_source("vicdemand")
)

# A Volume is a shared disk that outlives containers: the data cache, the forecast log and latest.json live here.
# The daily job writes it; the web endpoint only reads it.
volume = modal.Volume.from_name("vic-demand-data", create_if_missing=True)
ROOT = Path("/data")


@app.function(
    image=image,
    volumes={ROOT: volume},
    # 01:00 NEM time: AEMO's current-month file is updated just after midnight with the day that just finished.
    # "Australia/Brisbane" because the NEM runs on AEST all year (no daylight saving).
    schedule=modal.Cron("0 1 * * *", timezone="Australia/Brisbane"),
    timeout=15 * 60,  # the first run downloads ~8 years of data (about a minute); later runs fetch one month
    retries=2,  # AEMO or Open-Meteo hiccups shouldn't cost a day's forecast
)
def daily():
    from vicdemand.pipeline import run_daily

    latest = run_daily(ROOT)
    volume.commit()  # make the new files visible to the web endpoint's containers
    print("Forecast issued from", latest["origin"])


@app.function(
    image=image,
    volumes={ROOT: volume},
    min_containers=0,  # scale to zero between visits, so idle time costs nothing
    max_containers=1,  # hard ceiling on parallel machines, which also caps the bill
    scaledown_window=60,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="vic-demand-forecast")
def web():
    from vicdemand.api import create_app

    # Containers see the Volume as it was when they started; reload picks up the daily job's latest commit.
    return create_app(ROOT, refresh=volume.reload)
