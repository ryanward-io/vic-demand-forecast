"""Infrastructure for the Victorian demand forecast on Modal.

This file *is* the infrastructure: Modal reads it and creates the image, storage, schedule and web endpoint it describes.

    uv run modal deploy modal_app.py          # create or update the live app (the schedule starts on deploy)
    uv run modal run modal_app.py::daily      # run the daily job once, now (e.g. right after the first deploy)

The forecasting code lives in src/vicdemand and knows nothing about Modal.
"""

from pathlib import Path

import modal

app = modal.App("vic-demand-forecast")

# Two images (the software installed on the machine), versions pinned to match uv.lock so production runs what the tests ran.
# The web endpoint only reads a JSON file, so it gets a small image and wakes up fast.
web_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("fastapi[standard]==0.162.0")
    .add_local_python_source("vicdemand")
)

HF_CACHE = "/models"


def download_chronos():
    from huggingface_hub import snapshot_download

    snapshot_download("amazon/chronos-2")  # into $HF_HOME, where from_pretrained looks


# The nightly job runs the models. PyTorch comes from its CPU-only index (no 2 GB of CUDA libraries: we don't use a GPU).
# Chronos-2's weights (~480 MB) are downloaded once while the image is built, not on every run.
job_image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.14.1", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("pandas==3.1.0", "pyarrow==26.0.0", "numpy==2.5.4", "httpx==0.28.1", "fastapi[standard]==0.162.0",
                    "lightgbm==4.7.0", "statsforecast==2.0.1", "chronos-forecasting==2.3.2", "holidays==0.107")
    .env({"HF_HOME": HF_CACHE, "HF_HUB_OFFLINE": "1"})  # offline: use the baked-in weights, never re-download at run time
    .run_function(download_chronos, env={"HF_HUB_OFFLINE": "0"})
    .add_local_python_source("vicdemand")
)

# A Volume is a shared disk that outlives containers: the data cache, the forecast log and latest.json live here.
# The daily job writes it; the web endpoint only reads it.
volume = modal.Volume.from_name("vic-demand-data", create_if_missing=True)
ROOT = Path("/data")


@app.function(
    image=job_image,
    volumes={ROOT: volume},
    cpu=4,  # the weekly LSTM retrain takes a few minutes on 4 cores; daily runs take about a minute
    memory=4096,
    # 01:00 NEM time: AEMO's current-month file is updated just after midnight with the day that just finished.
    # "Australia/Brisbane" because the NEM runs on AEST all year (no daylight saving).
    schedule=modal.Cron("0 1 * * *", timezone="Australia/Brisbane"),
    timeout=30 * 60,  # the first run downloads ~8 years of data and trains everything; later runs fetch one month
    retries=2,  # AEMO or Open-Meteo hiccups shouldn't cost a day's forecast
)
def daily():
    from vicdemand.pipeline import run_daily

    latest = run_daily(ROOT)
    volume.commit()  # make the new files visible to the web endpoint's containers
    print("Forecast issued from", latest["origin"])


@app.function(
    image=web_image,
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
