"""Run the API locally without Modal: uv run python dev.py, then open http://localhost:8011/?api=http://localhost:8002
(serve the page with python3 -m http.server 8011). Runs the daily job first so there is something to show."""

from pathlib import Path

import uvicorn

from vicdemand.api import create_app
from vicdemand.pipeline import run_daily

ROOT = Path("data")

if __name__ == "__main__":
    run_daily(ROOT)
    uvicorn.run(create_app(ROOT, allow_local=True), host="127.0.0.1", port=8002)
