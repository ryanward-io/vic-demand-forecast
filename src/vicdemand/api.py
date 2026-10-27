"""Read-only web API: hands the latest forecast to the page. All the work happens in the daily job, not per request."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# CORS: which web pages browsers will let read our responses. This is not access control (the data is public anyway);
# it stops other sites embedding our API in their pages and running up our bill.
ALLOWED_ORIGINS = ["https://ryanward.dev"]
LOCAL_ORIGINS = r"http://(localhost|127\.0\.0\.1)(:\d+)?"


def create_app(root: Path, allow_local: bool = False, refresh: Callable[[], None] | None = None) -> FastAPI:
    """`refresh` runs before each read, e.g. to pick up a newer copy of the Modal Volume."""
    app = FastAPI(title="Victorian demand forecast", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=LOCAL_ORIGINS if allow_local else None,
                       allow_methods=["GET"])

    @app.get("/forecast")
    def forecast():
        if refresh:
            refresh()
        path = root / "latest.json"
        if not path.exists():
            raise HTTPException(503, "No forecast yet: the daily job hasn't run.")
        return json.loads(path.read_text())

    @app.get("/health")
    def health():
        return {"ok": True}

    return app
