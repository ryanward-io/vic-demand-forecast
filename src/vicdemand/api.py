"""Read-only web API: hands the latest forecast to the page. All the work happens in the daily job, not per request."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# CORS: which web pages browsers will let read our responses. This is not access control (the data is public anyway);
# it stops other sites embedding our API in their pages and running up our bill.
ALLOWED_ORIGINS = ["https://ryanward.dev"]
LOCAL_ORIGINS = r"http://(localhost|127\.0\.0\.1)(:\d+)?"
# The forecast changes once a night, so there's no need to recheck the disk on every request, and browsers can keep a copy.
REFRESH_SECONDS = 300
CACHE_HEADERS = {"Cache-Control": "public, max-age=600"}


def create_app(root: Path, allow_local: bool = False, refresh: Callable[[], None] | None = None) -> FastAPI:
    """`refresh` picks up a newer copy of the data, e.g. reloading the Modal Volume; it runs at most every REFRESH_SECONDS."""
    app = FastAPI(title="Victorian demand forecast", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_origin_regex=LOCAL_ORIGINS if allow_local else None,
                       allow_methods=["GET"])
    state = {"checked": float("-inf"), "body": None}

    def latest() -> dict | None:
        if time.monotonic() - state["checked"] >= REFRESH_SECONDS or state["body"] is None:
            state["checked"] = time.monotonic()
            if refresh:
                try:
                    refresh()
                except Exception as e:  # noqa: BLE001 - e.g. a reload clashing with an open file: serve the copy we have
                    print(f"refresh failed: {type(e).__name__}: {e}")
            path = root / "latest.json"
            if path.exists():
                state["body"] = json.loads(path.read_text())
        return state["body"]

    @app.get("/forecast")
    def forecast():
        body = latest()
        if body is None:
            raise HTTPException(503, "No forecast yet: the daily job hasn't run.")
        return JSONResponse(body, headers=CACHE_HEADERS)

    @app.get("/health")
    def health():
        return {"ok": True}

    return app
