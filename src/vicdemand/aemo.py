"""AEMO's own demand forecasts, as one more "model" to compare against.

AEMO publishes two forecasts that cover our 48 hours:
- Pre-dispatch (PD): run every half-hour; the run issued about 23:30 forecasts every half-hour from midnight to 04:00 the
  morning after next (56 of our 96 steps). Its TOTALDEMAND uses the same definition as the actual demand we forecast.
- ST PASA: run hourly, half-hourly demand for the next 7 days at 10/50/90% probability of exceedance (POE).
  We use the 50% POE value from the run issued about 22:00 (or the latest one before it), for the steps pre-dispatch
  doesn't reach. AEMO documents pre-dispatch TOTALDEMAND as the same measure as actual demand; we haven't confirmed that
  ST PASA's DEMAND50 is defined identically, and its bias does shift where the two meet (see the page).

Both live on NEMWEB: the last ~2 weeks as one zip per run under Reports/Current, older runs bundled into big weekly (PD,
~285 MB) or monthly (ST PASA, ~200 MB) zips under Reports/Archive. We read only the one run we need out of an archive
with HTTP range requests (a zip's table of contents is at its end), so a year of history is a few MB of downloads, not 15 GB.
"""

from __future__ import annotations

import csv
import io
import re
import time
import zipfile
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import httpx
import pandas as pd

from .data import HEADERS, REGION

NEMWEB = "https://nemweb.com.au/Reports"
PD_DIR, STPASA_DIR = "PredispatchIS_Reports", "Short_Term_PASA_Reports"
STEP = pd.Timedelta("30min")


def _get(url: str, **kw) -> httpx.Response:
    """GET with polite, slowing retries: NEMWEB answers 403/429 when it thinks we're asking too fast, so we back off
    (15 s, 1 min, 3 min) and give up after that rather than keep asking."""
    for wait in (0, 15, 60, 180):
        time.sleep(wait)
        r = _client().get(url, **kw)
        if r.status_code not in (403, 429, 503):
            break
    r.raise_for_status()
    return r


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests: enough for zipfile to pick single members out of a remote zip.
    Reads ahead in BLOCK-sized chunks, because zipfile makes many tiny reads and each would otherwise be a request."""

    BLOCK = 512 * 1024

    def __init__(self, url: str, client: httpx.Client):
        self.url, self.client, self.pos = url, client, 0
        self.size = int(client.head(url).headers["content-length"])
        self.buf_start, self.buf = 0, b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        if not (self.buf_start <= self.pos and end <= self.buf_start + len(self.buf)):
            fetch_end = min(self.size, max(end, self.pos + self.BLOCK))
            self.buf_start, self.buf = self.pos, _get(self.url, headers={"Range": f"bytes={self.pos}-{fetch_end - 1}"}).content
        data = self.buf[self.pos - self.buf_start:end - self.buf_start]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


@cache
def _client() -> httpx.Client:
    return httpx.Client(headers=HEADERS, follow_redirects=True, timeout=120)


@cache
def _listing(folder: str) -> tuple[str, ...]:
    html = _get(f"{NEMWEB}/{folder}/").text
    return tuple(sorted(set(re.findall(r"PUBLIC_[A-Z]+_[0-9_]+\.zip", html))))


@cache
def _archive(url: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(HttpFile(url, _client()))


def _run_csv(prefix: str, run: str, folder: str) -> str | None:
    """The CSV text of the run whose file name starts with `prefix + run`, from Current or Archive. None if not published."""
    def unpack(blob: bytes) -> str:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            return z.read(z.namelist()[0]).decode()

    for name in _listing(f"Current/{folder}"):
        if name.startswith(prefix + run):
            return unpack(_get(f"{NEMWEB}/Current/{folder}/{name}").content)
    # Archive zips are named by the dates they span; a run near midnight can sit in either neighbour, so try all that are close.
    day = pd.Timestamp(run[:8])
    for name in _listing(f"Archive/{folder}"):
        dates = [pd.Timestamp(d) for d in re.findall(r"_(\d{8})", name)]
        lo, hi = dates[0], dates[-1] if len(dates) > 1 else dates[0] + pd.offsets.MonthEnd(0)
        if not (lo - pd.Timedelta(days=1) <= day <= hi + pd.Timedelta(days=1)):
            continue
        z = _archive(f"{NEMWEB}/Archive/{folder}/{name}")
        member = next((m for m in z.namelist() if m.startswith(prefix + run)), None)
        if member:
            return unpack(z.read(member))
    return None


def _table(text: str, table: str) -> pd.DataFrame:
    """One table out of AEMO's multi-table CSV format: 'I' rows are headers, 'D' rows are data."""
    header, rows = None, []
    for row in csv.reader(line for line in text.splitlines() if line.startswith(("I,", "D,")) and f",{table}," in line[:40]):
        if row[0] == "I":
            header = row
        else:
            rows.append(row)
    return pd.DataFrame(rows, columns=header)


def predispatch(origin: pd.Timestamp) -> pd.Series | None:
    """VIC1 demand forecast from the pre-dispatch run for `origin` (published ~30 minutes before it), period-ending times."""
    text = _run_csv("PUBLIC_PREDISPATCHIS_", origin.strftime("%Y%m%d%H%M"), PD_DIR)
    if text is None:
        return None
    df = _table(text, "REGION_SOLUTION")
    df = df[(df["REGIONID"] == REGION) & (df["INTERVENTION"] == "0")]
    return pd.Series(df["TOTALDEMAND"].astype(float).to_numpy(), index=pd.to_datetime(df["DATETIME"]), name="aemo").sort_index()


def stpasa(origin: pd.Timestamp) -> pd.Series | None:
    """VIC1 50% POE demand from the last ST PASA run issued before `origin`."""
    for hours_back in range(2, 13, 1):  # the ~22:00 run, or earlier if that one's missing
        run = origin - pd.Timedelta(hours=hours_back)
        text = _run_csv("PUBLIC_STPASA_", run.strftime("%Y%m%d%H%M"), STPASA_DIR)
        if text is None:
            continue
        df = _table(text, "REGIONSOLUTION")
        df = df[df["REGIONID"] == REGION]
        return pd.Series(df["DEMAND50"].astype(float).to_numpy(), index=pd.to_datetime(df["INTERVAL_DATETIME"]), name="aemo").sort_index()
    return None


@dataclass
class AEMOForecast:
    """AEMO's forecast as a model: pre-dispatch where it reaches, ST PASA beyond. Needs no history; downloads instead."""

    name: str = "aemo"
    label: str = "AEMO's own forecast"
    cache: Path = field(default_factory=lambda: Path("data/cache/aemo"))  # one small file per origin: AEMO never revises old runs

    def forecast(self, demand, weather, origin, horizon):
        path = self.cache / f"{origin:%Y%m%d%H%M}_{horizon}.parquet"
        if path.exists():
            return pd.read_parquet(path)[self.name]
        targets = pd.date_range(origin + STEP, periods=horizon, freq=STEP, name="time")
        out = pd.Series(float("nan"), index=targets, name=self.name)
        for source in (stpasa(origin), predispatch(origin)):  # pre-dispatch last, so it wins where both exist
            if source is not None:
                hit = source.reindex(targets)
                out[hit.notna()] = hit[hit.notna()]
        if out.notna().all():  # only cache complete forecasts, so a run that wasn't published yet is retried next time
            self.cache.mkdir(parents=True, exist_ok=True)
            out.to_frame().to_parquet(path)
        return out
