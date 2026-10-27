"""A small LSTM sequence-to-sequence forecaster, trained from scratch on our data.

Encoder: an LSTM reads the last 7 days half-hour by half-hour (demand, temperature, solar, time of day).
Decoder: a second LSTM starts from the encoder's memory and walks through the 96 target half-hours, reading the weather
forecast and calendar for each, and outputs the gap from the 4-week average (the same target LightGBM learns).

Inputs are scaled to roughly unit size; everything is seeded, so a refit on the same data gives the same model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from . import features

STEP = pd.Timedelta("30min")
DAY, WEEK = 48, 336
MW = 1000.0  # demand scale


def _slot_angle(t: pd.DatetimeIndex) -> np.ndarray:
    s = t - STEP
    return 2 * np.pi * (s.hour * 2 + s.minute // 30).to_numpy() / DAY


class Seq2Seq(nn.Module):
    def __init__(self, n_past: int, n_future: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.LSTM(n_past, hidden, batch_first=True)
        self.decoder = nn.LSTM(n_future, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, past, future):
        _, state = self.encoder(past)
        out, _ = self.decoder(future, state)
        return self.head(out).squeeze(-1)


@dataclass
class LSTM:
    name: str = "lstm"
    label: str = "LSTM (seq2seq)"
    epochs: int = 8
    every_hours: int = 3  # training origins: every 3 hours, not just midnight, for ~8x more examples
    net: Seq2Seq | None = None

    def _arrays(self, demand: pd.Series, weather: pd.DataFrame, origins: pd.DatetimeIndex, horizon: int):
        d = demand.to_numpy() / MW
        pos = demand.index.get_indexer(origins)
        past_idx = pos[:, None] + np.arange(-WEEK + 1, 1)  # last 7 days up to and including the origin
        past_t = demand.index[past_idx.ravel()]
        temp, solar = weather["temp"].reindex(past_t).to_numpy() / 10, weather["solar"].reindex(past_t).to_numpy() / 1000
        a = _slot_angle(past_t)
        past = np.stack([d[past_idx].ravel(), temp, solar, np.sin(a), np.cos(a)], axis=1).reshape(len(origins), WEEK, 5)

        X = features.make(demand, weather, origins, horizon)
        t = X.index.get_level_values("time")
        a = _slot_angle(t)
        dow = np.eye(7)[X["dow"].to_numpy()]
        future = np.column_stack([X["temp"] / 10, X["solar"] / 1000, X["day_max_temp"] / 10, np.sin(a), np.cos(a), X["holiday"],
                                  X["mean_4w"] / MW, X["lag_recent_day"] / MW, dow]).reshape(len(origins), horizon, -1)
        base = X["mean_4w"].to_numpy().reshape(len(origins), horizon)
        y = demand.reindex(t).to_numpy().reshape(len(origins), horizon)
        f32 = lambda x: torch.tensor(np.nan_to_num(x.astype(np.float32)))
        return f32(past), f32(future), base, y

    def fit(self, demand, weather, horizon: int = 2 * DAY):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        start = demand.index[0].normalize() + pd.Timedelta(days=29)
        origins = pd.date_range(start, demand.index[-1] - pd.Timedelta(days=2), freq=f"{self.every_hours}h")
        past, future, base, y = self._arrays(demand, weather, origins, horizon)
        target = torch.tensor(np.nan_to_num((y - base) / MW).astype(np.float32))
        self.net = Seq2Seq(past.shape[-1], future.shape[-1])
        opt = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        for _ in range(self.epochs):
            self.net.train()
            for batch in np.array_split(rng.permutation(len(origins)), max(1, len(origins) // 256)):
                opt.zero_grad()
                loss = (self.net(past[batch], future[batch]) - target[batch]).abs().mean()  # L1, like LightGBM
                loss.backward()
                opt.step()
            sched.step()
        return self

    def forecast(self, demand, weather, origin, horizon):
        past, future, base, _ = self._arrays(demand, weather, pd.DatetimeIndex([origin]), horizon)
        self.net.eval()
        with torch.no_grad():
            gap = self.net(past, future).numpy()[0] * MW
        return pd.Series(base[0] + gap, index=pd.date_range(origin + STEP, periods=horizon, freq=STEP, name="time"), name=self.name)
