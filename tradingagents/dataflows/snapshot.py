"""Serve a run's data from a snapshot collected beforehand, instead of fetching it live.

Two runs of one ticker an hour apart see different headlines, a revised price, a feed that answered once and was
throttled the next time. Rerunning on the same inputs should differ only by the model, and a backtest should
read exactly what the live run read, so a caller can hand a run the data it collected for the date
(``config["data_snapshot"]``, a JSON file) and every tool answers from it:

- Prices: the symbol's price file as the collector left it. ``load_ohlcv`` reads it instead of refreshing, so price
  data, every indicator and the verified snapshot compute from the same bars as live, for any window.
- Fundamentals, statements, insider transactions, StockTwits: the stored answer to the same call on the date.
- News: the stored week. A narrower window keeps the dated headlines inside it (Yahoo's carry no date and stay);
  a wider one says the snapshot holds only that week.
- Market news: the stored answer, cut to the requested number of headlines.
- Prediction markets: a topic the collector asked about.
- A settled "no data" (no statements for a fund, no headlines that week) is served as collected; a throttle or
  an error is fetched again.
- Macro series stay live: FRED serves each series as known on the date, so the answer is already fixed.

A call the snapshot cannot answer is fetched live and recorded in ``misses``, so the run can say which inputs were
not the collected ones. The file::

    {"as_of": "2026-10-28", "ticker": "GOOGL", "prices_dir": "/path",
     "calls": [{"ticker": "GOOGL", "tool": "get_fundamentals", "args": {...}, "status": "ok", "output": "..."}]}
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import date

from tradingagents.dataflows.config import get_config

# Methods answered by their fixed source rather than from the snapshot, and so never counted as misses.
LIVE_BY_DESIGN = {"get_macro_indicators", "get_stock_data", "get_indicators"}

_lock = threading.Lock()
_loaded: dict[str, "Snapshot"] = {}


def active() -> "Snapshot | None":
    """The snapshot this run was handed, or None for a live run."""
    path = get_config().get("data_snapshot")
    if not path:
        return None
    with _lock:
        if path not in _loaded:
            with open(path, encoding="utf-8") as f:
                _loaded[path] = Snapshot(json.load(f))
        return _loaded[path]


def _blocks(text: str) -> tuple[str, list[str]]:
    """A news answer as its heading and its "### headline" blocks."""
    parts = re.split(r"\n(?=### )", text)
    return (parts[0], parts[1:]) if parts and not parts[0].startswith("### ") else ("", parts)


def _answered(call: dict) -> bool:
    """Whether a collected call holds the vendor's answer: data, or a settled "no data" (no statements for a fund,
    no headlines that week). A throttle or an error says nothing about the data, so that call is fetched again."""
    return call.get("status") in ("ok", "empty") or str(call.get("output") or "").startswith("NO_DATA_AVAILABLE")


class Snapshot:
    def __init__(self, data: dict):
        self.as_of = data["as_of"]
        self.ticker = data["ticker"].upper()
        self.prices_dir = data.get("prices_dir")
        self.calls = [c for c in data.get("calls", []) if _answered(c)]
        self.misses: list[dict] = []

    def _find(self, tool: str, ticker: str | None = None, **args):
        for c in self.calls:
            if c["tool"] != tool or (ticker is not None and c.get("ticker", "").upper() != ticker.upper()):
                continue
            if all(str(c["args"].get(k)).lower() == str(v).lower() for k, v in args.items()):
                return c["output"]
        return None

    def _same_day(self, day) -> bool:
        return day in (None, "", self.as_of)

    def answer(self, method: str, args: tuple, kwargs: dict) -> str | None:
        """The stored answer to this router call, or None (and a recorded miss) when the snapshot has none."""
        if method in LIVE_BY_DESIGN:
            return None
        out = self._answer(method, list(args) + list(kwargs.values()))
        if out is None:
            self.misses.append({"method": method, "args": [str(a) for a in args],
                                "kwargs": {k: str(v) for k, v in kwargs.items()}})
        return out

    def _answer(self, method: str, a: list) -> str | None:
        ours = lambda i: len(a) > i and str(a[i]).upper() == self.ticker
        if method == "get_fundamentals" and ours(0) and self._same_day(a[1] if len(a) > 1 else None):
            return self._find("get_fundamentals", self.ticker)
        if method in ("get_income_statement", "get_balance_sheet", "get_cashflow") and ours(0):
            freq = (a[1] if len(a) > 1 else "quarterly") or "quarterly"
            if self._same_day(a[2] if len(a) > 2 else None):
                return self._find(method, self.ticker, freq=freq)
        if method == "get_insider_transactions" and ours(0) and self._same_day(a[1] if len(a) > 1 else None):
            return self._find("get_insider_transactions", self.ticker)
        if method == "get_news" and ours(0) and len(a) >= 3 and a[2] == self.as_of:
            return self.news(str(a[1]))
        if method == "get_global_news" and len(a) >= 1 and a[0] == self.as_of:
            stored = self._find("get_global_news", "")
            limit = a[2] if len(a) > 2 else None
            if stored is not None and limit:
                head, blocks = _blocks(stored)
                stored = "\n".join([head, *blocks[:int(limit)]]) if head else "\n".join(blocks[:int(limit)])
            return stored
        if method == "get_prediction_markets" and len(a) >= 1:
            return self._find("get_prediction_markets", "", topic=a[0])
        return None

    def news(self, start: str) -> str | None:
        """The stored week of company news, kept to [start, as_of] where the headlines carry a date."""
        stored = next((c for c in self.calls if c["tool"] == "get_news" and c.get("ticker", "").upper() == self.ticker),
                      None)
        if stored is None:
            return None
        held_from = stored["args"].get("start_date") or start
        if start < held_from:
            return (f"(This run reads news collected on {self.as_of}, which covers {held_from} to {self.as_of} only;"
                    f" earlier headlines in the requested window are not in it.)\n\n{stored['output']}")
        head, blocks = _blocks(stored["output"])
        dated = lambda b: re.search(r"\(source: [^)]*?, (\d{4}-\d\d-\d\d)\)", b.splitlines()[0])
        kept = [b for b in blocks if not dated(b) or dated(b).group(1) >= start]
        return "\n".join([head, *kept]) if head else "\n".join(kept)

    def stocktwits(self, ticker: str) -> str | None:
        return self._find("stocktwits", ticker) if ticker.upper() == self.ticker else None

    def price_file(self, safe_symbol: str) -> str | None:
        """The collected price file for a symbol, when the snapshot holds one."""
        if not self.prices_dir:
            return None
        path = os.path.join(self.prices_dir, f"{safe_symbol}-YFin-data.csv")
        return path if os.path.exists(path) else None


def check_date(trade_date: str) -> None:
    """Refuse a snapshot collected for another date: its answers would be another day's data."""
    snap = active()
    if snap is not None and snap.as_of != str(date.fromisoformat(str(trade_date)[:10])):
        raise ValueError(f"data snapshot is for {snap.as_of}, not the run's date {trade_date}")
