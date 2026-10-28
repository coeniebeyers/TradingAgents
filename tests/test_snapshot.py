"""A run handed a data snapshot answers from it: the same inputs give the same answers, live or in a backtest."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tradingagents.dataflows import snapshot
from tradingagents.dataflows.config import run_config
from tradingagents.dataflows.router import route_to_vendor

AS_OF = "2026-10-28"
NEWS = """## SYG.JO (Sygnia) News from Google News, 2026-10-21 to 2026-10-28
Headlines only; the feed carries no article text.

### Shareholders approve share option scheme (source: News24, 2026-10-27)
Link: https://example.com/a

### Wierzycka weighs exit ahead of vote (source: IOL, 2026-10-22)
Link: https://example.com/b

### Sygnia unveils Singapore hub (source: Yahoo Finance)
Link: https://example.com/c
"""


def _calls():
    return [
        {"ticker": "SYG.JO", "tool": "get_fundamentals", "args": {"curr_date": AS_OF}, "status": "ok",
         "output": "# Company Fundamentals for SYG.JO\nMarket Cap: 5186838016 ZAR"},
        {"ticker": "SYG.JO", "tool": "get_balance_sheet", "args": {"freq": "annual", "curr_date": AS_OF},
         "status": "ok", "output": "# Balance Sheet for SYG.JO (annual)"},
        {"ticker": "SYG.JO", "tool": "get_news", "args": {"start_date": "2026-10-21", "end_date": AS_OF},
         "status": "ok", "output": NEWS},
        {"ticker": "SYG.JO", "tool": "stocktwits", "args": {"args": ["SYG.JO"]}, "status": "ok",
         "output": "<no StockTwits messages found for $SYG.JO>"},
        {"ticker": "", "tool": "get_global_news", "args": {"curr_date": AS_OF, "look_back_days": 7, "limit": 50},
         "status": "ok", "output": "## Global Market News\n\n### One (source: A)\nLink: x\n\n### Two (source: B)\nLink: y\n"},
        {"ticker": "", "tool": "get_prediction_markets", "args": {"topic": "Fed rate cut", "limit": 10},
         "status": "ok", "output": "Polymarket: Fed rate cut markets"},
        {"ticker": "SYG.JO", "tool": "get_insider_transactions", "args": {}, "status": "unavailable",
         "output": "DATA_UNAVAILABLE: throttled"},
    ]


@pytest.fixture
def snap_config(tmp_path):
    prices = tmp_path / "prices"
    prices.mkdir()
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"as_of": AS_OF, "ticker": "SYG.JO", "prices_dir": str(prices), "calls": _calls()}))
    snapshot._loaded.clear()
    with run_config({"data_snapshot": str(path), "data_cache_dir": str(tmp_path / "cache")}):
        yield prices
    snapshot._loaded.clear()


@pytest.mark.unit
def test_a_collected_call_is_answered_from_the_snapshot_not_the_vendor(snap_config, monkeypatch):
    monkeypatch.setattr("tradingagents.dataflows.router._route", lambda *a, **k: pytest.fail("fetched live"))
    assert route_to_vendor("get_fundamentals", "SYG.JO", AS_OF).endswith("Market Cap: 5186838016 ZAR")
    assert route_to_vendor("get_balance_sheet", "SYG.JO", "annual", AS_OF) == "# Balance Sheet for SYG.JO (annual)"
    assert route_to_vendor("get_prediction_markets", "fed rate cut", 6, AS_OF) == "Polymarket: Fed rate cut markets"
    assert snapshot.active().misses == []


@pytest.mark.unit
def test_what_the_snapshot_lacks_is_fetched_live_and_recorded(snap_config, monkeypatch):
    monkeypatch.setattr("tradingagents.dataflows.router._route", lambda method, *a, **k: f"live {method}")
    # Its insider call failed when collected, so it holds no answer; another date or topic isn't in it either.
    assert route_to_vendor("get_insider_transactions", "SYG.JO", AS_OF) == "live get_insider_transactions"
    assert route_to_vendor("get_fundamentals", "SYG.JO", "2026-10-20") == "live get_fundamentals"
    assert route_to_vendor("get_prediction_markets", "Strait of Hormuz", 6, AS_OF) == "live get_prediction_markets"
    # Macro series are point-in-time from FRED itself: live by design, not a miss.
    assert route_to_vendor("get_macro_indicators", "vix", AS_OF, 90) == "live get_macro_indicators"
    assert [m["method"] for m in snapshot.active().misses] == [
        "get_insider_transactions", "get_fundamentals", "get_prediction_markets"]


@pytest.mark.unit
def test_a_narrower_news_window_keeps_the_dated_headlines_inside_it(snap_config):
    out = route_to_vendor("get_news", "SYG.JO", "2026-10-25", AS_OF)
    assert "share option scheme" in out and "weighs exit" not in out
    assert "Singapore hub" in out  # undated (Yahoo): kept, as it can't be placed outside the window


@pytest.mark.unit
def test_a_wider_news_window_says_the_snapshot_holds_only_its_week(snap_config):
    out = route_to_vendor("get_news", "SYG.JO", "2026-09-28", AS_OF)
    assert out.startswith("(This run reads news collected on 2026-10-28, which covers 2026-10-21 to 2026-10-28 only")
    assert "weighs exit" in out


@pytest.mark.unit
def test_market_news_is_cut_to_the_requested_number(snap_config):
    out = route_to_vendor("get_global_news", AS_OF, 7, 1)
    assert "### One" in out and "### Two" not in out


@pytest.mark.unit
def test_stocktwits_and_prices_come_from_the_snapshot(snap_config, monkeypatch):
    from tradingagents.dataflows.vendors import stocktwits
    from tradingagents.dataflows.vendors.yahoo import ohlcv

    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: pytest.fail("fetched StockTwits live"))
    assert stocktwits.fetch_stocktwits_messages("SYG.JO") == "<no StockTwits messages found for $SYG.JO>"

    pd.DataFrame({"Date": ["2026-10-26", "2026-10-27", "2026-10-28"], "Open": [1.0, 2.0, 3.0],
                  "High": [1.0, 2.0, 3.0], "Low": [1.0, 2.0, 3.0], "Close": [10.0, 11.0, 12.0],
                  "Volume": [100, 100, 100]}).to_csv(snap_config / "SYG.JO-YFin-data.csv", index=False)
    monkeypatch.setattr(ohlcv.yf, "Ticker", lambda *a, **k: pytest.fail("downloaded prices"))
    frame = ohlcv.load_ohlcv("SYG.JO", "2026-10-27")
    assert list(frame["Close"]) == [10.0, 11.0]  # the collected bars, still cut at the as-of date


@pytest.mark.unit
def test_a_snapshot_from_another_date_is_refused(snap_config):
    snapshot.check_date(AS_OF)
    with pytest.raises(ValueError, match="for 2026-10-28, not the run's date 2026-10-29"):
        snapshot.check_date("2026-10-29")
