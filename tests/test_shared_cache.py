"""One fetch serves every analysis running side by side, and failures aren't remembered."""

import time

import pytest

from tradingagents.dataflows import router, shared_cache


@pytest.mark.unit
def test_a_second_caller_reads_the_first_fetch():
    calls = []
    fetch = lambda: calls.append(1) or "news"
    assert shared_cache.cached("ns", ["k"], 60, fetch) == "news"
    assert shared_cache.cached("ns", ["k"], 60, fetch) == "news"
    assert len(calls) == 1


@pytest.mark.unit
def test_an_entry_older_than_its_ttl_is_fetched_again():
    calls = []
    shared_cache.cached("ns", ["k"], 60, lambda: calls.append(1) or "old")
    assert shared_cache.cached("ns", ["k"], 0, lambda: calls.append(1) or "new") == "new"
    assert len(calls) == 2


@pytest.mark.unit
def test_a_failure_is_returned_but_not_kept():
    assert shared_cache.cached("ns", ["k"], 60, lambda: None) is None
    assert shared_cache.cached("ns", ["k"], 60, lambda: "fresh") == "fresh"


@pytest.mark.unit
def test_requests_to_one_host_are_spaced():
    start = time.time()
    for _ in range(3):
        with shared_cache.paced("example.org", 0.2):
            pass
    assert time.time() - start >= 0.4


@pytest.mark.unit
def test_global_news_is_shared_but_an_unavailable_answer_is_not(monkeypatch):
    calls = []

    def vendor(*args, **kwargs):
        calls.append(args)
        return "## Global news" if len(calls) > 1 else router.vendor_unavailable("get_global_news", RuntimeError("429"))

    monkeypatch.setitem(router.VENDOR_METHODS["get_global_news"], "yfinance", vendor)
    first = router.route_to_vendor("get_global_news", "2026-10-23", 7, 10)
    assert first.startswith("DATA_UNAVAILABLE")             # returned, not kept
    assert router.route_to_vendor("get_global_news", "2026-10-23", 7, 10) == "## Global news"
    assert router.route_to_vendor("get_global_news", "2026-10-23", 7, 10) == "## Global news"
    assert len(calls) == 2                                   # the third came from the cache
    router.route_to_vendor("get_global_news", "2026-10-24", 7, 10)
    assert len(calls) == 3                                   # another day is another question


@pytest.mark.unit
def test_reddit_feeds_are_shared_per_ticker(monkeypatch):
    from tradingagents.dataflows.vendors import reddit

    calls = []
    monkeypatch.setattr(reddit, "_fetch_subreddit_rss",
                        lambda ticker, sub, limit, timeout: calls.append(ticker) or [])
    reddit._shared_feed("GOOGL", "stocks", 25, 5)
    reddit._shared_feed("googl", "stocks", 25, 5)
    reddit._shared_feed("META", "stocks", 25, 5)
    assert calls == ["GOOGL", "META"]


@pytest.mark.unit
def test_prune_drops_entries_nobody_refreshed_and_keeps_the_rest():
    import os

    shared_cache.cached("ns", ["old"], 60, lambda: "old")
    shared_cache.cached("ns", ["new"], 60, lambda: "new")
    old = shared_cache._key_file("ns", ["old"])
    week_ago = time.time() - 8 * 86400
    os.utime(old, (week_ago, week_ago))
    assert shared_cache.prune(max_age_days=7) == 1
    assert not os.path.exists(old)
    assert os.path.exists(shared_cache._key_file("ns", ["new"]))
