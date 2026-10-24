"""Google News fills the news gap for listings Yahoo tags no articles with (most JSE shares).

It searches by company name, never leaks articles from after the window, and
reports itself unavailable, so the chain moves on, when it cannot search.
"""

import pytest
import requests

from tradingagents.dataflows import router
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.vendors import google_news as gn

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>"Omnia Holdings" - Google News</title>
<item><title>Omnia receives R21.8bn takeover bid from Solar Industries - Mining Review Africa</title>
  <link>https://news.google.com/rss/articles/a1</link><pubDate>Wed, 17 Sep 2026 08:00:00 GMT</pubDate>
  <source url="https://www.miningreview.com">Mining Review Africa</source></item>
<item><title>Omnia in advanced talks over potential buyout - Moneyweb</title>
  <link>https://news.google.com/rss/articles/a2</link><pubDate>Fri, 11 Sep 2026 06:30:00 GMT</pubDate>
  <source url="https://www.moneyweb.co.za">Moneyweb</source></item>
<item><title>Omnia in advanced talks over potential buyout - Business Day</title>
  <link>https://news.google.com/rss/articles/a2b</link><pubDate>Fri, 11 Sep 2026 07:00:00 GMT</pubDate>
  <source url="https://www.businesslive.co.za">Business Day</source></item>
<item><title>Omnia deal completes - Reuters</title>
  <link>https://news.google.com/rss/articles/a3</link><pubDate>Mon, 02 Nov 2026 09:00:00 GMT</pubDate>
  <source url="https://www.reuters.com">Reuters</source></item>
</channel></rss>"""


class _Response:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


@pytest.fixture
def omnia(monkeypatch):
    calls = []
    monkeypatch.setattr(gn, "_company_name", lambda ticker: "Omnia Holdings Ltd")

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(params)
        return _Response(FEED)

    monkeypatch.setattr(gn.requests, "get", fake_get)
    return calls


@pytest.mark.unit
@pytest.mark.parametrize("registered, searched", [
    ("Omnia Holdings Ltd", "Omnia Holdings"), ("Sygnia Limited", "Sygnia"), ("Alphabet Inc.", "Alphabet"),
    ("Berkshire Hathaway Inc. Class B", "Berkshire Hathaway"), ("AstraZeneca PLC", "AstraZeneca"),
])
def test_legal_form_words_are_dropped_from_the_search(registered, searched):
    assert gn.search_name(registered) == searched


@pytest.mark.unit
def test_searches_the_name_in_the_local_edition_within_the_window(omnia):
    out = gn.get_news_google("OMN.JO", "2026-09-01", "2026-09-30")
    (params,) = omnia
    assert params["q"] == '"Omnia Holdings" after:2026-09-01 before:2026-10-01'
    assert (params["hl"], params["gl"]) == ("en-ZA", "ZA")
    assert "R21.8bn takeover bid" in out and "(source: Mining Review Africa, 2026-09-17)" in out
    assert " - Mining Review Africa" not in out  # the source tail is stripped from the title


@pytest.mark.unit
def test_nothing_after_the_window_leaks_in(omnia):
    out = gn.get_news_google("OMN.JO", "2026-09-01", "2026-09-30")
    assert "deal completes" not in out


@pytest.mark.unit
def test_the_same_headline_from_two_outlets_appears_once(omnia):
    out = gn.get_news_google("OMN.JO", "2026-09-01", "2026-09-30")
    assert out.count("Omnia in advanced talks over potential buyout") == 1


@pytest.mark.unit
def test_an_empty_window_says_so_rather_than_failing(omnia):
    out = gn.get_news_google("OMN.JO", "2026-08-01", "2026-08-31")
    assert out.startswith('No Google News headlines naming "Omnia Holdings"')


@pytest.mark.unit
def test_no_name_or_a_failed_request_is_unavailable(monkeypatch):
    monkeypatch.setattr(gn, "_company_name", lambda ticker: None)
    with pytest.raises(VendorUnavailableError):
        gn.get_news_google("XYZ.JO", "2026-09-01", "2026-09-30")
    monkeypatch.setattr(gn, "_company_name", lambda ticker: "Omnia Holdings")
    monkeypatch.setattr(gn.requests, "get", lambda *a, **k: _Response("", status=503))
    with pytest.raises(VendorUnavailableError):
        gn.get_news_google("OMN.JO", "2026-09-01", "2026-09-30")


@pytest.mark.unit
def test_it_follows_yahoo_in_the_default_news_chain(omnia, monkeypatch):
    def yahoo_has_nothing(ticker, start, end):
        raise VendorUnavailableError("Yahoo Finance has no news articles tagged OMN.JO")

    monkeypatch.setitem(router.VENDOR_METHODS["get_news"], "yfinance", yahoo_has_nothing)
    out = router.route_to_vendor("get_news", "OMN.JO", "2026-09-01", "2026-09-30")
    assert "takeover bid" in out
    assert "google_news" not in router.VENDOR_METHODS["get_global_news"]
