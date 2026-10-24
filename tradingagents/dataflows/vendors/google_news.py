"""Company news from Google News RSS search, by company name. Keyless.

Yahoo tags no articles with most non-US listings (JSE, much of Asia), so the
news analyst saw nothing for them, even a takeover bid. Google News indexes the
local press, so searching the company's name finds what Yahoo's ticker tags miss.

The search is by name, not ticker: "OMN.JO" matches nothing, "Omnia Holdings"
does. The name comes from the same identity lookup that anchors the agents. The
query carries Google's after:/before: operators, and every item is filtered
again on its own publish date, so a historical window never sees later news.
Only headlines, sources and dates come back; the feed carries no article text.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import requests

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.date_window import in_window
from tradingagents.dataflows.errors import VendorUnavailableError

logger = logging.getLogger(__name__)

RSS_URL = "https://news.google.com/rss/search"
# Edition per listing suffix, so local outlets rank; anything else reads the US edition.
_EDITIONS = {
    ".JO": ("en-ZA", "ZA"), ".L": ("en-GB", "GB"), ".AX": ("en-AU", "AU"), ".TO": ("en-CA", "CA"),
    ".NS": ("en-IN", "IN"), ".BO": ("en-IN", "IN"), ".SI": ("en-SG", "SG"), ".HK": ("en-HK", "HK"),
}
# Legal-form words that only narrow an exact-phrase search ("Sygnia Ltd" misses "Sygnia").
_LEGAL_FORM = re.compile(
    r"[,\s]+(ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|plc|n\.?v\.?|s\.?a\.?|ag|se|"
    r"co\.?|company|class [a-z])$", re.IGNORECASE)


def search_name(company_name: str) -> str:
    """The name to search: the registered name without trailing legal-form words."""
    name = company_name.strip()
    while True:
        shorter = _LEGAL_FORM.sub("", name).strip()
        if shorter == name or not shorter:
            return name
        name = shorter


def _edition(ticker: str) -> tuple[str, str]:
    for suffix, edition in _EDITIONS.items():
        if ticker.upper().endswith(suffix):
            return edition
    return "en-US", "US"


def _company_name(ticker: str) -> str | None:
    from tradingagents.agents.context import resolve_instrument_identity

    return resolve_instrument_identity(ticker).get("company_name")


def parse_feed(xml_text: str) -> list[dict]:
    """Items of a Google News RSS feed: title (without the " - Source" tail), source, link, date."""
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        source = (item.findtext("source") or "").strip()
        title = (item.findtext("title") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].rstrip()
        pub = None
        raw = item.findtext("pubDate")
        if raw:
            try:
                pub = parsedate_to_datetime(raw)
            except (TypeError, ValueError):
                pub = None
        items.append({"title": title or "No title", "source": source or "Unknown",
                      "link": (item.findtext("link") or "").strip(), "pub_date": pub})
    return items


def get_news_google(ticker: str, start_date: str, end_date: str) -> str:
    """Headlines naming the company, published in [start_date, end_date]."""
    company = _company_name(ticker)
    if not company:
        raise VendorUnavailableError(f"Google News needs a company name and none resolved for {ticker}")
    name = search_name(company)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    language, country = _edition(ticker)
    query = f'"{name}" after:{start_date} before:{(end_dt + timedelta(days=1)).strftime("%Y-%m-%d")}'
    try:
        response = requests.get(
            RSS_URL,
            params={"q": query, "hl": language, "gl": country, "ceid": f"{country}:{language.split('-')[0]}"},
            headers={"User-Agent": "Mozilla/5.0 (TradingAgents news)"},
            timeout=20,
        )
        response.raise_for_status()
        items = parse_feed(response.text)
    except (requests.RequestException, ET.ParseError) as exc:
        raise VendorUnavailableError(f"Google News search failed for {name!r}: {exc}") from exc

    seen, kept = set(), []
    for item in sorted(items, key=lambda i: i["pub_date"].timestamp() if i["pub_date"] else 0.0, reverse=True):
        if not in_window(item["pub_date"], start_dt, end_dt) or item["title"].lower() in seen:
            continue
        seen.add(item["title"].lower())
        kept.append(item)
    kept = kept[: get_config()["news_article_limit"]]
    label = f"{ticker} ({name})"
    if not kept:
        return f"No Google News headlines naming \"{name}\" between {start_date} and {end_date} ({label})."
    lines = [f"## {label} News from Google News, {start_date} to {end_date}",
             "Headlines only; the feed carries no article text.", ""]
    for item in kept:
        when = item["pub_date"].strftime("%Y-%m-%d") if item["pub_date"] else "undated"
        lines.append(f"### {item['title']} (source: {item['source']}, {when})")
        if item["link"]:
            lines.append(f"Link: {item['link']}")
        lines.append("")
    return "\n".join(lines)
