"""
news.py — News Aggregation Module
===================================
Sources (all FREE, no API key):
  1. yfinance   — company-specific news headlines
  2. Yahoo RSS  — finance.yahoo.com RSS per symbol
  3. CNBC RSS   — market news
  4. Google News RSS — fallback per symbol

Sentiment tagging: keyword-based (no ML needed).
Cache TTL: 5 minutes per symbol.
"""

import re
import time
import logging
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, list]] = {}
NEWS_TTL = 300  # 5 minutes


def _cache_get(key: str) -> Optional[list]:
    if key in _cache:
        ts, val = _cache[key]
        if time.time() - ts < NEWS_TTL:
            return val
    return None


def _cache_set(key: str, val: list) -> list:
    _cache[key] = (time.time(), val)
    return val


# ──────────────────────────────────────────────────────────────
# Sentiment keywords
# ──────────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    "beat", "beats", "surges", "rally", "record", "upgrade", "buy",
    "strong", "gains", "rise", "rises", "soars", "profit", "growth",
    "outperform", "raised guidance", "exceeds", "breakthrough", "launches",
    "wins", "awarded", "positive", "bull", "expansion", "deal", "partnership",
    "revenue beat", "eps beat", "better than expected", "higher", "boost",
}
NEGATIVE_WORDS = {
    "miss", "misses", "falls", "drops", "decline", "declines", "cut",
    "downgrade", "sell", "weak", "loss", "lower", "disappoints", "plunges",
    "recall", "investigation", "fine", "penalty", "layoffs", "restructuring",
    "warning", "concern", "risk", "worse than expected", "below", "slump",
    "delay", "lawsuit", "fraud", "breach", "hack", "shortage",
}
HIGH_IMPACT_WORDS = {
    "earnings", "eps", "quarterly results", "guidance", "acquisition", "merger",
    "fda", "sec", "doj", "ftc", "investigation", "recall", "bankruptcy",
    "ceo", "cfo", "management change", "dividend", "buyback", "split",
}


def tag_sentiment(title: str, summary: str = "") -> tuple[str, str]:
    """
    Returns (sentiment, catalyst_type).
    sentiment: "positive" | "negative" | "neutral"
    catalyst:  "positive_catalyst" | "negative_catalyst" | "neutral" | "high_impact"
    """
    text = (title + " " + summary).lower()

    bull = sum(1 for w in POSITIVE_WORDS if w in text)
    bear = sum(1 for w in NEGATIVE_WORDS if w in text)
    high = any(w in text for w in HIGH_IMPACT_WORDS)

    if bull > bear:
        sentiment = "positive"
    elif bear > bull:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    if high and sentiment == "positive":
        catalyst = "positive_catalyst"
    elif high and sentiment == "negative":
        catalyst = "negative_catalyst"
    elif high:
        catalyst = "high_impact"
    elif sentiment == "positive":
        catalyst = "positive_catalyst"
    elif sentiment == "negative":
        catalyst = "negative_catalyst"
    else:
        catalyst = "neutral"

    return sentiment, catalyst


def _clean(text: str) -> str:
    """Strip HTML tags and extra whitespace."""
    text = re.sub(r"<[^>]+>", "", text or "")
    return re.sub(r"\s+", " ", text).strip()[:500]


def _parse_pubdate(raw: str) -> str:
    """Normalize pubdate to ISO string."""
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(raw.strip(), fmt).isoformat()
        except ValueError:
            continue
    return raw[:30]


# ──────────────────────────────────────────────────────────────
# Source adapters
# ──────────────────────────────────────────────────────────────

def _fetch_rss(url: str, source_name: str, symbol: str = "", max_items: int = 8) -> list[dict]:
    """Generic RSS fetcher."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 SwingTraderPro/1.0",
            "Accept": "application/rss+xml,application/xml,text/xml",
        })
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = resp.read()
        root = ET.fromstring(data)
        items = root.findall(".//item")
        results = []
        for item in items[:max_items]:
            title = _clean(item.findtext("title", ""))
            if not title:
                continue
            desc   = _clean(item.findtext("description", ""))
            link   = item.findtext("link", "")
            pub    = _parse_pubdate(item.findtext("pubDate", ""))
            sent, cat = tag_sentiment(title, desc)
            results.append({
                "title":       title,
                "summary":     desc[:200] if desc else "",
                "source":      source_name,
                "url":         link,
                "published_at": pub,
                "symbol":      symbol,
                "sentiment":   sent,
                "catalyst":    cat,
            })
        return results
    except Exception as e:
        log.debug(f"[news] RSS {source_name} ({url[:50]}): {e}")
        return []


def _from_yfinance(symbol: str) -> list[dict]:
    """yfinance news — company-specific, no key needed."""
    try:
        import yfinance as yf
        raw = yf.Ticker(symbol).news or []
        results = []
        for n in raw[:10]:
            title = _clean(n.get("title", ""))
            if not title:
                continue
            ts = n.get("providerPublishTime", 0)
            pub = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""
            sent, cat = tag_sentiment(title)
            results.append({
                "title":        title,
                "summary":      "",
                "source":       n.get("publisher", "Yahoo Finance"),
                "url":          n.get("link", ""),
                "published_at": pub,
                "symbol":       symbol,
                "sentiment":    sent,
                "catalyst":     cat,
            })
        return results
    except Exception as e:
        log.debug(f"[news] yfinance {symbol}: {e}")
        return []


def _from_yahoo_rss(symbol: str) -> list[dict]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    return _fetch_rss(url, "Yahoo Finance", symbol)


def _from_google_news(symbol: str, company: str = "") -> list[dict]:
    query = urllib.request.quote(f"{symbol} {company} stock".strip())
    url   = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    return _fetch_rss(url, "Google News", symbol)


def _from_cnbc_rss() -> list[dict]:
    """CNBC top market news — no symbol filter."""
    urls = [
        ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "CNBC Markets"),
        ("https://www.cnbc.com/id/10001147/device/rss/rss.html",  "CNBC Finance"),
    ]
    results = []
    for url, name in urls:
        results.extend(_fetch_rss(url, name, max_items=5))
    return results[:8]


def _from_seeking_alpha_rss(symbol: str) -> list[dict]:
    url = f"https://seekingalpha.com/api/sa/combined/{symbol}.xml"
    return _fetch_rss(url, "Seeking Alpha", symbol)


# ──────────────────────────────────────────────────────────────
# Deduplication
# ──────────────────────────────────────────────────────────────

def _dedupe(items: list[dict]) -> list[dict]:
    seen, out = set(), []
    for item in items:
        key = re.sub(r"\W+", "", item["title"][:50].lower())
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ──────────────────────────────────────────────────────────────
# Main public function
# ──────────────────────────────────────────────────────────────

def get_news(symbol: str, company: str = "", max_items: int = 12) -> list[dict]:
    """
    Fetch news for a symbol from multiple free sources.
    Results cached for NEWS_TTL seconds.
    """
    sym = symbol.upper()
    cached = _cache_get(sym)
    if cached is not None:
        log.debug(f"[news] Cache hit {sym}")
        return cached[:max_items]

    log.info(f"[news] Fetching {sym}...")
    items: list[dict] = []

    # Source 1: yfinance (most relevant — company-specific)
    items.extend(_from_yfinance(sym))

    # Source 2: Yahoo Finance RSS
    if len(items) < 6:
        items.extend(_from_yahoo_rss(sym))

    # Source 3: Google News RSS (fallback)
    if len(items) < 4:
        items.extend(_from_google_news(sym, company))

    # Deduplicate and sort by publish time (newest first)
    items = _dedupe(items)
    items.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    result = items[:max_items]
    _cache_set(sym, result)
    log.info(f"[news] {sym}: {len(result)} articles")
    return result


def get_market_news(max_items: int = 10) -> list[dict]:
    """General market news from CNBC RSS."""
    cached = _cache_get("__market__")
    if cached is not None:
        return cached[:max_items]
    items = _from_cnbc_rss()
    items = _dedupe(items)
    _cache_set("__market__", items)
    return items[:max_items]


def get_news_summary(symbol: str, company: str = "") -> dict:
    """
    Returns a short news summary for embedding in /api/watchlist.
    Keeps the response size small.
    """
    items = get_news(symbol, company, max_items=5)
    if not items:
        return {"count": 0, "top": [], "sentiment": "neutral", "has_catalyst": False}

    positive = sum(1 for n in items if n["sentiment"] == "positive")
    negative = sum(1 for n in items if n["sentiment"] == "negative")
    has_cat  = any(n["catalyst"] in ("positive_catalyst", "negative_catalyst") for n in items)

    if positive > negative:
        overall = "positive"
    elif negative > positive:
        overall = "negative"
    else:
        overall = "neutral"

    return {
        "count":       len(items),
        "sentiment":   overall,
        "has_catalyst": has_cat,
        "top": [
            {
                "title":    n["title"],
                "source":   n["source"],
                "url":      n["url"],
                "published_at": n["published_at"],
                "sentiment":    n["sentiment"],
                "catalyst":     n["catalyst"],
            }
            for n in items[:3]
        ],
    }
