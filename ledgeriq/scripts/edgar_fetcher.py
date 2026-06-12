"""EDGAR fetcher — a real client for the SEC EDGAR public API.

No API key required. SEC asks for a descriptive User-Agent and rate-limits to
~10 requests/second; this client respects both.

This module is the shared core. It is wrapped two ways:
  - mcp_server/server.py exposes it as an MCP server (tool the agent calls)
  - scripts/build_corpus.py uses it directly to build the local RAG corpus

Endpoints used (all public, documented at https://www.sec.gov/edgar/sec-api-documentation):
  - https://www.sec.gov/files/company_tickers.json      (ticker -> CIK)
  - https://data.sec.gov/submissions/CIK##########.json (filing history)
  - https://www.sec.gov/Archives/edgar/data/...         (the filing documents)
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional

# SEC requires a descriptive User-Agent identifying the requester.
# Replace the contact with your own email when you run this.
USER_AGENT = "LedgerIQ Portfolio Demo (contact: your-email@example.com)"
_HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

_MIN_INTERVAL = 0.12  # ~8 req/s, safely under SEC's 10 req/s limit
_last_call = [0.0]


def _throttle():
    dt = time.time() - _last_call[0]
    if dt < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - dt)
    _last_call[0] = time.time()


def _get(url: str, timeout: int = 30) -> bytes:
    _throttle()
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data


@dataclass
class Filing:
    ticker: str
    company: str
    cik: str
    form: str           # 10-K, 10-Q, 8-K
    filing_date: str
    accession: str
    primary_doc: str
    url: str            # direct URL to the primary document on SEC.gov


def ticker_to_cik(ticker: str) -> Optional[tuple[str, str]]:
    """Return (cik_10digit, company_name) for a ticker, or None."""
    raw = _get("https://www.sec.gov/files/company_tickers.json")
    table = json.loads(raw)
    t = ticker.upper()
    for v in table.values():
        if v["ticker"].upper() == t:
            return str(v["cik_str"]).zfill(10), v["title"]
    return None


def recent_filings(ticker: str, forms=("10-K", "10-Q"), limit: int = 4) -> list[Filing]:
    """List a company's most recent filings of the given form types."""
    res = ticker_to_cik(ticker)
    if not res:
        return []
    cik, company = res
    sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    recent = sub["filings"]["recent"]
    out: list[Filing] = []
    for form, acc, date, doc in zip(
        recent["form"], recent["accessionNumber"],
        recent["filingDate"], recent["primaryDocument"]):
        if form not in forms:
            continue
        acc_nodash = acc.replace("-", "")
        cik_int = str(int(cik))  # leading zeros removed for the Archives path
        url = (f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
               f"{acc_nodash}/{doc}")
        out.append(Filing(ticker.upper(), company, cik, form, date, acc, doc, url))
        if len(out) >= limit:
            break
    return out


def fetch_document_text(url: str, max_chars: int = 400_000) -> str:
    """Fetch a filing document and strip HTML to readable text."""
    raw = _get(url).decode("utf-8", errors="ignore")
    # remove scripts/styles, then tags
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = re.sub(r"&#160;|&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()[:max_chars]


# common 10-K / 10-Q sections worth isolating for finance-ops Q&A
SECTION_PATTERNS = {
    "Risk Factors": r"item\s*1a\.?\s*risk factors",
    "MD&A": r"item\s*7\.?\s*management.s discussion",
    "Liquidity & Capital Resources": r"liquidity and capital resources",
    "Revenue": r"(net revenues?|revenue recognition|net sales)",
    "Controls & Procedures": r"item\s*9a\.?\s*controls and procedures",
    "Legal Proceedings": r"item\s*3\.?\s*legal proceedings",
}


if __name__ == "__main__":
    # quick manual test (run on an unrestricted network)
    import sys
    tkr = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    for f in recent_filings(tkr, limit=3):
        print(f.form, f.filing_date, f.url)
