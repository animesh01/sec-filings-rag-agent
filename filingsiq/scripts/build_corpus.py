"""Build the FilingsIQ corpus.

Two modes:

    python scripts/build_corpus.py
        -> writes data/corpus.json from the bundled SAMPLE_DOCS (offline, instant)

    python scripts/build_corpus.py --live AAPL MSFT KO
        -> fetches REAL filings from SEC EDGAR (via edgar_fetcher), splits them into
           the standard sections, and writes data/corpus.json from genuine documents

The app reads data/corpus.json. Shipping the sample corpus means the app runs
out of the box; running --live on your own machine (unrestricted network) swaps in
real SEC filings so you can truthfully say it was built over real public data.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data" / "corpus.json"


def from_samples() -> list[dict]:
    from sample_corpus import SAMPLE_DOCS
    return list(SAMPLE_DOCS)


def from_live(tickers: list[str], per_company: int = 1) -> list[dict]:
    """Fetch real filings and split each into the standard sections."""
    import edgar_fetcher as ef
    docs: list[dict] = []
    for tkr in tickers:
        filings = ef.recent_filings(tkr, forms=("10-K", "10-Q"), limit=per_company)
        for f in filings:
            print(f"  fetching {f.ticker} {f.form} {f.filing_date} ...")
            text = ef.fetch_document_text(f.url)
            low = text.lower()
            # find each section's start, then slice up to the next section start
            marks = []
            for label, pat in ef.SECTION_PATTERNS.items():
                m = re.search(pat, low)
                if m:
                    marks.append((m.start(), label))
            marks.sort()
            for i, (start, label) in enumerate(marks):
                end = marks[i + 1][0] if i + 1 < len(marks) else min(len(text), start + 6000)
                section_text = text[start:end].strip()
                if len(section_text) < 200:
                    continue
                docs.append({
                    "company": f.company, "ticker": f.ticker, "form": f.form,
                    "filing_date": f.filing_date, "section": label,
                    "url": f.url, "text": section_text[:6000],
                })
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", nargs="*", metavar="TICKER",
                    help="fetch real filings for these tickers instead of using samples")
    ap.add_argument("--per-company", type=int, default=1)
    args = ap.parse_args()

    if args.live:
        print(f"Building corpus from LIVE SEC EDGAR filings: {args.live}")
        docs = from_live([t.upper() for t in args.live], per_company=args.per_company)
        source = "SEC EDGAR (live)"
    else:
        print("Building corpus from bundled sample filings.")
        docs = from_samples()
        source = "bundled samples (run --live to use real EDGAR filings)"

    payload = {
        "source": source,
        "n_docs": len(docs),
        "companies": sorted({d["company"] for d in docs}),
        "docs": docs,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"wrote {OUT} — {len(docs)} section-docs from {len(payload['companies'])} companies")


if __name__ == "__main__":
    main()
