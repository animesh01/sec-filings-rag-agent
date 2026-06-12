"""Build the LedgerIQ corpora — SEC filings and FP&A documents.

LedgerIQ runs two sources side by side, each with its own corpus file:

    data/corpus_sec.json   — SEC EDGAR filing sections
    data/corpus_fpa.json   — FP&A planning documents

Usage:

    python scripts/build_corpus.py
        -> rebuilds BOTH corpora from their bundled samples (offline, instant)

    python scripts/build_corpus.py --sec --live AAPL MSFT KO
        -> fetches REAL filings from SEC EDGAR and writes data/corpus_sec.json

    python scripts/build_corpus.py --fpa --csv budget.csv
        -> converts a public budget-vs-actuals CSV and writes data/corpus_fpa.json

The app reads both corpus files. Shipping the samples means it runs out of the
box; --live (SEC) and --csv (FP&A) swap in genuine public data on your machine.
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

SEC_OUT = ROOT / "data" / "corpus_sec.json"
FPA_OUT = ROOT / "data" / "corpus_fpa.json"


# ---------- SEC ----------
def sec_from_samples() -> list[dict]:
    from sample_corpus_sec import SAMPLE_DOCS
    return list(SAMPLE_DOCS)


def sec_from_live(tickers: list[str], per_company: int = 1) -> list[dict]:
    """Fetch real filings and split each into the standard sections."""
    import edgar_fetcher as ef
    docs: list[dict] = []
    for tkr in tickers:
        filings = ef.recent_filings(tkr, forms=("10-K", "10-Q"), limit=per_company)
        for f in filings:
            print(f"  fetching {f.ticker} {f.form} {f.filing_date} ...")
            text = ef.fetch_document_text(f.url)
            low = text.lower()
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


# ---------- FP&A ----------
def fpa_from_samples() -> list[dict]:
    import fpa_source as src
    return src.load_sample_docs()


def fpa_from_csv(csv_path: str) -> list[dict]:
    import fpa_source as src
    csv_text = Path(csv_path).read_text()
    return src.budget_csv_to_docs(csv_text)


# ---------- shared ----------
def write_corpus(out: Path, docs: list[dict], source: str) -> None:
    payload = {
        "source": source,
        "n_docs": len(docs),
        "companies": sorted({d["company"] for d in docs}),
        "docs": docs,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out} — {len(docs)} docs from {len(payload['companies'])} units")


def main():
    ap = argparse.ArgumentParser(description="Build LedgerIQ corpora (SEC + FP&A).")
    ap.add_argument("--sec", action="store_true", help="build only the SEC corpus")
    ap.add_argument("--fpa", action="store_true", help="build only the FP&A corpus")
    ap.add_argument("--live", nargs="*", metavar="TICKER",
                    help="SEC: fetch real filings for these tickers instead of samples")
    ap.add_argument("--csv", metavar="PATH",
                    help="FP&A: public budget-vs-actuals CSV to convert into docs")
    ap.add_argument("--per-company", type=int, default=1)
    args = ap.parse_args()

    # default: build both from samples
    do_sec = args.sec or not (args.sec or args.fpa)
    do_fpa = args.fpa or not (args.sec or args.fpa)

    if do_sec:
        if args.live:
            print(f"Building SEC corpus from LIVE EDGAR filings: {args.live}")
            docs = sec_from_live([t.upper() for t in args.live], per_company=args.per_company)
            source = "SEC EDGAR (live)"
        else:
            print("Building SEC corpus from bundled sample filings.")
            docs = sec_from_samples()
            source = "bundled samples (run --sec --live to use real EDGAR filings)"
        write_corpus(SEC_OUT, docs, source)

    if do_fpa:
        if args.csv:
            print(f"Building FP&A corpus from public budget CSV: {args.csv}")
            docs = fpa_from_csv(args.csv)
            source = f"public budget-vs-actuals dataset ({Path(args.csv).name})"
        else:
            print("Building FP&A corpus from bundled sample documents.")
            docs = fpa_from_samples()
            source = "bundled samples (run --fpa --csv FILE.csv to use a public dataset)"
        write_corpus(FPA_OUT, docs, source)


if __name__ == "__main__":
    main()
