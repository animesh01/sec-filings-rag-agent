"""FP&A document source for LedgerIQ.

In a real deployment this module connects to a finance team's document store
(planning docs, board decks, variance reports, close notes, policy pages) or a
data warehouse, and returns the text passages the RAG pipeline indexes.

For the offline demo it serves the bundled sample corpus. It also supports
loading a public, synthetic budget-vs-actuals dataset (CSV) and converting each
department's variance into a narrative document, so the corpus can be rebuilt
from genuinely public data.

No credentials required for the sample path. The structure mirrors a pluggable
connector so the same pipeline points at real FP&A sources in production.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass


@dataclass
class FpaDoc:
    company: str        # business unit / cost center
    ticker: str         # short code
    form: str           # doc type: Variance Commentary, Forecast Assumptions, Policy, ...
    filing_date: str    # period
    section: str
    url: str
    text: str


def load_sample_docs() -> list[dict]:
    """Return the bundled sample FP&A corpus."""
    from sample_corpus_fpa import SAMPLE_DOCS  # noqa
    return list(SAMPLE_DOCS)


def budget_csv_to_docs(csv_text: str) -> list[dict]:
    """Convert a public budget-vs-actuals CSV into FP&A variance narrative docs.

    Expected columns (case-insensitive): FiscalYear, Dept, Quarter, BudgetUSD,
    ForecastUSD, ActualUSD. Each row with a material variance becomes a short
    narrative document the RAG pipeline can retrieve and ground answers in.

    This lets the corpus be rebuilt from a genuinely public, synthetic dataset
    (e.g. the freely available department budget/forecast/actuals workbooks).
    """
    docs: list[dict] = []
    reader = csv.DictReader(io.StringIO(csv_text))
    # normalize header lookup
    def col(row, *names):
        for n in names:
            for k in row:
                if k and k.strip().lower() == n.lower():
                    return row[k]
        return None
    for row in reader:
        dept = col(row, "Dept", "Department")
        fy = col(row, "FiscalYear", "Year")
        q = col(row, "Quarter", "Qtr")
        try:
            budget = float(col(row, "BudgetUSD", "Budget") or 0)
            actual = float(col(row, "ActualUSD", "Actual") or 0)
        except ValueError:
            continue
        if not dept or budget == 0:
            continue
        var = actual - budget
        pct = var / budget * 100 if budget else 0
        direction = "over" if var > 0 else "under"
        text = (
            f"{dept} actuals for {fy} {q} were ${actual:,.0f} against a budget of "
            f"${budget:,.0f}, a variance of ${var:,.0f} ({abs(pct):.1f}% {direction} "
            f"budget). Variances beyond the 10% threshold require written commentary "
            f"from the budget owner per the reforecast cadence policy."
        )
        docs.append({
            "company": dept, "ticker": dept[:3].upper(),
            "form": "Variance (from data)", "filing_date": f"{fy} {q}",
            "section": f"{dept} Budget vs Actual", "url": "data://budget-vs-actuals",
            "text": text,
        })
    return docs


if __name__ == "__main__":
    docs = load_sample_docs()
    print(f"sample FP&A corpus: {len(docs)} documents")
    for d in docs[:3]:
        print(" -", d["company"], "|", d["section"])
