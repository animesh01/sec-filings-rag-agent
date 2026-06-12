"""Sample FP&A document corpus for LedgerIQ.

Realistic, representative FP&A documents modeled on the artifacts a finance
planning & analysis team actually produces and queries: budget-vs-actual variance
commentary, forecast assumptions, month-end close notes, and finance policies.

All content is synthetic and illustrative. Company/department names are
placeholders so the corpus is self-contained and non-attributive. Figures are
fabricated for demonstration.

This is the offline corpus; a finance team would point the same pipeline at its
own planning docs, board decks, and variance reports.
"""

SAMPLE_DOCS = [
    {
        "company": "Northwind BU",
        "ticker": "NW",
        "form": "Variance Commentary",
        "filing_date": "2025-Q3",
        "section": "Marketing OpEx Variance",
        "url": "internal://fpa/northwind/q3-variance#marketing",
        "text": (
            "Marketing operating expense came in at $4.6M for Q3, $0.5M (12%) over the "
            "budgeted $4.1M. The overspend was driven primarily by an unplanned brand "
            "campaign pulled forward from Q4 ($0.3M) and higher-than-expected digital "
            "media costs as CPMs rose across paid social ($0.2M). Headcount costs were "
            "on plan. Finance recommends reallocating the pulled-forward campaign spend "
            "in the Q4 reforecast and monitoring CPM trends, which are expected to "
            "normalize by early Q1. Year-to-date, Marketing remains 3% over budget."
        ),
    },
    {
        "company": "Northwind BU",
        "ticker": "NW",
        "form": "Variance Commentary",
        "filing_date": "2025-Q3",
        "section": "Revenue Variance",
        "url": "internal://fpa/northwind/q3-variance#revenue",
        "text": (
            "Net revenue for Q3 was $48.3M, favorable to the $46.5M budget by $1.8M "
            "(3.9%). The beat was driven by stronger subscription renewals (net revenue "
            "retention of 112% versus 108% planned) and faster-than-expected ramp in the "
            "enterprise segment. Transactional revenue was slightly below plan due to "
            "softer volumes in the SMB cohort. Finance has raised the full-year revenue "
            "forecast by $3.0M to reflect sustained renewal strength, while holding the "
            "SMB assumption flat pending two more months of data."
        ),
    },
    {
        "company": "Atlas BU",
        "ticker": "AT",
        "form": "Forecast Assumptions",
        "filing_date": "2025-Q4",
        "section": "Q4 Reforecast Assumptions",
        "url": "internal://fpa/atlas/q4-reforecast#assumptions",
        "text": (
            "The Q4 reforecast assumes flat sequential headcount (no new hires beyond "
            "backfills), a 2% sequential increase in cloud infrastructure cost tied to "
            "usage growth, and seasonal revenue uplift of 8% over Q3 consistent with "
            "prior-year holiday patterns. Foreign-exchange is held at the current quarter "
            "average. The reforecast does not include the proposed pricing change, which "
            "is modeled separately as upside scenario B. Key risks to the forecast are "
            "slower enterprise deal closure and higher-than-modeled cloud egress costs."
        ),
    },
    {
        "company": "Atlas BU",
        "ticker": "AT",
        "form": "Variance Commentary",
        "filing_date": "2025-Q3",
        "section": "Cloud Infrastructure Cost Variance",
        "url": "internal://fpa/atlas/q3-variance#infra",
        "text": (
            "Cloud and infrastructure cost was $3.2M in Q3, $0.4M (14%) above the $2.8M "
            "budget. The overage was driven by higher data-egress charges from a "
            "migration project and underestimated storage growth. Roughly half of the "
            "variance is one-time (migration) and is not expected to recur. Finance is "
            "working with Engineering to implement egress optimization and reserved-"
            "capacity commitments, projected to reduce run-rate infrastructure cost by "
            "approximately 9% beginning in Q1."
        ),
    },
    {
        "company": "Veridian BU",
        "ticker": "VR",
        "form": "Month-End Close",
        "filing_date": "2025-09",
        "section": "Close Notes & Accruals",
        "url": "internal://fpa/veridian/sep-close#notes",
        "text": (
            "September close completed on business day 4. Key accruals: $0.6M for "
            "professional services delivered but not yet invoiced, and $0.2M for "
            "year-end bonus true-up. A $0.15M prepaid software expense was reclassified "
            "to the correct cost center. The close included a manual revenue adjustment "
            "of $0.1M to defer subscription revenue billed but not yet earned, consistent "
            "with the revenue recognition policy. No material control exceptions were noted."
        ),
    },
    {
        "company": "Finance (Company-wide)",
        "ticker": "CO",
        "form": "Policy",
        "filing_date": "2025",
        "section": "Revenue Recognition Policy",
        "url": "internal://fpa/policy/revenue-recognition",
        "text": (
            "Subscription revenue is recognized ratably over the contractual service "
            "period, beginning when the service is made available to the customer. "
            "Implementation and onboarding fees are recognized over the expected customer "
            "life. Usage-based revenue is recognized as the usage occurs. Revenue billed "
            "in advance of being earned is recorded as deferred revenue on the balance "
            "sheet. Any non-standard contract terms must be reviewed by the revenue team "
            "before booking. This policy aligns with ASC 606 five-step recognition."
        ),
    },
    {
        "company": "Finance (Company-wide)",
        "ticker": "CO",
        "form": "Policy",
        "filing_date": "2025",
        "section": "Budget & Reforecast Cadence",
        "url": "internal://fpa/policy/budget-cadence",
        "text": (
            "The annual operating plan (AOP) is built bottoms-up each Q4 and approved by "
            "the board in December. Forecasts are refreshed quarterly via a reforecast "
            "cycle; budget owners submit updated department forecasts within five business "
            "days of quarter close. Variances greater than 10% or $0.25M against budget "
            "require written commentary from the budget owner. Rolling 12-month forecasts "
            "are maintained for revenue, headcount, and operating expense."
        ),
    },
    {
        "company": "Finance (Company-wide)",
        "ticker": "CO",
        "form": "Headcount Plan",
        "filing_date": "2025-Q4",
        "section": "Headcount & Workforce Plan",
        "url": "internal://fpa/plan/headcount#q4",
        "text": (
            "Ending headcount for Q3 was 612 against a plan of 628, a favorable variance "
            "of 16 driven by slower backfill of attrition in Engineering and Sales. The "
            "Q4 plan holds net headcount flat, prioritizing backfills of revenue-"
            "generating roles over net-new hiring. Fully loaded cost per head is modeled "
            "at $165K annually. Personnel costs represent approximately 58% of total "
            "operating expense, so hiring pace is the single largest driver of OpEx "
            "variance and is reviewed in every reforecast."
        ),
    },
]
