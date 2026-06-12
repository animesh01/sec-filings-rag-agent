# LedgerIQ — Finance-Ops RAG Agent (SEC Filings + FP&A)

> Ask a finance question, get an answer **grounded in retrieved document passages** — across **two sources**:
> real **SEC EDGAR filings** and **FP&A planning documents** (variance commentary, forecast assumptions, close
> notes, policy). Every answer cites the exact document and section, and the agent **refuses** rather than guess
> when the corpus doesn't cover the question. Built to demonstrate an end-to-end **RAG pipeline**, **token
> minimization**, **grounding + refusal**, and an **MCP-server** retrieval tool.

A portfolio piece bridging finance operations and applied AI: it shows how retrieval-augmented generation brings
**auditability** to finance — where every number in a board deck, reforecast, or filing summary must trace back
to a source, and a confident wrong answer is worse than no answer.

---

## 🧭 Two sources, one pipeline

LedgerIQ runs two document sources side by side as top-level tabs, both powered by the **same retrieval core**:

- **SEC Filings** — grounded in real SEC EDGAR filings (10-K / 10-Q / 8-K). Public, authoritative, auditable.
- **FP&A** — grounded in finance-planning documents modeled on what a real FP&A team produces.

The shared core uses **section-aware TF-IDF indexing** (the section label is folded into the embedding, so a
question phrased in section vocabulary still matches a body that words it differently) and a **strict refusal
gate** (a hit is kept only on genuine term overlap or a strong standalone retrieval score, so a name-match alone
can't carry an irrelevant question through).

---

## 🧩 What it demonstrates

**End-to-end RAG pipeline** — ingest → chunk → embed → retrieve → compress → ground → cite → score. Each stage
is inspectable in the UI, for both sources.

**Token minimization** (a first-class feature) — three live levers (top-k, chunk size, context compression)
with a **tokens-per-query meter** that shows savings against the naive "stuff the whole corpus" baseline. Maps
directly to real production RAG cost and latency concerns.

**Grounding & refusal** — a groundedness score per answer, and a refusal gate so out-of-corpus questions get
"I won't guess" rather than a hallucination. Critical for finance, where auditability is the product.

**MCP servers** — each source is also exposed as a [Model Context Protocol](https://modelcontextprotocol.io)
server, so any MCP-capable agent can retrieve filings or FP&A documents as a tool — the agentic-RAG pattern.
See [`mcp_server/`](mcp_server/).

---

## 📊 The data — real where it can be, synthetic where it must be

**SEC source — real, public, defensible.** Built to run over **SEC EDGAR**, the official U.S. public-company
filings database. Free, no API key, filings from 1993 to present.
- Source: [sec.gov/search-filings](https://www.sec.gov/search-filings) · [EDGAR API docs](https://www.sec.gov/edgar/sec-api-documentation)
- Ships a sample corpus (representative, paraphrased filing text) for offline review.
- `python scripts/build_corpus.py --sec --live AAPL MSFT KO` fetches **genuine filings** from EDGAR.

**FP&A source — synthetic, rebuildable from public data.** Ships a fully synthetic FP&A corpus (placeholder unit
names; no proprietary content). `python scripts/build_corpus.py --fpa --csv budget.csv` converts a public
budget-vs-actuals CSV into variance-narrative documents the pipeline retrieves over.

---

## 🚀 Run locally

```bash
pip install -r requirements.txt

# build BOTH corpora from bundled samples (instant, offline)...
python scripts/build_corpus.py

# ...or rebuild a single source from real public data:
python scripts/build_corpus.py --sec --live AAPL MSFT KO   # real SEC filings
python scripts/build_corpus.py --fpa --csv budget.csv      # public budget CSV

streamlit run streamlit_app.py
```

Retrieval and answering run **locally with no API key** (TF-IDF embeddings + extractive grounded synthesis), so
it deploys free on Streamlit Cloud. Swapping in a real LLM for synthesis is a one-function change in
`scripts/rag_pipeline.py`.

---

## 🗂 Structure

```
streamlit_app.py            # the app — two source tabs (SEC | FP&A), each: Ask · Inspector · Pipeline · MCP · Data
scripts/
  rag_pipeline.py           # shared core: chunk · embed · retrieve · compress · ground · score
  edgar_fetcher.py          # real SEC EDGAR API client (SEC source)
  fpa_source.py             # FP&A document source (samples + public-CSV → docs)
  build_corpus.py           # build either/both corpora from samples or real public data
data/
  sample_corpus_sec.py      # bundled sample SEC filings
  sample_corpus_fpa.py      # bundled sample FP&A documents
  corpus_sec.json           # active SEC corpus the app reads
  corpus_fpa.json           # active FP&A corpus the app reads
mcp_server/
  server_sec.py             # SEC EDGAR fetcher as an MCP server (live tools)
  server_fpa.py             # FP&A document source as an MCP server
  README.md                 # MCP tools + registration
```

---

## ⚠️ Disclaimer

The SEC sample corpus uses representative, paraphrased filing text with placeholder company names; `--live`
fetches genuine public filings from SEC EDGAR. The FP&A corpus is **fully synthetic** — illustrative text with
placeholder unit names, containing no proprietary, confidential, or employer-specific information. Nothing here
is investment or financial advice.
