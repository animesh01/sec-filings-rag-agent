# FilingsIQ — Finance-Ops RAG Agent over SEC Filings

> Ask a finance question, get an answer **grounded in real SEC filing passages**, with citations to the exact filing and section — and an honest **refusal** when the answer isn't in the documents. Built to demonstrate an end-to-end **RAG pipeline**, **token minimization**, and an **MCP-server** retrieval tool.

A portfolio piece for AI product management: it shows the judgment behind retrieval-augmented generation in a regulated, audit-heavy domain — where a confident wrong answer is worse than no answer, and every claim must trace back to a source.

---

## 🎯 Why finance + RAG

Financial institutions are among the heaviest adopters of RAG precisely because **accuracy, auditability, and explainability are non-negotiable**. FilingsIQ leans into that: the citation isn't a nice-to-have, it's the product. Every answer points to the exact 10-K / 10-Q section it came from, and the agent refuses rather than hallucinate when the corpus doesn't cover the question.

---

## 📊 The data — real, public, defensible

Built to run over **SEC EDGAR** — the official U.S. public-company filings database (10-K annual, 10-Q quarterly, 8-K current reports). It's **free, requires no API key**, and covers filings from 1993 to present.

- **Source:** [sec.gov/search-filings](https://www.sec.gov/search-filings) · [EDGAR API docs](https://www.sec.gov/edgar/sec-api-documentation)
- The repo ships with a **sample corpus** (representative, paraphrased filing text) so the app runs offline for review.
- Running `python scripts/build_corpus.py --live AAPL MSFT KO` fetches **genuine filings** from EDGAR — so the pipeline operates on real public data end to end.

---

## 🧩 What it demonstrates

**End-to-end RAG pipeline** — ingest (EDGAR) → chunk → embed → retrieve → compress → ground → cite → score. Each stage is inspectable in the UI.

**Token minimization** (a first-class feature) — three live levers (top-k, chunk size, context compression) with a **tokens-per-query meter** that shows savings against the naive "stuff the whole corpus" baseline. Maps directly to real production RAG cost/latency concerns.

**Grounding & refusal** — a groundedness score per answer, and a refusal gate so out-of-corpus questions get "I won't guess" rather than a hallucination.

**MCP server** — the EDGAR fetcher is also exposed as a [Model Context Protocol](https://modelcontextprotocol.io) server, so any MCP-capable agent can retrieve filings as a tool — the agentic-RAG pattern. See [`mcp_server/`](mcp_server/).

---

## 🚀 Run locally

```bash
pip install -r requirements.txt

# build the corpus from bundled samples (instant, offline)...
python scripts/build_corpus.py

# ...or from REAL SEC filings (unrestricted network):
python scripts/build_corpus.py --live AAPL MSFT KO

streamlit run streamlit_app.py
```

Retrieval and answering run **locally with no API key** (TF-IDF embeddings + extractive grounded synthesis), so it deploys free on Streamlit Cloud. Swapping in a real LLM for synthesis is a one-function change in `scripts/rag_pipeline.py`.

---

## 🗂 Structure

```
streamlit_app.py                 # the app (Ask · Retrieval inspector · Pipeline · MCP · Data)
scripts/edgar_fetcher.py         # real SEC EDGAR API client (shared core)
scripts/rag_pipeline.py          # chunk · embed · retrieve · compress · ground · score
scripts/build_corpus.py          # build corpus from samples or live EDGAR
data/sample_corpus.py            # bundled sample filings
data/corpus.json                 # the active corpus the app reads
mcp_server/server.py             # EDGAR fetcher exposed as an MCP server
mcp_server/README.md             # MCP tools + registration
```

---

## ⚠️ Disclaimer

The bundled sample corpus uses representative, paraphrased filing text with **placeholder company names** so the app is self-contained. It is not copied from any specific company's filing. When run with `--live`, the app fetches genuine public filings from SEC EDGAR. Nothing here is investment advice, and the app contains no proprietary or confidential data.
