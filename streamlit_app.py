"""FilingsIQ - a finance-ops RAG agent over real SEC filings.

Ask a finance question; get an answer grounded in retrieved filing passages, with
citations to the exact filing and section - and a refusal when the answer isn't in
the corpus. Built to demonstrate an end-to-end RAG pipeline, token minimization,
and an MCP-server retrieval tool.

Data: SEC EDGAR (public, no API key). Ships with sample filings; run
scripts/build_corpus.py --live TICKER... to populate real documents.
Retrieval + answering run locally (no API key, free deploy).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from rag_pipeline import (chunk_corpus, RagIndex, answer_question, estimate_tokens)
except ModuleNotFoundError as e:
    import streamlit as _st
    _st.error(
        "Could not import the RAG pipeline. This usually means the `scripts/` folder "
        "didn't get pushed to the repo. The repo root must contain `streamlit_app.py`, "
        "`scripts/`, and `data/` side by side. Missing module: " + str(e))
    _st.stop()

CORPUS = ROOT / "data" / "corpus.json"

# ---- palette: Option C light system, EMERALD/TEAL accent (finance) ----
INK = "#10241f"
MUTED = "#5f7268"
TEAL = "#0e8f6f"
TEAL_DK = "#0a6f57"
GOLD = "#c08a2d"
RED = "#d65745"
GREEN = "#1aa06a"
DEEP = "#0c2a23"
CARD = "#ffffff"
LINE = "#dfeae5"
BG = "#e8efea"
MINT = "#e3f3ec"

st.set_page_config(page_title="FilingsIQ - SEC filings RAG agent",
                   page_icon="search", layout="wide")


def esc(s) -> str:
    return html.escape(str(s))


@st.cache_data
def load_corpus() -> dict:
    return json.loads(CORPUS.read_text())


@st.cache_resource
def build_index(chunk_size: int, overlap: int):
    data = load_corpus()
    chunks = chunk_corpus(data["docs"], chunk_size=chunk_size, overlap=overlap)
    return RagIndex(chunks), data, chunks


def inject_styles() -> None:
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
        .stApp {{ background:
            radial-gradient(1200px 500px at 18% -8%, #f0f7f3 0%, rgba(240,247,243,0) 60%),
            linear-gradient(180deg, #e8efea 0%, #e0e9e4 100%); }}
        html, body, [class*="css"] {{ font-family:"DM Sans",sans-serif; color:{INK}; }}
        h1,h2,h3,h4 {{ font-family:"Manrope",sans-serif !important; letter-spacing:-0.025em; color:{INK} !important; }}
        .stCaption, [data-testid="stCaptionContainer"] {{ color:{MUTED} !important; }}
        .hero {{ position:relative; overflow:hidden; border-radius:24px; padding:32px 36px; margin-bottom:18px; color:#fff;
            background:linear-gradient(135deg,#0a6f57 0%,#0e8f6f 52%,#3fb98e 100%);
            box-shadow:0 2px 2px rgba(10,50,40,.10), 0 16px 32px -10px rgba(14,120,90,.4), 0 44px 70px -24px rgba(14,120,90,.42); }}
        .hero::before {{ content:""; position:absolute; inset:0 0 auto 0; height:1px; border-radius:24px 24px 0 0;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,.5),transparent); }}
        .hero h1, .hero h1 * {{ color:#fff !important; -webkit-text-fill-color:#fff !important; font-size:2.0rem; margin:8px 0 8px; }}
        .hero p {{ color:#e7f6ef !important; font-size:1.0rem; line-height:1.5; max-width:78%; margin:0; }}
        .hero p b {{ color:#fff !important; }}
        .hero .pill {{ display:inline-block; padding:6px 14px; border-radius:999px;
            background:rgba(255,255,255,0.22); color:#fff !important; font-size:0.72rem; font-weight:700;
            letter-spacing:0.06em; text-transform:uppercase; box-shadow:inset 0 1px 0 rgba(255,255,255,.3); }}
        .hero-art {{ position:absolute; right:28px; top:22px; opacity:0.96; filter:drop-shadow(0 12px 18px rgba(10,50,40,.3)); }}
        .section-title {{ font-family:Manrope; font-weight:800; font-size:1.26rem; margin:12px 0 2px; letter-spacing:-0.015em; }}
        .section-copy {{ color:{MUTED}; font-size:0.96rem; margin-bottom:14px; max-width:840px; line-height:1.5; }}
        .panel {{ position:relative; background:linear-gradient(180deg,#fff,#fafdfb); border:1px solid {LINE};
            border-radius:20px; padding:22px 24px; margin-top:14px;
            box-shadow:0 1px 1px rgba(10,50,40,.04), 0 6px 12px -3px rgba(10,50,40,.10), 0 20px 34px -12px rgba(10,50,40,.16); }}
        .panel::before {{ content:""; position:absolute; inset:0 0 auto 0; height:1px; border-radius:20px 20px 0 0;
            background:linear-gradient(90deg,transparent,rgba(255,255,255,.9),transparent); }}
        .answer {{ font-size:1.05rem; line-height:1.6; color:{INK}; }}
        .answer sup, .answer .cref {{ color:{TEAL_DK}; font-weight:700; }}
        .ground-chip {{ display:inline-block; font-weight:800; font-size:0.8rem; padding:5px 13px; border-radius:10px; box-shadow:inset 0 1px 0 #fff; }}
        .cite {{ border:1px solid {LINE}; border-left:3px solid {TEAL}; border-radius:12px; padding:12px 15px; margin:8px 0;
            background:#fff; box-shadow:0 4px 10px -8px rgba(10,50,40,.3); }}
        .cite .meta {{ font-size:0.78rem; color:{MUTED}; margin-bottom:4px; }}
        .cite .meta b {{ color:{TEAL_DK}; font-family:Manrope; }}
        .cite .snip {{ font-size:0.9rem; color:{INK}; line-height:1.45; }}
        .cite a {{ color:{TEAL_DK}; font-size:0.78rem; text-decoration:none; font-weight:600; }}
        .refuse {{ border:1px solid #f1d9c9; border-left:3px solid {GOLD}; background:#fdf6ee; border-radius:12px;
            padding:14px 18px; color:#7a5a1e; font-size:0.98rem; line-height:1.5; }}
        .kpi-row {{ display:flex; gap:13px; flex-wrap:wrap; margin-top:4px; }}
        .kpi {{ flex:1; min-width:150px; position:relative; background:linear-gradient(180deg,#fff,#fafdfb); border:1px solid {LINE};
            border-radius:16px; padding:14px 17px;
            box-shadow:0 1px 1px rgba(10,50,40,.04), 0 6px 12px -3px rgba(10,50,40,.10); }}
        .kpi .lbl {{ font-size:0.72rem; font-weight:700; color:{MUTED}; text-transform:uppercase; letter-spacing:0.04em; }}
        .kpi .val {{ font-family:Manrope; font-weight:800; font-size:1.5rem; margin:6px 0 0; letter-spacing:-0.02em; }}
        .kpi .sub {{ font-size:0.74rem; color:#9aa8a0; margin-top:2px; }}
        .takeaway {{ background:linear-gradient(180deg,#fff,#fafdfb); border:1px solid {LINE}; border-left:3px solid {TEAL};
            border-radius:14px; padding:14px 18px; font-size:0.95rem; line-height:1.5; margin:12px 0 4px; color:#4c5a52;
            box-shadow:0 1px 1px rgba(10,50,40,.04), 0 6px 12px -3px rgba(10,50,40,.10); }}
        .takeaway b {{ font-family:Manrope; color:{INK}; }}
        .stepcard {{ background:linear-gradient(180deg,#fff,#fafdfb); border:1px solid {LINE}; border-radius:16px;
            padding:15px 17px; box-shadow:0 1px 1px rgba(10,50,40,.04), 0 6px 12px -3px rgba(10,50,40,.10); }}
        .stepcard .n {{ font-size:0.72rem; font-weight:800; color:{TEAL}; text-transform:uppercase; letter-spacing:0.05em; }}
        .stepcard .t {{ font-weight:800; font-family:Manrope; margin:4px 0 6px; }}
        .stepcard .d {{ color:{MUTED}; font-size:0.84rem; line-height:1.4; }}
        .stTabs [data-baseweb="tab-list"] {{ gap:9px; border-bottom:none !important; padding:4px 0 6px; flex-wrap:wrap; }}
        .stTabs [data-baseweb="tab"] {{ background:#fff !important; color:{MUTED} !important; border:1px solid {LINE} !important;
            border-radius:999px !important; padding:8px 17px !important; height:auto !important;
            box-shadow:0 4px 10px -6px rgba(10,50,40,.25), inset 0 1px 0 #fff !important;
            transition:transform .25s cubic-bezier(.2,.7,.2,1), box-shadow .25s, background .25s !important; }}
        .stTabs [data-baseweb="tab"] * {{ color:{MUTED} !important; }}
        .stTabs [data-baseweb="tab"]:hover {{ transform:translateY(-2px) !important; box-shadow:0 8px 16px -8px rgba(10,50,40,.3) !important; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] {{ background:{MINT} !important; border-color:#bfe6d5 !important; transform:translateY(-3px) !important;
            box-shadow:0 10px 20px -8px rgba(14,143,111,.4), inset 0 1px 0 #fff !important; }}
        .stTabs [data-baseweb="tab"][aria-selected="true"] * {{ color:{TEAL_DK} !important; font-weight:700 !important; }}
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none !important; background:transparent !important; }}
        [data-testid="stExpander"] {{ border:1px solid {LINE} !important; border-radius:14px; background:#fff; box-shadow:0 6px 14px -10px rgba(10,50,40,.3); overflow:hidden; }}
        [data-testid="stExpander"] summary {{ background:{MINT} !important; border-bottom:1px solid #cfe9dd !important; }}
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color:{INK} !important; -webkit-text-fill-color:{INK} !important; font-weight:700 !important; }}
        [data-testid="stExpander"] summary:hover * {{ color:{TEAL_DK} !important; -webkit-text-fill-color:{TEAL_DK} !important; }}
        [data-testid="stExpander"] p, [data-testid="stExpander"] li {{ color:{INK} !important; }}
        .stButton button[kind="primary"] {{ background:linear-gradient(180deg,{TEAL},{TEAL_DK}) !important; color:#fff !important;
            border:none !important; font-weight:700 !important; border-radius:12px !important;
            box-shadow:0 8px 18px -6px rgba(14,143,111,.5), inset 0 1px 0 rgba(255,255,255,.2) !important; }}
        .stButton button[kind="primary"]:hover {{ filter:brightness(1.07); transform:translateY(-1px); }}
        [data-testid="stTextInput"] input {{ border-radius:12px !important; border-color:{LINE} !important; }}
        [data-baseweb="slider"] [role="slider"] {{ background:{TEAL} !important; }}
        [data-testid="stSidebar"] {{ background:#f3f8f5 !important; }}
        </style>
    """, unsafe_allow_html=True)


inject_styles()

# ---- sidebar: retrieval + token-minimization controls ----
st.sidebar.markdown("### Retrieval settings")
st.sidebar.caption("Tune the pipeline and watch the token cost change.")
top_k = st.sidebar.slider("Top-k passages", 1, 6, 3,
                          help="How many filing passages to retrieve and ground on.")
chunk_size = st.sidebar.select_slider("Chunk size (words)", options=[80, 100, 120, 160, 200], value=120,
                                      help="Smaller chunks = tighter, cheaper context; larger = more coverage.")
token_budget = st.sidebar.slider("Context token budget", 100, 1200, 500, step=50,
                                 help="Hard cap on retrieved context tokens - the core cost lever.")
compress = st.sidebar.toggle("Context compression", value=True,
                             help="Sentence-level pruning: keep only query-relevant sentences. Cuts tokens.")
st.sidebar.markdown("---")
st.sidebar.caption("Retrieval & answering run locally - no API key. "
                   "Swap in a real LLM by editing rag_pipeline.answer_question.")

index, corpus, chunks = build_index(chunk_size, 25)

# ---- hero ----
art = (
    "<svg class='hero-art' width='148' height='104' viewBox='0 0 148 104' fill='none'>"
    "<circle cx='116' cy='38' r='56' fill='rgba(255,255,255,0.06)'/>"
    # filing (the hero element)
    "<rect x='20' y='16' width='62' height='78' rx='9' fill='#fff'/>"
    "<rect x='30' y='30' width='36' height='5' rx='2.5' fill='#0a6f57'/>"
    "<rect x='30' y='42' width='42' height='4' rx='2' fill='#bfe0d3'/>"
    "<rect x='30' y='52' width='42' height='4' rx='2' fill='#bfe0d3'/>"
    "<rect x='30' y='62' width='28' height='4' rx='2' fill='#bfe0d3'/>"
    # dollar coin emblem
    "<circle cx='104' cy='44' r='22' fill='#fff'/>"
    "<circle cx='104' cy='44' r='22' fill='none' stroke='#bfe0d3' stroke-width='2'/>"
    "<text x='104' y='53' text-anchor='middle' font-family='Manrope,sans-serif' "
    "font-weight='800' font-size='26' fill='#0a6f57'>$</text>"
    # citation check
    "<circle cx='120' cy='74' r='11' fill='#1aa06a' stroke='#fff' stroke-width='2.4'/>"
    "<path d='M115 74 l3.5 3.5 l6 -7' stroke='#fff' stroke-width='2.2' fill='none' "
    "stroke-linecap='round' stroke-linejoin='round'/></svg>"
)
companies = ", ".join(corpus["companies"])
st.markdown(f"""
    <div class="hero">{art}
      <span class="pill">Finance-ops RAG agent . grounded in SEC filings</span>
      <h1>FilingsIQ</h1>
      <p>Ask a finance question and get an answer <b>grounded in real SEC filing passages</b>,
      with citations to the exact filing and section - and an honest refusal when the answer
      isn't in the documents. <b>Auditable by design.</b></p>
    </div>
""", unsafe_allow_html=True)

tabs = st.tabs(["Ask", "Retrieval inspector", "Pipeline", "MCP server", "About the data"])

# ============================================================ ASK
with tabs[0]:
    st.markdown('<div class="section-title">Ask the filings</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-copy">Corpus: {corpus["n_docs"]} filing sections from '
                f'{len(corpus["companies"])} companies ({esc(companies)}). '
                f'Source: {esc(corpus["source"])}.</div>', unsafe_allow_html=True)

    examples = [
        "What drove Northwind's revenue growth?",
        "How much liquidity and credit capacity does Northwind have?",
        "What are Northwind's key risk factors?",
        "How does Veridian recognize revenue?",
        "What were Atlas's operating margin trends?",
    ]
    ex = st.selectbox("Try an example, or type your own below:", [""] + examples)
    q = st.text_input("Your question", value=ex, placeholder="e.g. What did the company say about liquidity?")
    go = st.button("Ask FilingsIQ", type="primary")

    if go and q.strip():
        ans = answer_question(q, index, top_k=top_k, token_budget=token_budget, compress=compress)

        # token meter
        q_tokens = estimate_tokens(q)
        naive_tokens = sum(c.tokens for c in chunks)  # if you dumped the whole corpus
        saved = max(0, naive_tokens - ans.context_tokens)
        pct = (saved / naive_tokens * 100) if naive_tokens else 0

        if ans.refused:
            st.markdown(f'<div class="refuse">{esc(ans.text)}</div>', unsafe_allow_html=True)
        else:
            gcol = GREEN if ans.groundedness >= 0.8 else (GOLD if ans.groundedness >= 0.5 else RED)
            gbg = "#e1f5ee" if ans.groundedness >= 0.8 else ("#fbf1dc" if ans.groundedness >= 0.5 else "#fbe9e6")
            # render answer with citation superscripts
            import re as _re
            disp = _re.sub(r"\[(\d+)\]", r"<sup class='cref'>[\1]</sup>", esc(ans.text))
            st.markdown(f'<div class="panel"><div class="answer">{disp}</div>'
                        f'<div style="margin-top:12px"><span class="ground-chip" style="background:{gbg};color:{gcol}">'
                        f'Groundedness {ans.groundedness*100:.0f}%</span></div></div>', unsafe_allow_html=True)

            # token meter KPIs
            st.markdown(f"""<div class="kpi-row" style="margin-top:12px">
                <div class="kpi"><div class="lbl">Context tokens used</div><div class="val" style="color:{TEAL_DK}">{ans.context_tokens:,}</div><div class="sub">retrieved + compressed</div></div>
                <div class="kpi"><div class="lbl">vs. dumping whole corpus</div><div class="val" style="color:{GREEN}">-{pct:.0f}%</div><div class="sub">{naive_tokens:,} -> {ans.context_tokens:,} tokens</div></div>
                <div class="kpi"><div class="lbl">Passages retrieved</div><div class="val">{len(ans.citations)}</div><div class="sub">top-{top_k}, budget {token_budget}</div></div>
                <div class="kpi"><div class="lbl">Compression</div><div class="val">{'ON' if compress else 'OFF'}</div><div class="sub">sentence pruning</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-title" style="margin-top:18px">Citations</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-copy">Every answer sentence is numbered to the passage it came from. '
                        'In finance, the citation is the point - an answer you can audit back to the filing.</div>',
                        unsafe_allow_html=True)
            for c in ans.citations:
                st.markdown(f"""<div class="cite">
                    <div class="meta"><b>[{c['n']}] {esc(c['company'])}</b> . {esc(c['form'])} . {esc(c['filing_date'])}
                    . {esc(c['section'])} . match {c['score']:.2f}</div>
                    <div class="snip">{esc(c['snippet'])}</div>
                    <a href="{esc(c['url'])}" target="_blank">View source filing -&gt;</a>
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="takeaway" style="margin-top:18px">Try a question outside the filings (e.g. '
                '<i>"what is the CEO\'s favorite color?"</i>) - FilingsIQ <b>refuses rather than guessing</b>. '
                'In a finance setting, a confident wrong answer is worse than no answer.</div>',
                unsafe_allow_html=True)

# ============================================================ RETRIEVAL INSPECTOR
with tabs[1]:
    st.markdown('<div class="section-title">What did retrieval actually pull?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">RAG quality lives or dies on retrieval. This shows the ranked passages '
                'for a query, their similarity scores, and the token cost of each - so you can see the '
                'token-vs-coverage tradeoff directly.</div>', unsafe_allow_html=True)
    iq = st.text_input("Inspect retrieval for:", value="liquidity and credit capacity", key="inspect")
    if iq.strip():
        hits = index.search(iq, top_k=top_k, token_budget=token_budget)
        if not hits:
            st.markdown('<div class="refuse">No passages matched.</div>', unsafe_allow_html=True)
        else:
            cum = 0
            for rank, (ch, score) in enumerate(hits, 1):
                cum += ch.tokens
                bar = int(score / hits[0][1] * 100) if hits[0][1] else 0
                st.markdown(f"""<div class="cite">
                    <div class="meta"><b>#{rank} {esc(ch.company)}</b> . {esc(ch.form)} . {esc(ch.section)}
                    . score {score:.3f} . {ch.tokens} tokens (cumulative {cum})</div>
                    <div class="hbar" style="height:6px;border-radius:99px;background:#e3efe9;margin:6px 0">
                      <span style="display:block;height:100%;border-radius:99px;width:{bar}%;background:{TEAL}"></span></div>
                    <div class="snip">{esc(ch.text[:260])}...</div>
                </div>""", unsafe_allow_html=True)
            st.markdown(f'<div class="takeaway">Total retrieved context: <b>{cum} tokens</b> '
                        f'(budget {token_budget}). Lowering top-k or chunk size, or enabling compression, '
                        f'cuts this further - the live levers are in the sidebar.</div>', unsafe_allow_html=True)

# ============================================================ PIPELINE
with tabs[2]:
    st.markdown('<div class="section-title">The RAG pipeline, end to end</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Six stages from raw filing to grounded, cited answer. '
                'Token minimization is built into stages 3-4.</div>', unsafe_allow_html=True)
    steps = [
        ("1 . Ingest", "SEC EDGAR", "Fetch real 10-K / 10-Q filings from the official EDGAR API (no key). Bundled samples for instant deploy."),
        ("2 . Chunk", "Section passages", "Split filings into overlapping word-windows, tagged with company, form, and section."),
        ("3 . Embed + Retrieve", "TF-IDF + cosine", "Vectorize locally; cosine-similarity top-k with a company/ticker boost - capped by a token budget."),
        ("4 . Compress", "Sentence pruning", "Keep only query-relevant sentences in each passage - a model-free way to cut context tokens."),
        ("5 . Ground", "Cite every claim", "Stitch the answer from retrieved sentences, each numbered to its source passage."),
        ("6 . Score + Gate", "Groundedness + refusal", "Score how well the answer is supported; refuse when nothing relevant is retrieved."),
    ]
    cols = st.columns(3)
    for i, (n, t, d) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(f'<div class="stepcard" style="margin-bottom:12px"><div class="n">{n}</div>'
                        f'<div class="t">{esc(t)}</div><div class="d">{d}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:14px">Token minimization - why it matters</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">In production RAG, context tokens are the dominant cost and latency driver. '
                'FilingsIQ exposes three levers - top-k, chunk size, and context compression - and meters the savings '
                'against the naive "stuff the whole corpus" baseline on every answer.</div>', unsafe_allow_html=True)
    with st.expander("Reproduce the pipeline locally"):
        st.markdown(
            "```bash\n"
            "pip install -r requirements.txt\n\n"
            "# build the corpus from bundled samples...\n"
            "python scripts/build_corpus.py\n\n"
            "# ...or from REAL SEC filings (run on an unrestricted network):\n"
            "python scripts/build_corpus.py --live AAPL MSFT KO\n\n"
            "streamlit run streamlit_app.py\n"
            "```")

# ============================================================ MCP SERVER
with tabs[3]:
    st.markdown('<div class="section-title">EDGAR retrieval as an MCP server</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">The filing-fetcher is also exposed as a <b>Model Context Protocol (MCP) '
                'server</b>, so an agent (Claude or any MCP client) can retrieve filings on demand as a tool - the '
                'agentic-RAG pattern, where retrieval is a tool the model calls rather than hard-wired code.</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (name, desc) in zip([c1, c2, c3], [
        ("list_filings", "List a company's recent 10-K / 10-Q / 8-K filings with direct document URLs."),
        ("fetch_filing", "Fetch a filing by URL and return cleaned, readable text."),
        ("find_sections", "Locate standard sections (Risk Factors, MD&A, Liquidity, Revenue, Controls)."),
    ]):
        with col:
            st.markdown(f'<div class="stepcard"><div class="n">MCP tool</div><div class="t">{name}</div>'
                        f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title" style="margin-top:16px">Run &amp; register</div>', unsafe_allow_html=True)
    st.code("# start the MCP server (stdio transport)\npython mcp_server/server.py", language="bash")
    st.markdown('<div class="section-copy" style="margin-top:8px">Register it with an MCP client by pointing the '
                'client at that command. Example for Claude Desktop\'s config:</div>', unsafe_allow_html=True)
    st.code(json.dumps({
        "mcpServers": {
            "filingsiq-edgar": {
                "command": "python",
                "args": ["/absolute/path/to/mcp_server/server.py"]
            }
        }
    }, indent=2), language="json")
    st.markdown('<div class="takeaway">Why this matters for an agentic product: exposing retrieval as an MCP tool '
                'means <b>any</b> MCP-capable agent can ground itself in SEC filings without bespoke integration - '
                'the same capability powers this app and an external agent.</div>', unsafe_allow_html=True)

# ============================================================ ABOUT THE DATA
with tabs[4]:
    st.markdown('<div class="section-title">About the data</div>', unsafe_allow_html=True)
    st.markdown("""<div class="section-copy">
        FilingsIQ is designed to run over <b>real SEC EDGAR filings</b> - the official, public U.S. corporate
        filings database (10-K annual, 10-Q quarterly, 8-K current reports), free and without an API key.
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="panel">
        <b>Source:</b> SEC EDGAR - <a href="https://www.sec.gov/search-filings" target="_blank">sec.gov/search-filings</a>
        &nbsp;|&nbsp; API docs: <a href="https://www.sec.gov/edgar/sec-api-documentation" target="_blank">EDGAR API</a><br><br>
        <b>This deployment's corpus:</b> {esc(corpus['source'])} - {corpus['n_docs']} filing sections
        from {len(corpus['companies'])} companies.<br><br>
        The bundled sample corpus uses representative, paraphrased filing text with placeholder company names so the
        app runs offline for review. Running <code>scripts/build_corpus.py --live TICKER...</code> swaps in genuine
        documents fetched live from EDGAR, so the pipeline operates on real public data end to end.
    </div>""", unsafe_allow_html=True)
    st.markdown('<div class="takeaway">SEC EDGAR is the standard public corpus for finance NLP precisely because '
                'it is authoritative and auditable - every claim traces to a filed document. That auditability is '
                'the whole value proposition of grounded RAG in finance.</div>', unsafe_allow_html=True)

st.markdown(f"<div style='margin-top:26px;color:{MUTED};font-size:0.82rem'>"
            "FilingsIQ - a finance-ops RAG agent over SEC EDGAR filings. Retrieval &amp; answering run locally; "
            "the EDGAR fetcher is real and also exposed as an MCP server. A portfolio piece illustrating "
            "retrieval-augmented generation, grounding &amp; citation, token minimization, and MCP tooling.</div>",
            unsafe_allow_html=True)
