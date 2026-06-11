"""FilingsIQ RAG pipeline — the retrieval-augmented core.

Deliberately dependency-light so it deploys free on Streamlit with no API key and
no large model download: embeddings are local TF-IDF vectors (scikit-learn), and
answer synthesis is extractive + templated (grounded strictly in retrieved text).

The pipeline stages, each independently inspectable in the UI:

    1. chunk      — split filings into overlapping passages
    2. embed      — TF-IDF vectorize the corpus (fit once, cached)
    3. retrieve   — cosine similarity, top-k, with a TOKEN BUDGET cap
    4. compress   — optional sentence-level pruning to cut context tokens
    5. ground     — assemble a grounded answer with inline citations
    6. score      — a groundedness score (is every answer sentence supported?)

Token minimization is a first-class feature: top_k, chunk_size, and a
context-compression toggle all trade tokens against coverage, and the app
surfaces a live tokens-per-query meter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------- token est.
def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token, the common heuristic)."""
    return max(1, round(len(text) / 4))


# ----------------------------------------------------------------- chunking
@dataclass
class Chunk:
    chunk_id: int
    company: str
    ticker: str
    form: str
    filing_date: str
    section: str
    url: str
    text: str
    tokens: int = 0


def chunk_corpus(docs: list[dict], chunk_size: int = 120, overlap: int = 25) -> list[Chunk]:
    """Split each doc into word-window chunks (chunk_size words, `overlap` words shared)."""
    chunks: list[Chunk] = []
    cid = 0
    step = max(1, chunk_size - overlap)
    for d in docs:
        words = d["text"].split()
        if not words:
            continue
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_size])
            if not piece.strip():
                continue
            chunks.append(Chunk(
                chunk_id=cid, company=d["company"], ticker=d["ticker"],
                form=d["form"], filing_date=d["filing_date"], section=d["section"],
                url=d["url"], text=piece, tokens=estimate_tokens(piece)))
            cid += 1
            if start + chunk_size >= len(words):
                break
    return chunks


# ----------------------------------------------------------------- index
class RagIndex:
    """TF-IDF index over the chunked corpus."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int = 4,
               token_budget: Optional[int] = None) -> list[tuple[Chunk, float]]:
        """Return top-k (chunk, score), optionally capped by a context token budget.

        Scores are TF-IDF cosine similarity, with a small boost when the query names
        the chunk's company or ticker (so 'Northwind revenue' prefers Northwind chunks).
        """
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel().astype(float)
        ql = query.lower()
        for i, ch in enumerate(self.chunks):
            name_first = ch.company.split()[0].lower()
            if name_first in ql or ch.ticker.lower() in ql:
                sims[i] += 0.15  # company/ticker affinity
        order = np.argsort(-sims)
        results: list[tuple[Chunk, float]] = []
        used_tokens = 0
        for idx in order:
            score = float(sims[idx])
            if score <= 0:
                break
            ch = self.chunks[idx]
            if token_budget is not None and used_tokens + ch.tokens > token_budget:
                # if nothing fits yet, allow the single best chunk through so a
                # tight budget degrades gracefully rather than refusing everything
                if results:
                    continue
            results.append((ch, score))
            used_tokens += ch.tokens
            if len(results) >= top_k:
                break
        return results


# ----------------------------------------------------------------- compression
def compress_context(query: str, chunk_text: str, keep_sentences: int = 3) -> str:
    """Sentence-level pruning: keep only the sentences most relevant to the query.

    A cheap, model-free way to cut context tokens — a real token-minimization lever.
    """
    sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
    if len(sentences) <= keep_sentences:
        return chunk_text
    q_terms = set(re.findall(r"\w+", query.lower()))
    scored = []
    for s in sentences:
        s_terms = set(re.findall(r"\w+", s.lower()))
        overlap = len(q_terms & s_terms)
        scored.append((overlap, s))
    top = sorted(scored, key=lambda x: -x[0])[:keep_sentences]
    # preserve original order
    keep = [s for _, s in top]
    return " ".join(s for s in sentences if s in keep)


# ----------------------------------------------------------------- answer + grounding
@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    groundedness: float = 0.0
    refused: bool = False
    context_tokens: int = 0
    retrieved: list = field(default_factory=list)


REFUSAL = ("I couldn't find this in the retrieved filings, so I won't guess. "
           "Try rephrasing, or ask about revenue, margins, liquidity, risk factors, "
           "or controls for the companies in the corpus.")


def answer_question(query: str, index: RagIndex, top_k: int = 4,
                    token_budget: Optional[int] = None,
                    compress: bool = False, min_score: float = 0.06) -> Answer:
    """End-to-end: retrieve -> (compress) -> ground -> score."""
    hits = index.search(query, top_k=top_k, token_budget=token_budget)

    # Refusal gate: keep only hits with genuine query-term overlap with the chunk,
    # so a company-name boost alone can't carry an otherwise-irrelevant question
    # (e.g. "what is the CEO's favorite color" must refuse).
    q_terms = {t for t in re.findall(r"\w+", query.lower()) if len(t) > 3}
    kept = []
    for ch, score in hits:
        if score < min_score:
            continue
        chunk_terms = set(re.findall(r"\w+", ch.text.lower()))
        content_overlap = len(q_terms & chunk_terms)
        if content_overlap >= 1:
            kept.append((ch, score))
    hits = kept

    if not hits:
        return Answer(text=REFUSAL, refused=True)

    pieces, citations, ctx_tokens = [], [], 0
    for ch, score in hits:
        ctx = compress_context(query, ch.text) if compress else ch.text
        ctx_tokens += estimate_tokens(ctx)
        pieces.append(ctx)
        citations.append({
            "n": len(citations) + 1,
            "company": ch.company, "ticker": ch.ticker, "form": ch.form,
            "filing_date": ch.filing_date, "section": ch.section,
            "url": ch.url, "score": round(score, 3),
            "snippet": ctx[:240] + ("..." if len(ctx) > 240 else ""),
        })

    # extractive grounded synthesis: stitch the most relevant sentences, cite each.
    answer_sents = []
    for i, ctx in enumerate(pieces, start=1):
        sents = re.split(r"(?<=[.!?])\s+", ctx)
        best = _most_relevant_sentence(query, sents)
        if best:
            answer_sents.append(f"{best} [{i}]")
    answer_text = " ".join(answer_sents) if answer_sents else pieces[0] + " [1]"

    grounded = _groundedness(answer_text, pieces)
    return Answer(text=answer_text, citations=citations, groundedness=grounded,
                  refused=False, context_tokens=ctx_tokens, retrieved=hits)


def _most_relevant_sentence(query: str, sentences: list[str]) -> str:
    q = set(re.findall(r"\w+", query.lower()))
    best, best_score = "", -1
    for s in sentences:
        s_terms = set(re.findall(r"\w+", s.lower()))
        sc = len(q & s_terms)
        if sc > best_score and len(s.split()) > 3:
            best, best_score = s.strip(), sc
    return best


def _groundedness(answer: str, contexts: list[str]) -> float:
    """Fraction of answer sentences whose content words are covered by the context."""
    ctx_terms = set(re.findall(r"\w+", " ".join(contexts).lower()))
    sents = [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", answer)) if s.strip()]
    if not sents:
        return 0.0
    covered = 0
    for s in sents:
        terms = [t for t in re.findall(r"\w+", s.lower()) if len(t) > 3]
        if not terms:
            covered += 1
            continue
        hit = sum(1 for t in terms if t in ctx_terms) / len(terms)
        if hit >= 0.8:
            covered += 1
    return round(covered / len(sents), 3)
