"""Hybrid retrieval over KB chunks: BM25 (lexical) + dense embeddings (semantic), fused with RRF.

Chunk scores are aggregated to article level (max over an article's chunks) because an article
is the unit an agent links to; the best chunks are still returned as context for generation.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .kb import Chunk
from .text import tokenize

_STOPWORDS = frozenset(
    "a about all am an and any are as at be been but by can could did do does for from get got had has have how "
    "i if in is it its just me my of on or our so that the there this to want was we what when where which will "
    "with would you your im need please help hi hello thanks num".split()
)


def _stem(token: str) -> str:
    """Very light suffix stripping (weeks→week, refunds→refund) — enough for short support queries."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _terms(text: str) -> list[str]:
    return [_stem(t) for t in tokenize(text) if t not in _STOPWORDS]


class BM25:
    """Okapi BM25 over a small in-memory corpus."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_terms = [Counter(_terms(d)) for d in docs]
        self.doc_len = np.array([sum(c.values()) for c in self.doc_terms], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(docs) else 0.0
        df: Counter[str] = Counter()
        for counts in self.doc_terms:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.doc_terms))
        for term in set(_terms(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, counts in enumerate(self.doc_terms):
                tf = counts.get(term, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * tf * (self.k1 + 1) / denom
        return out


class Encoder(Protocol):
    def encode(self, texts: list[str], **kwargs) -> np.ndarray: ...


class DenseIndex:
    def __init__(self, docs: list[str], encoder: Encoder):
        self.encoder = encoder
        self.matrix = self._embed(docs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def scores(self, query: str) -> np.ndarray:
        return self.matrix @ self._embed([query])[0]

    def scores_batch(self, queries: list[str]) -> np.ndarray:
        return self._embed(queries) @ self.matrix.T


def load_encoder(model_name: str) -> Encoder:
    from sentence_transformers import SentenceTransformer  # heavy import, only when dense is enabled

    return SentenceTransformer(model_name, device="cpu")


def rrf(rankings: list[np.ndarray], k: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """Weighted Reciprocal Rank Fusion over several score vectors (higher score = better)."""
    weights = weights or [1.0] * len(rankings)
    fused = np.zeros(len(rankings[0]))
    for scores, weight in zip(rankings, weights):
        order = np.argsort(-scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(order) + 1)
        fused += weight / (k + ranks)
    return fused


@dataclass
class ArticleHit:
    article_id: str
    title: str
    score: float
    chunks: list[Chunk]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        encoder: Encoder | None = None,
        bm25_weight: float = 0.5,
        prior_weight: float = 0.5,
    ):
        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self.prior_weight = prior_weight
        texts = [c.search_text for c in chunks]
        self.bm25 = BM25(texts)
        self.dense = DenseIndex(texts, encoder) if encoder is not None else None
        self.article_ids = sorted({c.article_id for c in chunks})
        self._article_index = {a: i for i, a in enumerate(self.article_ids)}
        self._chunk_article = np.array([self._article_index[c.article_id] for c in chunks])

    @property
    def mode(self) -> str:
        if self.dense is None:
            return "bm25"
        return "dense" if self.bm25_weight == 0 else "hybrid"

    def chunk_scores(self, query: str, method: str | None = None, dense_scores: np.ndarray | None = None) -> np.ndarray:
        method = method or self.mode
        if method == "bm25":
            return self.bm25.scores(query)
        if self.dense is None:
            raise RuntimeError("Dense retrieval requested but no encoder is configured")
        dense = dense_scores if dense_scores is not None else self.dense.scores(query)
        if method == "dense" or self.bm25_weight == 0:
            return dense
        return rrf([self.bm25.scores(query), dense], weights=[self.bm25_weight, 1 - self.bm25_weight])

    def article_scores(self, chunk_scores: np.ndarray) -> np.ndarray:
        out = np.full(len(self.article_ids), -np.inf)
        np.maximum.at(out, self._chunk_article, chunk_scores)
        return out

    def search(
        self,
        query: str,
        top_k: int = 3,
        method: str | None = None,
        intent_prior: dict[str, float] | None = None,
        prior_weight: float | None = None,
        dense_scores: np.ndarray | None = None,
    ) -> list[ArticleHit]:
        """Rank KB articles for a query.

        ``intent_prior`` maps article_id → probability (from the intent classifier). When given, the
        final ranking fuses retrieval with the classifier: ``(1-w)·rrf_retrieval + w·rrf_prior``.
        """
        chunk_scores = self.chunk_scores(query, method, dense_scores)
        art = self.article_scores(chunk_scores)
        if intent_prior:
            prior = np.array([intent_prior.get(a, 0.0) for a in self.article_ids])
            # Confidence gating: a peaked prior (confident classifier) gets the full weight, a flat
            # one (unsure classifier) barely moves the ranking, so a wrong guess can't bury the answer.
            w = (self.prior_weight if prior_weight is None else prior_weight) * min(1.0, float(prior.max()))
            art = rrf([art, prior], weights=[1 - w, w])

        hits = []
        for idx in np.argsort(-art)[:top_k]:
            article_id = self.article_ids[idx]
            members = [i for i in np.argsort(-chunk_scores) if self._chunk_article[i] == idx][:2]
            hits.append(
                ArticleHit(
                    article_id=article_id,
                    title=self.chunks[members[0]].article_title,
                    score=float(art[idx]),
                    chunks=[self.chunks[i] for i in members],
                )
            )
        return hits
