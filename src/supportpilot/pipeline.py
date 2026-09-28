"""End-to-end ticket pipeline: classify → triage → retrieve → draft reply."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import numpy as np

from .classifier import IntentClassifier
from .config import Settings
from .generator import ClaudeGenerator, ExtractiveGenerator, build_generator
from .kb import chunk_articles, intent_to_articles, load_articles
from .retrieval import HybridRetriever, load_encoder
from .triage import triage

log = logging.getLogger(__name__)


@dataclass
class TicketResult:
    intent: str
    confidence: float
    alternatives: list[dict]
    category: str
    team: str
    priority: str
    priority_reasons: list[str]
    escalate: bool
    escalation_reasons: list[str]
    articles: list[dict]
    reply: str
    citations: list[str]
    reply_backend: str
    grounded: bool
    notes: list[str]
    latency_ms: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class SupportPipeline:
    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: HybridRetriever,
        generator: ClaudeGenerator | ExtractiveGenerator,
        intent_articles: dict[str, set[str]],
        confidence_threshold: float = 0.4,
        top_k: int = 3,
        retrieval_intent_weight: float = 0.6,
        retrieval_temperature: float = 0.1,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        self.intent_articles = intent_articles
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.retrieval_intent_weight = retrieval_intent_weight
        self.retrieval_temperature = retrieval_temperature
        self.article_intents: dict[str, list[str]] = {}
        for intent, article_ids in intent_articles.items():
            for article_id in article_ids:
                self.article_intents.setdefault(article_id, []).append(intent)

    @classmethod
    def from_settings(cls, settings: Settings) -> SupportPipeline:
        if not settings.model_path.exists():
            raise FileNotFoundError(
                f"Intent model not found at {settings.model_path}. Run `make train` (or scripts/train_classifier.py) first."
            )
        articles = load_articles(settings.kb_dir)
        encoder = None
        if settings.use_dense:
            try:
                encoder = load_encoder(settings.embedding_model)
            except Exception as exc:  # missing package, no network for first download, etc.
                log.warning("Dense retrieval disabled (%s); falling back to BM25 only", exc)
        return cls(
            # One sentence encoder is shared by the classifier's semantic head and dense retrieval.
            classifier=IntentClassifier.load(settings.model_path, encoder=encoder),
            retriever=HybridRetriever(
                chunk_articles(articles), encoder, bm25_weight=settings.bm25_weight, prior_weight=settings.prior_weight
            ),
            generator=build_generator(settings.use_llm, settings.llm_model, settings.llm_effort),
            intent_articles=intent_to_articles(articles),
            confidence_threshold=settings.confidence_threshold,
            top_k=settings.top_k,
            retrieval_intent_weight=settings.retrieval_intent_weight,
            retrieval_temperature=settings.retrieval_temperature,
        )

    def article_prior(self, probs: dict[str, float]) -> dict[str, float]:
        """Turn intent probabilities into a probability mass per KB article."""
        prior: dict[str, float] = {}
        for intent, p in probs.items():
            for article_id in self.intent_articles.get(intent, ()):
                prior[article_id] = prior.get(article_id, 0.0) + p
        return prior

    def retrieval_intent_distribution(self, dense_scores: np.ndarray) -> np.ndarray:
        """Intent distribution implied by KB similarity: softmax over articles, split across their intents.

        This covers gaps in the classifier's training data: e.g. "I was charged twice" never appears
        in the training set, but it is a near-verbatim match for a KB section on payment issues.
        """
        scores = self.retriever.article_scores(dense_scores)
        weights = np.exp((scores - scores.max()) / self.retrieval_temperature)
        weights /= weights.sum()
        labels = self.classifier.labels
        out = np.zeros(len(labels))
        for article_id, w in zip(self.retriever.article_ids, weights):
            intents = self.article_intents.get(article_id, [])
            for intent in intents:
                out[labels.index(intent)] += w / len(intents)
        return out

    def classify(self, text: str) -> tuple[dict[str, float], dict[str, float], np.ndarray | None]:
        """Return (final intent probs, classifier-only probs, dense chunk scores or None)."""
        clf_probs = self.classifier.predict_proba([text])[0]
        dense_scores = self.retriever.dense.scores(text) if self.retriever.dense is not None else None
        final = clf_probs
        if dense_scores is not None and self.retrieval_intent_weight > 0:
            w = self.retrieval_intent_weight
            final = (1 - w) * clf_probs + w * self.retrieval_intent_distribution(dense_scores)
        labels = self.classifier.labels
        return dict(zip(labels, final)), dict(zip(labels, clf_probs)), dense_scores

    def process(self, text: str) -> TicketResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        probs, clf_probs, dense_scores = self.classify(text)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
        intent, confidence = ranked[0]
        timings["classify"] = (time.perf_counter() - t0) * 1000

        tri = triage(text, intent, float(confidence), self.confidence_threshold)

        t0 = time.perf_counter()
        # The article prior uses classifier-only probabilities so retrieval evidence isn't counted twice.
        hits = self.retriever.search(
            text, top_k=self.top_k, intent_prior=self.article_prior(clf_probs), dense_scores=dense_scores
        )
        timings["retrieve"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        draft = self.generator.generate(text, hits, tri.team, tri.escalate)
        timings["generate"] = (time.perf_counter() - t0) * 1000
        timings["total"] = sum(timings.values())

        escalation_reasons = list(tri.escalation_reasons)
        if not draft.grounded:
            escalation_reasons.append("reply could not be grounded in the knowledge base")

        return TicketResult(
            intent=intent,
            confidence=round(float(confidence), 4),
            alternatives=[{"intent": i, "confidence": round(float(p), 4)} for i, p in ranked[1:3]],
            category=tri.category,
            team=tri.team,
            priority=tri.priority,
            priority_reasons=tri.reasons,
            escalate=bool(escalation_reasons),
            escalation_reasons=escalation_reasons,
            articles=[{"id": h.article_id, "title": h.title, "score": round(h.score, 4)} for h in hits],
            reply=draft.reply,
            citations=draft.citations,
            reply_backend=draft.backend,
            grounded=draft.grounded,
            notes=draft.notes,
            latency_ms={k: round(v, 1) for k, v in timings.items()},
        )
