"""Intent classifier: a probability-averaging ensemble of two complementary models.

* Lexical head — TF-IDF over word 1-2-grams and character 2-5-grams → logistic regression.
  Character n-grams make it robust to typos; it excels on phrasing seen in training.
* Semantic head — sentence embeddings (MiniLM, the same encoder the retriever uses) → logistic
  regression. It generalizes to phrasings never seen in the (templated) training data.

On hand-written realistic tickets the ensemble beats either head alone (see reports/). If no
encoder is available at load time, the classifier degrades gracefully to the lexical head.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .text import normalize

log = logging.getLogger(__name__)


@dataclass
class Prediction:
    intent: str
    confidence: float
    alternatives: list[tuple[str, float]]


def build_tfidf_pipeline(c: float = 10.0) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True
                ),
            ),
        ]
    )
    return Pipeline([("features", features), ("clf", LogisticRegression(C=c, max_iter=2000))])


def embed(encoder, texts: list[str]) -> np.ndarray:
    return np.asarray(encoder.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False))


class IntentClassifier:
    def __init__(
        self,
        tfidf: Pipeline,
        embed_head: LogisticRegression | None = None,
        encoder=None,
        encoder_name: str | None = None,
        tfidf_weight: float = 0.3,
    ):
        self.tfidf = tfidf
        self.embed_head = embed_head
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.tfidf_weight = tfidf_weight
        if embed_head is not None and list(embed_head.classes_) != list(tfidf.classes_):
            raise ValueError("Ensemble heads were trained on different label sets")

    @property
    def mode(self) -> str:
        return "ensemble" if self.embed_head is not None and self.encoder is not None else "tfidf"

    @property
    def labels(self) -> list[str]:
        return list(self.tfidf.classes_)

    @classmethod
    def train(
        cls,
        texts: list[str],
        labels: list[str],
        c: float = 10.0,
        encoder=None,
        encoder_name: str | None = None,
        embed_c: float = 10.0,
        tfidf_weight: float = 0.3,
    ) -> IntentClassifier:
        tfidf = build_tfidf_pipeline(c).fit(texts, labels)
        head = None
        if encoder is not None:
            head = LogisticRegression(C=embed_c, max_iter=3000).fit(embed(encoder, texts), labels)
        return cls(tfidf, head, encoder, encoder_name, tfidf_weight)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "tfidf": self.tfidf,
                "embed_head": self.embed_head,
                "encoder_name": self.encoder_name,
                "tfidf_weight": self.tfidf_weight,
            },
            path,
            compress=3,
        )

    @classmethod
    def load(cls, path: Path, encoder=None) -> IntentClassifier:
        """Load a saved model. Pass the shared sentence encoder to enable the semantic head."""
        state = joblib.load(path)
        if state["embed_head"] is not None and encoder is None:
            log.warning("No encoder supplied; intent classifier running lexical head only")
        return cls(state["tfidf"], state["embed_head"], encoder, state["encoder_name"], state["tfidf_weight"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        lexical = self.tfidf.predict_proba(texts)
        if self.mode == "tfidf":
            return lexical
        semantic = self.embed_head.predict_proba(embed(self.encoder, texts))
        return self.tfidf_weight * lexical + (1 - self.tfidf_weight) * semantic

    def predict(self, text: str, top_n: int = 3) -> Prediction:
        probs = self.predict_proba([text])[0]
        order = np.argsort(probs)[::-1][:top_n]
        ranked = [(self.labels[i], float(probs[i])) for i in order]
        return Prediction(intent=ranked[0][0], confidence=ranked[0][1], alternatives=ranked[1:])
