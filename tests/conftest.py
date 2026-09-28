from pathlib import Path

import pandas as pd
import pytest

from supportpilot.classifier import IntentClassifier
from supportpilot.generator import ExtractiveGenerator
from supportpilot.kb import chunk_articles, intent_to_articles, load_articles
from supportpilot.pipeline import SupportPipeline
from supportpilot.retrieval import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
KB_DIR = ROOT / "data" / "kb"
FIXTURE = Path(__file__).parent / "fixtures" / "sample_tickets.csv"


@pytest.fixture(scope="session")
def articles():
    return load_articles(KB_DIR)


@pytest.fixture(scope="session")
def classifier():
    # A small TF-IDF model trained on a 405-row sample keeps tests fast and offline.
    df = pd.read_csv(FIXTURE)
    return IntentClassifier.train(df["text"].tolist(), df["intent"].tolist())


@pytest.fixture(scope="session")
def model_path(classifier, tmp_path_factory):
    path = tmp_path_factory.mktemp("models") / "intent_classifier.joblib"
    classifier.save(path)
    return path


@pytest.fixture(scope="session")
def retriever(articles):
    return HybridRetriever(chunk_articles(articles), encoder=None)  # BM25 only: no model download in CI


@pytest.fixture(scope="session")
def pipeline(classifier, retriever, articles):
    return SupportPipeline(
        classifier=classifier,
        retriever=retriever,
        generator=ExtractiveGenerator(),
        intent_articles=intent_to_articles(articles),
    )
