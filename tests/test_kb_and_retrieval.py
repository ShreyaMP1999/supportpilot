import numpy as np

from supportpilot.kb import chunk_articles, intent_to_articles
from supportpilot.retrieval import BM25, HybridRetriever, rrf
from supportpilot.triage import INTENT_CATEGORY


def test_every_intent_has_a_kb_article(articles):
    mapping = intent_to_articles(articles)
    assert set(INTENT_CATEGORY) <= set(mapping), set(INTENT_CATEGORY) - set(mapping)


def test_chunks_carry_article_metadata(articles):
    chunks = chunk_articles(articles)
    assert len(chunks) > len(articles)
    assert all(c.heading and c.text and c.article_id.startswith("KB-") for c in chunks)


def test_bm25_prefers_matching_document():
    bm25 = BM25(["refund policy thirty days", "track your parcel with the carrier", "reset your password"])
    assert int(np.argmax(bm25.scores("how do I reset my password"))) == 2


def test_rrf_rewards_agreement_between_rankers():
    fused = rrf([np.array([3.0, 2.0, 1.0]), np.array([1.0, 3.0, 2.0])])
    assert int(np.argmax(fused)) == 1


def test_bm25_retrieval_finds_relevant_article(retriever):
    assert retriever.search("what is the status of my refund", top_k=3)[0].article_id == "KB-009"
    assert retriever.search("I forgot my password", top_k=1)[0].article_id == "KB-015"


def test_intent_prior_reorders_results(retriever):
    query = "question about my account"
    base = retriever.search(query, top_k=1)[0].article_id
    boosted = retriever.search(query, top_k=1, intent_prior={"KB-016": 1.0}, prior_weight=0.9)[0].article_id
    assert boosted == "KB-016" and base != "KB-016"


def test_dense_path_works_with_any_encoder(articles):
    class HashEncoder:  # deterministic stand-in for a sentence-transformer
        def encode(self, texts, **_):
            vecs = np.array([[hash(w) % 97 for w in (t.lower().split() + [""] * 8)[:8]] for t in texts], float)
            return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    retriever = HybridRetriever(chunk_articles(articles), encoder=HashEncoder())
    assert retriever.mode == "hybrid"
    assert len(retriever.search("cancel my order", top_k=3)) == 3
