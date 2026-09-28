"""Benchmark KB retrieval: BM25 vs dense vs hybrid (RRF) vs hybrid + intent-classifier prior.

Relevance labels come for free: every KB article declares which intents it answers, so a query's
relevant articles are those mapped to its gold intent. Reports Recall@1, Recall@3 and MRR on the
in-distribution test split and on the hand-written realistic ticket set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportpilot.classifier import IntentClassifier  # noqa: E402
from supportpilot.config import get_settings  # noqa: E402
from supportpilot.kb import chunk_articles, intent_to_articles, load_articles  # noqa: E402
from supportpilot.retrieval import HybridRetriever, load_encoder  # noqa: E402


def evaluate(retriever, classifier, intent_articles, df: pd.DataFrame) -> dict:
    texts, gold = df["text"].tolist(), df["intent"].tolist()
    dense = retriever.dense.scores_batch(texts)
    probs = classifier.predict_proba(texts)

    methods = ["bm25", "dense", "hybrid", "dense+intent", "hybrid+intent"]
    bm25_weight = retriever.bm25_weight or 0.2  # evaluate hybrid variants at 0.2 even when disabled in prod
    retriever.bm25_weight = bm25_weight
    ranks: dict[str, list[int]] = {m: [] for m in methods}
    for i, (text, intent) in enumerate(zip(texts, gold)):
        relevant = intent_articles[intent]
        prior: dict[str, float] = {}
        for label, p in zip(classifier.labels, probs[i]):
            for article_id in intent_articles.get(label, ()):
                prior[article_id] = prior.get(article_id, 0.0) + float(p)
        for method in methods:
            base, _, boosted = method.partition("+")
            retriever.bm25_weight = 0.0 if base == "dense" else bm25_weight
            hits = retriever.search(
                text,
                top_k=len(retriever.article_ids),
                method=base,
                intent_prior=prior if boosted else None,
                dense_scores=dense[i],
            )
            ranks[method].append(next(r for r, h in enumerate(hits, 1) if h.article_id in relevant))

    out = {}
    for method, r in ranks.items():
        r = np.array(r)
        out[method] = {
            "recall@1": round(float((r <= 1).mean()), 4),
            "recall@3": round(float((r <= 3).mean()), 4),
            "mrr": round(float((1 / r).mean()), 4),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "processed" / "test.csv")
    parser.add_argument("--ood", type=Path, default=ROOT / "data" / "eval" / "realistic_tickets.csv")
    args = parser.parse_args()

    settings = get_settings()
    articles = load_articles(settings.kb_dir)
    encoder = load_encoder(settings.embedding_model)
    retriever = HybridRetriever(
        chunk_articles(articles), encoder, bm25_weight=settings.bm25_weight, prior_weight=settings.prior_weight
    )
    classifier = IntentClassifier.load(settings.model_path, encoder=encoder)
    intent_articles = intent_to_articles(articles)

    report = {
        "kb": {"articles": len(articles), "chunks": len(retriever.chunks)},
        "config": {
            "production_bm25_weight": settings.bm25_weight,
            "hybrid_variants_bm25_weight": settings.bm25_weight or 0.2,
            "prior_weight": settings.prior_weight,
        },
        "in_distribution_test": evaluate(retriever, classifier, intent_articles, pd.read_csv(args.test)),
        "realistic_tickets": evaluate(retriever, classifier, intent_articles, pd.read_csv(args.ood)),
    }
    out = ROOT / "reports" / "retrieval_metrics.json"
    out.write_text(json.dumps(report, indent=2))

    for split in ("in_distribution_test", "realistic_tickets"):
        print(f"\n{split}")
        print(f"  {'method':15s} {'R@1':>7s} {'R@3':>7s} {'MRR':>7s}")
        for method, m in report[split].items():
            print(f"  {method:15s} {m['recall@1']:7.3f} {m['recall@3']:7.3f} {m['mrr']:7.3f}")
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()
