"""End-to-end system evaluation on the hand-written realistic tickets.

Runs the full production pipeline (classifier ensemble + retrieval-augmented classification +
retrieval + triage) and reports what matters operationally: intent accuracy, routing accuracy,
top-article accuracy, and the automation/accuracy trade-off of the escalation threshold.

The realistic set is split in two halves (alternating examples per intent): pipeline
hyper-parameters are tuned on `dev` only; `test` is reported as the held-out number.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportpilot.config import get_settings  # noqa: E402
from supportpilot.generator import ExtractiveGenerator  # noqa: E402
from supportpilot.pipeline import SupportPipeline  # noqa: E402
from supportpilot.triage import CATEGORY_TEAM, INTENT_CATEGORY  # noqa: E402


def run(pipeline: SupportPipeline, df: pd.DataFrame) -> dict:
    labels = pipeline.classifier.labels
    rows = []
    for text, gold in zip(df["text"], df["intent"]):
        probs, clf_probs, _ = pipeline.classify(text)
        result = pipeline.process(text)
        rows.append(
            {
                "gold": gold,
                "classifier_only": max(clf_probs, key=clf_probs.get),
                "final": result.intent,
                "confidence": result.confidence,
                "team_ok": result.team == CATEGORY_TEAM[INTENT_CATEGORY[gold]],
                "article_ok": result.articles[0]["id"] in pipeline.intent_articles[gold],
                "escalate": result.escalate,
                "latency_ms": result.latency_ms["total"],
            }
        )
    r = pd.DataFrame(rows)
    correct = (r["final"] == r["gold"]).to_numpy()
    conf = r["confidence"].to_numpy()
    tradeoff = []
    for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]:
        keep = conf >= t
        tradeoff.append(
            {
                "threshold": t,
                "auto_handled": round(float(keep.mean()), 3),
                "accuracy_on_auto_handled": round(float(correct[keep].mean()), 3) if keep.any() else None,
            }
        )
    assert len(labels) == 27
    return {
        "n": len(r),
        "intent_accuracy_classifier_only": round(float((r["classifier_only"] == r["gold"]).mean()), 3),
        "intent_accuracy_final": round(float(correct.mean()), 3),
        "routing_accuracy": round(float(r["team_ok"].mean()), 3),
        "top_article_accuracy": round(float(r["article_ok"].mean()), 3),
        "escalation_rate": round(float(r["escalate"].mean()), 3),
        "p50_latency_ms": round(float(np.percentile(r["latency_ms"], 50)), 1),
        "p95_latency_ms": round(float(np.percentile(r["latency_ms"], 95)), 1),
        "confidence_threshold_tradeoff": tradeoff,
    }


def main() -> None:
    settings = replace(get_settings(), use_llm=False)
    pipeline = SupportPipeline.from_settings(settings)
    pipeline.generator = ExtractiveGenerator()  # measure triage/retrieval, not LLM latency

    df = pd.read_csv(ROOT / "data" / "eval" / "realistic_tickets.csv")
    half = df.groupby("intent").cumcount() % 2
    report = {
        "config": {
            "classifier": pipeline.classifier.mode,
            "retrieval": pipeline.retriever.mode,
            "retrieval_intent_weight": pipeline.retrieval_intent_weight,
            "retrieval_temperature": pipeline.retrieval_temperature,
            "confidence_threshold": pipeline.confidence_threshold,
        },
        "dev": run(pipeline, df[half == 0]),
        "test": run(pipeline, df[half == 1]),
        "all": run(pipeline, df),
    }
    out = ROOT / "reports" / "pipeline_metrics.json"
    out.write_text(json.dumps(report, indent=2))

    keys = [
        "intent_accuracy_classifier_only", "intent_accuracy_final", "routing_accuracy",
        "top_article_accuracy", "escalation_rate", "p50_latency_ms", "p95_latency_ms",
    ]
    print(f"{'metric':34s} {'dev':>8s} {'test':>8s} {'all':>8s}")
    for k in keys:
        print(f"{k:34s} {report['dev'][k]:8} {report['test'][k]:8} {report['all'][k]:8}")
    print("\nThreshold trade-off (all):")
    for row in report["all"]["confidence_threshold_tradeoff"]:
        print(f"  t={row['threshold']:<5} auto-handled={row['auto_handled']:.3f} accuracy={row['accuracy_on_auto_handled']}")
    print(f"\nReport → {out}")


if __name__ == "__main__":
    main()
