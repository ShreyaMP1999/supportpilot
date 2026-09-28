"""Train the intent classifier ensemble and write an evaluation report to reports/.

1. Tune the TF-IDF head's C on the validation split.
2. Retrain on train+val: TF-IDF head, embedding head, and their ensemble.
3. Evaluate all three once on (a) the held-out in-distribution test split and (b) 108 hand-written
   realistic tickets, the number that best predicts production behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supportpilot.classifier import IntentClassifier  # noqa: E402

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Bitext tags each utterance with the linguistic phenomena it contains.
FLAG_NAMES = {
    "Z": "typos / noise",
    "Q": "colloquial",
    "K": "keyword-only",
    "L": "lexical variation",
    "M": "morphological variation",
    "W": "offensive language",
    "C": "complex / multi-clause",
}


def selective_curve(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray) -> list[dict]:
    """Accuracy on auto-handled tickets vs. share escalated to humans, per confidence threshold."""
    rows = []
    for t in [0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]:
        keep = conf >= t
        rows.append(
            {
                "threshold": t,
                "auto_handled": round(float(keep.mean()), 4),
                "accuracy_on_auto_handled": round(float((y_true[keep] == y_pred[keep]).mean()), 4) if keep.any() else None,
            }
        )
    return rows


def score(model: IntentClassifier, df: pd.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    probs = model.predict_proba(df["text"].tolist())
    ms = (time.perf_counter() - t0) / len(df) * 1000
    pred = np.array(model.labels)[probs.argmax(axis=1)]
    metrics = {
        "accuracy": round(accuracy_score(df["intent"], pred), 4),
        "macro_f1": round(f1_score(df["intent"], pred, average="macro"), 4),
        "ms_per_ticket_batched": round(ms, 3),
    }
    return metrics, pred, probs.max(axis=1)


def plot_confusion(y_true, y_pred, labels, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("True intent")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--ood", type=Path, default=ROOT / "data" / "eval" / "realistic_tickets.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "intent_classifier.joblib")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--tfidf-weight", type=float, default=0.3)
    parser.add_argument("--no-embeddings", action="store_true", help="train the lexical head only (fast)")
    args = parser.parse_args()

    train = pd.read_csv(args.data / "train.csv")
    val = pd.read_csv(args.data / "val.csv")
    test = pd.read_csv(args.data / "test.csv")
    ood = pd.read_csv(args.ood)

    print("Tuning TF-IDF regularization C on the validation split…")
    best_c, best_f1 = 10.0, -1.0
    for c in [1.0, 3.0, 10.0, 30.0]:
        model = IntentClassifier.train(train["text"].tolist(), train["intent"].tolist(), c=c)
        f1 = f1_score(val["intent"], model.tfidf.predict(val["text"].tolist()), average="macro")
        print(f"  C={c:<5} val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    encoder = None
    if not args.no_embeddings:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(args.embedding_model, device="cpu")

    full = pd.concat([train, val])
    t0 = time.perf_counter()
    model = IntentClassifier.train(
        full["text"].tolist(),
        full["intent"].tolist(),
        c=best_c,
        encoder=encoder,
        encoder_name=args.embedding_model if encoder else None,
        tfidf_weight=args.tfidf_weight,
    )
    train_seconds = time.perf_counter() - t0
    model.save(args.out)
    print(f"Saved {model.mode} model → {args.out} ({train_seconds:.0f}s)")

    variants = {"tfidf_lr": IntentClassifier(model.tfidf)}
    if encoder is not None:
        variants["embedding_lr"] = IntentClassifier(model.tfidf, model.embed_head, encoder, tfidf_weight=0.0)
        variants["ensemble"] = model

    comparison = {}
    for name, variant in variants.items():
        test_m, _, _ = score(variant, test)
        ood_m, _, _ = score(variant, ood)
        comparison[name] = {"in_distribution_test": test_m, "realistic_tickets": ood_m}
        print(f"  {name:13s} test acc={test_m['accuracy']:.4f}  realistic acc={ood_m['accuracy']:.4f}")

    _, test_pred, _ = score(model, test)
    _, ood_pred, ood_conf = score(model, ood)
    y_test, y_ood = test["intent"].to_numpy(), ood["intent"].to_numpy()

    by_flag = {}
    for flag, name in FLAG_NAMES.items():
        mask = test["flags"].fillna("").str.contains(flag).to_numpy()
        if mask.sum() >= 30:
            by_flag[name] = {"n": int(mask.sum()), "accuracy": round(float((y_test[mask] == test_pred[mask]).mean()), 4)}

    report = {
        "dataset": {
            "train_plus_val": len(full),
            "test": len(test),
            "realistic_tickets": len(ood),
            "intents": len(model.labels),
        },
        "production_model": {
            "mode": model.mode,
            "tfidf_C": best_c,
            "embedding_model": model.encoder_name,
            "tfidf_weight": model.tfidf_weight,
            "train_seconds": round(train_seconds, 1),
        },
        "model_comparison": comparison,
        "robustness_by_variation": by_flag,
        "realistic_tickets_threshold_tradeoff": selective_curve(y_ood, ood_pred, ood_conf),
        "realistic_ticket_errors": [
            {"text": t, "true": y, "pred": p, "confidence": round(float(c), 3)}
            for t, y, p, c in zip(ood["text"], y_ood, ood_pred, ood_conf)
            if y != p
        ],
    }

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "classifier_metrics.json").write_text(json.dumps(report, indent=2))
    plot_confusion(
        y_ood, ood_pred, sorted(model.labels), REPORTS / "confusion_matrix_realistic.png",
        f"Intent {model.mode} — confusion matrix on realistic tickets (n={len(ood)})",
    )
    print(f"Report → {REPORTS / 'classifier_metrics.json'}")


if __name__ == "__main__":
    main()
