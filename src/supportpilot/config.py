"""Runtime configuration, read from environment variables (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    kb_dir: Path = field(default_factory=lambda: Path(os.getenv("SP_KB_DIR", PROJECT_ROOT / "data" / "kb")))
    model_path: Path = field(
        default_factory=lambda: Path(os.getenv("SP_MODEL_PATH", PROJECT_ROOT / "models" / "intent_classifier.joblib"))
    )
    db_path: Path = field(default_factory=lambda: Path(os.getenv("SP_DB_PATH", PROJECT_ROOT / "supportpilot.db")))

    # Retrieval
    use_dense: bool = field(default_factory=lambda: _env_bool("SP_USE_DENSE", True))
    embedding_model: str = field(default_factory=lambda: os.getenv("SP_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    top_k: int = field(default_factory=lambda: int(os.getenv("SP_TOP_K", "3")))
    # Fusion weights (see README "Retrieval"). BM25 did not improve on dense retrieval on any split,
    # so its fusion weight defaults to 0 (BM25 is still the fallback when dense is disabled). The
    # intent-classifier prior is capped at 0.5 so a wrong intent can't override retrieval.
    bm25_weight: float = field(default_factory=lambda: float(os.getenv("SP_BM25_WEIGHT", "0.0")))
    prior_weight: float = field(default_factory=lambda: float(os.getenv("SP_PRIOR_WEIGHT", "0.5")))

    # Retrieval-augmented classification: final intent = (1-w)·classifier + w·KB-similarity intent
    # distribution. Tuned on the dev half of the realistic ticket set (see README).
    retrieval_intent_weight: float = field(default_factory=lambda: float(os.getenv("SP_RETRIEVAL_INTENT_WEIGHT", "0.6")))
    retrieval_temperature: float = field(default_factory=lambda: float(os.getenv("SP_RETRIEVAL_TEMPERATURE", "0.1")))

    # Triage
    confidence_threshold: float = field(default_factory=lambda: float(os.getenv("SP_CONFIDENCE_THRESHOLD", "0.4")))

    # Generation. LLM drafting is used only when enabled AND Anthropic credentials are available;
    # otherwise the pipeline falls back to a deterministic extractive reply.
    use_llm: bool = field(default_factory=lambda: _env_bool("SP_USE_LLM", True))
    llm_model: str = field(default_factory=lambda: os.getenv("SP_LLM_MODEL", "claude-opus-5"))
    llm_effort: str = field(default_factory=lambda: os.getenv("SP_LLM_EFFORT", "low"))


def get_settings() -> Settings:
    return Settings()
