"""Text normalization shared by training and inference, so both see identical inputs."""

from __future__ import annotations

import re

_PLACEHOLDER = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
_ORDER_REF = re.compile(r"#\s?[a-z]*-?\d[\w-]*", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d[\d,.-]*\b")
_TOKEN = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and replace dataset slot placeholders / IDs with stable tokens.

    The Bitext training data contains slots like ``{{Order Number}}`` while real customers
    write ``#A12345``. Mapping both to the same token keeps train and serve distributions aligned.
    """
    text = _PLACEHOLDER.sub(lambda m: " slot_" + re.sub(r"\W+", "_", m.group(1).lower()).strip("_") + " ", text)
    text = _ORDER_REF.sub(" slot_ref ", text)
    text = _NUMBER.sub(" num ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize(text))
