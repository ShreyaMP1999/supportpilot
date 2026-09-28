"""PII redaction applied before any customer text leaves the service (e.g. to an LLM API)."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: card numbers must be matched before generic phone numbers.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # A country code must be introduced by "+" or followed by a separator, so long bare digit
    # strings (order numbers) are not mistaken for phone numbers.
    ("PHONE", re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?|\d{1,3}[\s.-])?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")),
]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class RedactionResult:
    text: str
    found: dict[str, int]


def redact(text: str) -> RedactionResult:
    found: dict[str, int] = {}

    for label, pattern in _PATTERNS:

        def _sub(match: re.Match[str], label: str = label) -> str:
            if label == "CARD":
                digits = re.sub(r"\D", "", match.group(0))
                if not _luhn_ok(digits):
                    return match.group(0)
            found[label] = found.get(label, 0) + 1
            return f"[{label}]"

        text = pattern.sub(_sub, text)

    return RedactionResult(text=text, found=found)
