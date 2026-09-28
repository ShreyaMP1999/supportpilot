"""Explainable triage: category, owning team, priority (P1–P4) and escalation decision.

Priority is a transparent points system rather than a black box, so support leads can audit
and tune it. Every point added is recorded as a human-readable reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INTENT_CATEGORY = {
    "cancel_order": "ORDER", "change_order": "ORDER", "place_order": "ORDER", "track_order": "ORDER",
    "change_shipping_address": "SHIPPING", "set_up_shipping_address": "SHIPPING",
    "delivery_options": "DELIVERY", "delivery_period": "DELIVERY",
    "check_refund_policy": "REFUND", "get_refund": "REFUND", "track_refund": "REFUND",
    "check_payment_methods": "PAYMENT", "payment_issue": "PAYMENT",
    "check_invoice": "INVOICE", "get_invoice": "INVOICE",
    "create_account": "ACCOUNT", "delete_account": "ACCOUNT", "edit_account": "ACCOUNT",
    "recover_password": "ACCOUNT", "registration_problems": "ACCOUNT", "switch_account": "ACCOUNT",
    "newsletter_subscription": "SUBSCRIPTION",
    "check_cancellation_fee": "CANCEL",
    "contact_customer_service": "CONTACT", "contact_human_agent": "CONTACT",
    "complaint": "FEEDBACK", "review": "FEEDBACK",
}

CATEGORY_TEAM = {
    "ORDER": "Fulfillment", "SHIPPING": "Fulfillment", "DELIVERY": "Fulfillment", "CANCEL": "Fulfillment",
    "REFUND": "Billing", "PAYMENT": "Billing", "INVOICE": "Billing",
    "ACCOUNT": "Accounts & Security", "SUBSCRIPTION": "Accounts & Security",
    "CONTACT": "Customer Care", "FEEDBACK": "Customer Care",
}

# Base severity of an intent: money movement, account access and complaints matter most.
INTENT_BASE_POINTS = {
    "payment_issue": 3, "complaint": 3, "get_refund": 2, "track_refund": 2, "recover_password": 2,
    "delete_account": 2, "registration_problems": 1, "cancel_order": 1, "change_order": 1,
    "change_shipping_address": 1, "track_order": 1, "contact_human_agent": 1,
}

_SIGNALS: list[tuple[str, int, re.Pattern[str]]] = [
    ("urgency language", 2, re.compile(r"\b(urgent(ly)?|asap|immediately|right now|emergency)\b", re.I)),
    ("legal / chargeback threat", 3, re.compile(r"\b(lawyer|legal action|sue|chargeback|dispute|fraud|scam|bbb)\b", re.I)),
    ("repeat contact", 2, re.compile(r"\b(again|still (not|no|haven'?t)|(second|third|3rd|2nd) time|for (days|weeks)|no one|nobody)\b", re.I)),
    ("charged incorrectly", 2, re.compile(r"\b(charged (twice|double|two times)|double charged|overcharged|unauthori[sz]ed)\b", re.I)),
    ("account security", 3, re.compile(r"\b(hacked|compromised|someone (else )?(logged|accessed|used))\b", re.I)),
]

_NEGATIVE = frozenset(
    "angry furious terrible horrible awful worst useless ridiculous unacceptable disgusted frustrated "
    "disappointed annoyed pathetic incompetent rude waste".split()
)


@dataclass
class TriageResult:
    category: str
    team: str
    priority: str
    priority_points: int
    reasons: list[str] = field(default_factory=list)
    escalate: bool = False
    escalation_reasons: list[str] = field(default_factory=list)


def _priority_label(points: int) -> str:
    if points >= 6:
        return "P1"
    if points >= 4:
        return "P2"
    if points >= 2:
        return "P3"
    return "P4"


def triage(text: str, intent: str, confidence: float, confidence_threshold: float) -> TriageResult:
    category = INTENT_CATEGORY.get(intent, "CONTACT")
    reasons: list[str] = []

    points = INTENT_BASE_POINTS.get(intent, 0)
    if points:
        reasons.append(f"intent '{intent}' (+{points})")

    for name, weight, pattern in _SIGNALS:
        if pattern.search(text):
            points += weight
            reasons.append(f"{name} (+{weight})")

    negative_hits = sum(1 for w in re.findall(r"[a-z']+", text.lower()) if w in _NEGATIVE)
    shouting = sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", text)) >= 2 or "!!" in text
    if negative_hits or shouting:
        weight = min(2, negative_hits + int(shouting))
        points += weight
        reasons.append(f"negative sentiment (+{weight})")

    priority = _priority_label(points)

    escalation_reasons = []
    if confidence < confidence_threshold:
        escalation_reasons.append(f"low classifier confidence ({confidence:.2f} < {confidence_threshold:.2f})")
    if intent == "contact_human_agent":
        escalation_reasons.append("customer asked for a human")
    if priority == "P1":
        escalation_reasons.append("P1 priority")

    return TriageResult(
        category=category,
        team=CATEGORY_TEAM[category],
        priority=priority,
        priority_points=points,
        reasons=reasons,
        escalate=bool(escalation_reasons),
        escalation_reasons=escalation_reasons,
    )
