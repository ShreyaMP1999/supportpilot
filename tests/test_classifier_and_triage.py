from supportpilot.triage import triage


def test_classifier_predicts_clear_intents(classifier):
    assert classifier.predict("I forgot my password and cannot log in").intent == "recover_password"
    assert classifier.predict("please cancel my order").intent == "cancel_order"


def test_prediction_confidence_is_a_probability(classifier):
    pred = classifier.predict("where is my package")
    assert 0.0 < pred.confidence <= 1.0
    assert len(pred.alternatives) == 2
    assert all(pred.confidence >= p for _, p in pred.alternatives)


def test_calm_question_is_low_priority():
    result = triage("what payment methods do you accept?", "check_payment_methods", 0.95, 0.55)
    assert result.priority == "P4"
    assert result.team == "Billing"
    assert not result.escalate


def test_angry_double_charge_is_p1_and_escalated():
    text = "I was charged twice AGAIN!! This is ridiculous, I'm filing a chargeback"
    result = triage(text, "payment_issue", 0.9, 0.55)
    assert result.priority == "P1"
    assert result.escalate
    assert any("chargeback" in r for r in result.reasons)


def test_low_confidence_triggers_escalation():
    result = triage("hmm", "review", 0.2, 0.55)
    assert result.escalate
    assert "low classifier confidence" in result.escalation_reasons[0]


def test_human_request_is_escalated():
    assert triage("let me talk to a person", "contact_human_agent", 0.99, 0.55).escalate
