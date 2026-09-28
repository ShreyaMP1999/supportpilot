from supportpilot.pii import redact
from supportpilot.text import normalize


def test_normalize_maps_placeholders_and_ids_to_same_token():
    assert normalize("Cancel order {{Order Number}}") == "cancel order slot_order_number"
    assert "slot_ref" in normalize("Cancel order #A-48213")
    assert normalize("I paid 25.99 dollars") == "i paid num dollars"


def test_redacts_email_phone_and_valid_card():
    result = redact("Email jane.doe@example.com, call (415) 555-0132, card 4111 1111 1111 1111")
    assert "jane.doe" not in result.text
    assert "[EMAIL]" in result.text and "[PHONE]" in result.text and "[CARD]" in result.text
    assert result.found == {"EMAIL": 1, "CARD": 1, "PHONE": 1}


def test_does_not_redact_order_numbers_that_fail_luhn():
    result = redact("My order number is 1234567890123")
    assert "1234567890123" in result.text
    assert "CARD" not in result.found
