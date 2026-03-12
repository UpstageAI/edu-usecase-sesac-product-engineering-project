from __future__ import annotations

from backend.agents.email_agent.tools import detect_email_risks


def _risk_types(payload):
    return {item.get("type") for item in payload if isinstance(item, dict)}


def test_detect_email_risks_does_not_flag_missing_when_details_exist():
    text = (
        "Payment terms: L/C at sight. "
        "Quantity: 100 units. "
        "Delivery date: 2026-03-01."
    )

    result = detect_email_risks.invoke({"email_content": text})
    types = _risk_types(result)

    assert "payment_missing" not in types
    assert "quantity_missing" not in types
    assert "date_missing" not in types


def test_detect_email_risks_flags_missing_when_trigger_terms_have_no_details():
    text = "결제 조건은 추후 협의하겠습니다. 수량과 납기도 추후 안내 예정입니다."

    result = detect_email_risks.invoke({"email_content": text})
    types = _risk_types(result)

    assert "payment_missing" in types
    assert "quantity_missing" in types
    assert "date_missing" in types
