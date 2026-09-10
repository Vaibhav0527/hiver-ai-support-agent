"""
tests/test_reply_generator.py

Unit tests for ReplyGenerator.
The OpenAI client is injected via mock — no real API calls are made.

Scenarios tested:
  1. Grounded reply from good historical evidence
  2. Insufficient evidence → model asks clarifying question
  3. Conflicting historical examples → low-confidence reply
  4. Missing customer information → model asks for device/OS details
  5. Empty customer message → early-exit fallback
  6. Malformed LLM response → graceful error handling
  7. API error → retry fallback
  8. Evidence items are correctly parsed into EvidenceItem objects
"""

import json
import os
import sys
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent.reply_generator import (
    ReplyGenerator,
    ReplyResult,
    EvidenceItem,
    _parse_response,
    _format_historical,
)


# ---------------------------------------------------------------------------
# Helper — build a mock OpenAI response
# ---------------------------------------------------------------------------

def _mock_openai_response(payload: dict):
    """Wrap a dict as a mock OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(payload)
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Fixture — ReplyGenerator with mocked client
# ---------------------------------------------------------------------------

@pytest.fixture
def gen():
    """ReplyGenerator with injected mock client (no real API calls)."""
    import yaml
    g = ReplyGenerator.__new__(ReplyGenerator)
    model_cfg = yaml.safe_load(open("configs/model.yaml"))
    g.model_name   = model_cfg["model"]["name"]
    g.temperature  = 0.0
    g.max_tokens   = 512
    g.timeout      = 30
    g.max_attempts = 3
    g.backoff      = 0          # no sleeping in tests
    g.prompt_template = open("prompts/reply_generation.txt").read()
    g._client = MagicMock()
    g._openai_available = True
    return g


# ---------------------------------------------------------------------------
# Fake historical examples
# ---------------------------------------------------------------------------

GOOD_EVIDENCE = [
    {
        "conversation_id": "synth_001",
        "customer_message": "My iPhone battery drains really fast",
        "historical_brand_response": "Try Settings > Battery > Battery Health. If below 80%, visit Genius Bar.",
        "similarity_score": 0.91,
        "intent": "battery_power_issue",
    }
]

CONFLICTING_EVIDENCE = [
    {
        "conversation_id": "synth_001",
        "customer_message": "Battery draining fast",
        "historical_brand_response": "Visit a Genius Bar for a battery replacement.",
        "similarity_score": 0.70,
        "intent": "battery_power_issue",
    },
    {
        "conversation_id": "synth_002",
        "customer_message": "Battery draining fast",
        "historical_brand_response": "Turn off Background App Refresh in Settings.",
        "similarity_score": 0.68,
        "intent": "battery_power_issue",
    },
]


# ---------------------------------------------------------------------------
# _parse_response unit tests (no network)
# ---------------------------------------------------------------------------

def test_parse_valid_grounded_response():
    payload = {
        "reply": "Try checking your battery health in Settings.",
        "confidence": 0.88,
        "evidence": [{"conversation_id": "synth_001", "reason": "Similar battery issue resolved with health check."}],
        "unsupported_claims": [],
    }
    result = _parse_response(json.dumps(payload))
    assert result["reply"].startswith("Try")
    assert result["confidence"] == 0.88
    assert len(result["evidence"]) == 1
    assert isinstance(result["evidence"][0], EvidenceItem)
    assert result["evidence"][0].conversation_id == "synth_001"


def test_parse_clamps_confidence_above_one():
    payload = {"reply": "ok", "confidence": 5.0, "evidence": [], "unsupported_claims": []}
    result = _parse_response(json.dumps(payload))
    assert result["confidence"] == 1.0


def test_parse_malformed_json_raises():
    with pytest.raises(ValueError, match="Malformed JSON"):
        _parse_response("not json at all")


def test_parse_missing_fields_get_defaults():
    result = _parse_response(json.dumps({"reply": "ok"}))
    assert result["evidence"] == []
    assert result["unsupported_claims"] == []
    assert 0.0 <= result["confidence"] <= 1.0


def test_format_historical_empty():
    output = _format_historical([])
    assert "no similar" in output.lower()


def test_format_historical_single_example():
    output = _format_historical(GOOD_EVIDENCE)
    assert "synth_001" in output
    assert "Battery Health" in output


# ---------------------------------------------------------------------------
# ReplyGenerator integration tests (mocked client)
# ---------------------------------------------------------------------------

def test_empty_message_returns_fallback(gen):
    result = gen.generate("", intent="battery_power_issue")
    assert result.error is not None or "describe" in result.reply.lower()


def test_grounded_reply_with_good_evidence(gen):
    payload = {
        "reply": "Please check Settings > Battery > Battery Health. If it's below 80%, a replacement is recommended.",
        "confidence": 0.90,
        "evidence": [{"conversation_id": "synth_001", "reason": "Direct match for battery health issue."}],
        "unsupported_claims": [],
    }
    gen._client.chat.completions.create = MagicMock(return_value=_mock_openai_response(payload))

    result = gen.generate(
        customer_message="My iPhone battery drains really fast",
        intent="battery_power_issue",
        historical_examples=GOOD_EVIDENCE,
    )
    assert result.confidence == 0.90
    assert len(result.evidence) == 1
    assert result.evidence[0].conversation_id == "synth_001"
    assert result.unsupported_claims == []


def test_insufficient_evidence_returns_low_confidence(gen):
    payload = {
        "reply": "Could you tell me which iPhone model and iOS version you're using? That will help me assist you better.",
        "confidence": 0.2,
        "evidence": [],
        "unsupported_claims": ["Cannot find a historical example matching this issue."],
    }
    gen._client.chat.completions.create = MagicMock(return_value=_mock_openai_response(payload))

    result = gen.generate(
        customer_message="My phone is doing something weird",
        intent="software_bug_glitch",
        historical_examples=[],
    )
    assert result.confidence < 0.5
    assert len(result.unsupported_claims) > 0


def test_conflicting_evidence_lowers_confidence(gen):
    payload = {
        "reply": "There are a couple of things to try: first check Background App Refresh, and if that doesn't help, visit a Genius Bar.",
        "confidence": 0.55,
        "evidence": [
            {"conversation_id": "synth_001", "reason": "Suggested Genius Bar visit."},
            {"conversation_id": "synth_002", "reason": "Suggested Background App Refresh setting."},
        ],
        "unsupported_claims": [],
    }
    gen._client.chat.completions.create = MagicMock(return_value=_mock_openai_response(payload))

    result = gen.generate(
        customer_message="Battery draining fast",
        intent="battery_power_issue",
        historical_examples=CONFLICTING_EVIDENCE,
    )
    assert result.confidence < 0.8
    assert len(result.evidence) == 2


def test_missing_customer_info_triggers_clarification(gen):
    payload = {
        "reply": "I'd be happy to help! Could you please tell me which iPhone model and iOS version you're running?",
        "confidence": 0.5,
        "evidence": [{"conversation_id": "synth_001", "reason": "Apple agents always ask for device model first."}],
        "unsupported_claims": [],
    }
    gen._client.chat.completions.create = MagicMock(return_value=_mock_openai_response(payload))

    result = gen.generate(
        customer_message="My update failed",
        intent="software_bug_glitch",
        historical_examples=GOOD_EVIDENCE,
    )
    # The reply should be asking for more info
    assert "?" in result.reply
    assert result.confidence <= 0.7


def test_api_error_returns_fallback(gen):
    gen.max_attempts = 2
    gen._client.chat.completions.create = MagicMock(side_effect=Exception("Timeout"))

    result = gen.generate(
        customer_message="My iPhone won't turn on",
        intent="battery_power_issue",
        historical_examples=GOOD_EVIDENCE,
    )
    assert result.error is not None
    assert "support.apple.com" in result.reply


def test_evidence_items_are_parsed_as_dataclass(gen):
    payload = {
        "reply": "Please visit iforgot.apple.com to reset your password.",
        "confidence": 0.95,
        "evidence": [{"conversation_id": "synth_006", "reason": "Exact match for Apple ID reset."}],
        "unsupported_claims": [],
    }
    gen._client.chat.completions.create = MagicMock(return_value=_mock_openai_response(payload))

    result = gen.generate(
        customer_message="Forgot my Apple ID password",
        intent="account_id_access",
        historical_examples=GOOD_EVIDENCE,
    )
    assert all(isinstance(e, EvidenceItem) for e in result.evidence)
