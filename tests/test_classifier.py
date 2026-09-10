"""
tests/test_classifier.py

Unit tests for the LLMIntentClassifier.
Tests do NOT call the real OpenAI API — the client is monkey-patched.
"""
import json
import sys
import os
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent.classifier import LLMIntentClassifier, _parse_response, ClassificationResult

# ---------------------------------------------------------------------------
# Helper — build a mock OpenAI response
# ---------------------------------------------------------------------------

def _mock_response(intent, confidence=0.9, reason="test reason"):
    payload = json.dumps({"intent": intent, "confidence": confidence, "reason": reason})
    choice = MagicMock()
    choice.message.content = payload
    resp = MagicMock()
    resp.choices = [choice]
    return resp

# ---------------------------------------------------------------------------
# _parse_response unit tests (no network needed)
# ---------------------------------------------------------------------------

ALLOWED = {
    "battery_power_issue", "software_bug_glitch", "physical_hardware_damage",
    "account_id_access", "billing_subscription", "device_lost_stolen",
    "purchase_shipping_inquiry", "feature_how_to", "general_complaint",
}

def test_parse_valid_intent():
    raw = json.dumps({"intent": "battery_power_issue", "confidence": 0.95, "reason": "mentions battery"})
    result = _parse_response(raw, ALLOWED)
    assert result["intent"] == "battery_power_issue"
    assert result["confidence"] == 0.95

def test_parse_unsupported_intent_falls_back():
    raw = json.dumps({"intent": "made_up_intent", "confidence": 0.8, "reason": "..."})
    result = _parse_response(raw, ALLOWED)
    assert result["intent"] == "needs_clarification"

def test_parse_needs_clarification_is_valid():
    raw = json.dumps({"intent": "needs_clarification", "confidence": 0.3, "reason": "ambiguous"})
    result = _parse_response(raw, ALLOWED)
    assert result["intent"] == "needs_clarification"

def test_parse_malformed_json_raises():
    with pytest.raises(ValueError, match="Malformed JSON"):
        _parse_response("this is not json", ALLOWED)

def test_parse_markdown_fences_stripped():
    raw = "```json\n{\"intent\": \"feature_how_to\", \"confidence\": 0.7, \"reason\": \"how-to question\"}\n```"
    result = _parse_response(raw, ALLOWED)
    assert result["intent"] == "feature_how_to"

def test_parse_confidence_clamped():
    raw = json.dumps({"intent": "general_complaint", "confidence": 99.0, "reason": "..."})
    result = _parse_response(raw, ALLOWED)
    assert result["confidence"] == 1.0

# ---------------------------------------------------------------------------
# LLMIntentClassifier integration tests (mocked client)
# ---------------------------------------------------------------------------

@pytest.fixture
def clf():
    """
    Create classifier with a fully mocked openai client.
    We inject the mock client directly to avoid dealing with lazy imports
    and SDK credential checks.
    """
    classifier = LLMIntentClassifier.__new__(LLMIntentClassifier)
    # Manually set the attributes that __init__ would set
    import yaml
    model_cfg  = yaml.safe_load(open("configs/model.yaml"))
    intents_cfg = yaml.safe_load(open("configs/intents.yaml"))
    classifier.model_name = model_cfg["model"]["name"]
    classifier.temperature = 0.0
    classifier.max_tokens = 256
    classifier.timeout = 30
    classifier.max_attempts = 3
    classifier.backoff = 0  # no sleep in tests
    classifier.allowed_intents = {i["name"] for i in intents_cfg["intents"]}
    classifier.intent_list_str = ""
    classifier.prompt_template = open("prompts/classification.txt").read()
    classifier._client = MagicMock()
    classifier._openai_available = True
    return classifier

def test_empty_message_returns_needs_clarification(clf):
    result = clf.classify("")
    assert result.intent == "needs_clarification"

def test_none_stripped_message_returns_needs_clarification(clf):
    result = clf.classify("   ")
    assert result.intent == "needs_clarification"

def test_valid_intent_returned(clf):
    mock_resp = _mock_response("battery_power_issue", confidence=0.95)
    clf._client.chat.completions.create = MagicMock(return_value=mock_resp)
    clf._openai_available = True

    result = clf.classify("My iPhone battery drains super fast")
    assert result.intent == "battery_power_issue"
    assert result.confidence == 0.95
    assert isinstance(result.latency_ms, float)

def test_ambiguous_message_returns_needs_clarification(clf):
    mock_resp = _mock_response("needs_clarification", confidence=0.4, reason="message is too vague")
    clf._client.chat.completions.create = MagicMock(return_value=mock_resp)
    clf._openai_available = True

    result = clf.classify("Something is wrong")
    assert result.intent == "needs_clarification"
    assert result.confidence < 0.5

def test_unsupported_intent_from_llm_falls_back(clf):
    mock_resp = _mock_response("some_random_intent_the_llm_invented")
    clf._client.chat.completions.create = MagicMock(return_value=mock_resp)
    clf._openai_available = True

    result = clf.classify("Tell me about AppleCare+")
    assert result.intent == "needs_clarification"

def test_api_error_falls_back_after_retries(clf):
    clf._openai_available = True
    clf.max_attempts = 2
    clf.backoff = 0  # no sleep in tests
    clf._client.chat.completions.create = MagicMock(side_effect=Exception("Connection timeout"))

    result = clf.classify("My phone won't turn on")
    assert result.intent == "needs_clarification"
    assert result.error is not None

def test_context_passed_to_prompt(clf):
    """Ensure conversation context is included in the prompt text."""
    captured = {}

    def capture_call(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return _mock_response("software_bug_glitch")

    clf._client.chat.completions.create = MagicMock(side_effect=capture_call)
    clf._openai_available = True

    context = [{"role": "brand", "text": "Hi, what device are you using?"}]
    clf.classify("My app keeps crashing", context=context)
    assert "Hi, what device are you using?" in captured["prompt"]
