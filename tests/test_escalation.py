"""
tests/test_escalation.py

Tests for the EscalationPolicy engine.
No mocking needed — the policy is deterministic rule-based logic.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent.escalation import EscalationPolicy, EscalationDecision

CONFIG = "configs/escalation.yaml"

@pytest.fixture
def policy():
    return EscalationPolicy(CONFIG)


# Helper
def _decide(policy, intent="software_bug_glitch", intent_conf=0.85,
            msg="", sim=0.70, count=2, reply_conf=0.80, claims=None):
    return policy.decide(
        intent=intent,
        intent_confidence=intent_conf,
        customer_message=msg,
        top_retrieval_similarity=sim,
        retrieved_count=count,
        reply_confidence=reply_conf,
        unsupported_claims=claims or [],
    )


# ---- Rule 1: uncertainty intent ----
def test_uncertainty_intent_always_escalates(policy):
    d = _decide(policy, intent="needs_clarification", intent_conf=0.95)
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "uncertainty_intent"


# ---- Rule 2: always-escalate intents ----
def test_physical_damage_always_escalates(policy):
    d = _decide(policy, intent="physical_hardware_damage", intent_conf=0.99,
                sim=0.95, count=5, reply_conf=0.99)
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "always_escalate_intent"

def test_general_complaint_always_escalates(policy):
    d = _decide(policy, intent="general_complaint", intent_conf=0.90)
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "always_escalate_intent"


# ---- Rule 3: frustration signals ----
def test_frustration_keyword_triggers_escalation(policy):
    d = _decide(policy, intent="software_bug_glitch", intent_conf=0.90,
                msg="This is absolutely ridiculous, I want a lawyer!")
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "frustration_signal"
    assert "ridiculous" in d.details.get("matched_signals", []) or \
           "lawyer" in d.details.get("matched_signals", [])

def test_legal_threat_escalates(policy):
    d = _decide(policy, intent="billing_subscription",
                msg="This is fraud and I'm calling my attorney.")
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "frustration_signal"


# ---- Rule 4: low intent confidence ----
def test_low_intent_confidence_escalates(policy):
    d = _decide(policy, intent_conf=0.45)
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "low_intent_confidence"

def test_intent_confidence_at_threshold_passes(policy):
    # Exactly at threshold should NOT escalate (>= boundary)
    d = _decide(policy, intent_conf=0.60)
    assert d.decision == "AUTO_HANDLE"


# ---- Rule 5: insufficient retrieval ----
def test_zero_retrieved_examples_escalates(policy):
    d = _decide(policy, count=0, sim=0.0)
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "no_retrieval_evidence"

def test_low_similarity_escalates(policy):
    d = _decide(policy, count=3, sim=0.10)
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "low_retrieval_similarity"

def test_similarity_at_threshold_passes(policy):
    d = _decide(policy, count=2, sim=0.25)
    assert d.decision == "AUTO_HANDLE"


# ---- Rule 6: low reply confidence ----
def test_low_reply_confidence_escalates(policy):
    d = _decide(policy, reply_conf=0.30)
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "low_reply_confidence"


# ---- Rule 7: unsupported claims ----
def test_unsupported_claims_trigger_escalation(policy):
    d = _decide(policy, claims=["Claimed a 14-day refund window not found in evidence."])
    assert d.decision == "ESCALATE"
    assert d.rule_triggered == "unsupported_claims"


# ---- Rule 8: auto-handle ----
def test_all_green_signals_auto_handle(policy):
    d = _decide(policy, intent="software_bug_glitch", intent_conf=0.90,
                msg="My WiFi keeps disconnecting",
                sim=0.75, count=3, reply_conf=0.85, claims=[])
    assert d.decision == "AUTO_HANDLE"
    assert d.rule_triggered == "auto_handle"

def test_feature_how_to_auto_handles(policy):
    d = _decide(policy, intent="feature_how_to", intent_conf=0.88,
                sim=0.60, count=2, reply_conf=0.80)
    assert d.decision == "AUTO_HANDLE"


# ---- Output structure ----
def test_decision_has_required_fields(policy):
    d = _decide(policy)
    assert isinstance(d, EscalationDecision)
    assert d.decision in ("AUTO_HANDLE", "ESCALATE")
    assert isinstance(d.reason, str) and len(d.reason) > 10
    assert 0.0 <= d.confidence <= 1.0
    assert isinstance(d.rule_triggered, str)

def test_combined_confidence_is_weighted_average(policy):
    d = policy.decide(
        intent="feature_how_to",
        intent_confidence=1.0,
        top_retrieval_similarity=1.0,
        retrieved_count=3,
        reply_confidence=1.0,
    )
    # All signals at max → combined ≈ 1.0
    assert d.confidence >= 0.99
