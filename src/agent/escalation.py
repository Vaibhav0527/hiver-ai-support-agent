"""
src/agent/escalation.py

Transparent, rule-based escalation decision system.

The escalation policy is deliberately rule-based (not another ML model) because:
  1. The rules are inspectable and auditable by a human reviewer.
  2. Each escalation carries an explicit reason — no black-box decisions.
  3. Thresholds are externalised to configs/escalation.yaml so they can be
     tuned without touching code.
  4. False automation (incorrectly auto-handling a case) is more dangerous than
     unnecessary escalation — see reports/escalation_policy.md.

Evaluation order (first matching rule wins):
  1. Uncertainty intent    → always escalate
  2. Always-escalate intent → always escalate
  3. Frustration signal     → always escalate
  4. Low intent confidence  → escalate
  5. Insufficient retrieval → escalate
  6. Low reply confidence   → escalate
  7. Unsupported claims     → escalate
  8. Otherwise             → AUTO_HANDLE
"""

from dataclasses import dataclass, field
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class EscalationDecision:
    decision: str           # "AUTO_HANDLE" or "ESCALATE"
    reason: str
    confidence: float       # combined signal confidence (0-1)
    rule_triggered: str     # which rule caused the decision
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

class EscalationPolicy:
    """
    Evaluates all available signals and returns a transparent escalation decision.

    Signals evaluated:
      - intent + intent_confidence  (from LLMIntentClassifier)
      - top_retrieval_similarity    (from ConversationRetriever)
      - retrieved_count             (number of useful examples found)
      - reply_confidence            (from ReplyGenerator)
      - unsupported_claims          (list of claims ReplyGenerator flagged)
      - customer_message            (raw text, checked for frustration signals)
    """

    AUTO   = "AUTO_HANDLE"
    ESCL   = "ESCALATE"

    def __init__(self, config_path: str = "configs/escalation.yaml"):
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        t = cfg["thresholds"]
        self.min_intent_conf     = t["min_intent_confidence"]
        self.min_reply_conf      = t["min_reply_confidence"]
        self.min_similarity      = t["min_retrieval_similarity"]
        self.min_evidence_count  = t["min_evidence_count"]
        self.max_unsupported     = t["max_unsupported_claims"]

        self.always_escalate     = set(cfg.get("always_escalate_intents", []))
        self.uncertainty_intent  = cfg.get("uncertainty_intent", "needs_clarification")
        self.frustration_signals = [
            w.lower() for w in cfg.get("frustration_signals", [])
        ]

    # ------------------------------------------------------------------

    def decide(
        self,
        intent: str,
        intent_confidence: float,
        customer_message: str = "",
        top_retrieval_similarity: float = 0.0,
        retrieved_count: int = 0,
        reply_confidence: float = 1.0,
        unsupported_claims: Optional[list] = None,
    ) -> EscalationDecision:
        """
        Evaluate all signals and return an escalation decision.

        Args:
            intent:                    Classified intent label.
            intent_confidence:         Confidence from the intent classifier (0-1).
            customer_message:          Raw customer text (for frustration detection).
            top_retrieval_similarity:  Cosine similarity of best retrieved example.
            retrieved_count:           Number of retrieved examples available.
            reply_confidence:          Confidence from the reply generator (0-1).
            unsupported_claims:        Claims the reply generator flagged as ungrounded.

        Returns:
            EscalationDecision with decision, reason, rule_triggered, and confidence.
        """
        unsupported_claims = unsupported_claims or []
        msg_lower = customer_message.lower()

        # Combined "overall confidence" for reporting purposes
        combined = round(
            (intent_confidence * 0.4) +
            (min(top_retrieval_similarity, 1.0) * 0.3) +
            (reply_confidence * 0.3),
            4,
        )

        # ---- Rule 1: Uncertainty intent ----
        if intent == self.uncertainty_intent:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    "The intent classifier could not reliably identify the customer's "
                    "issue. A human agent should clarify the request before proceeding."
                ),
                confidence=combined,
                rule_triggered="uncertainty_intent",
                details={"intent": intent, "intent_confidence": intent_confidence},
            )

        # ---- Rule 2: Always-escalate intents ----
        if intent in self.always_escalate:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    f"Intent '{intent}' always requires human handling. "
                    "This class of issue cannot be safely resolved by an automated agent."
                ),
                confidence=combined,
                rule_triggered="always_escalate_intent",
                details={"intent": intent},
            )

        # ---- Rule 3: Frustration / high-risk language ----
        matched_signals = [w for w in self.frustration_signals if w in msg_lower]
        if matched_signals:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    "Customer message contains high-emotion or legally sensitive language "
                    f"({', '.join(matched_signals)}). A human agent should respond with empathy."
                ),
                confidence=combined,
                rule_triggered="frustration_signal",
                details={"matched_signals": matched_signals},
            )

        # ---- Rule 4: Low intent confidence ----
        if intent_confidence < self.min_intent_conf:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    f"Intent classifier confidence is {intent_confidence:.2f}, "
                    f"below the threshold of {self.min_intent_conf}. "
                    "Auto-handling an uncertain intent risks giving irrelevant advice."
                ),
                confidence=combined,
                rule_triggered="low_intent_confidence",
                details={
                    "intent_confidence": intent_confidence,
                    "threshold": self.min_intent_conf,
                },
            )

        # ---- Rule 5: Insufficient retrieval evidence ----
        if retrieved_count < self.min_evidence_count:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    f"No historical examples were retrieved (count={retrieved_count}). "
                    "Without grounding evidence the reply would be speculative."
                ),
                confidence=combined,
                rule_triggered="no_retrieval_evidence",
                details={"retrieved_count": retrieved_count},
            )

        if top_retrieval_similarity < self.min_similarity:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    f"Top retrieval similarity is {top_retrieval_similarity:.3f}, "
                    f"below the threshold of {self.min_similarity}. "
                    "The retrieved examples are too dissimilar to ground a reliable reply."
                ),
                confidence=combined,
                rule_triggered="low_retrieval_similarity",
                details={
                    "top_similarity": top_retrieval_similarity,
                    "threshold": self.min_similarity,
                },
            )

        # ---- Rule 6: Low reply confidence ----
        if reply_confidence < self.min_reply_conf:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    f"Reply generator confidence is {reply_confidence:.2f}, "
                    f"below the threshold of {self.min_reply_conf}. "
                    "The model doubts its own grounding — human review is safer."
                ),
                confidence=combined,
                rule_triggered="low_reply_confidence",
                details={
                    "reply_confidence": reply_confidence,
                    "threshold": self.min_reply_conf,
                },
            )

        # ---- Rule 7: Unsupported claims detected ----
        if len(unsupported_claims) > self.max_unsupported:
            return EscalationDecision(
                decision=self.ESCL,
                reason=(
                    f"The reply generator flagged {len(unsupported_claims)} "
                    "unsupported claim(s) that cannot be grounded in historical evidence. "
                    "Sending unverified information to a customer is unacceptable."
                ),
                confidence=combined,
                rule_triggered="unsupported_claims",
                details={"unsupported_claims": unsupported_claims},
            )

        # ---- Rule 8: Auto-handle ----
        return EscalationDecision(
            decision=self.AUTO,
            reason=(
                f"All signals are within acceptable thresholds. "
                f"Intent '{intent}' is well-classified (conf={intent_confidence:.2f}), "
                f"retrieval evidence is adequate (sim={top_retrieval_similarity:.2f}, "
                f"count={retrieved_count}), and the reply is well-grounded "
                f"(reply_conf={reply_confidence:.2f})."
            ),
            confidence=combined,
            rule_triggered="auto_handle",
            details={
                "intent": intent,
                "intent_confidence": intent_confidence,
                "top_retrieval_similarity": top_retrieval_similarity,
                "retrieved_count": retrieved_count,
                "reply_confidence": reply_confidence,
            },
        )
