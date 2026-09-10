# Escalation Policy: AppleSupport Agent

## Core Principle: Conservative by Design

> **False automation is more dangerous than unnecessary escalation.**

When an AI agent incorrectly auto-handles a case that should have been escalated, the consequences are concrete and severe:
- A customer with a cracked screen is told to "reset network settings" — wasted time and eroded trust.
- A billing dispute gets an incorrect refund policy stated as fact — potential legal liability.
- A stolen device owner receives incomplete lock instructions — real security risk.
- A legal threat is met with a scripted reply — escalates the situation further.

When an agent unnecessarily escalates a case that could have been auto-handled, the only consequence is extra human workload. That is a recoverable operational problem. The others are not.

**Therefore: when in doubt, escalate.**

---

## Decision Output Schema

```json
{
  "decision": "AUTO_HANDLE | ESCALATE",
  "reason": "Human-readable explanation",
  "confidence": 0.0–1.0,
  "rule_triggered": "<rule name>",
  "details": {}
}
```

---

## The 8-Rule Evaluation Chain

Rules are evaluated in order. The first matching rule determines the outcome.

| Priority | Rule | Decision | Justification |
|----------|------|----------|---------------|
| 1 | Intent = `needs_clarification` | ESCALATE | Classifier cannot identify the issue — auto-reply would be irrelevant |
| 2 | Intent in `always_escalate_intents` | ESCALATE | Physical repair and high-emotion complaints require human judgment |
| 3 | Frustration/legal keywords detected | ESCALATE | Emotionally sensitive or legally risky situations need empathy and expertise |
| 4 | Intent confidence < 0.60 | ESCALATE | Unreliable classification means any reply is a guess |
| 5 | Retrieved examples = 0 | ESCALATE | No historical precedent means the reply would be speculative |
| 6 | Top retrieval similarity < 0.25 | ESCALATE | Evidence exists but is too dissimilar to ground a reliable response |
| 7 | Reply confidence < 0.55 | ESCALATE | The model itself doubts its grounding |
| 8 | Unsupported claims > 0 | ESCALATE | The model flagged claims it cannot verify — sending them is unacceptable |
| — | None of the above | AUTO_HANDLE | All signals pass; proceed with the grounded reply |

---

## Configurable Thresholds (`configs/escalation.yaml`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `min_intent_confidence` | 0.60 | Below this, the intent is unreliable |
| `min_reply_confidence` | 0.55 | Below this, the reply is poorly grounded |
| `min_retrieval_similarity` | 0.25 | Below this, retrieved examples are irrelevant |
| `min_evidence_count` | 1 | Minimum examples needed to auto-handle |
| `max_unsupported_claims` | 0 | Any unverified claim triggers escalation |

All thresholds are tunable in `configs/escalation.yaml` — no code changes required.

---

## Always-Escalate Intents

These intents **always** trigger escalation regardless of confidence scores:

| Intent | Reason |
|--------|--------|
| `physical_hardware_damage` | Requires a physical Genius Bar appointment — the AI cannot arrange repairs |
| `general_complaint` | Frustrated customers need human empathy and de-escalation skills |

---

## Combined Confidence Formula

The `confidence` field in the output is a weighted average of three signals:

```
combined = (intent_confidence × 0.4) + (retrieval_similarity × 0.3) + (reply_confidence × 0.3)
```

Intent confidence is weighted most heavily (0.4) because an incorrect intent means every downstream step is wrong.

---

## Why Not a Learned Escalation Model?

A machine-learned escalation classifier would require labelled escalation/non-escalation examples, which we do not have. More importantly:

1. **Interpretability**: Stakeholders must be able to audit every decision.
2. **Edge case safety**: Rule-based systems can guarantee coverage of known high-risk cases (legal threats, physical damage) that a statistical model might miss at low frequency.
3. **Debuggability**: When an escalation decision is wrong, the `rule_triggered` field tells you exactly which rule to adjust.
