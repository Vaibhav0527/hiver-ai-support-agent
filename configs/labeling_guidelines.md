# Golden Set Labeling Guidelines

These guidelines are designed for human annotators to accurately label customer support messages for the AppleSupport evaluation dataset. Do not rely exclusively on automated systems for these labels.

## Labeling Schema
For each JSONL entry in `data/golden/golden_set.jsonl`, you must review the `customer_message` and `conversation_context` to fill in:
1. `intent`: Choose one of the 9 strict taxonomy labels below.
2. `should_escalate`: `true` if a human agent must step in (or a store appointment is needed), `false` if an AI agent can resolve it.
3. `escalation_reason`: If `true`, explain why (e.g., "Physical damage requires Genius Bar appointment").
4. `expected_reply_characteristics`: A list of strings detailing what a good AI response should look like (e.g., `["ask for iOS version", "express empathy"]`).
5. `source`: Change from `"unlabeled"` or `"heuristic_draft"` to `"human_labeled"` when finished.

## Intent Taxonomy

### 1. battery_power_issue
- **Description:** Device rapid drain, failure to charge, unexpected shutdown.
- **Example:** "My iPhone dies at 20%."
- **Escalation Rules:** Only escalate if software troubleshooting (restart/update) has already failed.

### 2. software_bug_glitch
- **Description:** App crashes, update failures, UI freezes, WiFi drops.
- **Example:** "Safari freezes every time."
- **Escalation Rules:** Generally `false`. Resolvable via troubleshooting steps.

### 3. physical_hardware_damage
- **Description:** Cracked screens, water damage, physically broken buttons.
- **Example:** "I dropped my phone and shattered the screen."
- **Escalation Rules:** ALWAYS `true`. Requires an Apple Store repair appointment.

### 4. account_id_access
- **Description:** Apple ID login issues, forgotten passwords, activation lock.
- **Example:** "I forgot my Apple ID password."
- **Escalation Rules:** `false`. AI should provide iForgot or account recovery link.

### 5. billing_subscription
- **Description:** Unrecognized charges, refund requests, Apple Music issues.
- **Example:** "I got charged $9.99 for Apple Music but I canceled."
- **Escalation Rules:** `true` if it requires account verification to process a refund.

### 6. device_lost_stolen
- **Description:** Lost/stolen device, Find My inquiries.
- **Example:** "My iPhone was stolen, how do I lock it?"
- **Escalation Rules:** `false`. AI should provide Find My instructions immediately.

### 7. purchase_shipping_inquiry
- **Description:** Order tracking, trade-ins, store availability.
- **Example:** "When will my iPhone 15 ship?"
- **Escalation Rules:** `false`. Provide link to Apple Order Status.

### 8. feature_how_to
- **Description:** Instructions on how to use a feature or backup.
- **Example:** "How do I take a screenshot on iPad?"
- **Escalation Rules:** `false`. Highly resolvable by AI.

### 9. general_complaint
- **Description:** Frustration/anger without actionable technical issue.
- **Example:** "You guys are the worst."
- **Escalation Rules:** `true`. A human agent should handle severe dissatisfaction.

## Handling Ambiguous Cases & Multi-Intent
- **Multi-Intent**: If a user says "My phone is slow and the screen is cracked", ALWAYS prioritize the most critical intent. Physical hardware damage trumps software slowness. The label would be `physical_hardware_damage` and `should_escalate=true`.
- **Ambiguity**: If the user just says "Help", label it as `feature_how_to` but note in `expected_reply_characteristics` that the AI must ask clarifying questions.
