# Intent Taxonomy: AppleSupport

This document outlines the intent taxonomy designed specifically for the AppleSupport Twitter interactions. This taxonomy simplifies the massive range of customer support queries into 9 highly distinct, actionable intents.

## Design Philosophy
1. **Mutually Understandable**: Intents like `physical_hardware_damage` vs `software_bug_glitch` have clear dividing lines.
2. **Escalation Support**: Intents are mapped to distinct resolution types. A `physical_hardware_damage` intent will almost always require escalation (Store Appointment), whereas a `feature_how_to` intent can be resolved entirely by the AI agent with a support article.
3. **Appropriate Granularity**: We avoid overly fine-grained labels (e.g. `iphone_battery_drain` vs `macbook_battery_drain`) in favor of broad, actionable categories (`battery_power_issue`).

## Taxonomy Overview

### 1. battery_power_issue
- **Definition**: Device experiencing rapid battery drain, failing to charge, shutting down unexpectedly, or not turning on.
- **Expected Resolution**: Troubleshooting steps or repair assessment.

### 2. software_bug_glitch
- **Definition**: Software-related malfunction (app crashing, update failures, UI freezes, WiFi/Bluetooth drops).
- **Expected Resolution**: Troubleshooting steps (Restart, update, reset settings).

### 3. physical_hardware_damage
- **Definition**: Physical damage to the device (cracked screen, water damage, broken port).
- **Expected Resolution**: Apple Store appointment / repair escalation.

### 4. account_id_access
- **Definition**: Issues logging into an Apple ID, activation lock, or forgotten passwords.
- **Expected Resolution**: Direct link to iForgot or account recovery guide.

### 5. billing_subscription
- **Definition**: Inquiries regarding iTunes/App Store purchases, unrecognized charges, or Apple Music subscriptions.
- **Expected Resolution**: Support article on refunds or billing escalation.

### 6. device_lost_stolen
- **Definition**: Customer has lost their device or it was stolen (Find My network tracking).
- **Expected Resolution**: Find My iPhone guide / remote lock instructions.

### 7. purchase_shipping_inquiry
- **Definition**: Questions regarding buying a physical product, tracking an order, or trade-in values.
- **Expected Resolution**: Informational reply or link to Order Status page.

### 8. feature_how_to
- **Definition**: Asking for instructions on how to use a feature, change a setting, or perform a backup.
- **Expected Resolution**: Support article link with brief instructions.

### 9. general_complaint
- **Definition**: Expressing frustration or anger without describing a specific actionable technical issue.
- **Expected Resolution**: Empathy, de-escalation, and request for DM to resolve the issue.

*For full definitions, inclusions, exclusions, and real examples, see `configs/intents.yaml`.*
