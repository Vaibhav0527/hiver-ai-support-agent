"""
evaluate_baseline.py

Evaluates the MajorityBaselineClassifier against the golden evaluation set.

LEAKAGE PREVENTION:
  - Golden set conversation_ids are collected first.
  - Training labels are derived ONLY from conversations NOT in the golden set.
  - Even pseudo-labels (heuristic) are excluded for golden-set conversations.

Usage:
  python scripts/evaluate_baseline.py
"""

import json
import os
import sys
import random

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.baseline_majority import MajorityBaselineClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list:
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def heuristic_intent(text: str) -> str:
    """
    Keyword-based pseudo-labeller used ONLY to populate the training
    distribution when real human labels are absent. Never used on the
    golden/test set.
    """
    text = str(text).lower()
    if any(w in text for w in ["battery", "draining", "won't charge", "power", "dies"]):
        return "battery_power_issue"
    if any(w in text for w in ["crash", "freeze", "frozen", "update fail", "wifi", "bluetooth", "bug"]):
        return "software_bug_glitch"
    if any(w in text for w in ["cracked", "broken", "shattered", "water damage", "dropped"]):
        return "physical_hardware_damage"
    if any(w in text for w in ["password", "apple id", "locked out", "forgot", "activation lock"]):
        return "account_id_access"
    if any(w in text for w in ["charged", "bill", "refund", "subscription", "purchase"]):
        return "billing_subscription"
    if any(w in text for w in ["lost", "stolen", "find my", "missing"]):
        return "device_lost_stolen"
    if any(w in text for w in ["ship", "order", "delivery", "trade-in", "store hours"]):
        return "purchase_shipping_inquiry"
    if any(w in text for w in ["how do i", "how to", "where is", "backup", "screenshot", "setting"]):
        return "feature_how_to"
    return "general_complaint"


# ---------------------------------------------------------------------------
# Synthetic fallback dataset (used when real data is insufficient)
# ---------------------------------------------------------------------------
SYNTHETIC_TRAIN = [
    ("My iPhone battery drains in 2 hours", "battery_power_issue"),
    ("Battery dies at 40% constantly", "battery_power_issue"),
    ("My phone won't charge at all", "battery_power_issue"),
    ("Battery health dropped to 60%", "battery_power_issue"),
    ("Safari keeps crashing on iOS 17", "software_bug_glitch"),
    ("WiFi disconnects every 5 minutes", "software_bug_glitch"),
    ("Update failed and phone won't boot", "software_bug_glitch"),
    ("App freezes when I open it", "software_bug_glitch"),
    ("Dropped my phone and the screen cracked", "physical_hardware_damage"),
    ("Water spilled on my MacBook", "physical_hardware_damage"),
    ("Forgot my Apple ID password", "account_id_access"),
    ("Account locked after too many attempts", "account_id_access"),
    ("Got charged $9.99 I don't recognize", "billing_subscription"),
    ("How do I cancel Apple Music?", "billing_subscription"),
    ("iPhone stolen last night", "device_lost_stolen"),
    ("Lost my AirPods, can Find My track them?", "device_lost_stolen"),
    ("When will my order ship?", "purchase_shipping_inquiry"),
    ("How do I take a screenshot on iPad?", "feature_how_to"),
    ("Where is dark mode setting?", "feature_how_to"),
    ("Apple products are terrible, switching to Android", "general_complaint"),
    ("Your customer service is the worst", "general_complaint"),
    ("I hate this phone", "general_complaint"),
    ("Very disappointed with Apple", "general_complaint"),
    ("general issue with my device", "general_complaint"),
    ("Something is wrong with my iPhone", "general_complaint"),
]

SYNTHETIC_TEST = [
    ("Battery draining super fast after update", "battery_power_issue"),
    ("Phone dies overnight even on charge", "battery_power_issue"),
    ("Bluetooth keeps dropping connection", "software_bug_glitch"),
    ("iOS 17.1 update keeps failing", "software_bug_glitch"),
    ("Screen shattered after fall", "physical_hardware_damage"),
    ("Forgot password and can't log in", "account_id_access"),
    ("Mystery charge on my credit card from Apple", "billing_subscription"),
    ("My phone was stolen", "device_lost_stolen"),
    ("How do I enable two-factor auth?", "feature_how_to"),
    ("Apple is the worst company ever", "general_complaint"),
]


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def run():
    os.makedirs("reports", exist_ok=True)
    print("=" * 60)
    print("  Majority Baseline Evaluation")
    print("=" * 60)

    # ---- Load real data ----
    all_convs = load_jsonl("data/processed/AppleSupport_conversations.jsonl")
    golden = load_jsonl("data/golden/golden_set.jsonl")

    golden_conv_ids = {item["conversation_id"] for item in golden}

    # Build real train labels (exclude golden conv ids to prevent leakage)
    real_train_labels = []
    for conv in all_convs:
        if conv["conversation_id"] in golden_conv_ids:
            continue
        for msg in conv["messages"]:
            if msg["role"] == "customer":
                real_train_labels.append(heuristic_intent(msg["text"]))
                break

    real_y_test = [item["intent"] for item in golden]
    real_X_test = [item["customer_message"] for item in golden]

    # ---- Decide whether to use real or synthetic data ----
    USE_SYNTHETIC = len(real_train_labels) < 5 or len(real_y_test) < 5

    if USE_SYNTHETIC:
        print("\n[INFO] Real dataset too small for meaningful evaluation.")
        print("[INFO] Using synthetic demonstration data to show pipeline mechanics.\n")
        y_train = [label for _, label in SYNTHETIC_TRAIN]
        X_test  = [text  for text, _  in SYNTHETIC_TEST]
        y_test  = [label for _, label in SYNTHETIC_TEST]
        mode    = "synthetic_demo"
    else:
        y_train = real_train_labels
        X_test  = real_X_test
        y_test  = real_y_test
        mode    = "real_data"

    print(f"  Mode          : {mode}")
    print(f"  Train labels  : {len(y_train)}")
    print(f"  Test samples  : {len(y_test)}")

    # ---- Fit baseline ----
    clf = MajorityBaselineClassifier()
    clf.fit(y_train)
    print(f"\n  Majority class: '{clf.majority_class}'")

    # ---- Predict ----
    y_pred = clf.predict(X_test)

    # ---- Metrics ----
    acc        = accuracy_score(y_test, y_pred)
    macro_f1   = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1= f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_str = classification_report(y_test, y_pred, zero_division=0)
    all_labels = sorted(set(y_test) | set(y_pred))
    conf       = confusion_matrix(y_test, y_pred, labels=all_labels)

    print(f"\n  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"\n  Per-class report:\n{report_str}")
    print(f"  Confusion matrix (labels: {all_labels}):\n{conf}\n")

    # ---- Save JSON (machine-readable) ----
    per_class = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    results = {
        "model": "MajorityBaselineClassifier",
        "mode": mode,
        "majority_class": clf.majority_class,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "leakage_prevention": "split by conversation_id; golden conv_ids excluded from train",
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": all_labels,
            "matrix": conf.tolist(),
        },
    }

    with open("reports/baseline_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ---- Save Markdown ----
    conf_rows = "\n".join(
        f"  {lbl:30s} | {' '.join(str(v).rjust(3) for v in row)}"
        for lbl, row in zip(all_labels, conf.tolist())
    )
    col_header = "  " + " " * 30 + " | " + "  ".join(lbl[:3] for lbl in all_labels)

    md = f"""# Baseline Results: Majority Class Classifier

## Summary

| Field | Value |
|-------|-------|
| Model | `MajorityBaselineClassifier` |
| Mode | `{mode}` |
| Majority class predicted | `{clf.majority_class}` |
| Training label count | {len(y_train)} |
| Test sample count | {len(y_test)} |

> **Leakage prevention**: The golden evaluation set `conversation_id`s were
> collected before any training-set label derivation. No conversation appearing
> in the golden set contributed to the training majority count.

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Accuracy | {acc:.4f} |
| Macro F1 | {macro_f1:.4f} |
| Weighted F1 | {weighted_f1:.4f} |

## Per-Class Report

```text
{report_str}
```

## Confusion Matrix

Labels (columns): `{all_labels}`

```
{col_header}
{conf_rows}
```

## Interpretation

The majority baseline always predicts `{clf.majority_class}`.
It achieves **{acc*100:.1f}% accuracy** purely because that class dominates.

- **Macro F1 = {macro_f1:.4f}** – The macro average penalises the model heavily for
  ignoring minority classes, which is the expected behaviour for a trivial baseline.
- **Weighted F1 = {weighted_f1:.4f}** – Slightly higher because it weights by class frequency.

Any intent classifier we build next (LLM zero-shot, fine-tuned) must beat these
numbers to justify its complexity.

---
*Results also saved to `reports/baseline_results.json` for automated comparison.*
"""

    with open("reports/baseline_results.md", "w", encoding="utf-8") as fh:
        fh.write(md)

    print("  Results saved to reports/baseline_results.md and reports/baseline_results.json")
    print("=" * 60)


if __name__ == "__main__":
    run()
