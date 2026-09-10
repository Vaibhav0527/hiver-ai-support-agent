"""
train_tfidf.py

Trains the TF-IDF + Logistic Regression intent classifier and evaluates it
against the held-out golden set.

LEAKAGE PREVENTION:
  - Golden set conversation_ids are loaded FIRST.
  - Training examples are drawn ONLY from conversations whose id is NOT in
    the golden set.
  - No individual message can appear in both splits.

Usage:
  python scripts/train_tfidf.py
"""

import json
import os
import sys
import time

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.tfidf_classifier import TFIDFIntentClassifier

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
    """Keyword pseudo-labeller — only used to build training data."""
    t = str(text).lower()
    if any(w in t for w in ["battery", "draining", "won't charge", "power", "dies"]):
        return "battery_power_issue"
    if any(w in t for w in ["crash", "freeze", "frozen", "update fail", "wifi", "bluetooth", "bug"]):
        return "software_bug_glitch"
    if any(w in t for w in ["cracked", "broken", "shattered", "water damage", "dropped"]):
        return "physical_hardware_damage"
    if any(w in t for w in ["password", "apple id", "locked out", "forgot", "activation lock"]):
        return "account_id_access"
    if any(w in t for w in ["charged", "bill", "refund", "subscription", "purchase"]):
        return "billing_subscription"
    if any(w in t for w in ["lost", "stolen", "find my", "missing"]):
        return "device_lost_stolen"
    if any(w in t for w in ["ship", "order", "delivery", "trade-in", "store hours"]):
        return "purchase_shipping_inquiry"
    if any(w in t for w in ["how do i", "how to", "where is", "backup", "screenshot", "setting"]):
        return "feature_how_to"
    return "general_complaint"


# ---------------------------------------------------------------------------
# Synthetic dataset (covers all 9 intents; used when real data is too small)
# ---------------------------------------------------------------------------
SYNTHETIC_TRAIN = [
    # battery_power_issue
    ("My iPhone battery drains in 2 hours", "battery_power_issue"),
    ("Battery dies at 40% constantly", "battery_power_issue"),
    ("My phone won't charge at all", "battery_power_issue"),
    ("Battery health dropped to 60%", "battery_power_issue"),
    ("Phone shuts off even when charged", "battery_power_issue"),
    ("Charging takes forever and stops at 80%", "battery_power_issue"),
    # software_bug_glitch
    ("Safari keeps crashing on iOS 17", "software_bug_glitch"),
    ("WiFi disconnects every 5 minutes", "software_bug_glitch"),
    ("Update failed and phone won't boot", "software_bug_glitch"),
    ("App freezes when I open it", "software_bug_glitch"),
    ("Bluetooth disconnects randomly", "software_bug_glitch"),
    ("iOS update stuck on the Apple logo", "software_bug_glitch"),
    # physical_hardware_damage
    ("Dropped my phone and the screen cracked", "physical_hardware_damage"),
    ("Water spilled on my MacBook keyboard", "physical_hardware_damage"),
    ("The home button is completely broken", "physical_hardware_damage"),
    ("Camera lens is shattered", "physical_hardware_damage"),
    # account_id_access
    ("Forgot my Apple ID password", "account_id_access"),
    ("Account locked after too many attempts", "account_id_access"),
    ("Can't get past activation lock on a used phone", "account_id_access"),
    ("Two-factor code not arriving", "account_id_access"),
    # billing_subscription
    ("Got charged $9.99 I don't recognize", "billing_subscription"),
    ("How do I cancel Apple Music?", "billing_subscription"),
    ("My kid bought an app by accident, need a refund", "billing_subscription"),
    ("iCloud storage charged twice this month", "billing_subscription"),
    # device_lost_stolen
    ("iPhone stolen last night", "device_lost_stolen"),
    ("Lost my AirPods, can Find My track them?", "device_lost_stolen"),
    ("How do I remotely lock my stolen phone?", "device_lost_stolen"),
    # purchase_shipping_inquiry
    ("When will my order ship?", "purchase_shipping_inquiry"),
    ("Do you take trade-ins for iPhone 12?", "purchase_shipping_inquiry"),
    ("What are the Genius Bar walk-in hours?", "purchase_shipping_inquiry"),
    # feature_how_to
    ("How do I take a screenshot on iPad?", "feature_how_to"),
    ("Where is dark mode setting in iOS 17?", "feature_how_to"),
    ("How do I back up my phone to iCloud?", "feature_how_to"),
    ("Can you walk me through setting up Face ID?", "feature_how_to"),
    # general_complaint
    ("Apple products are terrible, switching to Android", "general_complaint"),
    ("Your customer service is the worst I've experienced", "general_complaint"),
    ("I hate this phone, total waste of money", "general_complaint"),
    ("Very disappointed with Apple's quality control", "general_complaint"),
    ("This is my third time contacting you for the same issue", "general_complaint"),
]

SYNTHETIC_TEST = [
    ("Battery draining super fast after update", "battery_power_issue"),
    ("Phone dies overnight even on charge", "battery_power_issue"),
    ("Bluetooth keeps dropping connection", "software_bug_glitch"),
    ("iOS 17.1 update keeps failing to install", "software_bug_glitch"),
    ("Screen shattered after small fall", "physical_hardware_damage"),
    ("Forgot password and can't log in to Apple ID", "account_id_access"),
    ("Mystery charge on my credit card from Apple", "billing_subscription"),
    ("My phone was stolen, help me lock it", "device_lost_stolen"),
    ("How do I enable two-factor authentication?", "feature_how_to"),
    ("Apple is the worst company, totally useless support", "general_complaint"),
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run():
    os.makedirs("reports", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("  TF-IDF + Logistic Regression Baseline")
    print("=" * 60)

    # ---- Load real data ----
    all_convs = load_jsonl("data/processed/AppleSupport_conversations.jsonl")
    golden    = load_jsonl("data/golden/golden_set.jsonl")
    golden_ids = {item["conversation_id"] for item in golden}

    # Build real training set (strict conversation-level split)
    real_X_train, real_y_train = [], []
    for conv in all_convs:
        if conv["conversation_id"] in golden_ids:
            continue
        for msg in conv["messages"]:
            if msg["role"] == "customer":
                real_X_train.append(msg["text"])
                real_y_train.append(heuristic_intent(msg["text"]))
                break

    real_X_test = [item["customer_message"] for item in golden]
    real_y_test = [item["intent"] for item in golden]

    # ---- Fall back to synthetic if real data is too small ----
    USE_SYNTHETIC = len(real_y_train) < 10 or len(real_y_test) < 5

    if USE_SYNTHETIC:
        print("\n[INFO] Real dataset too small — using synthetic demo data.\n")
        X_train = [x for x, _ in SYNTHETIC_TRAIN]
        y_train = [y for _, y in SYNTHETIC_TRAIN]
        X_test  = [x for x, _ in SYNTHETIC_TEST]
        y_test  = [y for _, y in SYNTHETIC_TEST]
        mode    = "synthetic_demo"
    else:
        X_train, y_train = real_X_train, real_y_train
        X_test,  y_test  = real_X_test,  real_y_test
        mode = "real_data"

    print(f"  Mode          : {mode}")
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test samples  : {len(X_test)}")
    print(f"  Unique intents in train: {sorted(set(y_train))}")

    # ---- Train ----
    clf = TFIDFIntentClassifier()
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    print(f"\n  Training time : {train_time*1000:.1f} ms")

    # Save model
    model_path = "models/tfidf_classifier.pkl"
    clf.save(model_path)
    model_size_kb = os.path.getsize(model_path) / 1024
    print(f"  Model saved   : {model_path} ({model_size_kb:.1f} KB)")

    # ---- Predict + timing ----
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    infer_time = time.perf_counter() - t0
    ms_per_sample = (infer_time / len(X_test)) * 1000

    # ---- Metrics ----
    acc         = accuracy_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_str  = classification_report(y_test, y_pred, zero_division=0)
    all_labels  = sorted(set(y_test) | set(y_pred))
    conf        = confusion_matrix(y_test, y_pred, labels=all_labels)

    print(f"\n  Accuracy    : {acc:.4f}")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"  Inference   : {ms_per_sample:.2f} ms/sample")
    print(f"\n{report_str}")

    # ---- Machine-readable JSON ----
    per_class_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    results = {
        "model": "TFIDFLogisticRegression",
        "mode": mode,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "leakage_prevention": "split by conversation_id; golden ids excluded from train",
        "training_time_ms": round(train_time * 1000, 2),
        "inference_ms_per_sample": round(ms_per_sample, 4),
        "model_size_kb": round(model_size_kb, 1),
        "metrics": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
        },
        "per_class": per_class_dict,
        "confusion_matrix": {
            "labels": all_labels,
            "matrix": conf.tolist(),
        },
    }

    with open("reports/tfidf_results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # ---- Append to baseline_results.md ----
    conf_rows = "\n".join(
        f"  {lbl:30s} | {' '.join(str(v).rjust(3) for v in row)}"
        for lbl, row in zip(all_labels, conf.tolist())
    )
    col_header = "  " + " " * 30 + " | " + "  ".join(l[:3] for l in all_labels)

    section = f"""
---

# Baseline 2: TF-IDF + Logistic Regression

## Why This Baseline?

| Reason | Detail |
|--------|--------|
| Fast to train | Milliseconds on sample; seconds on full dataset |
| Interpretable | Top TF-IDF tokens per class are human-readable |
| Strong on keywords | Apple support queries rely on product names & error keywords |
| Classic NLP bar | Standard second step after majority baseline |

## Model Details

| Field | Value |
|-------|-------|
| Vectoriser | `TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)` |
| Classifier | `LogisticRegression(C=1.0, class_weight='balanced', solver='lbfgs')` |
| Mode | `{mode}` |
| Train samples | {len(X_train)} |
| Test samples | {len(X_test)} |
| Model size | {model_size_kb:.1f} KB |
| Training time | {train_time*1000:.1f} ms |
| Inference speed | {ms_per_sample:.2f} ms / sample |

> **Leakage prevention**: Golden `conversation_id`s excluded from training.
> Split is at the conversation level, not the tweet level.

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

## Analysis

### Where it works
- **Keyword-rich intents**: `battery_power_issue`, `physical_hardware_damage`,
  `device_lost_stolen` all have distinctive vocabulary that TF-IDF captures well.
- **Speed**: At ~{ms_per_sample:.2f} ms/sample, it can classify thousands of
  messages in real time with no GPU.

### Where it fails
- **Vague complaints**: `general_complaint` overlaps with almost every other
  class because frustrated users always include issue keywords.
- **Short, context-dependent messages**: "It's not working" gives TF-IDF nothing
  useful; the LLM baseline will handle these far better.
- **Class imbalance**: Even with `class_weight='balanced'`, rare intents with
  fewer than ~5 training examples get poor recall.

### Verdict
TF-IDF + LR is a strong, lightweight baseline but it is fundamentally limited
by its bag-of-words representation. It cannot reason about context, paraphrase,
or implicit meaning. The LLM classifier should outperform it significantly on
ambiguous and short messages.
"""

    with open("reports/baseline_results.md", "a", encoding="utf-8") as fh:
        fh.write(section)

    print("\n  Results appended to reports/baseline_results.md")
    print("  Machine-readable results saved to reports/tfidf_results.json")
    print("=" * 60)


if __name__ == "__main__":
    run()
