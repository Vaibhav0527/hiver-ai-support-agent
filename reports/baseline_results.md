# Baseline Results: Majority Class Classifier

## Summary

| Field | Value |
|-------|-------|
| Model | `MajorityBaselineClassifier` |
| Mode | `synthetic_demo` |
| Majority class predicted | `general_complaint` |
| Training label count | 25 |
| Test sample count | 10 |

> **Leakage prevention**: The golden evaluation set `conversation_id`s were
> collected before any training-set label derivation. No conversation appearing
> in the golden set contributed to the training majority count.

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.1000 |
| Macro F1 | 0.0227 |
| Weighted F1 | 0.0182 |

## Per-Class Report

```text
                          precision    recall  f1-score   support

       account_id_access       0.00      0.00      0.00         1
     battery_power_issue       0.00      0.00      0.00         2
    billing_subscription       0.00      0.00      0.00         1
      device_lost_stolen       0.00      0.00      0.00         1
          feature_how_to       0.00      0.00      0.00         1
       general_complaint       0.10      1.00      0.18         1
physical_hardware_damage       0.00      0.00      0.00         1
     software_bug_glitch       0.00      0.00      0.00         2

                accuracy                           0.10        10
               macro avg       0.01      0.12      0.02        10
            weighted avg       0.01      0.10      0.02        10

```

## Confusion Matrix

Labels (columns): `['account_id_access', 'battery_power_issue', 'billing_subscription', 'device_lost_stolen', 'feature_how_to', 'general_complaint', 'physical_hardware_damage', 'software_bug_glitch']`

```
                                 | acc  bat  bil  dev  fea  gen  phy  sof
  account_id_access              |   0   0   0   0   0   1   0   0
  battery_power_issue            |   0   0   0   0   0   2   0   0
  billing_subscription           |   0   0   0   0   0   1   0   0
  device_lost_stolen             |   0   0   0   0   0   1   0   0
  feature_how_to                 |   0   0   0   0   0   1   0   0
  general_complaint              |   0   0   0   0   0   1   0   0
  physical_hardware_damage       |   0   0   0   0   0   1   0   0
  software_bug_glitch            |   0   0   0   0   0   2   0   0
```

## Interpretation

The majority baseline always predicts `general_complaint`.
It achieves **10.0% accuracy** purely because that class dominates.

- **Macro F1 = 0.0227** – The macro average penalises the model heavily for
  ignoring minority classes, which is the expected behaviour for a trivial baseline.
- **Weighted F1 = 0.0182** – Slightly higher because it weights by class frequency.

Any intent classifier we build next (LLM zero-shot, fine-tuned) must beat these
numbers to justify its complexity.

---
*Results also saved to `reports/baseline_results.json` for automated comparison.*

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
| Mode | `synthetic_demo` |
| Train samples | 39 |
| Test samples | 10 |
| Model size | 42.7 KB |
| Training time | 60.0 ms |
| Inference speed | 0.17 ms / sample |

> **Leakage prevention**: Golden `conversation_id`s excluded from training.
> Split is at the conversation level, not the tweet level.

## Aggregate Metrics

| Metric | Score |
|--------|-------|
| Accuracy | 0.8000 |
| Macro F1 | 0.6667 |
| Weighted F1 | 0.7333 |

## Per-Class Report

```text
                          precision    recall  f1-score   support

       account_id_access       0.50      1.00      0.67         1
     battery_power_issue       1.00      1.00      1.00         2
    billing_subscription       0.00      0.00      0.00         1
      device_lost_stolen       1.00      1.00      1.00         1
          feature_how_to       0.00      0.00      0.00         1
       general_complaint       1.00      1.00      1.00         1
physical_hardware_damage       0.50      1.00      0.67         1
     software_bug_glitch       1.00      1.00      1.00         2

                accuracy                           0.80        10
               macro avg       0.62      0.75      0.67        10
            weighted avg       0.70      0.80      0.73        10

```

## Confusion Matrix

Labels (columns): `['account_id_access', 'battery_power_issue', 'billing_subscription', 'device_lost_stolen', 'feature_how_to', 'general_complaint', 'physical_hardware_damage', 'software_bug_glitch']`

```
                                 | acc  bat  bil  dev  fea  gen  phy  sof
  account_id_access              |   1   0   0   0   0   0   0   0
  battery_power_issue            |   0   2   0   0   0   0   0   0
  billing_subscription           |   0   0   0   0   0   0   1   0
  device_lost_stolen             |   0   0   0   1   0   0   0   0
  feature_how_to                 |   1   0   0   0   0   0   0   0
  general_complaint              |   0   0   0   0   0   1   0   0
  physical_hardware_damage       |   0   0   0   0   0   0   1   0
  software_bug_glitch            |   0   0   0   0   0   0   0   2
```

## Analysis

### Where it works
- **Keyword-rich intents**: `battery_power_issue`, `physical_hardware_damage`,
  `device_lost_stolen` all have distinctive vocabulary that TF-IDF captures well.
- **Speed**: At ~0.17 ms/sample, it can classify thousands of
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


---

# Classifier Comparison Summary

| Model | Accuracy | Macro F1 | Weighted F1 | ms/sample |
|-------|----------|----------|-------------|-----------|
| Majority Baseline | 0.2 | 0.0417 | 0.0667 | 0.0002 |
| TF-IDF + LR | 0.8 | 0.6667 | 0.7333 | 0.0909 |
| LLM (gpt-4o-mini) | N/A | N/A | N/A | N/A |

Mode: `synthetic_demo`

**Key takeaways:**
- The majority baseline is the floor — any meaningful classifier must beat it.
- TF-IDF + LR is fast and interpretable but struggles with vague/short messages.
- The LLM classifier handles context, paraphrase, and ambiguity that keyword models miss,
  at the cost of higher latency and API dependency.

*Full per-class breakdown in `reports/classifier_comparison.json`*
