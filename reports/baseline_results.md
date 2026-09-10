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
