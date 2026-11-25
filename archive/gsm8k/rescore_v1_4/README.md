# Offline answer rescoring (posthoc sensitivity)

This result reuses existing generated text. No model, GPU, training or network request is needed.

Original archived files are unchanged. Original generation settings/code/protocol remain in each manifest; scoring_revision identifies the new verifier. Each row retains original_score.

Original training rewards and the originally selected test adapter are retained. This is a posthoc sensitivity analysis, not a fresh experiment under the revised reward.

Frozen test checkpoint: 384. Revised dev ranking would choose: 384. No automatic reselection was performed.

| Evaluation | Samples | Original correct | Revised correct | Changed scores |
|---|---:|---:|---:|---:|
| test_base | 3200 | 1451 | 2074 | 889 |
| test_rl | 3200 | 2530 | 2530 | 0 |
| dev/checkpoint-128 | 800 | 608 | 608 | 0 |
| dev/checkpoint-256 | 800 | 636 | 636 | 0 |
| dev/checkpoint-384 | 800 | 652 | 652 | 0 |

See main_analysis/report.md for paired question-level bootstrap statistics, changed_scores.jsonl for changed decisions, selection_sensitivity.json for all development candidates, and ready.json for integrity hashes.
