# Project Organization Implementation Plan

Goal: Retain both the final MATH Level 4 experiment and the earlier GSM8K experiment while exposing the selected RL adapter, original answers and final scores through clear project-root paths.

Architecture: Preserve immutable experiment bundles byte-for-byte and move them as complete trees. Publish hash-identical copies of the two raw answer files and the selected adapter as convenient exports. Store relocation and cleanup records outside immutable bundles.

Constraints: No training, model loading or scorer execution. Keep the source paper and proposal. Keep GSM8K training code, weights, answers and evaluation. Delete only approved legacy recovery states and regenerable/temporary material. Do not copy the public Base weights; retain pinned revision and downloader. Downloads is outside the project cleanup scope.

- [x] Inventory all original files before moving or deleting; record relative path and size.
- [x] Move immutable MATH evidence to archive/math_l4 and current v4 scores to results/final; preserve ready.json and original identities unchanged.
- [x] Move GSM8K cloud results and corrected scores to archive/gsm8k; retain original training source and experimental evidence.
- [x] Group difficulty probes and their evaluation evidence under archive/difficulty_probes.
- [x] Export raw baseline/RL answer files to results/raw and selected LoRA parameters to models/rl_checkpoint384 with matching hashes.
- [x] Remove approved GSM8K optimizer/scheduler/RNG states, cached wheels/partial downloads, duplicate deployment payload, Python caches and Finder metadata.
- [x] Retain frozen training packages and scorer sources; add portable entrypoints and a hash-only project checker outside them.
- [x] Write README.md, ANALYSIS.md, paper provenance and memory.md with current paths, actual metrics and limitations.
- [x] Verify all immutable hashes, 6,400 paired answers, selected adapter, retained GSM8K inputs/results, relocation map and document links. Record exact deleted paths and before/after inventory.
