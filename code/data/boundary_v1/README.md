# Baseline calibration data v1

100 verbatim questions in five groups of20. Only baseline inference is intended. Seed20261031; selection did not use model outputs.

GSM-Symbolic main/P1/P2 use the same20 source families and matching instance indices, excluding known previously evaluated GSM8K test families. MATH-500 uses scalar, text-only questions with level quotas6/7/7/10/10. Non-scalar answers and diagrams are excluded before random selection. No40-50% accuracy is promised before measurement.

Questions: questions.md. Exact scoring gold/source answers: questions.jsonl. Provenance, exclusions, hashes and limitations: manifest.json. Original files and source licenses: raw/.

Sources: https://huggingface.co/datasets/apple/GSM-Symbolic (CC-BY-NC-ND-4.0); https://huggingface.co/datasets/HuggingFaceH4/MATH-500 (derived from OpenAI PRM800K MATH split; original MATH repository MIT license included separately). The MATH source card does not declare a separate license. Questions are preserved unchanged; only metadata and gold canonicalization are added.

Run from code: python3 scripts/prepare_boundary_data.py --verify. Verification is offline and reruns deterministic selection. Do not pass source solutions, gold answers or canary metadata into the model prompt.
