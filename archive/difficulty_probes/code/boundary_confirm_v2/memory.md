# Version history

## 2026-11-01 — boundary-confirm-2.0

User requested one more baseline difficulty calibration: 100 GSM-Symbolic main and 100 MATH Level 4 questions, with the strict native output allowance. The user explicitly selected `262144 - actual prompt tokens` instead of a 32,768-output-token ceiling.

This isolated package reuses the existing BF16 Qwen3.5-2B-Base and Python environment. It loads no trained adapter. Four seeded responses per question produce 800 responses. Two independent processes share one GPU; a sequential option retains the same shards and sampling identities.

Data selection uses seed 20261101. GSM has 64 previously unused original families, with 28 represented once and 36 twice. MATH comes from the full fixed-revision original-test mirror, because the remaining eligible MATH-500 Level 4 pool contained only 77 rows. The full mirror offers 832 unseen Level 4 scalar-answer, text-only candidates. Exactly 100 are sampled, preserving source question text. The main dataset manifest records source revision, raw hashes, exclusions and selection provenance. Source-defined Level 4 is not a guarantee of difficulty for this model.

The per-answer generation limit is recorded and checked by generation, cache reuse, merge and analysis. Only EOS or exhaustion of the native context ends normal generation; no answer-marker or repetition early stop was introduced. Completed answers are synced individually. An interrupted in-flight answer restarts from its fixed seed.

The independent v2 scorer repairs known extraction cases: final numerical answer labels, complete paired think spans, empty trailing hash markers, multiline boxed display answers and Unicode currency signs. Contradictory answers and malformed wrappers remain rejected. Numeric equality uses exact rational arithmetic. It does not assess the correctness of the reasoning. The original production scorer, training code, v1 package and downloaded archives are preserved.

Validation uses CPU-only fixtures and the frozen local dataset. GPU native-context memory, latency and accuracy remain unmeasured. The package includes static capacity evidence and an executable CPU test runner. Read `validation/release_check.json` for the actual release checks and `validation/test_results.txt` for their output.

Deployment target at preparation time: `root@xj-member.bitahub.com`, SSH port `42172`, package path `/root/code/boundary_confirm_v2`, environment `../.venv`, model `../models/Qwen3.5-2B-Base`.
