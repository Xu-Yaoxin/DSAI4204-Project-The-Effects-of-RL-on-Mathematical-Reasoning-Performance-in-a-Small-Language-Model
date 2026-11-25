# Dataset provenance

- `math_source_tree.json`: recorded source tree at fixed public revision `21a5633873b6a120296cce3e2df9d5550074f4a3`; its 14 LFS SHA-256 values and byte sizes bind every original parquet file.
- `history_exclusions.json`: frozen prior question-text hash set, with hashes and row counts of the 46 source files used to construct it. Evaluation outcomes are not selection inputs. Original history files need not be uploaded to use the frozen ledger.
- `gold_parser.py`: the earlier exact-rational parser used only to filter source boxed answers. Keeping this parser fixed prevents answer-presentation fixes from silently changing the eligible dataset pool. Runtime reward/evaluation uses the new scorer in `scripts/verify_answers.py`.
- `eligible_pool.jsonl`: all 1,830 eligible, previously unseen exact question texts, with pinned file/row source identity and canonical scalar gold. No model output is used in this pool.

`data/processed/manifest.json` binds all these files, the preparation script, all original sources, each selected split, deterministic subject quotas and the source filtering policy. `scripts/doctor.py` checks the source bytes, canonical gold, official split membership, source-pool bindings, history exclusion, no train/dev/test overlap, nested training subsets and deterministic selection. Its normal data-only mode does not import model packages or pyarrow.

The mirror supplies file/row positions, not original MATH JSON filenames. The schema labels this explicitly. Raw source solutions remain in the parquet provenance; selected JSONL rows only carry the canonical answer and original boxed scalar, never teacher reasoning.

The source filter is a parsing and provenance check, not a complete mathematical or semantic audit. Exact-text family IDs cannot establish independence of all mathematically similar problems.
