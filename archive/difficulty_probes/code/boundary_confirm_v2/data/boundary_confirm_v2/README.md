# Fresh source-group calibration v2

200 verbatim, previously unused question texts:100 GSM-Symbolic main and100 original MATH test Level4 scalar text-only questions. Seed20261101. The two source groups were chosen after the prior screen; individual questions were not selected using model scores. This confirms source-group difficulty, not a formal Base/RL hypothesis test.

All previous boundary questions and complete GSM source families are excluded, as are known previously evaluated GSM8K test families and exact normalized train/dev/test questions. The GSM100 rows cover64 families:36 families contribute2 rows,28 contribute1. Treat shared-family responses as clustered. MATH uses all seven original-test subjects before fixed random selection; scalar and diagram filtering changes benchmark coverage.

Sources: https://huggingface.co/datasets/apple/GSM-Symbolic (CC-BY-NC-ND-4.0); https://huggingface.co/datasets/EleutherAI/hendrycks_math (full MATH mirror at pinned revision); https://github.com/hendrycks/math (original MIT license included separately). Source bytes, revisions, hashes, filters and history hashes are in manifest.json. Raw pools and licenses are in raw/. The full-MATH mirror does not retain original JSON filenames; source.original_id is explicitly a pinned mirror file/row locator, and family_id hashes normalized problem text.

Gold is extracted from the final balanced boxed/fbox source-solution answer and validated as an exact scalar by the frozen project scorer. No model is called. This is parsing validation, not a semantic review of200 source answers. No target accuracy is guaranteed before inference. Source gold, solutions and canaries never belong in generation prompts.

From the main code directory: python3 scripts/prepare_confirmation_data.py --verify. Preparation uses pyarrow already present in the GPU requirements; the uploaded inference package only needs questions.jsonl, manifest.json, questions.md and licenses, not raw parquet pools.
