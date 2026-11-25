# Version history

## 2026-11-01 · boundary-guarded-3.0

- User replaced same-GPU parallel evaluation with strictly sequential inference and requested a 300-second limit per response, with unfinished responses scored incorrect. Retained the explicitly selected native context allowance.
- Added live token traces, special-token-visible previews and an exact periodic token suffix guard. Fixed guard thresholds are evaluated every 32 tokens, minimum span 1,024, period up to 256, at least 4 repeats. No gold-aware early stop or sampling-policy change.
- Added an independent generation supervisor: 300-second answer deadline, 120-second no-token diagnosis, at most 15-second stop grace. Deadline results score zero; other guarded stops retain unknown mathematical correctness. Distinct end reasons and unknown bounds are reported.
- Token stream is checked against generate return values. Durable terminal rows are validated and fsynced before a forced exit75. The parent validates the cache before a bounded restart; complete shards need no restart. Runtime exceptions fail explicitly, preserving old samples.
- Reused the exact v2 dataset (200 questions, 800 attempts), model identity and scalar scorer. Old packages, old training outputs and earlier responses remain separate. This version starts a fresh stopping-policy cohort.
- Seven CPU test suites cover threaded streaming, subprocess hard-exit and resume, negative repetition controls, timeout score consistency, conservative unknown metrics, serial execution and live monitoring. Actual CUDA inference and timeout recovery remain untested locally.
- Cloud directory: /root/code/boundary_guarded_v3. SSH port: 42172. Reuses ../.venv and ../models/Qwen3.5-2B-Base; no added dependencies. Commands in COMMANDS.md.
