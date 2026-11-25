# Boundary probe version history

## 2026-10-31 · boundary-1.0

- Baseline only: original BF16 Qwen3.5-2B-Base, pinned content identity; no adapter or training.
- 100 fixed questions: GSM-Symbolic main/P1/P2 and scalar MATH-500 Levels1-3/4-5, 20 per group. Selection did not use model outputs.
- Two independent processes on one GPU, each assigned 10 questions per group; four responses per question, 400 in total. Independent durable progress; exact full-grid merge; group uncertainty and diagnostic reports.
- Shared launch lock, CPU thread limits, GPU/container memory headroom checks, child failure cleanup, complete-cache verification without CUDA, and sequential continuation of the same shards.
- Defaults: temperature0.6, top_p0.95, top_k0, BF16, max_output_tokens1024. This length budget differs from the original 512-token experiment.
- 24 local CPU suites passed; actual evaluator-to-merger-to-analyzer integration used mocked generation. GPU inference, GPU peak usage and parallel speedup remain unmeasured.
- Packaged runtime uses ../.venv/bin/python and ../models/Qwen3.5-2B-Base. All scripts/scorer/data are frozen here; original training files are untouched.
- Exploratory group calibration only. Future formal comparison requires fresh questions/template families, not selecting individual failures from this set.

## 2026-10-31 · SSH connection update

- User updated the server SSH port from 42258 to 42260: `ssh -p 42260 root@xj-member.bitahub.com`.
- Upload/download instructions updated. Remote project remains `/root/code/boundary_probe_v1/`; no runtime code or data changed.
