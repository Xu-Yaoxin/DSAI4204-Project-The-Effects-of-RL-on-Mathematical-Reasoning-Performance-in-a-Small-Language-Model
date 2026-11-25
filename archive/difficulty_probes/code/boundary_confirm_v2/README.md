# Baseline confirmation v2 / 基线难度复核

This isolated package evaluates the unchanged Qwen3.5-2B-Base. It does not train or load an RL adapter. Place it at `/root/code/boundary_confirm_v2`, beside the existing `/root/code/.venv` and `/root/code/models`.

## Frozen protocol / 固定设置

- 100 GSM-Symbolic **main** questions and 100 original MATH **test Level 4** questions.
- Only exact scalar numerical answers, and no required diagrams. This is a selected numeric subset, not the complete MATH Level 4 benchmark.
- GSM uses 64 previously unused template families, with at most two instances per family. Confidence intervals resample families rather than treating all instances as independent.
- Previously evaluated question texts and GSM families are excluded. No model response was used to choose individual questions. The two source groups were chosen after the earlier exploratory run; this remains calibration data, not a final held-out training comparison.
- Four independent samples per question: **800 responses**. The main result is mean single-sample accuracy, `pass@1`; `pass@4` is secondary.
- Same BF16 base weights, prompt, temperature 0.6, top-p 0.95, and per-question deterministic sampling. No adapter, quantization, repetition penalty change, or early stop after an answer marker.
- **Native context budget:** `generation_limit = 262144 - actual_prompt_tokens`, including prompt special tokens. The loaded model's context setting is checked. EOS can finish a response earlier. The per-response limit is saved alongside `prompt_tokens`, `output_tokens`, and `truncated`.
- New standalone scorer v2 handles previously identified extraction issues. The original training scorer and archived scores remain unchanged. Because both output budget and scorer changed, do not interpret a difference from the earlier 1,024-token run as a training effect.

Scoring remains conservative: percent suffixes and units in an explicit final scalar are rejected. Extraction failures are reported separately and must not be read as proof of a mathematical error. A post-hoc replay of the previous 400 responses recovered 17 correct finals and rejected one previously accepted percent-formatted response; those archived official scores were not changed.

## Launch / 执行

Existing GPU environment and model are reused. No package or model downloads are needed.

```bash
cd /root/code/boundary_confirm_v2
bash run_confirmation_parallel.sh --plan
nohup bash run_confirmation_parallel.sh > confirmation.log 2>&1 < /dev/null &
echo $! > confirmation.pid
tail -n 50 -f confirmation.log
```

Default: two independent workers on the same visible GPU, one response at a time per worker, four CPU threads per worker where quota permits. Startup requires at least 20 GiB free GPU memory and 32 GiB available system RAM. These are conservative admission thresholds, not proof that every native-length generation fits. The 262k context path has not been executed on a local GPU.

If parallel execution fails for GPU memory, allow the supervisor to stop, inspect the worker logs, and restart the **same frozen task** sequentially:

```bash
cd /root/code/boundary_confirm_v2
nohup bash run_confirmation_parallel.sh --sequential >> confirmation.log 2>&1 < /dev/null &
echo $! > confirmation.pid
```

This keeps the same two shards, question seeds and output directory. It does not shorten the output budget. Never run two supervisors at the same time.

## Progress and resume / 进度与续跑

```bash
cd /root/code/boundary_confirm_v2
tail -n 50 -f confirmation.log
tail -n 30 -f outputs/confirmation_v2_parallel/worker_0.log outputs/confirmation_v2_parallel/worker_1.log
```

The parent reports completed answers every 15 seconds. Each worker prints `[ANSWER START]` with the question, sample and exact budget before generation. A repeated answer can run for hours before producing the next saved sample; the original short-run ETA is not a valid prediction for this run. Native context is finite and does not guarantee zero truncation or eliminate repetition.

Each completed response is flushed and synced to disk. Running the same command resumes verified completed responses; an interrupted response starts again with the same seed. It does **not** resume a partially generated token stream. Completed outputs are hash-verified before reuse. Model, data, scorer, protocol or sample-count changes are rejected when resuming.

To stop the supervisor and its two workers cleanly:

```bash
cd /root/code/boundary_confirm_v2
kill -TERM "$(cat confirmation.pid)"
```

The supervisor sends termination to the worker process groups and waits for them. Preserve all partial results. A hard interruption during a JSONL write is detected rather than silently repaired.

## Results / 结果

All 800 responses must finish and pass integrity checks before merge and analysis:

- `outputs/confirmation_v2_parallel/analysis/report.md`: group results and interpretation limits.
- `outputs/confirmation_v2_parallel/analysis/summary.json`: estimates and confidence intervals.
- `outputs/confirmation_v2_parallel/base/samples.jsonl`: all question texts, raw responses, scores, token counts, budgets and timings.
- `outputs/confirmation_v2_parallel/worker_0.log`, `worker_1.log`: diagnostic logs.
- `data/boundary_confirm_v2/manifest.json`: fixed sources, selection seed, exclusions and dataset hash.

Use the source-group pass@1 estimates and uncertainty to choose the next experiment. Select fresh, disjoint questions for future training/dev/final evaluation; do not select individual questions because the baseline happened to fail them in this run.

## Local verification / 本地检查

```bash
python3 -S tests/run_all.py
bash run_confirmation_parallel.sh --plan
```

The runtime tests use isolated CPU fixtures and mocked model generation. They validate per-prompt context budgets, score replay, interruption/resume, disjoint merges, resource gates and sibling process cleanup. They do not establish actual GPU memory, speed or model accuracy at native context. See `validation/release_check.json` for the checks performed on the frozen package.
