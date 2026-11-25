# Version history

## 2026-11-04 — math-l4-rl-split-1.0

- Added two independent 100-question RL test batches for the frozen checkpoint-384 selection; each retains all 16 samples and original per-question seeds.
- Preserved original experiment files and bound the scheduling extension explicitly in shard manifests. Added complete-grid merge validation and source provenance.
- Added incremental download, full selection-evidence verification, upload staging and separate commands for the known server and a second server placeholder.
- Added a Base-completion handoff for the existing machine, pinned environment preparation for the new machine, and matched RTX 4090/thread/software checks.
- Local checks use frozen data and CPU fixtures. Actual CUDA installation, remote handoff and model generation remain to be verified on the target machines.
- Known source scoring/admission issues remain recorded; this migration does not silently revise historical scores or checkpoint selection.


## 2026-11-04 — math-l4-rl-split-1.1

- Updated deployment topology: the original server finishes its current Base test and stops; two additional RTX 4090 servers run RL shards 0 and 1.
- Added watch_base_only.py and --stop-only mode. The controller can safely replace a matching old Base-to-RL watcher without restarting Base, and writes a durable stop receipt after complete Base validation.
- Added upload_base_stop.sh to transfer only external control scripts/references, preserving original code, models, active answers and run state.
- Updated upload/nohup/log/result-retrieval commands for all three machines; nohup logs append across restarts.
- Validation now covers 134 CPU scenarios, including real evaluator cache resume from 1598 to 1600 rows, zero generation for completed cache, stop-only behavior and repeated control-only uploads.
- evaluate_shard.py retains SHA256 2c181d6aacc06a7d34ebde71473d7e663b86983075757764e0912e9ef0a73564, preserving existing shard cache compatibility.
- No live cloud processes were signalled and no GPU inference or network transfer was executed locally. Resume is per complete saved answer; an unfinished answer is generated again.

## 2026-11-04 — math-l4-rl-split-1.1.1: Python pidfd compatibility

- User runtime evidence: conda-forge Python 3.11.15 on Linux 5.15.0-105-generic x86_64 exposes neither os.pidfd_open nor signal.pidfd_send_signal. The previous attribute gate exited before any process signal; the original Base evaluator was unaffected, but automatic stop was not armed.
- Added pidfd_compat.py, preferring native Python wrappers and otherwise using the same Linux x86_64 pidfd syscalls. Both controllers probe their own PID with signal 0 before monitoring or replacing anything. ENOSYS or permission failures still fail explicitly; there is no PID-only kill fallback.
- Updated the control-only upload filter to include the helper. The frozen experiment and evaluate_shard.py remain byte-identical, so saved answers and shard resume identities are unchanged.
- Local validation uses mocked kernel operations and process fixtures. No remote deployment, real Linux signals or GPU inference were performed by the assistant; the server must print [PIDFD READY] and [WAIT BASE] after uploading and restarting only the controller.
- Validation completed: 185 CPU checks (55 shard/resume, 45 transfer, 36 controller, 49 pidfd compatibility); shell/Python/documented commands parse; all 81 original frozen files and the shard evaluator hash remain unchanged.

## 2026-11-04 — math-l4-rl-split-1.2: direct server transfer

- Added upload_direct_tools.sh: sync the small independent shard code and four installation inputs to the existing Base server (port 42172) without model weights, active outputs, logs or restarts. The existing server becomes a file source; it continues only Base.
- Added pull_from_source.sh and a fixed 110-file inventory. Each new server pulls Base weights plus frozen experiment and all three checkpoint-selection receipts directly from the old server, then runs full build_payload validation. No local model relay or pre-existing local payload is needed.
- Destination uses the existing single-GPU experiment lock; active Base/RL answers and run state are excluded and existing destination results are preserved. SSH authentication remains interactive; new workers still run shard 0/1 with the existing nohup commands after [DIRECT READY].
- Added DIRECT_TRANSFER.md and updated the AI handoff. New 45 CPU direct-transfer checks passed; the existing 45 transfer checks passed again. The earlier 185 checks remain recorded; source, evaluator fingerprint and GPU protocol are unchanged. Cross-server SSH transfer and new-machine inference have not been executed by the assistant.
- User has confirmed the existing Base stop controller's pidfd probe and [WAIT BASE] logs after release 1.1.1. Leave it running and keep the old machine available until Base and both worker downloads finish.

## 2026-11-04 — worker endpoints confirmed

- User supplied worker A: root@xj-member.bitahub.com port 42369, shard 0; worker B: same host port 42186, shard 1. Existing Base/source server remains port 42172.
- Updated direct/relay command guides and portable handoff. Remote upload, installation and GPU execution on these endpoints remain unverified. Evaluation code and cache identity were not changed.

## 2026-11-04 — math-l4-rl-split-1.2.1: openrsync deployment path fix

- User executed the source upload but worker 42369 received only the old 45.47 KB controller directory and could not find pull_from_source.sh. Reproduced locally: macOS openrsync retains the complete absolute path prefix for --relative /absolute/code/./... sources and exits zero. Exact remote misplaced location was not inspected.
- Fixed upload_direct_tools.sh by changing cwd to the code directory and passing genuinely relative sources. Added local pinned-release/installer checks and remote sha256sum verification before [DIRECT TOOLS READY]. No runtime or model files are deleted; previously misnested remote copies are left untouched.
- Prior fixtures used hand-written relative arguments and missed the production absolute-source bug. Regression now derives source arguments from the production script and exercises the actual transfer paths with spaces and protected payload/output fixtures.
- Deploy this correction to source port 42172, then let worker 42369 retry its code pull. Worker 42186 remains shard 1. Current Base and the already running stop controller require no restart.
