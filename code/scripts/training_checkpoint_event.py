"""Seal a completed Trainer checkpoint after every rank has saved its RNG state."""
from pathlib import Path
import runpy

if "CHECKPOINT_EVENT" not in globals():
    raise SystemExit("Internal script: called by the training on_save callback.")
event = CHECKPOINT_EVENT
trainer = event["trainer"]
state = event["state"]
CONTROL = event["control"]
torch = event["torch"]
distributed = event["world_size"] > 1
if distributed:
    torch.distributed.barrier()
packet = [None]
if event["rank"] == 0:
    try:
        expected_step = state.global_step * event["accumulation_per_rank"]
        if trainer._step != expected_step or expected_step % event["steps_per_generation"]:
            raise RuntimeError("Recovery checkpoints must fall on complete GRPO rollout/update boundaries.")
        sealed = runpy.run_path(str(Path(__file__).with_name("training_resume.py")),
            init_globals={"RESUME_REQUEST": {"operation": "seal", "output_dir": event["output_dir"],
                "binding": event["binding"], "step": state.global_step,
                "extra_state": {"grpo_step": trainer._step, "optimizer_step": state.global_step}}})["RESUME_RESULT"]
        packet[0] = {"checkpoint": sealed["checkpoint"]}
    except Exception as error:
        packet[0] = {"error": f"{type(error).__name__}: {error}"}
if distributed:
    torch.distributed.broadcast_object_list(packet, src=0)
if packet[0] is None or "error" in packet[0]:
    raise RuntimeError(f"Checkpoint sealing failed: {packet[0]}")
if event["rank"] == 0:
    print(f"[CHECKPOINT SAVED] step={state.global_step}/{state.max_steps} | "
          f"{100 * state.global_step / state.max_steps:.1f}% | {packet[0]['checkpoint']}", flush=True)
