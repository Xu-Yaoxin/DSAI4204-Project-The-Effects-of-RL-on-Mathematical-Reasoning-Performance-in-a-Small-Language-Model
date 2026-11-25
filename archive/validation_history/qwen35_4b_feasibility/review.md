# Independent 4B memory and protocol audit

Scope: read-only inspection of the current 2B project and the official 4B configuration. No model weights were downloaded and no GPU allocation or training occurred in this audit. This is component accounting, not a measured or guaranteed peak.

## Architecture and current safeguards

- Official Qwen3.5-4B-Base text config: hidden size 2560, 32 layers, 8 full-attention and 24 linear-attention layers, intermediate size 9216, vocabulary 248320, tied word embeddings, BF16. Source: https://huggingface.co/Qwen/Qwen3.5-4B-Base/raw/main/config.json (consulted 2026-10-30; parent audit records a pinned revision).
- The project loader currently accepts only hidden size 2048 and 24 layers. The downloader is also pinned to the 2B repository. The current executable pipeline therefore does not yet support 4B by changing a directory name.
- Text-only loading excludes the visual encoder and MTP heads. For 4B, official full-checkpoint bytes are not the same as resident text-backbone bytes.
- LoRA covers language q_proj/k_proj/v_proj/o_proj plus gate_proj/up_proj/down_proj. Linear-attention in_proj_* and out_proj are currently excluded. Retain the same target-selection policy for a model-size comparison, and audit the resulting module list.
- Both official 2B and 4B use tied embeddings. Keep cast_lm_head_to_fp32=False (the pinned TRL default), preserve tied parameter identity, and verify that the tied tensor is counted once. There is no need for a separate reference model at beta=0.

## Known memory components

Generation uses batch 8 and at most 768 prompt + 512 completion tokens. At a conservative total context of 1280, the 4B cache component arithmetic is:

| Component | Bytes | GiB |
|---|---:|---:|
| full_attention_kv_bf16 | 335,544,320 | 0.312500 |
| linear_recurrent_fp32 | 402,653,184 | 0.375000 |
| linear_conv_bf16 | 12,582,912 | 0.011719 |
| total_component_bytes | 750,780,416 | 0.699219 |

These entries include full-attention key/value state and the linear-attention recurrent/convolution state. They exclude weights, LoRA state, context/allocator overhead, projection and attention temporaries, and autograd activations. They are not a lower/upper bound on total observed VRAM because allocation lifetime and allocator reservation differ across stages.

The pinned TRL 1.16.0 source unconditionally installs its fused/chunked LM-head path for training, even with use_liger_kernel=False. It scores only nonmasked completion tokens on CUDA and uses vocabulary tiles of 32768. With 512 completion tokens, one BF16 projection tile occupies 32 MiB. Do not assume the full batch-by-sequence-by-vocabulary logits tensor is retained during the current training path. Generation uses the original model forward.

Gradient checkpointing and training microbatch 1 reduce retained activations. They do not eliminate per-layer recomputation buffers. The Torch fallback for gated DeltaNet explicitly uses FP32 temporaries; GPU-specific kernel behavior still needs real measurement.

## Minimum preparation for a separate 4B experiment

1. Pin the official 4B revision and all required file hashes in a new manifest; preserve the existing 2B manifest and outputs.
2. Add explicit 4B model-profile acceptance to loading/readiness and validate the text-only checkpoint conversion, tied embeddings, tokenizer, and LoRA modules.
3. Keep the experiment settings initially fixed: BF16, rank 16, 8 answers/question, 32 answers/update, microbatch 1, max completion 512, same data/prompt/reward. Use a separate configuration and output tree.
4. Run a fresh 4B baseline and engineering/pilot tests. The 2B checkpoint and its completed pilot cannot initialize or authorize 4B formal training.

## Actual GPU acceptance evidence still required

- A fresh 4B load on an otherwise idle target GPU with the locked package stack; include actual device total/free memory and package/model identity.
- Generation with 8 concurrent answers, including the longest real prompts and a distinct capacity stress test reaching the declared 768+512 token bounds.
- Forward/backward and optimizer update using current GRPO semantics; finite metrics, LoRA parameter changes, unchanged frozen base, valid mixed-reward groups, and completed optimizer/checkpoint save.
- Record both torch.cuda.max_memory_allocated and max_memory_reserved, plus process/device memory. The existing pilot/report.json already supplies the PyTorch peaks for the 2B reference.
- Benchmark steady-state optimizer-step time and generation token throughput; do not extrapolate runtime from current utilization or parameter count alone.
- For deployment, test checkpoint restoration and evaluation as well as a training step. Formal training and evaluation must have enough capacity with useful remaining headroom, not merely fit one short prompt.

Verdict: the architecture, frozen-backbone training method, and component accounting make 4B a plausible single-4090 candidate. They do not establish a measured peak, a speed guarantee, or a passed target-GPU preflight.
