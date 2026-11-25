Managed directory for the official `Qwen/Qwen3.5-2B-Base` Transformers snapshot.

Run `bash setup_gpu.sh` from the code directory on the Linux GPU server. It downloads the official BF16 snapshot from `Qwen/Qwen3.5-2B-Base` at revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` into this directory, with progress and file verification against `../qwen3.5-2b-base-manifest.json`.

Required: `config.json`, tokenizer files, `.safetensors` weights and the shard index when sharded. An existing matching snapshot is reused; conflicting complete files are preserved and rejected. Only the official snapshot passes the setup manifest. Scripts hash the checkpoint content before every real run. `source_manifest.json` records successful verification; an incomplete or failed download is not ready for training.

No weights are included in this project. See `../../READM.md` for the workflow.
