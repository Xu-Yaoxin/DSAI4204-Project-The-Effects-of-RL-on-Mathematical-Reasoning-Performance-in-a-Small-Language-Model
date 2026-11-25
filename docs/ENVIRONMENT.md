# 训练环境与可复现范围

根目录 [requirement.txt](../requirement.txt) 是已使用环境的完整二进制依赖锁，包含 84 个包的精确版本和 SHA-256。它与 [requirements-lock.txt](../code/packages/linux-cp311-cu128/requirements-lock.txt) 字节相同；[manifest.json](../code/packages/linux-cp311-cu128/manifest.json) 另记录每个 wheel 的下载地址、目标平台和完整依赖闭包。`code/requirements-gpu.txt` 只是直接依赖说明，完整复现应使用根目录的锁文件和安装入口。

| 项目 | 固定要求／记录 |
|---|---|
| 操作系统与架构 | Linux x86_64，glibc ≥ 2.28；可使用 Ubuntu 22.04 |
| Python | 64 位 CPython 3.11；已有训练环境记录使用 3.11 系列 |
| GPU | 本次记录为 RTX 4090，BF16；MATH 流程一次使用一张卡 |
| NVIDIA 驱动 | ≥ 570.26，由服务器提供，pip 不安装驱动 |
| PyTorch / CUDA runtime | `torch==2.10.0+cu128` / CUDA 12.8 |
| Transformers / TRL | `transformers==5.3.0` / `trl==1.16.0` |
| PEFT / Accelerate | `peft==0.18.1` / `accelerate==1.10.1` |
| Datasets / Safetensors | `datasets==4.8.5` / `safetensors==0.8.0` |
| 可选加速内核 | 原训练未安装 flash-linear-attention、fla-core、causal-conv1d；不额外引入 |

在项目根目录执行：

```bash
RLVR_SYSTEM_PYTHON=python3.11 bash setup_environment.sh
```

安装入口核对根锁文件后，复用已保留的 GPU 安装代码，在 `code/.venv/` 创建环境；按原 URL 下载约 4.27 GB 的 84 个 wheel，校验哈希，再用 `--no-index --require-hashes` 安装。随后下载固定 Base、验证数据和 GPU，最后执行 MATH 预检。进度显示在终端；不会启动训练。网络断开后可重跑同一命令复用已验证的下载。`--offline` 仅适用于 wheel 和 Base 均已下载完整的机器。

安装使用的核心 pip 命令如下，执行前必须已准备 Python 环境和全部 wheel：

```bash
code/.venv/bin/python -m pip install --no-cache-dir --no-index \
  --find-links code/packages/linux-cp311-cu128/wheels \
  --require-hashes -r requirement.txt
```

Base 为 [Qwen/Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base)，固定 revision 为 `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`。文件清单和哈希见 [模型清单](../code/models/qwen3.5-2b-base-manifest.json)。环境已有时，可单独下载：

```bash
code/.venv/bin/python -u code/scripts/download_model.py
```

新预检结果位于 `code/math_l4_v1/outputs/environment/reproduction_preflight-<唯一标识>.json`，实际文件名会打印到终端，重复安装不会覆盖先前记录。MATH 预检默认只使用第 0 张可见 GPU；可通过 `CUDA_VISIBLE_DEVICES` 指定一张卡。原运行环境记录保存在 [supplementary/environment](../archive/math_l4/supplementary/environment/)，原 formal 报告与评测 manifests 还记录实际包版本、设备、线程和生成设置。

依赖锁固定的是已记录的软件包内容；驱动、硬件、并发负载和时间限制仍影响实际运行。项目提供原配置、种子、数据、模型身份和检查点证据，不能承诺不同机器上逐 token、逐浮点位完全相同。安装脚本此次只做静态／语法检查，未在当前 Mac 重新执行 Linux GPU 安装。
