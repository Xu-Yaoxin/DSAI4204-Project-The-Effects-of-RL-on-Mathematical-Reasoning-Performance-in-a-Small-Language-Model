# DSAI4207 RLVR v1.2.2 · RTX 4090

完整说明见 [READM.md](READM.md)，版本记录见 [memory.md](memory.md)。数据已准备在 `data/`；GPU 安装包下载目录、完整版本锁和校验清单位于 `packages/linux-cp311-cu128/`。

当前位于课程目录，在 Linux x86_64、Python 3.11、NVIDIA 驱动 ≥ 570.26 的服务器执行：

```bash
bash project/code/setup_gpu.sh
bash project/code/run.sh all
```

第一条命令自动准备依赖、固定版本的官方 Qwen3.5-2B-Base、GSM8K 和固定划分，并检查 tokenizer、模型文件、数据及 GPU 通信；不需要先手动下载模型。终端显示阶段、依赖下载、模型下载和校验进度。只有全部通过才会显示 `✅ [SETUP COMPLETE]`，报告保存在 `outputs/environment/setup-XXXXXXXX/ready.json`。第二条命令开始真实工程检查和主实验。

启动器自动使用 1/2/4 张可见 GPU；也可用 `--gpus 1`、`--gpus 2` 或 `--gpus 4`。训练采用 DDP，各卡保留模型副本；全局每次更新固定 4 道题。评估在第一张卡上串行执行。

补齐或检查安装包、运行 CPU 检查：

```bash
bash project/code/download_packages.sh
bash project/code/download_packages.sh --verify
bash project/code/run.sh check
```

已经进入 `project/code` 时去掉命令中的 `project/code/` 前缀。下载中断可重跑第一条命令继续；已有文件与固定清单冲突时会停止并保留文件。`--offline` 要求完整的依赖包、模型和原始数据及来源记录。真实 GPU 训练、峰值显存和速度仍待服务器实测。

2026-10-30 v1.2.1 审查修复了梯度重复缩放、安装前置检查顺序和评估缓存溯源检查。原验证证据见 `outputs/validation/final_2026_10_30/`；v1.2.2 安装器验证见 `outputs/validation/v1_2_2/`。
