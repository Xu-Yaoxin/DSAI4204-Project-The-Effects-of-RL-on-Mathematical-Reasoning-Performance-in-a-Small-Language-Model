# DSAI4207 RLVR v1.4.0 · RTX 4090 / 本地离线复核

**当前 MATH Level 4 最终结果入口（2026-11-06）：[FINAL_RESULTS.md](outputs/FINAL_RESULTS.md)。** 原始 Base/RL 回答和最终 v4 评分的完整地址均在该索引中，评分目录为 `outputs/rescore_20261106/`。本文件后文的 `outputs/rescore_v1_4/` 与 `rescore_local.sh` 属于之前的 GSM8K 实验。

## 100 题 Baseline 难度摸底 / Boundary probe

新增独立评测包 `boundary_probe_v1/`：5 个题组各 20 题，只测原始 Base；支持同一张 4090 上两个进程分片生成、断点恢复、完整合并与分组成功率统计。服务器现有环境与模型可直接复用。上传、nohup、日志和结果下载命令见 [README_BOUNDARY.md](README_BOUNDARY.md)。本地 GPU 推理与实际加速尚未验证。

本项目使用 Qwen3.5-2B-Base，比较答案奖励强化学习前后的数学单次成功率、有限采样覆盖和结构变化后的表现。方法是可验证奖励强化学习（RLVR）+ 组相对策略优化（GRPO）+ 低秩适配（LoRA）。主评测每题 16 次，依据上一级 `DSAI4207_LLM_Proposal_Overleaf(4).pdf`。

**当前状态：下载的 v1.2.2 云端实验已完成并通过归档完整性复核。v1.4 修复判分器，新增无需模型的本地重评分。** 旧实验的训练奖励和 checkpoint-384 选择保持原样；新评分属于事后敏感性分析（posthoc sensitivity analysis）。v1.3 的真实 GPU 中断恢复仍未实测，不能用旧版完整训练代替恢复验证。

## 本地一键复核：不加载模型

在本地 `project/code` 目录运行：

```bash
bash rescore_local.sh
```

只需本机 Python 3.10+；已在 Python 3.11 验证。入口使用 `python3 -S`，不加载 site-packages，不需要安装 PyTorch、下载权重、使用 GPU 或调用模型 API。可用 `RLVR_PYTHON=/path/to/python3 bash rescore_local.sh` 指定本机解释器，不使用从 Linux 复制的虚拟环境。

命令先运行判分、训练奖励接口、统计和离线复核的回归测试，再读取相邻目录 `../server_results/run_20261031/code/`。核对数据、生成记录、原判分、冻结选点和 adapter 哈希后，用同一规则重评 Base/RL 的 6,400 条测试回答及三个开发候选的 2,400 条回答。标准答案来自原 GSM8K 数据；没有生成新回答。

默认结果位于 `outputs/rescore_v1_4/`：

- `main_analysis/report.md`、`summary.json`：pass@1/2/4/8/16 和 2,000 次按题配对 bootstrap。
- `changed_scores.jsonl`、`diagnostics.json`：逐条判分变化及汇总；每条重评回答保留 `original_score`。
- `selection_sensitivity.json`：全部开发候选的新旧得分。主测试仍使用原冻结选点，不自动换模型。
- `provenance.json`、`ready.json`：原生成与新判分各自的身份、输入/输出哈希和完成状态。

完整相同结果会校验后复用；输入、判分器或选项发生变化时拒绝覆盖。未完成的输出保留供排查，需要用新目录重跑，例如：

```bash
bash rescore_local.sh --output outputs/rescore_review_2
```

归档放在其他位置时可传 `--archive /absolute/path/to/downloaded/code`。原归档不被改写；本地无需基础模型权重，但须保留已下载的候选 adapter 文件用于身份校验，不会加载它们进行推理。

只测试代码、不重评分真实回答：

```bash
python3 -S tests/run_all.py
```

完整测试报告写入 `outputs/validation/v1_4/test_report.json`。数值判分比较已保存的最终数值与标准答案，例如 `1/2` 与 `0.5` 等价；它不验证整段推理逻辑。新规则也不会追溯修正旧训练收到的奖励。

## 1. 云端启动与升级

以下服务器命令默认已进入 `/root/code`，无需额外的 `project` 父目录。本地代码仍位于课程目录的 `project/code`。上传时保留 `data/`、`configs/`、`scripts/`、`tests/`、`models/` 和 `packages/` 中的版本锁与清单，不从 Mac 复制 `.venv`。

### v1.4 与旧训练的边界

本次本地复核不需要上传或启动云端任务。`deploy_resume.sh` 是此前 v1.2→v1.3 的训练恢复升级入口，会停止旧任务；不要为重评分执行它。

v1.4 改变了判分规则。旧训练的检查点、pilot 验收、冻结选点和评测缓存不可直接混入新训练；源码与配置身份检查会拒绝混用。将来验证新奖励下的训练时，须使用新运行目录，重新经过工程、pilot、正式训练和开发集选点，例如 `bash run.sh all --run-dir outputs/v1_4_fresh`。本次没有启动该训练。

### 后台启动或恢复

在服务器执行：

```bash
cd /root/code
bash resume_run.sh
```

该命令用 `nohup` 和独立会话运行 `bash run.sh all --resume`。没有现成结果时从头执行；已有结果时检查并复用已完成阶段，为中断评测补齐样本，为训练读取最近一个完整检查点。发现项目仍在运行时会拒绝再次启动，避免两个任务写入同一目录。关闭本地电脑或断开 SSH 不影响已经启动的云端进程。

每次运行保留独立的 `logs/train-*.log`。查看当前日志：

```bash
tail -n 50 -f /root/code/train-current.log
```

兼容入口默认输出仍为 `outputs/v1_2/`；版本相同且身份校验通过才可恢复。新 v1.4 训练必须明确指定独立 `--run-dir`，不能复用旧版评分缓存。主流程依次执行代码检查、数据准备、工程采样与训练、pilot、正式训练、开发集选点、Base/RL 测试和统计绘图。任一步失败即停止。结构题须独立审阅后单独运行 `structural`，不包含在主流程中。

### 新服务器首次安装

服务器需要 Linux x86_64、64-bit CPython 3.11（含 venv/ensurepip）、glibc ≥ 2.28、NVIDIA 驱动 ≥ 570.26，并能访问依赖和模型下载站点。服务器的 Python 3.11 不在默认位置时，用 `RLVR_SYSTEM_PYTHON` 指定现有解释器。

```bash
bash setup_gpu.sh
bash resume_run.sh
```

安装脚本完成系统与虚拟环境检查、84 个锁定依赖的下载校验与安装、CUDA BF16 运算、官方模型下载、GSM8K 准备，以及 tokenizer、模型文件布局和单卡/多卡通信检查。模型下载到 `models/Qwen3.5-2B-Base/`，无需手动激活 `.venv`。

安装进度分为 7 个阶段；依赖、模型下载及校验另有字节进度。阶段百分比表示完成步骤，不是剩余时间估计。下载中断可重跑，已校验文件会复用；完整文件若与固定清单冲突，则保留并报错。

只有全部通过才会显示 `✅ [SETUP COMPLETE]`，并保存 `outputs/environment/setup-XXXXXXXX/ready.json`。该记录证明准备检查通过；正式模型训练和显存需求仍由工程阶段验证。

本机检查或预览命令，在本地 `project/code` 目录执行：

```bash
bash run.sh check
bash run.sh prepare
bash run.sh all --resume --gpus 4 --plan
```

`check` 运行 CPU 测试并报告环境状态；`prepare` 校验并复用现有数据；`--plan` 只打印计划，不加载模型或启动训练。

## 2. 数据已经放在哪里

官方数据来自 [OpenAI GSM8K 仓库](https://github.com/openai/grade-school-math)，固定 commit 为 `3101c7d5072418e28b9008a6636bde82a006892c`。

| 路径，均相对 `code` 根目录 | 内容 |
|---|---|
| `data/raw/gsm8k/train.jsonl` | 官方训练集，7,473 题 |
| `data/raw/gsm8k/test.jsonl` | 官方测试集，1,319 题 |
| `data/raw/gsm8k/LICENSE` | 官方 MIT 许可证 |
| `data/raw/gsm8k/source_manifest.json` | commit、来源 URL、题数、文件大小与 SHA256 |
| `data/processed/train.jsonl` | 正式训练，512 题 |
| `data/processed/pilot.jsonl` | 预实验，128 题，属于正式训练子集 |
| `data/processed/engineering.jsonl` | 工程检查，32 题，属于预实验子集 |
| `data/processed/dev.jsonl` | 开发集，200 题，从官方训练集独立留出 |
| `data/processed/test.jsonl` | 主测试集，200 题，来自官方测试集 |
| `data/processed/manifest.json` | 划分种子、来源、题号、哈希与重叠检查 |
| `data/processed/structural.jsonl` | 自建结构题 160 道，不是下载的 GSM-Symbolic |

原始答案推理只保留在 raw 数据中。训练和评测提示仅包含题目；规范化数值答案只传给判分器。

下载器验证 HTTPS，不关闭证书检查；本机 Python 缺少默认 CA 时可使用系统 CA。文件采用临时文件后替换的写入方式，已存在且哈希正确的文件直接复用；部分下载可补齐，内容冲突会报错。数据准备默认要求可核查的来源和许可证；`--allow-unverified-source` 仅供离线合成测试，不可用来伪装正式实验数据。

重复运行 `prepare` 会核对相同输入、种子与代码生成的结果。变更数据规模、种子或判分规则时，使用新配置和新的 `data.output_dir`，不覆盖旧划分。规范化文本重叠检查不能识别所有语义改写或预训练污染。

## 3. 模型和 GPU 环境

`setup_gpu.sh` 调用 `scripts/download_model.py`，将官方原始模型快照下载到 `models/Qwen3.5-2B-Base/`，包含 `config.json`、完整 tokenizer 文件、safetensors 权重和分片索引。本机交付目录不含大模型权重，下载在服务器执行；训练和评测仍只读取本地模型。

模型固定为官方 `Qwen/Qwen3.5-2B-Base` 原始 BF16 快照，revision 为 `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`。清单 `models/qwen3.5-2b-base-manifest.json` 来自该版本的官方 API，包含 12 个文件，共 4,571,206,192 bytes（约 4.57GB）。LFS 文件按官方 SHA256 校验，普通 Git 文件按 blob SHA1 校验；全部通过后在模型目录保存 `source_manifest.json`。Instruct、GGUF、GPTQ、Ollama 导出包不属于当前实验。训练程序另外计算文件内容身份，防止 Base 和 RL 混用不同底座。

目标环境固定为 **Linux x86_64、Python 3.11、glibc ≥ 2.28、NVIDIA 驱动 ≥ 570.26**。建议使用服务器现成的 Python 3.11 环境；可以通过 `RLVR_SYSTEM_PYTHON` 指定其解释器。当前 Mac 只用于开发和准备文件，安装脚本会拒绝在 Mac 上安装 Linux GPU 包。

安装包位于 `packages/linux-cp311-cu128/`：`manifest.json` 记录来源、目标平台、文件大小与 SHA256，`requirements-lock.txt` 固定完整版本与哈希，`wheels/` 保存 Linux wheel。锁定的 84 个 wheel 共约 3.98GiB。默认安装会复用完整文件并补齐缺失下载。离线安装要求完整的 wheel、模型文件、GSM8K 原始数据及来源清单；仅下载完依赖包还不够：

```bash
bash setup_gpu.sh --offline
```

如果安装包缺失，可在有网络的机器补齐，再复制到服务器：

```bash
bash download_packages.sh
```

默认 `setup_gpu.sh` 先验证虚拟环境、venv/ensurepip 和 pip，再下载缺失的包。`--offline` 会传给模型和数据准备程序，缺文件就停止，不会联网补齐；该模式可根据完整原始数据重新生成固定划分。安装器创建独立 `.venv`，核对哈希、依赖关系与实际版本。`RLVR_PYTHON` 可指定启动实验时的解释器。系统 NVIDIA 驱动不包含在 wheel 包内，需要服务器提供。

核心版本为 PyTorch 2.10.0 CUDA 12.8、Transformers 5.3.0、PEFT 0.18.1、TRL 1.16.0、Accelerate 1.10.1。依赖解析与文件校验不等于 GPU 上的安装或运行验证。Qwen3.5 使用 Transformers 的 PyTorch 线性注意力实现；可选 FLA/causal-conv1d 加速内核未作为默认依赖，避免不匹配的编译包改变环境。不要在一次已开始的实验中途切换内核。

4090 每卡 24GB，分布式数据并行（Distributed Data Parallel，DDP）在每张卡保留一份模型，显存不会相加。启动器尊重服务器的 `CUDA_VISIBLE_DEVICES`，自动选择不超过可见数量的 1、2 或 4 张卡；3 张时使用 2 张，超过 4 张时使用 4 张，以保持固定的全局更新批量。也可显式选择：

```bash
bash run.sh engineering --gpus 1
bash run.sh engineering --gpus 2
bash run.sh engineering --gpus 4
```

上述命令是三种替代运行方式，不要依次写入同一目录。不同硬件实验使用不同 `--run-dir`。例如仅使用服务器的第 2、3 张卡：

```bash
CUDA_VISIBLE_DEVICES=1,2 bash run.sh all --gpus 2 --run-dir outputs/two_gpu_run
```

每个训练阶段先进行小规模 BF16 运算和 NCCL 通信检查，再加载模型。也可以单独执行硬件检查，不加载模型或读取训练答案：

```bash
bash run.sh gpucheck
```

RTX 40 系列多卡采用保守的通信默认值，关闭 NCCL P2P 和 InfiniBand 通道；已有显式环境变量会被保留。报告记录实际 GPU、驱动、CUDA、通信设置与逐卡显存。硬件检查只验证小运算和通信；真实训练是否能装入 24GB 仍以工程阶段为准。多卡时开发集选择与正式采样仍串行使用第一张选中 GPU，保证评估随机种子和采样流程一致。

模型加载使用官方文本接口；不加载视觉塔参与数学训练。缺失文本权重、分片缺失或架构不匹配会报错。LoRA 覆盖实际存在的语言层 `q/k/v/o_proj` 和 `gate/up/down_proj`，线性注意力的其他投影保持冻结。具体参数名称和数量由运行时审计保存。

## 4. 训练配置与卡数

默认保持无 KL（`beta=0`）、每题 8 条回答、每次更新 4 道不同题，共 32 条回答。多卡只分配计算，不扩大这个批量：

| 选中 GPU 数 | 每卡 microbatch | 每卡梯度累积 | 每卡每批生成回答 | 全局回答/更新 | 全局题目/更新 |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 32 | 8 | 32 | 4 |
| 2 | 1 | 16 | 8 | 32 | 4 |
| 4 | 1 | 8 | 8 | 32 | 4 |

`steps_per_generation=8`，每张卡每批收到同一道题的完整 8 条回答。各卡的奖励审计分别落盘，主进程合并记录；模型保存和公共报告由主进程负责。程序核对完整题组、每次更新的不同题数、更新总步数和题目覆盖。

v1.2.1 修正了固定依赖组合中的梯度重复缩放：GRPO 已按累积步数规范化损失，Accelerate 不再重复相除。Trainer 的实际微批次、同步、优化器更新边界保持不变。程序检查修正前后的状态，并在训练清单与报告中记录 `loss_normalization`；升级依赖前必须重新验证。

当前配置为 `configs/v1.json`，版本号 1.4.0；`configs/archive/v1.3.0.json` 保留升级前配置。模型、数据和训练数值超参数未变；判分器的提取规则已修正，因此奖励协议发生变化。`training.recovery_save_steps=16` 沿用 v1.3，控制恢复检查点保存频率。无 KL 主候选参照 Yue et al. 控制实验，仍有模型规模、LoRA、长度和数据量差异。

LoRA rank 16、alpha 32、学习率 `5e-6`、训练温度 1.0、3 个 epoch、`loss_type="grpo"`、`scale_rewards="group"`、裁剪阈值 0.2、`num_iterations=1` 保持不变。训练与评测的生成长度上限仍为 512 tokens，须通过截断率判断是否调整。这些是预实验起点，没有实测证据说明已经最优。

训练不使用 SFT、教师推理或格式奖励；只训练 LoRA 参数。预实验和正式训练都从同一 Base 新建 adapter，正式训练不接着训练 pilot adapter。

## 5. 分阶段执行与输出

默认运行目录为 `outputs/v1_2/`。各阶段可单独启动，前置结果缺失时会直接报错。

| 命令后缀 | 工作 | 前置条件 |
|---|---|---|
| `check` | CPU 测试和环境报告 | 不需要模型 |
| `prepare` | 下载或校验原始数据、生成划分与结构题 | 首次下载需要联网 |
| `gpucheck` | BF16 运算与多卡通信检查 | 已安装 GPU 依赖，不需要模型 |
| `engineering` | GPU 检查、Base 工程采样和 2 步 GRPO | 已准备数据、GPU、模型 |
| `pilot` | 128 题预实验 | 已准备数据、GPU、模型 |
| `formal` | 512 题正式训练 | 相同协议下通过的 pilot 报告 |
| `select` | 开发集选择检查点 | 已完成且验收通过的 formal |
| `evaluate` | Base/RL 主测试采样 | 已冻结的 selection |
| `analyze` | 配对统计和 2×2 图 | 两组完整测试样本 |
| `structural` | 结构题采样、统计和图表 | selection 和完整独立审阅记录 |
| `all` | check 到 analyze 的主流程 | 会进行真实 GPU 工作 |

完整阶段命令示例：

```bash
bash run.sh engineering
bash run.sh pilot
bash run.sh formal
bash run.sh select
bash run.sh evaluate
bash run.sh analyze
```

运行另一个配置时，配置文件位置和输出目录均相对 `code` 根目录：

```bash
bash run.sh pilot --config configs/v1.json --run-dir outputs/another_run
```

### 训练断点恢复（checkpoint resume）

前台运行完整恢复流程，或只恢复一个训练阶段：

```bash
bash run.sh all --resume
bash run.sh formal --resume
```

这两条是替代用法，不要同时执行；通常使用第 1 节的 `bash resume_run.sh` 后台入口。`engineering` 和 `pilot` 也支持 `--resume`。不加该参数时，训练仍要求新输出目录。

| 当前状态 | `--resume` 的处理 |
|---|---|
| 尚未创建训练目录 | 从同一 Base 新建 LoRA adapter 开始训练 |
| 存在完整、已封存的检查点 | 校验后从最近的检查点继续 |
| 已中断但尚无完整检查点 | 归档旧尝试，从第 0 步重新开始 |
| 阶段已完成且验收通过 | 检查配置、状态和产物后跳过重复训练 |
| 配置、源码、数据、模型、依赖或 GPU 拓扑变化 | 保留现有文件并拒绝恢复 |
| 旧 v1.2 的训练目录 | 不能直接恢复；由升级工具归档后重新训练 |

默认 engineering 每个优化器步保存，pilot/formal 每 16 个优化器步保存。检查点包含 LoRA 权重、Trainer 状态、优化器、调度器、各 rank 的随机状态和对应奖励审计快照。所有 rank 保存完成后才计算文件哈希并写入 `resume_state.json`；只有该封存记录完整且校验通过的检查点可用于恢复。最新已封存检查点若损坏，程序报错，不静默退回更早版本。

例如正式训练在第 53 步中断，最近成功保存的是第 48 步，重启后从第 48 步的状态继续，第 49 步起重算。尚未保存的部分可能丢失；`nohup` 只负责断开 SSH 后继续运行，不能代替检查点。

恢复时同时衔接 GRPO 微步计数、rollout 缓冲边界、采样器的独立随机状态及每个 rank 的奖励记录。检查点之后的未完成审计和派生产物先备份，再恢复到一致的边界，避免重复计数。旧 report/manifest 和失败记录保留在恢复归档中；report/manifest 采用临时文件写入后替换，降低中断造成半份 JSON 的风险。

`all --resume` 仍会运行代码、数据和硬件检查；已经完成的模型训练通过验证后复用。pilot 和正式训练各自从 Base 新建 adapter，正式训练的恢复仅加载它自己的检查点，不把 pilot adapter 当作正式训练起点。

### 检查点选择与评测恢复

```bash
bash run.sh select --resume
bash run.sh evaluate --resume
bash run.sh analyze
```

完整评测缓存先校验配置、源码、依赖、模型身份、样本网格与判分；一致时不重新生成回答。部分缓存补齐缺失样本。已冻结的检查点选择只允许验证复用，不能重新选择或改写；协议变化后不能混用旧缓存。

所有恢复检查点都会保留，但开发集选择仍只使用 epoch 结束时的候选。当前正式训练共 384 步，候选仍为第 128、256、384 步三个检查点，每题采样 4 次；第 16、32 等恢复检查点不增加开发集搜索次数，同一步的最终 adapter 也不会重复评测。

训练记录包括 manifest、report、模块审计、奖励日志、逐步日志、恢复检查点和最终 adapter。`resume_history` 记录恢复来源；`attempt_start_step` 记录本次启动位置。耗时、显存峰值和原始 Trainer 吞吐指标描述当前启动尝试；跨恢复后的逐步日志平均损失使用 `mean_logged_step_loss`。不能将最后一次尝试的耗时或原始 `train_loss` 当成整段训练的完整统计。

## 6. 训练验收与资源判断

验收检查包括：完成训练、有限日志和参数、出现混合奖励、LoRA 参数实际变化、非零有限梯度更新、冻结 Base 保持一致、完整检查点。报告记录显存峰值、耗时、回答长度和截断数量。

这些是工程完整性检查，不保证数学提升或成本可接受。只有少量混合奖励也可能不足以支持有效学习；需要结合混合奖励比例、开发集表现和错误样本判断。不得把没有训练起来的运行解释为 RL 无效。

正式训练必须与 pilot 的配置、数据、模型身份、关键源码、软件版本和训练卡数匹配。换卡数后需要在新运行目录重新做 pilot。相同全局批量不保证单卡和多卡得到逐位相同的随机轨迹。修改任何影响实验的设置后，应重新运行预实验。

## 7. 判分与评测

评测统一使用温度 0.6、top-p 0.95、top-k 0、每题 16 条回答和 512-token 上限。每条回答根据全局种子、题号、样本序号确定随机种子。问题不会因长度上限被静默截断。

判分器采用精确有理数：`1/2`、`0.5`、`\frac{1}{2}` 数值等价。识别 `####`、`\boxed{...}`、明确最终答案、精确的 `<answer>数值</answer>` 包装，以及没有其他标记时的末尾独立数值行。修复 `Answer: #### 5` 的重复候选误判和 `The answer is: 5` 的分隔符解析；支持有后续答案的空标题、CRLF 换行和完整粗体答案标签，避免 `#### 5` 后的 `**Final answer:** 6` 漏过矛盾检测。矛盾最终值记 ambiguous；相同值重复可合并。符合要求的独立 `#### <answer>` 行才计入格式合规，其他可解析的正确最终值仍可获得数值奖励。

例如 `#### 5` 和随后独立出现的 `6` 会判为矛盾，而不会选取较有利的一个。表达式 `2+3`、带单位的 `5 apples`、未约定的百分比形式不会被猜测转换。保守提取可能漏掉可理解的答案，需对两组同样进行人工抽检。空 `Answer:` 标题必须有紧随的可解析答案；`Final Answer:\nThe result is 5 apples.\n#### 5` 仍会提取失败，因为标题后的候选不是单一数值。该边界对两组一致适用，详见 `outputs/validation/v1_4/scoring_review.md` 的四条真实漏判记录。

数值正确、格式合规、提取失败、矛盾答案和截断分别记录。基础设施错误中止运行并写独立日志，不当作数学错误计入分母。最终数值正确不证明推理正确；截断保留独立标志，若已有完整且无冲突的数值答案，沿用原协议计分。

主测试需要 6,400 份回答，生成上限约 328 万 tokens；默认三个开发候选另需 2,400 份回答。这不包含训练与结构测试，也不是速度承诺。

## 8. 结构题

结构题为四个模板家族：折扣、库存、共享资金、路程。40 个配对实例各包含 original、numerical、relational、recombined，共 160 道题。`pair_id` 标识配对实例，`family_id` 标识真正的模板家族，不能把 40 组参数实例当作 40 种独立结构。

每题保留精确算术 oracle、参数与两次运算记录。运算次数相同不保证难度完全一致；仅有 4 个家族，聚类置信区间属于探索性证据。未审计全部训练题结构，因此不声称 recombined 是训练中未见过的组合。

`prepare` 生成待审阅模板。复制 `data/processed/structural_review.template.json` 为同目录的 `structural_review.json`，独立确认题意与答案后填写每条 `reviewed_answer` 和 `verified`、总记录的 `reviewer` 与 `approved`。程序检查题号、题目/数据哈希、独立答案和重复审阅项；不能只把总开关改为 true。数据要在选择检查点之前生成，才能冻结其哈希。

```bash
bash run.sh structural
```

## 9. 统计输出

每题采样 n 次、其中 c 次正确，使用组合式估计：

```math
\widehat{\operatorname{pass@k}}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
```

例如 16 次答对 4 次：

```math
\operatorname{pass@1}=4/16=0.25,\qquad
\operatorname{pass@2}=1-\frac{66}{120}=0.45,\qquad
\operatorname{pass@16}=1.
```

输出包含 pass@1/2/4/8/16、RL−Base 差值、两个端点增益之差、95% 配对 bootstrap 区间、both/base_only/rl_only/neither 覆盖集合、格式/截断诊断，以及人工抽检 CSV。原始测试按题重采样，结构题按模板家族聚类，关系与数值变体的增益差保留配对关系。

有限采样覆盖不等于绝对能力上限；区间跨零不等于没有变化。单个训练种子的置信区间不包含训练随机性。所有结论都应限定在此模型、LoRA 范围、训练规模和采样预算内。

## 10. 文件与验证证据

- `configs/v1.json`：v1.4 判分协议版本及现有恢复配置；`configs/archive/`：旧版配置。
- `run.sh`：统一相对路径入口；`run_v1.sh`：兼容旧入口，转发至主流程。
- `deploy_resume.sh`：Mac 上传、停止旧运行及升级重启入口；`resume_run.sh`：服务器后台恢复入口。
- `scripts/training_resume.py`、`training_checkpoint_event.py`：检查点封存、校验、恢复和奖励审计衔接。
- `scripts/stop_run.py`、`prepare_resume_upgrade.py`：项目进程停止和旧输出迁移。
- `scripts/`：数据、训练、评测、统计的顺序脚本；无自定义函数/类封装；库接口回调使用必要的 lambda。
- `tests/`：无需模型的测试。`outputs/validation/v1_4/test_report.json` 保存当前测试输出和源码哈希。
- `outputs/validation/v1_0/`、`outputs/validation/v1_1/`：保留旧版验证记录。
- `packages/linux-cp311-cu128/`：Linux GPU 依赖锁、来源清单和安装包。
- `memory.md`：版本更新、依据、实际验证与未完成的 GPU 实验。

v1.3 的历史 CPU 测试报告位于 `outputs/validation/v1_3/`，恢复开发证据另存于 `outputs/validation/v1_3_resume/`。验证覆盖完整检查点、未完成写入、文件篡改、奖励日志回退、1/2/4 卡参数传递和实际生产回调。固定上游代码的 186 个 CPU 场景验证保存在 `outputs/validation/v1_3_resume_upstream/`，包含采样顺序、随机状态及优化器恢复检查。没有执行真实 RTX 4090 上的完整训练中断恢复，不能把 CPU 检查当作 GPU 恢复成功证明。

旧版证据保留：v1.2.1 在 `outputs/validation/final_2026_10_30/`，v1.2.2 安装器在 `outputs/validation/v1_2_2/`，独立 `code` 目录修复在 `outputs/validation/v1_2_2_pathfix/`。

v1.2.2 的 11 个 CPU 测试套件全部通过，其中模型下载新增 28 项检查，安装流程和就绪判断新增 24 项检查。18 个 Python 脚本、4 个 Bash 入口及安装器内嵌 Python 语法检查通过。已有真实 GSM8K 的离线复用和固定划分也已核验，证据为同目录 `data_readiness.json`。官方权重端点的 16 字节范围请求通过；完整下载及 GPU 执行仍未在本机进行。

v1.2.1 的 9 个 CPU 测试套件通过。该版本另执行了锁定版本的实际采样器代码（9 个分组场景）和 Accelerate 的实际 backward 方法（6 个标量梯度场景），并保存可复现脚本与源文件哈希。这些检查验证分组和缩放逻辑，没有执行完整目标模型或真实 DDP。

合成演示的独立运行命令（当前位于 `code` 目录）：

```bash
python3 scripts/smoke_demo.py --output outputs/my_synthetic_demo --plot
```

`--plot` 需要 Matplotlib。所有演示结果标注 synthetic；输出目录存在时拒绝覆盖。

## 11. 参考来源

Yue, Yang, et al. 2025. “Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?” *arXiv*, 2504.13837v5. [论文](https://arxiv.org/abs/2504.13837v5)。无 KL 主候选参照其控制实验，不代表复现其全部设置。

Shao, Zhihong, et al. 2024. “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.” *arXiv*, 2402.03300. [GRPO](https://arxiv.org/abs/2402.03300)。

Hu, Edward J., et al. 2021. “LoRA: Low-Rank Adaptation of Large Language Models.” *arXiv*, 2106.09685. [LoRA](https://arxiv.org/abs/2106.09685)。

Cobbe, Karl, et al. 2021. “Training Verifiers to Solve Math Word Problems.” *arXiv*, 2110.14168. [固定版本数据](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)。

Chen, Mark, et al. 2021. “Evaluating Large Language Models Trained on Code.” *arXiv*, 2107.03374. [pass@k](https://arxiv.org/abs/2107.03374)。

Qwen Team. 2026. “Qwen3.5-2B-Base.” [官方模型](https://huggingface.co/Qwen/Qwen3.5-2B-Base)。

Hugging Face. 2026. “GRPOConfig.” [固定 TRL 版本的生成批量定义](https://github.com/huggingface/trl/blob/v1.16.0/trl/trainer/grpo_config.py)。

NVIDIA. n.d. “GeForce RTX 4090.” [24GB 与硬件规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)。

PyTorch. 2026. “Previous PyTorch Versions.” [2.10.0 CUDA 12.8 官方安装版本](https://pytorch.org/get-started/previous-versions/)。

PyTorch. 2026. “DistributedDataParallel.” [每进程/每卡模型副本](https://docs.pytorch.org/docs/2.10/generated/torch.nn.parallel.DistributedDataParallel.html)。


## 2026-11-01：200 题原生长度复核

独立上传包在 `boundary_confirm_v2/`。两组各 100 道新题（GSM-Symbolic main、原始 MATH test Level 4 标量无图子集），只测 Base，每题 4 次，共 800 个回答。输出额度按用户确认设置为 262,144 减实际输入 token 数。复用服务器环境和模型，默认同卡两进程，按完整回答续跑。

上传和 nohup 命令见 [COMMANDS.md](boundary_confirm_v2/COMMANDS.md)，协议与运行说明见 [README.md](boundary_confirm_v2/README.md)。新版判分器仅在该独立包使用，旧训练与结果不变。原生长度 GPU 占用与速度尚未实测，长时间重复输出仍可能达到上限；不可用上一轮短输出耗时推算本轮完成时间。


## 2026-11-01：当前边界复核使用 guarded v3

新的入口为 `boundary_guarded_v3/run_guarded.sh`：同一时间仅一个模型进程；每次回答限时 300 秒，超时计错、0 分；固定阈值检测精确重复并提前停止，每 5 秒显示 token 进度与实际输出。复用 v2 相同的 200 题、每题 4 次与原生长度额度，使用新的结果目录，不混入旧停止规则下的回答。

无需新增依赖。上传、结束旧任务、nohup 串行启动与实时查看命令见 [guarded v3 COMMANDS](boundary_guarded_v3/COMMANDS.md)，协议及验证范围见 [guarded v3 README](boundary_guarded_v3/README.md)。下面/上面的 v2 并行说明仅保留为历史版本记录；当前建议以本节为准。v3 已做 CPU 线程、子进程与缓存回归，真实 4090 推理仍需部署验证。


## 2026-11-01：MATH Level 4 下一轮正式实验

新的独立包为 `math_l4_v1/`：Qwen3.5-2B-Base、LoRA-GRPO，512训练/128验证/200测试，判分器3.0.0；用户确认Base/RL各16次采样，比较pass@1/4/16。训练2048-token上限，评测保留原生长度与300秒超时，严格单卡串行。数据排除历史校准题，验证集冻结checkpoint后才测试。

完整上传与nohup命令见 [MATH COMMANDS](math_l4_v1/COMMANDS.md)，协议见 [MATH READM](math_l4_v1/READM.md)。包内已含实际数据，复用服务器现有环境和模型。11套件658项CPU回归及上传异地目录计划通过；实际新GPU训练未在本机执行，`bash run.sh all`会先工程/pilot验收。旧GSM和边界复核包继续保留为历史版本。
