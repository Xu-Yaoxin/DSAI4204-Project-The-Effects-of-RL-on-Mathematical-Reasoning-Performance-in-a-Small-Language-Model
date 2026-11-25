# DSAI4207 RLVR v1.2.2 · RTX 4090

本项目使用 Qwen3.5-2B-Base，比较答案奖励强化学习前后的数学单次成功率、有限采样覆盖和结构变化后的表现。方法是可验证奖励强化学习（RLVR）+ 组相对策略优化（GRPO）+ 低秩适配（LoRA）。主评测每题 16 次，依据上一级 `DSAI4207_LLM_Proposal_Overleaf(4).pdf`。

**当前状态：官方 GSM8K 和固定划分已准备；RTX 4090 单机 1/2/4 卡启动，以及依赖、模型和数据的一键准备已实现。真实 Qwen 推理、GPU 训练、显存和吞吐尚未验证。** 合成测试输出不能用作模型实验结果。

## 1. 两条命令启动

当前云端使用 `/root/code`：先执行 `cd /root/code`，然后运行 `bash setup_gpu.sh` 和 `bash run.sh all`。无需建立 `project` 父目录；下方 `project/code/` 示例用于本地课程目录。2026-10-30 的部署路径补丁修复了 `tests/test_pipeline.py` 对外部目录结构的错误假设，并覆盖课程目录、独立 `code` 目录和含空格的重命名目录。若旧版本在该测试第 33 行报错，只需更新测试文件，再运行 `bash run.sh all`；无需重装依赖或重新下载模型。

命令默认你位于包含 `project` 文件夹的课程目录。将整个 `project/code` 复制到服务器，保留 `data/`、`configs/`、`scripts/`、`tests/`、`models/`、`packages/` 中的版本锁与清单；不要从 Mac 复制 `.venv`。安装脚本会自动把官方原始模型下载到 `project/code/models/Qwen3.5-2B-Base/`。

服务器需要 Linux x86_64、64-bit CPython 3.11（含 venv/ensurepip）、glibc ≥ 2.28、NVIDIA 驱动 ≥ 570.26，且可以访问依赖下载站点。

一键准备环境：

```bash
bash project/code/setup_gpu.sh
```

这条命令会完成以下准备，不需要手动激活 `.venv` 或另行下载模型：

1. 检查服务器、Python 3.11 和本项目虚拟环境。
2. 下载、校验并安装 84 个锁定依赖。
3. 核对依赖版本、Qwen3.5/GRPO 导入和 CUDA BF16 运算。
4. 下载固定版本的官方 Qwen3.5-2B-Base，逐文件核对大小和哈希。
5. 下载或复用 GSM8K，生成或核对固定数据划分和结构题文件。
6. 检查 tokenizer、数据来源、模型分片布局，以及自动选中的单卡/多卡通信。

终端显示 7 个准备阶段的进度条；依赖下载显示分块字节进度，模型下载和校验显示各自的进度条。阶段百分比表示完成了多少准备步骤，不是剩余时间估计。下载中断后可重跑同一命令，已校验的文件会复用。已有完整文件与固定版本冲突时保留文件并报错。

只有全部检查通过才会显示 `✅ [SETUP COMPLETE]`，并保存 `outputs/environment/setup-XXXXXXXX/ready.json`。每次安装检查保留独立报告。准备成功表示依赖、模型文件、数据和小型 GPU 检查通过；模型加载、真实训练显存和吞吐由随后 `all` 中的工程阶段验证。

一键运行主实验：

```bash
bash project/code/run.sh all
```

启动器自动识别 1/2/4 张可见 GPU，依次执行代码检查、数据准备、GPU/通信检查、工程运行、pilot、正式训练、开发集选点、Base/RL 测试和统计绘图。任一步失败立即停止。默认输出为 `project/code/outputs/v1_2/`；已有训练目录不会覆盖。结构题需独立审阅后另行运行 `structural`，不包含在主流程中。

如果已经进入 `project/code`，两条命令分别是 `bash setup_gpu.sh` 和 `bash run.sh all`。服务器的 Python 3.11 不在默认位置时，用 `RLVR_SYSTEM_PYTHON` 指定现有解释器。

本机检查或预览命令，不需要 GPU：

```bash
bash project/code/run.sh check
bash project/code/run.sh prepare
bash project/code/run.sh all --gpus 4 --plan
```

`check` 运行 CPU 测试并报告环境状态；`prepare` 完整校验并复用现有数据；`--plan` 只打印执行计划，不加载模型或启动训练。当前已完成的是代码、数据和 CPU 检查，真实 4090 训练仍需在服务器执行。

## 2. 数据已经放在哪里

官方数据来自 [OpenAI GSM8K 仓库](https://github.com/openai/grade-school-math)，固定 commit 为 `3101c7d5072418e28b9008a6636bde82a006892c`。

| 路径，均相对 `project/code` | 内容 |
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

`setup_gpu.sh` 调用 `scripts/download_model.py`，将官方原始模型快照下载到 `project/code/models/Qwen3.5-2B-Base/`，包含 `config.json`、完整 tokenizer 文件、safetensors 权重和分片索引。本机交付目录不含大模型权重，下载在服务器执行；训练和评测仍只读取本地模型。

模型固定为官方 `Qwen/Qwen3.5-2B-Base` 原始 BF16 快照，revision 为 `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`。清单 `models/qwen3.5-2b-base-manifest.json` 来自该版本的官方 API，包含 12 个文件，共 4,571,206,192 bytes（约 4.57GB）。LFS 文件按官方 SHA256 校验，普通 Git 文件按 blob SHA1 校验；全部通过后在模型目录保存 `source_manifest.json`。Instruct、GGUF、GPTQ、Ollama 导出包不属于当前实验。训练程序另外计算文件内容身份，防止 Base 和 RL 混用不同底座。

目标环境固定为 **Linux x86_64、Python 3.11、glibc ≥ 2.28、NVIDIA 驱动 ≥ 570.26**。建议使用服务器现成的 Python 3.11 环境；可以通过 `RLVR_SYSTEM_PYTHON` 指定其解释器。当前 Mac 只用于开发和准备文件，安装脚本会拒绝在 Mac 上安装 Linux GPU 包。

安装包位于 `packages/linux-cp311-cu128/`：`manifest.json` 记录来源、目标平台、文件大小与 SHA256，`requirements-lock.txt` 固定完整版本与哈希，`wheels/` 保存 Linux wheel。锁定的 84 个 wheel 共约 3.98GiB。默认安装会复用完整文件并补齐缺失下载。离线安装要求完整的 wheel、模型文件、GSM8K 原始数据及来源清单；仅下载完依赖包还不够：

```bash
bash project/code/setup_gpu.sh --offline
```

如果安装包缺失，可在有网络的机器补齐，再复制到服务器：

```bash
bash project/code/download_packages.sh
```

默认 `setup_gpu.sh` 先验证虚拟环境、venv/ensurepip 和 pip，再下载缺失的包。`--offline` 会传给模型和数据准备程序，缺文件就停止，不会联网补齐；该模式可根据完整原始数据重新生成固定划分。安装器创建独立 `.venv`，核对哈希、依赖关系与实际版本。`RLVR_PYTHON` 可指定启动实验时的解释器。系统 NVIDIA 驱动不包含在 wheel 包内，需要服务器提供。

核心版本为 PyTorch 2.10.0 CUDA 12.8、Transformers 5.3.0、PEFT 0.18.1、TRL 1.16.0、Accelerate 1.10.1。依赖解析与文件校验不等于 GPU 上的安装或运行验证。Qwen3.5 使用 Transformers 的 PyTorch 线性注意力实现；可选 FLA/causal-conv1d 加速内核未作为默认依赖，避免不匹配的编译包改变环境。不要在一次已开始的实验中途切换内核。

4090 每卡 24GB，分布式数据并行（Distributed Data Parallel，DDP）在每张卡保留一份模型，显存不会相加。启动器尊重服务器的 `CUDA_VISIBLE_DEVICES`，自动选择不超过可见数量的 1、2 或 4 张卡；3 张时使用 2 张，超过 4 张时使用 4 张，以保持固定的全局更新批量。也可显式选择：

```bash
bash project/code/run.sh engineering --gpus 1
bash project/code/run.sh engineering --gpus 2
bash project/code/run.sh engineering --gpus 4
```

上述命令是三种替代运行方式，不要依次写入同一目录。不同硬件实验使用不同 `--run-dir`。例如仅使用服务器的第 2、3 张卡：

```bash
CUDA_VISIBLE_DEVICES=1,2 bash project/code/run.sh all --gpus 2 --run-dir outputs/two_gpu_run
```

每个训练阶段先进行小规模 BF16 运算和 NCCL 通信检查，再加载模型。也可以单独执行硬件检查，不加载模型或读取训练答案：

```bash
bash project/code/run.sh gpucheck
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

当前配置为 `configs/v1.json`，版本号 1.2.2；旧版保存在 `configs/archive/`。本次更新环境准备流程，训练参数沿用 v1.2.1。无 KL 主候选参照 Yue et al. 控制实验，仍有模型规模、LoRA、长度和数据量差异。

LoRA rank 16、alpha 32、学习率 `5e-6`、训练温度 1.0、3 个 epoch、`loss_type="grpo"`、`scale_rewards="group"`、裁剪阈值 0.2、`num_iterations=1` 保持不变。训练与评测的生成长度上限仍为 512 tokens，须通过截断率判断是否调整。这些是预实验起点，没有实测证据说明已经最优。

训练不使用 SFT、教师推理或格式奖励；只训练 LoRA 参数。预实验和正式训练都从同一 Base 新建 adapter，正式训练不接着训练 pilot adapter。

## 5. 分阶段执行与输出

默认运行目录为 `project/code/outputs/v1_2/`。各阶段可单独启动，前置结果缺失时会直接报错。

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
bash project/code/run.sh engineering
bash project/code/run.sh pilot
bash project/code/run.sh formal
bash project/code/run.sh select
bash project/code/run.sh evaluate
bash project/code/run.sh analyze
```

运行另一个配置时，配置文件位置和输出目录均相对 `project/code`：

```bash
bash project/code/run.sh pilot --config configs/v1.json --run-dir outputs/another_run
```

不要把新训练写入已有目录。训练失败时保留现场，用新输出目录重跑。v1.2 不提供训练中途恢复；后续阶段可以直接复用已完成训练，不需要从 `all` 重新开始。

评测和开发集选择支持恢复：

```bash
bash project/code/run.sh select --resume
bash project/code/run.sh evaluate --resume
bash project/code/run.sh analyze
```

完整缓存先校验配置、源码、依赖、模型文件身份、样本网格与判分；一致时不重新加载 GPU 模型或生成回答。部分缓存通过检查后补齐缺失样本。已冻结的检查点选择只允许验证复用，不允许重新选择或改写。改变协议后不能混用旧缓存。

训练记录包括 manifest、report、模块审计、奖励日志、逐步训练日志、每个 epoch 的 checkpoint 和最终 adapter。报告中的项目内可执行路径按 code 根目录保存为相对路径，复制整个目录到服务器后仍可解析。默认 3 个候选检查点的开发集每题采样 4 次；同一步的最终 adapter 不重复评测。

## 6. 训练验收与资源判断

验收检查包括：完成训练、有限日志和参数、出现混合奖励、LoRA 参数实际变化、非零有限梯度更新、冻结 Base 保持一致、完整检查点。报告记录显存峰值、耗时、回答长度和截断数量。

这些是工程完整性检查，不保证数学提升或成本可接受。只有少量混合奖励也可能不足以支持有效学习；需要结合混合奖励比例、开发集表现和错误样本判断。不得把没有训练起来的运行解释为 RL 无效。

正式训练必须与 pilot 的配置、数据、模型身份、关键源码、软件版本和训练卡数匹配。换卡数后需要在新运行目录重新做 pilot。相同全局批量不保证单卡和多卡得到逐位相同的随机轨迹。修改任何影响实验的设置后，应重新运行预实验。

## 7. 判分与评测

评测统一使用温度 0.6、top-p 0.95、top-k 0、每题 16 条回答和 512-token 上限。每条回答根据全局种子、题号、样本序号确定随机种子。问题不会因长度上限被静默截断。

判分器采用精确有理数：`1/2`、`0.5`、`\frac{1}{2}` 数值等价。识别 `####`、`\boxed{...}`、明确最终答案，以及没有其他标记时的末尾独立数值行。矛盾最终值记 ambiguous；相同值重复可合并。符合要求的独立 `#### <answer>` 行才计入格式合规，其他可解析的正确最终值仍可获得数值奖励。

例如 `#### 5` 和随后独立出现的 `6` 会判为矛盾，而不会选取较有利的一个。表达式 `2+3`、带单位的 `5 apples`、未约定的百分比形式不会被猜测转换。保守提取可能漏掉可理解的答案，需对两组同样进行人工抽检。

数值正确、格式合规、提取失败、矛盾答案和截断分别记录。基础设施错误中止运行并写独立日志，不当作数学错误计入分母。最终数值正确不证明推理正确。

主测试需要 6,400 份回答，生成上限约 328 万 tokens；默认三个开发候选另需 2,400 份回答。这不包含训练与结构测试，也不是速度承诺。

## 8. 结构题

结构题为四个模板家族：折扣、库存、共享资金、路程。40 个配对实例各包含 original、numerical、relational、recombined，共 160 道题。`pair_id` 标识配对实例，`family_id` 标识真正的模板家族，不能把 40 组参数实例当作 40 种独立结构。

每题保留精确算术 oracle、参数与两次运算记录。运算次数相同不保证难度完全一致；仅有 4 个家族，聚类置信区间属于探索性证据。未审计全部训练题结构，因此不声称 recombined 是训练中未见过的组合。

`prepare` 生成待审阅模板。复制 `data/processed/structural_review.template.json` 为同目录的 `structural_review.json`，独立确认题意与答案后填写每条 `reviewed_answer` 和 `verified`、总记录的 `reviewer` 与 `approved`。程序检查题号、题目/数据哈希、独立答案和重复审阅项；不能只把总开关改为 true。数据要在选择检查点之前生成，才能冻结其哈希。

```bash
bash project/code/run.sh structural
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

- `configs/v1.json`：v1.2 实验候选；`configs/archive/`：旧版配置。
- `run.sh`：统一相对路径入口；`run_v1.sh`：兼容旧入口，转发至主流程。
- `scripts/`：数据、训练、评测、统计的顺序脚本；无自定义函数/类封装；库接口回调使用必要的 lambda。
- `tests/`：无需模型的测试。`outputs/validation/v1_2_2/test_report.json` 保存当前测试输出和源码哈希。
- `outputs/validation/v1_0/`、`outputs/validation/v1_1/`：保留旧版验证记录。
- `packages/linux-cp311-cu128/`：Linux GPU 依赖锁、来源清单和安装包。
- `memory.md`：版本更新、依据、实际验证与未完成的 GPU 实验。

2026-10-30 v1.2.1 审查证据保存在 `outputs/validation/final_2026_10_30/`；本次安装器更新及后续 `check` 的结果写入 `outputs/validation/v1_2_2/`。测试覆盖单卡/多卡计划、逐卡奖励记录与公共文件写入、评测单进程限制、数据与缓存完整性。模型训练和 NCCL 实际通信须在目标服务器执行。v1.1 已完成的全量数据审计仍保存在原证据目录。

v1.2.2 的 11 个 CPU 测试套件全部通过，其中模型下载新增 28 项检查，安装流程和就绪判断新增 24 项检查。18 个 Python 脚本、4 个 Bash 入口及安装器内嵌 Python 语法检查通过。已有真实 GSM8K 的离线复用和固定划分也已核验，证据为同目录 `data_readiness.json`。官方权重端点的 16 字节范围请求通过；完整下载及 GPU 执行仍未在本机进行。

v1.2.1 的 9 个 CPU 测试套件通过。该版本另执行了锁定版本的实际采样器代码（9 个分组场景）和 Accelerate 的实际 backward 方法（6 个标量梯度场景），并保存可复现脚本与源文件哈希。这些检查验证分组和缩放逻辑，没有执行完整目标模型或真实 DDP。

合成演示的独立运行命令（当前位于课程目录）：

```bash
python3 project/code/scripts/smoke_demo.py --output outputs/my_synthetic_demo --plot
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
