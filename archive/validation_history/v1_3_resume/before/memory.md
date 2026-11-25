# 项目版本与决策记录 / Project memory

本文件只记录当前 RLVR 项目的决策、代码变更、实际验证和待办。新增版本追加记录，保留旧记录；真实实验结果必须附运行目录和原始证据，不能用合成测试代替。

## 2026-10-22 · v1.0.0 · 首版实验代码

### 项目方向回忆与依据

- 最终选择是小语言模型 RLVR 的概念复现与结构扩展：同一 Qwen3.5-2B-Base，对比 Base 与答案奖励 LoRA-GRPO 的单次成功率、有限采样覆盖和关系变化后的表现。
- 已重新读取本地 `../DSAI4207_LLM_Proposal_Overleaf(4).pdf` 全文，并核对先前“推荐优质选题”讨论。用户最后明确使用下载的原始模型包，不使用 Ollama；可租至少一张 4090，必要时考虑更多 GPU。
- 早期记忆中的每题 128 次采样已被本次指定 PDF 收敛：主实验每题 16 次，报告 pass@1/2/4/8/16；64–100 次只是预算允许时提前登记的扩展。v1 默认 16 次。
- DataPilot 属于更早的另一项目方向，未迁入本项目。当前创建 `project/code` 时，此目录不存在，没有覆盖旧代码。
- 模型目录固定为 `models/Qwen3.5-2B-Base/`；默认工程/预实验/正式训练分别 32/128/512 题，开发和测试各 200 题。规模可在正式测试前根据资源测量调整并记录。

### 已实现

- `download_data.py`：显式下载官方 GSM8K，先固定 Git commit，再保存文件哈希。准备和训练脚本不会自动联网下载。
- `prepare_data.py`：固定随机划分、嵌套训练子集、官方训练/测试边界、规范化文本重叠检查和不可覆盖的数据清单。
- `verify_answers.py`：有理数精确判分、整数/小数/分数等价、显式最终答案矛盾检查、独立格式和截断标记。仅最终数值奖励。
- `load_model.py`：官方文本主干加载、本地权重和 tokenizer 内容身份、BF16/CUDA 检查、禁止缺失文本权重后继续。视觉塔不参与实验。
- `train_grpo.py`：单 GPU 的真实 TRL LoRA-GRPO 代码；工程、pilot、formal 模式；实际参数更新、冻结 Base 哈希、有效梯度、混合奖励、有限日志、显存/耗时与检查点完整性审计。正式训练要求同一配置、源代码、环境、数据和底座下通过的 pilot；正式 adapter 从 Base 新建。
- `select_checkpoint.py`：仅依据开发集数值 pass@1 选择，固定同分规则，冻结模型/软件/评测/数据身份。默认每 epoch 保存，3 个候选；最终同一步 adapter 不重复做开发集采样。
- `evaluate.py`：固定每条回答种子、同一提示和生成预算、逐条落盘、明确基础设施失败、重验内容后续跑。正式测试必须提供开发集选择文件。
- `analyze.py` / `plot_results.py`：组合式 pass@k、配对 bootstrap、四种覆盖集合、诊断率、结构家族聚类和增益对比、2×2 分析图、人工抽检表。
- `build_variants.py`：160 道精确算术结构题，40 个 pair_id，只有 4 个真实 family_id；外部独立审阅记录与数据哈希绑定。不能声称未见组合泛化。
- `READM.md` 为完整说明，`README.md` 为常规入口；`setup_gpu.sh` 安装独立环境，`run_v1.sh` 串联主实验。

### 本轮审核修正

1. 按 PDF 使用 16 次主采样，修正沿用旧讨论 128 次的风险。
2. 将结构 bootstrap 单位由 40 个参数实例改为 4 个真正模板家族；40 组配对关系单独保留。
3. 修正共享资金题中代数等价的“关系变化”，使用同一费用数值分别作为总费用和个人费用，测试每组原题/关系题答案确实不同。
4. 将 `format_compliant` 限定为提示要求的独立 `#### <answer>` 行；其他明确最终值可以正确，但不计为格式合规。
5. 测试数据、代码/软件/判分器在开发集选定后发生变化会被拒绝。续跑会重新判分缓存文本，拒绝重复或不一致记录。
6. 每 epoch 保存候选检查点，减少开发集重复采样；默认 3 个候选约 2,400 个开发回答，主测试另需 6,400 个回答。

### 已执行的验证

- `python3 tests/run_all.py`：4 个测试套件通过。包含 37 个判分案例、21 个统计案例、10 个评测计划/冻结案例、11 个训练奖励接口案例，以及数据划分、哈希、防篡改、160 道算术 oracle 和泄漏拒绝检查。
- 13 个 Python 脚本语法/AST 检查通过，没有自定义 `def` 或 `class`；训练仅保留 TRL 必需的一个奖励接口 lambda。
- 两个 Bash 脚本 `bash -n` 检查通过；配置 JSON 解析通过。
- 合成端到端测试：8 道人工题，Base/RL 各 16 次回答，共 256 条；经真实判分器、2,000 次 bootstrap 和绘图程序导出 `outputs/validation/demo/analysis/`。所有结果均标为 synthetic，不能用于论文实验结论。
- `doctor.py` 正确报告本机未就绪：没有目标模型或 GSM8K 文件，当前库为旧版本且缺 TRL。未自动修改当前 Python 环境。
- 证据：`outputs/validation/test_report.json` 保存测试输出和源文件哈希；`outputs/validation/environment.json` 保存环境检查；示例目录保存 JSONL、summary、report 和 overview.png。

### 尚未执行或不在 v1 范围

- 未下载或加载目标 2B 权重，未下载真实 GSM8K，未租 GPU，未运行 GPU 安装、真实推理、GRPO 训练或目标模型评测。
- 固定的 Transformers 5.3.0 / PEFT 0.18.1 / TRL 1.16.0 接口按官方源码核对；完整依赖组合和显存/吞吐仍待目标 Linux/CUDA 环境验证。PyTorch 候选版本为 2.10.0 CUDA 12.8。
- 结构题尚待独立逐题审阅；只有 4 个模板家族，聚类区间只能作探索性分析。没有审计训练题的完整操作结构，没有“训练中未见组合”证据。
- 没有多训练种子、双卡训练、训练中断恢复、vLLM 加速或高采样扩展。评测支持续跑，训练失败需用新目录重跑。
- 预实验门槛只确认基本训练完整性，不能保证数学收益、统计显著性或成本可接受。下一步是在 GPU 上检查工程运行，再根据实测决定预实验与正式规模。

### 后续版本记录模板

```text
日期与版本：
修改原因：
修改文件和行为：
数据、模型、配置和环境身份：
实际执行的命令与证据目录：
测试和实验结果：
未验证内容与下一步：
```

## 2026-10-26 · v1.1.0 · 真实数据、批量修正与相对路径启动

### 修改原因与范围

- 用户要求检查和优化现有代码，下载所需数据，并在当前课程目录使用相对路径启动。本次保留原研究问题、Qwen3.5-2B-Base、纯数值奖励、LoRA 和每题 16 次主评测。
- v1.0 配置存入 `configs/archive/v1.0.json`，原测试与环境记录另存于 `outputs/validation/v1_0/`。旧合成演示和原验证记录保留；新证据写入 `outputs/validation/v1_1/`。

### 真实数据已下载与准备

- 下载 [OpenAI 官方 GSM8K](https://github.com/openai/grade-school-math)，固定 commit `3101c7d5072418e28b9008a6636bde82a006892c`，保存于 `data/raw/gsm8k/`。包含 train 7,473 题、test 1,319 题、MIT `LICENSE` 和 `source_manifest.json`。
- train SHA256：`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`；test SHA256：`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
- 实际生成 engineering 32、pilot 128、train 512、dev 200、test 200；前三者嵌套，训练/开发/测试边界与内容哈希均通过检查。结构题生成 160 道，40 个配对实例、4 个模板家族，仍待独立审阅。
- 全量检查 8,792 个原始数值答案，零解析失败；全量规范化题目零重复组。这不证明语义无重叠或不存在预训练污染。
- 发现真实题目含 Unicode U+2028；原先的 `splitlines()` 会在字符串内部错误拆分 JSONL。所有生产 JSONL 读取改为仅按换行符 `\n` 分割，保留原文；新增 U+2028/U+0085 回归测试。判分器对回答文本本身的分行逻辑不受此修改影响。
- 下载器保持 TLS 校验，在本机默认 CA 不可用时使用系统 CA；采用临时文件、完整性检查、冲突拒绝和失败回滚。重复准备已验证无需联网或新建数据文件。正式数据准备和训练绑定来源清单、许可证、原始与划分文件哈希。

### 代码和默认候选配置

- 新增 `run.sh` / `scripts/run_pipeline.py`，提供 check、prepare、engineering、pilot、formal、select、evaluate、analyze、structural 和 all 阶段，支持无副作用的 `--plan`。从课程目录执行 `bash project/code/run.sh ...`；在 code 目录执行 `bash run.sh ...`。
- 默认输出改为 `outputs/v1_1/`。保存的项目内训练、adapter、checkpoint 和 selection 操作路径均相对 code 根目录；目录迁移与空格路径已有测试。`run_v1.sh` 保留为兼容入口。
- 默认 KL 系数由 0.04 改为 0，每题回答由 4 改为 8，梯度累积由 4 改为 32，microbatch 仍为 1，显式 `steps_per_generation=8`。每批生成同一题的 8 条回答，四批、四道不同题合计 32 条回答后更新一次。已核对 TRL 1.16.0 配置、采样器和生成缓冲实现支持此组合。
- 无 KL 候选更接近 Yue et al. v5 的控制实验；2B、LoRA、数据规模和 512-token 上限等差异仍然存在。这是待预实验验证的配置，不是已证明最优的参数。
- `train_grpo.py --plan` 不导入模型/GPU 库、不写训练目录。实际数据下工程/预实验/正式训练分别计划 2/96/384 次更新。新增完整更新步数、每步题组数、题目覆盖检查，以及正式训练与 pilot 的来源身份核对。
- 模型预检查识别缺失分片、损坏的 safetensors 头、索引与实际张量不一致。冻结参数哈希减少一次大块内存复制。软件身份同时记录可选线性注意力内核，避免改变后端后混用评测记录。
- 完整评测缓存及已写完全部样本但未更新完成标志的缓存，可在校验后无需加载 GPU 模型直接复用。检查样本网格、严格类型、种子、判分结果、模型/adapter 身份和协议；部分缓存经检查后补齐缺失样本。
- 已冻结的开发集选择支持只读 `--resume`，验证原候选缓存、训练报告与清单哈希，不重新选点。结构题审阅要求逐题填写独立 `reviewed_answer`、严格布尔确认和完整题号覆盖，拒绝重复审阅项与非数值答案。
- `setup_gpu.sh` 支持从课程目录启动，检查 Linux/NVIDIA/Python 条件，不覆盖已有 `.venv`。本次未执行环境安装。

### 本次实际执行与证据

以下命令默认当前目录是课程目录：

```bash
bash project/code/run.sh prepare
bash project/code/run.sh check
python3 project/code/scripts/train_grpo.py --mode pilot --output outputs/v1_1/pilot --plan
python3 project/code/scripts/evaluate.py --model base --split engineering --samples 8 --output outputs/v1_1/engineering_base --plan
python3 project/code/scripts/smoke_demo.py --output outputs/validation/v1_1/demo --plot
```

- `prepare` 实际成功；重复运行记录 `network_files=0`、`installed_files=0`，所有处理文件 `new_files=0`。
- 8 个 CPU 测试套件全部通过：24 个统计、37 个判分及数据隔离/oracle 检查、14 个下载/来源/Unicode、23 个评测计划/审阅、16 个缓存/迁移、17 个模型布局/数据/训练计划、20 个启动入口、16 个训练奖励场景。
- 14 个生产 Python 脚本 AST 检查通过，无自定义函数或类；3 个 Bash 脚本语法检查通过。测试报告记录的 26 个配置、脚本、测试和启动文件哈希与最终文件一致。
- 新合成端到端演示完成：8 题、Base/RL 各 16 份回答，共 256 条，经真实判分器、2,000 次 bootstrap 和 2×2 绘图流程；图像已检查，清楚标记 synthetic。
- `outputs/validation/v1_1/test_report.json`：完整测试输出与源码哈希。
- `outputs/validation/v1_1/environment.json`：真实本机环境及数据检查，`data_ready=true`、`ready=false`。
- `outputs/validation/v1_1/real_data_audit.json`：全量原始答案、重复题检查、数据来源和划分数量。
- `outputs/validation/v1_1/{engineering,pilot,formal}_plan.json`：真实数据对应的批量与更新计划。
- `outputs/validation/v1_1/demo/analysis/`：仅用于验证统计与绘图链路的合成输出。

### 当前限制与下一步

- 当前 Mac 未安装约定的 GPU 依赖，模型目录只有放置说明，没有目标权重。未下载模型、未执行真实 Qwen 推理、GRPO 训练、显存/吞吐测量或模型效果评测。
- GPU 环境依赖仍为候选组合；将完整项目复制至 Linux NVIDIA 服务器、放好原始模型后，先运行 engineering，再运行 pilot；formal 必须通过同一协议下的 pilot 验收。
- 结构题仍需独立逐题审阅。训练中断恢复、多卡、vLLM 和多训练种子仍不在本版范围；失败训练保留现场并用新目录重跑，评测可续跑。
- CPU 测试与合成结果只证明所测软件行为，不证明 2B 模型能够获得学习收益。

## 2026-10-26 · v1.2.0 · RTX 4090 单机多卡与服务器安装

### 用户要求与最终选择

- 最终运行平台为 RTX 4090，事前不确定卡数，因此启动器自动识别单机可见设备，支持 1/2/4 卡 DDP。
- 用户在获知完整依赖包约 4GB、本机下载较慢后，明确选择“在 4090 服务器上一键下载并安装”。因此以服务器在线安装为默认，停止本机大包下载；已完成文件与部分下载保留。不能把本机目录描述为完整离线包。
- 原始模型仍由用户放入 `models/Qwen3.5-2B-Base/`。本次未下载模型、未修改本机 Python 环境，也未启动真实 GPU 工作。

### 依赖与安装

- 目标固定为 Linux x86_64、CPython 3.11、glibc ≥ 2.28、NVIDIA 驱动 ≥ 570.26。驱动需服务器提供，安装器不修改系统 Python 或安装系统驱动。
- 核心版本：PyTorch 2.10.0+cu128、Transformers 5.3.0、PEFT 0.18.1、TRL 1.16.0、Accelerate 1.10.1。完整目标平台依赖闭包含 84 个 wheel，共 4,269,992,994 bytes，约 3.98GiB。
- `packages/linux-cp311-cu128/manifest.json` 保存目标平台、每个包的 URL、版本、大小、SHA256 与依赖关系；`requirements-lock.txt` 保存所有精确版本与安装哈希；`resolution-report.json` 保存解析证据。
- 针对跨平台准备时 pip 仍可能按 macOS 解释条件依赖的问题，解析阶段显式使用 Linux/Python 3.11 marker 环境，并独立复核完整依赖闭包，避免漏掉 CUDA/NVIDIA wheel。
- `download_packages.sh` / `scripts/prepare_packages.py` 支持校验、断点续传、最多 8 个下载任务和大文件范围分块。请求检查范围和 ETag，最终校验整个 wheel 的 SHA256 后才提交为完成文件；不会把部分文件作为可安装 wheel。
- `setup_gpu.sh` 默认下载缺失的锁定文件，创建本项目 `.venv`，离线从已校验的 wheel 安装并运行 `pip check`、精确版本核对、Qwen3.5/GRPO 导入及小型 BF16 GPU 运算。只有服务器执行后才能声称这些 GPU 环境检查通过。`--offline` 仅供完整包通过 `--verify` 后使用。
- 已有同一安装锁的受管 `.venv` 可继续安装；不匹配或不受管的环境会保留并拒绝覆盖。可用 `RLVR_SYSTEM_PYTHON` 指定服务器现有 Python 3.11。
- 默认不安装 FLA/causal-conv1d 源码编译内核，使用固定 Transformers 的 PyTorch 实现。当前最新 FLA 与本项目 Transformers 版本要求不一致，causal-conv1d 没有可直接采用的 PyPI 二进制 wheel。运行身份同时记录 `fla-core`、`flash-linear-attention` 和 `causal-conv1d`，防止中途换内核后混用结果。

### 单卡与多卡行为

- `run.sh` 不再把可见设备强制限制为 0 号卡；新增 `--gpus auto|1|2|4`。auto 选择不超过可见数量的最大支持卡数：3 卡使用 2 卡，超过 4 卡使用 4 卡；尊重服务器已有的 `CUDA_VISIBLE_DEVICES` 索引或 UUID。
- 固定全局每次更新 32 条回答、4 道不同题。1/2/4 卡时，每卡 microbatch 都为 1，梯度累积分别为 32/16/8，`steps_per_generation=8`，每卡每批生成同一题的完整 8 条回答。训练框架显式设置 `split_batches=False`、`dispatch_batches=False`。
- 多卡通过 `torch.distributed.run --standalone` 启动一张卡一个进程；每卡保留完整文本模型与 adapter。DDP 不合并显存，也不保证线性加速。显存报告给出各卡峰值及最大单卡峰值。
- 通过 `LOCAL_RANK` 选择设备；各卡独立写 `reward_audit.rank-N.jsonl`，主进程合并；共享报告、tokenizer 和最终模型保存由主进程负责。核对所有 rank 的初始/最终参数身份、有效更新、有限指标、冻结底座和完整问题覆盖。
- 异常路径不调用 collective，以便 torchrun 终止失败作业的其他进程。各卡保留错误记录；若主进程提前退出，启动器将残留 running 报告标为 failed。已有训练目录不覆盖，重试硬件检查则保存新的带时间戳报告。
- pilot 与 formal 要求相同 GPU 拓扑。固定全局批量不意味着跨卡数的随机轨迹逐位相同。开发集选点和正式评估始终使用第一张选中卡、单进程执行；选点核查多卡训练报告和逐卡证据。
- `resolve_gpus.py` 在独立短进程中探测 GPU，避免启动器长期占用 CUDA 上下文。`check_gpu.py` 在加载模型前执行小型 BF16 运算和 NCCL all-reduce；该检查无法证明完整模型的显存需求。
- RTX 40 系列多卡采用保守通信默认值 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`，保留用户显式覆盖。已拒绝遗留多进程环境变量，避免误把入口本身放进 torchrun 后重复启动工作进程。
- 完整评测缓存的 `--resume` 仍可先在无 GPU 环境校验复用；存在缺失采样时才需要单卡加载。

### 实际验证与证据

- `bash project/code/run.sh check`：9 个 CPU 测试套件通过，覆盖真实数据、下载/依赖、1/2/4 卡计划、并发奖励文件、缓存、选点、启动器工作进程失败和重试保留。
- 17 个生产 Python 脚本 AST 检查通过，无自定义函数或类封装；4 个 Bash 脚本语法检查通过。
- 对实际准备的数据执行 1/2/4 卡 × engineering/pilot/formal 共 9 种计划，均保持全局 32 条回答、4 道题；更新数分别为 2/96/384。只生成计划，没有初始化 CUDA 或训练模型。
- `outputs/validation/v1_2/test_report.json` 保存测试输出、代码、依赖声明、完整包清单及锁文件哈希；`distributed_plans.json` 保存 9 种计划；`environment.json` 明确记录本机数据就绪、目标 GPU 环境未就绪。
- `outputs/validation/v1_2/package_inventory.json` 保存本机下载停止时的完成文件、部分文件和缺失包状态。完整目标依赖已锁定，不等于本机完整离线包已下载。
- `configs/archive/v1.1.json` 保留旧配置；v1.0/v1.1 证据目录不覆盖。默认实验输出改为 `outputs/v1_2/`。

### 服务器启动与验证边界

默认位于包含 `project` 的目录，服务器已提供合适的系统环境并放好模型：

```bash
bash project/code/setup_gpu.sh
bash project/code/run.sh all
```

第一条下载缺失依赖并安装，第二条自动选择卡数，经硬件检查、工程运行与 pilot 验收后执行正式实验。可先用 `bash project/code/run.sh all --gpus 4 --plan` 查看四卡命令，不会启动训练。

依赖解析、CPU 测试、分组计划与代码审查已完成；Linux GPU 实际安装、4090 上的 BF16/NCCL 运算、模型加载、训练速度、峰值显存与学习收益尚未实测。没有把多进程 CPU 文件测试当成 DDP 梯度训练证据。

## 2026-10-30 · v1.2.1 · 最终代码审查与两条启动命令

### 用户要求与范围

- 用户要求最后检查总体代码，提供一键环境安装命令和一键运行命令。本次执行代码、数据、依赖、入口与真实上游库关键行为的最终检查，未租用或连接 GPU 服务器。
- 保留服务器在线下载安装依赖的既定选择；未恢复本机约 4GB 的批量下载。模型仍由用户提供原始本地快照。
- `configs/archive/v1.2.json` 保存原配置；当前 `configs/v1.json` 版本为 1.2.1，实验数值参数、数据划分和默认输出 `outputs/v1_2/` 保持一致。旧验证证据保留，新审查写入 `outputs/validation/final_2026_10_30/`；后续普通 `check` 写入 `outputs/validation/v1_2_1/`。

### 已修复的问题

1. **固定依赖组合重复缩放梯度。** 精确核对锁定的 TRL 1.16.0、Transformers 5.3.0 和 Accelerate 1.10.1 wheel 后，确认 GRPO 已按 Trainer 累积窗口规范化损失，Accelerate 的 backward 又按其插件步数相除。单卡/双卡/四卡分别多除以 32/16/8，导致不同拓扑的更新幅度不一致。修复仅把 Accelerate 的额外 backward 除数设为 1；Trainer 的微批次数、同步边界、优化器/调度器步数及 GRPO 生成缓冲节奏不变。程序严格检查固定实现的预期状态，拒绝未验证的 DeepSpeed/FSDP 路径，并在 manifest/report 中记录 `loss_normalization`。训练与选点均检查每个 rank 的归一化验收结果。
2. **安装前置检查太晚。** 以前先下载依赖，再检查已有虚拟环境和 pip。现在 64-bit CPython 3.11、venv/ensurepip、受管 `.venv`、解释器归属和 pip 都先通过检查，才下载。缺环境支持或损坏环境直接报错，避免白等大文件；有效的受管环境可继续安装。
3. **评测恢复与选点来源未完全绑定。** 冻结协议新增 `select_checkpoint.py` 源码哈希；正式评估核查训练 report/manifest 的冻结哈希；恢复缓存核查冻结选择和结构题审阅记录。项目整体移动后的相对路径复用仍通过测试。旧来源发生变化时会拒绝混用，不会静默继续。

### 实际验证结果

- 最终 9 个 CPU 套件全部通过：24 个统计场景、37 个判分案例及数据隔离/oracle 检查、20 个依赖/下载/安装检查、14 个数据下载/来源检查、27 个评测检查、43 个缓存/选择检查、23 个模型布局/数据/计划检查、44 个启动入口检查、30 个训练奖励/归一化审计检查。
- 17 个生产 Python 脚本 AST 检查通过，无自定义函数或类封装；4 个 Bash 脚本语法检查通过。最终报告所列 35 个代码、配置、测试、安装与依赖锁文件哈希均与交付文件匹配。
- 再次运行实际 `prepare`：GSM8K 7,473/1,319 行及许可证哈希一致，处理划分保持 32/128/512/200/200；结构题 160 行，仍待独立审阅；网络下载和新增数据文件均为 0。
- 执行锁定 TRL/Accelerate wheel 的真实采样器类：32/128/512 道题 × 1/2/4 个分区共 9 个场景通过，验证每卡完整的 8 回答组、每次更新 4 道不同题及完整 epoch 覆盖。该检查使用已有 CPU PyTorch 2.6.0，没有执行 DDP 或模型。
- 执行锁定 Accelerate 1.10.1 的实际 backward 方法，使用已有 CPU PyTorch 2.1.2 做标量 autograd。目标全局平均梯度为 -33；修复前 1/2/4 分区分别得到 -1.03125/-2.0625/-4.125，修复后均为 -33。跨 rank 平均由算术模拟，6 个场景通过；没有执行 NCCL、BF16 或 GPU 训练。
- 新的合成端到端演示完成：8 题、两组各 16 条回答，共 256 条，经判分、2,000 次 bootstrap 和 2×2 绘图。所有结果均标记 synthetic，不是模型成绩。

### 可核查证据

- `outputs/validation/final_2026_10_30/test_report.json`：最终测试结果与 35 个文件哈希。
- 同目录 `baseline_tests.json`：本轮修复前的基线测试，说明原测试通过并不覆盖此次发现的上游组合问题。
- 同目录 `sampler_audit.py` / `sampler_audit.json`：真实上游采样器的可复现 CPU 检查，含 wheel、成员源码和复现脚本哈希。
- 同目录 `gradient_scaling_audit.py` / `gradient_scaling_audit.json`：六组标量梯度检查、上游来源和适用边界。
- 同目录 `environment.json`：当前 Mac 数据已就绪，目标模型与 CUDA 训练环境未就绪。
- 同目录 `demo/analysis/`：合成流程的 JSON、报告与图表。

### 最终启动方式

在包含 `project` 的目录执行，服务器需 Linux x86_64、CPython 3.11（含 venv/ensurepip）、glibc ≥ 2.28、NVIDIA 驱动 ≥ 570.26；原始模型放在 `project/code/models/Qwen3.5-2B-Base/`：

```bash
bash project/code/setup_gpu.sh
bash project/code/run.sh all
```

不需要手动激活环境；自动选择单机 1/2/4 卡。第一条准备依赖，第二条执行主实验与图表；结构题的独立审阅和 `structural` 阶段不包含在主流程内。真实 4090 安装、模型加载、显存、吞吐、完整 GRPO 训练及模型收益尚待实机验证，不能把 CPU 审查当作 GPU 成功运行证明。

## 2026-10-30 · v1.2.2 · 环境、模型和数据一键准备

### 用户要求与范围

- 用户要求把模型下载加入 `setup_gpu.sh`，运行一条准备命令后具备开始实验所需的依赖、模型和数据，并显示进度条。此要求替代先前“假设模型已经下载”的安装前提。
- 完整下载在 Linux 4090 服务器执行。本机获取官方小型文件元数据，并对权重进行仅 16 字节的 HTTPS Range 探测；不下载完整模型或继续下载 Linux GPU 依赖。
- `configs/archive/v1.2.1.json` 保留上一版配置；当前配置版本更新为 1.2.2，训练和评测参数不变。

### 修改内容

- `setup_gpu.sh` 增加七阶段进度显示，依次准备系统与虚拟环境、依赖下载、依赖安装、CUDA 检查、模型、数据及最终就绪检查。
- 模型固定为官方 `Qwen/Qwen3.5-2B-Base` 的 `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`。新增 `models/qwen3.5-2b-base-manifest.json`，记录官方 API 返回的 12 个文件和大小，总计 4,571,206,192 bytes；LFS 文件使用 SHA256，普通 Git 文件使用 Git blob SHA1。
- 新增 `scripts/download_model.py`，显示下载和字节校验进度，先验证已有文件，再补齐缺失文件。完整但冲突的文件保留并报错；成功凭据为模型目录 `source_manifest.json` 的 `status=verified` 和 `all_files_verified=true`。
- 检查实际锁定的 `huggingface_hub==1.33.0` wheel 后发现其 HTTP 下载失败时会删除未完成的临时文件，因此模型下载采用显式 HTTPS Range 续传。部分内容保存在模型目录 `.cache/rlvr-download/`，核对响应范围、长度和最终文件哈希后才安装为正式文件；网络失败和重跑均保留已完成字节。
- `prepare_packages.py` 增加依赖下载字节进度条，保留原有分块、断点续传与最终哈希校验。
- 在线模式下载或复用 GSM8K 并准备固定划分；`--offline` 同时约束依赖、模型和原始数据，缺失文件时停止。现有数据下载器已支持离线检查，本次直接复用该接口。
- 安装结尾检查本地 tokenizer、模型分片布局、数据来源、自动选中的 GPU 和 NCCL 通信。GPU 排序与实验启动器一致，显式 `CUDA_DEVICE_ORDER` 和设备掩码会保留。
- 每次检查输出到独立的 `outputs/environment/setup-XXXXXXXX/`；只有所有步骤成功才保存 `ready.json` 并打印 `✅ [SETUP COMPLETE]`。准备过程不运行模型推理或训练。
- 更新 `READM.md`、`README.md` 和模型目录说明；训练命令仍为 `bash project/code/run.sh all`。

### 验证边界

- `tests/test_download_model.py` 使用小型模拟文件验证离线、哈希、冲突、中断和续传行为；`tests/test_setup_workflow.py` 执行实际 shell 入口并模拟外部环境，验证在线/离线参数、失败停止和就绪报告。最终测试证据保存于 `outputs/validation/v1_2_2/`。
- 最终执行 `python3 project/code/tests/run_all.py`，11 个 CPU 套件全部通过；模型下载检查 28 项，安装流程和就绪检查 24 项。18 个生产 Python 脚本保持无自定义函数或类，4 个 Bash 入口和安装器内嵌 Python 语法检查通过。
- 实际核对已有 GSM8K，`network_files=0`、`installed_files=0`，数据划分和结构题均 `new_files=0`，`doctor --data-only` 通过。结构题仍需独立审阅。
- 官方权重的有界 HTTPS 探测请求 `bytes=8-23`，返回 HTTP 206、`Content-Range: bytes 8-23/4548221488`，只读取 16 字节后关闭连接；这验证下载源的范围请求能力，不代表完整模型已经下载或 GPU 运行成功。
- 未执行真实权重完整下载、目标 Linux GPU 安装、Qwen 推理或 GRPO 训练；峰值显存、吞吐与模型收益仍需服务器工程阶段验证。

## 2026-10-30 · v1.2.2 部署路径补丁 · 独立 code 目录

- 用户将云端部署目录指定为 `/root/code`，并提供安装完成日志。随后 `run.sh all` 在 CHECK 阶段的 `tests/test_pipeline.py` 第 33 行失败，尚未进入模型推理或训练。
- 已读取用户上传的 `test_report.json`，其中 39 个源码、配置和依赖清单哈希均与本地修复前文件一致。失败由测试中的 `cwd=root.parents[1]` 与硬编码 `project/code/run.sh` 组合导致：部署在 `/root/code` 时，该测试从 `/` 查找不存在的 `/project/code/run.sh`。
- 修复 `tests/test_pipeline.py`：在临时目录中分别构建 `project/code`、独立 `code` 和含空格的重命名目录，运行实际启动器的 `prepare --plan`。测试显式使用当前 Python，验证相对命令、零执行和无输出写入；失败时保留子进程 stdout/stderr。生产代码、实验配置、依赖锁及模型清单不变，配置版本仍为 1.2.2。
- 更新 `README.md` 和 `READM.md`，说明 `/root/code` 部署方式及旧测试错误的恢复方法。云端只需同步修复后的测试文件，再运行 `bash run.sh all`，不需要重新安装环境或下载模型。
- 本地先在搬迁后的目录复现旧测试第 33 行失败及 `No such file or directory`，然后针对修复运行启动测试，46 个场景通过。另将整个项目复制到独立临时 `code` 目录执行全部 11 个 CPU 测试套件，全部通过，用时 32.49 秒；报告中的 39 个文件哈希与修复后的本地文件一致。
- 证据保存在 `outputs/validation/v1_2_2_pathfix/`：`server_test_report.before.json` 为用户提供的原报告，`test_pipeline.before.py` 保存旧测试，`reproduction.before.json` 记录复现，`test_report.relocated.json` 和 `full_suite.relocated.log` 记录搬迁后的完整检查。
- 云端环境准备成功来自用户提供的日志；本次修复验证仅执行本地 CPU 测试，没有通过 SSH 修改服务器或启动 GPU 训练。真实模型显存、吞吐和训练结果仍待后续工程阶段验证。
