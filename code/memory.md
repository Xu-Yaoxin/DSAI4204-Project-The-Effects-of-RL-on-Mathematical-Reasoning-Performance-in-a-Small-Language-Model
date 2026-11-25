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

## 2026-10-30 · v1.3.0 · 训练断点恢复与整套流程恢复

### 用户要求与已知云端状态

- 用户提供的日志显示 `/root/code` 已完成安装，随后完成了一部分 `engineering_base` 回答生成。该步骤是未经本项目训练的 Base 评测；日志不能证明正式训练已经开始或完成。
- 用户明确要求加入断点续跑后重新启动。新入口保留已保存的 baseline 样本和现有安装环境，恢复任务不会重新下载模型。
- 配置版本更新为 1.3.0，新增 `training.recovery_save_steps=16`；`configs/archive/v1.2.2.json` 保存旧配置。模型、数据划分、奖励、LoRA 超参数、每次更新的题数和回答数保持不变。

### 训练与检查点

- `run.sh all --resume` 将恢复参数传给 engineering、pilot、formal、baseline、选点和评测。训练仍检查 GPU 和对应拓扑；独立选点/评测的完整缓存可在不加载模型的情况下验证复用。
- `train_grpo.py` 接入 `trainer.train(resume_from_checkpoint=...)`。已有阶段完成且全部验收通过时验证并跳过；中断阶段从最近一个完整、已封存的检查点继续；尚无完整检查点时归档旧尝试，从第 0 步重新训练。配置、源码、数据、模型、依赖或 GPU 拓扑不一致时拒绝恢复。
- 默认工程训练每步保存，pilot/formal 每 16 个优化器步保存。`training_checkpoint_event.py` 在所有 rank 保存后调用 `training_resume.py`，校验权重、优化器、调度器、Trainer/RNG 状态及奖励记录，计算文件哈希并写入 `resume_state.json`。未封存检查点不能被当作可恢复状态；已封存检查点损坏会报错，不静默回退。
- 恢复衔接 GRPO 微步计数和 rollout 边界，并重建 RepeatSampler 独立随机数发生器已跨过的 epoch 状态。每个 rank 的奖励日志恢复到同一检查点边界；超出边界的日志、未完成检查点和派生产物先归档，避免重复计数。report/manifest 采用临时文件后替换的写入方式。
- 保留所有恢复检查点，但开发集选择仍只评估 epoch 结束候选：当前 formal 的第 128、256、384 步，共 3 个。增加恢复频率不扩大开发集搜索预算。
- `resume_history` 和 `attempt_start_step` 记录每次恢复。耗时、峰值显存和原始 Trainer 吞吐是当前尝试的遥测；全程逐步损失日志平均值记录为 `mean_logged_step_loss`，不能将恢复后最后一段的原始 `train_loss` 当作全程均值。

### 云端迁移与启动

- 默认服务器为 `root@xj-member.bitahub.com`，SSH 端口 `42258`，目录 `/root/code`。Mac 本地 `code` 目录执行 `bash deploy_resume.sh`，先停止该项目的旧进程，再同步源码并调用服务器后台恢复入口；环境、模型、输出和旧日志不随源码上传覆盖。
- `resume_run.sh` 先检查是否已有同项目进程，再准备旧输出迁移，以 `nohup setsid` 启动 `all --resume`。每次日志写入 `logs/train-*.log`，`train-current.log` 指向当前日志；任务已停止后再次执行该脚本即可恢复。
- `stop_run.py` 按项目工作目录和实际运行脚本定位进程，并处理该流水线的子进程；`prepare_resume_upgrade.py` 备份后修复可确认的 baseline JSONL 尾部中断记录，保留完整样本。
- 旧 v1.2 训练没有新恢复协议，不能直接续训。升级工具将旧 engineering/pilot/formal 及其关联下游结果保留归档，再按 v1.3 从 Base 开始对应训练。已有 v1.3 状态保留给恢复逻辑处理。

### 验证与边界

- CPU 回归覆盖真实生产回调、检查点封存和恢复、篡改拒绝、半写入、奖励日志备份回退、目录搬迁，以及 1/2/4 卡的启动参数和恢复分发。`tests/test_training_resume_integration.py` 的 57 项检查执行实际回调和恢复代码块，使用小文件、标准库替身和模拟 collective，不执行 GPU。
- `outputs/validation/v1_3_resume_upstream/` 保存固定上游代码的 186 个 CPU 场景：147 个真实采样器/分片场景、27 个 GRPO 缓冲与随机状态场景、12 个真实 Trainer 优化器/调度器加载及 CPU AdamW 继续更新场景；三个未修复的跨 epoch 恢复错误也已复现。
- 完整当前测试报告写入 `outputs/validation/v1_3/test_report.json`；开发过程及升级前文件保存在 `outputs/validation/v1_3_resume/`。旧版本报告不覆盖。
- 最终运行 `python3 tests/run_all.py`：16 个测试套件全部通过，用时 42.60 秒；报告记录 50 个生产脚本、测试、配置、启动脚本和依赖清单的 SHA256。包含 56 个检查点恢复、57 个回调/恢复集成、17 个运行目录保护、24 个迁移/进程选择及 22 个部署/后台启动检查。六个 Bash 入口语法、22 个无自定义函数或类的生产 Python 脚本 AST 检查通过。
- 部署测试只调用模拟 SSH/scp/rsync；后台启动测试使用独立的短时等待进程并在结束时终止。没有通过测试连接服务器、上传文件或停止真实训练。
- 没有执行真实 Qwen/RTX 4090 训练中断后的完整恢复，没有实测 GPU 训练峰值显存、吞吐或数学收益。部署脚本运行成功与模型训练完成是不同状态，后者必须根据云端训练报告和验收门槛确认。

## 2026-10-30 · v1.3.0 验证记录 · 4B 可行性与模型规模选择

- 用户提供新的云端日志：2B pilot 已完成 96 步，formal 已进入 13/384 步。用户另贴回 pilot 报告字段：`peak_cuda_allocated_bytes=4307286528`（4.0115 GiB）、`peak_cuda_reserved_bytes=4435476480`（4.1309 GiB）、`training_seconds=13121.818159399554`。这是用户提供的真实运行报告字段，不是本地 GPU 复测；本次只读 SSH 尝试未通过认证，没有执行远端命令。
- 用户要求验证 Qwen3.5-4B-Base 是否适合下次运行，并判断 2B/4B 哪种更适合检验当前研究假设。本次没有改动生产训练脚本、配置、依赖锁或当前实验。
- 已读取官方 4B revision `1001bb4d826a52d1f399e183466143f4da7b741b` 的配置、文件索引和全部权重文件头，校验 Git blob 身份及 Range 响应范围/长度。完整权重未下载，其全文件 SHA256 未验证。
- 在独立 Mac 临时环境中使用 torch 2.10.0、Transformers 5.3.0、PEFT 0.18.1、Accelerate 1.10.1，构建完整尺寸的 meta 模型并注入项目同规则的 rank 16 LoRA。2B 的 10,911,744 个 LoRA 参数与用户日志一致；4B 的 426 个文本张量键名和形状全部匹配，128 个目标模块、21,233,664 个 LoRA 参数，文本骨干 4,205,751,296 个参数，BF16 权重约 7.8338 GiB，输入输出嵌入共享保持。
- 技术结论：4B 已通过当前可执行的结构/容量预检查，结合 2B 实测峰值，值得在单张 24GB 4090 上做短程验收。meta 验证没有实际权重存储或前后向计算，不能称为已完成 4B GPU 训练、显存峰值或速度验证。当前生产加载器和下载器仍锁定 2B，尚未交付可直接运行的 4B 全流程。
- 重新核对 proposal 第 1–2 页：检验单次增益为正且大于 pass@16 增益，并比较数值/关系变体。建议保留 2B 为主实验，将 4B 作为下一组规模对照候选。2B 当前已有混合奖励信号；4B 若 Base pass@16 接近满分，会受到天花板效应，不能把较小的高采样增益直接当作机制证明。最终选择应根据独立开发集上的地板/天花板诊断、耗时与预算，不能根据测试集哪组更符合假设来选模型。
- 证据及可复查脚本保存于 `outputs/validation/qwen35_4b_feasibility/`，汇总为 `summary.md`，包含官方固定版本、meta 架构检查、组件内存审计、用户 pilot 峰值来源和实验设计建议。没有启动 4B 训练，没有更改正在运行的 2B 进程。

## 2026-10-31 · 云端 2B 主实验归档复核 · 判分敏感性

- 用户下载服务器 `/root/code` 至 `project/server_results/run_20261031/code/` 后要求检查。实际归档配置是 **v1.2.2**，没有 v1.3 恢复协议；不能将本次完整运行作为 v1.3 GPU 断点恢复的证据。
- 工程、pilot、formal 已分别完成 2、96、384 步，全部工程验收通过。formal 用时 48,698.883 秒（13.5275 小时），allocated 峰值 4.0141 GiB、reserved 峰值 4.1543 GiB。开发集 128/256/384 步得分为 76.0%/79.5%/81.5%，冻结选择 checkpoint-384。
- 复核原始/处理数据哈希、数据 ID/题目/答案和划分隔离、源码和奖励日志身份、8 个检查点关键文件、3 个候选 adapter 哈希，以及 6,400 条测试记录的网格、协议和种子。重放原判分器全部一致，独立组合数计算一致；重跑 2,000 次配对 bootstrap 后 summary 一致，其余统计报告逐字节一致。基础模型权重按备份约定未下载，未重新计算其哈希；优化器/RNG 文件未反序列化。
- 原始结果：Base/RL pass@1 为 45.34375%/79.0625%，增益 33.71875 pp（95% CI 30.75 至 36.65625）；pass@16 为 94%/95.5%，增益 1.5 pp（CI -1 至 4）。覆盖 both/base_only/rl_only/neither 为 186/2/5/7。
- 发现实际判分缺陷：`Answer: #### 5` 会同时提取合法标记答案 `5` 和无效重复候选 `#### 5`，后者使本应可判的输出成为 extraction_failed。真实样本 `gsm8k-test-001010` sample 0 的计算和答案 5 正确，仍被漏判。
- 在独立诊断副本中仅跳过这类重复候选，同规则重评全部 Base/RL 输出，原始归档和主结果未改动。Base 601 条状态变化，434 条恢复正确、167 条明确判错；RL 不变。事后诊断 pass@1 为 58.90625%/79.0625%，增益 **20.15625 pp（CI 17.84297 至 22.5）**；pass@16 为 94.5%/95.5%，增益 **1 pp（CI -1.5 至 3.5）**；两种增益之差为 19.15625 pp（CI 15.65625 至 22.4375）。覆盖为 187/2/4/7。
- 13.5625 pp 的变化是对该评分修正的敏感程度，不是训练收益的因果分解。其他提取边界和推理正确性尚未完全审查，剩余增益不能直接称为纯数学推理提升。52 条抽样回答由助手做定性复核，不是独立人工盲审。
- 同一判分器用于训练奖励；窄修正会改变已保存 pilot 3,072 条回答中的 94 条奖励、formal 12,288 条中的 118 条。评测重评分不会追溯性修正训练过程；原 checkpoint-384 也仍是旧开发集评分选出，未重新选点。
- 当前数值关系支持有限预算下的操作性假设，但不能确立“没有新增推理能力”的机制。Base pass@16 接近天花板，高采样增益空间有限；区间跨零不等于覆盖没有变化，观察结果也不支持覆盖收缩。
- 建议优先完善评分规范与对称敏感性分析，再完成结构题独立审阅和评测，最后安排 4B/新训练版本。当前归档无 `structural_review.json`，`all` 未运行结构实验。
- 全部复核证据保存到 `outputs/validation/server_results_review_20261031/`，主报告 `review_report.md`，图 `review_overview.png`，包含完整性、原报告重放、独立统计、回答抽检和训练奖励敏感性。生产代码和归档未修改，没有启动云端任务。

## 2026-10-31 · v1.4.0 · 生产判分器修复与本地离线重评分

- 用户要求完善判分器和回归测试，并给出无需模型的本地运行命令。已修改生产 `scripts/verify_answers.py`，修复 `Answer: #### 5` 重复候选、`answer is: 5` 分隔符；支持精确 XML 数值包装、CRLF、有后续答案的空标题和完整粗体答案标签。保留精确有理数、矛盾/非法候选拒绝、数值与格式/截断分开记录。
- 配置标识升至 1.4.0，旧配置保留 `configs/archive/v1.3.0.json`；模型、数据和训练数值超参数不变，但提取规则改变，所以不能混用旧奖励训练/选点/评测身份。新训练须另开运行目录并重新做工程和 pilot；本次未启动云端训练。
- 新增 `rescore_local.sh`、`scripts/rescore_saved.py`、`tests/test_rescore_saved.py`、`tests/test_verifier_regressions.py`。本地 `bash rescore_local.sh` 使用 `python3 -S`，只依赖标准库，不需要模型权重、PyTorch、GPU 或网络。
- 入口先执行五组相关回归，再核验下载归档的原始/处理数据、样本网格、原判分重放、原代码/配置/协议、训练报告和 adapter 哈希、原冻结选点及配对种子。同规则重评 Base/RL 6,400 条及全部开发候选 2,400 条。原生成身份保留；新评分通过 scoring_revision 单独记录，每条保留 original_score。
- 原归档未修改；默认新结果 `outputs/rescore_v1_4/`，主分析显著标记 posthoc。完成标志最后写入；同输入、代码和参数的完整输出经哈希验证后复用，改变或损坏时拒绝覆盖。主测试仍使用旧规则冻结的 checkpoint-384；新开发集排名仅作敏感性记录。
- 实际一键重评 8,800 条成功，2,000 次按题配对 bootstrap 成功，第二次运行完整校验并复用成功。Base 正确数 1,451→2,074，RL 2,530 不变；Base 889 条评分变化，其中 627 条恢复正确、4 条因保守标题/非数值规则变为 extraction_failed。三个开发候选完全不变，仍选 384。
- v1.4 事后结果：pass@1 Base/RL 为 64.8125%/79.0625%，增益 14.25 pp（95% CI 11.96875 至 16.4375）；pass@16 95%/95.5%，增益 0.5 pp（CI -1.5 至 3）；两种增益之差 13.75 pp（CI 10.5 至 16.71953）。覆盖 both/base_only/rl_only/neither=188/2/3/7。不得与上一版仅修嵌套标记的 58.90625% Base 诊断混称。
- 4 条降级样本的数值本身与 gold 一致，原因是空 Answer 标题或标题后的自然语言/单位句；它们是保守提取边界，不是数学错误，全文和说明保存到 `outputs/validation/v1_4/scoring_review.json`、`scoring_review.md`。当前没有按这四条定向修改分数。
- 全部 18 个 CPU 套件通过（37.74 秒），23 个线性生产脚本语法通过；包括原 37 判分案例、新 118 判分场景、31 统计、27 离线复核及 30 训练奖励接口场景。离线 fixture 主动阻断网络和模型包导入。报告和源码哈希在 `outputs/validation/v1_4/test_report.json`。
- 判分器检查最终数值，不验证自然语言推理。重评分不能修正过去的训练奖励，也不是新评分协议下的独立确认实验。README/READM 已更新本地命令、输出路径、协议边界及真实云端训练完成状态；真实 v1.3 GPU 中断恢复仍未实测。

## 2026-10-31 · boundary-1.0 · 100 题 Base 难度校准与单卡双进程评测

- 用户决定只依据原始 Base 寻找 pass@1 约 40%–50% 的题组，并将规模改为约 100 题。新集固定为 5×20：GSM-Symbolic main/P1/P2，以及 MATH-500 的数值、无图 Level 1–3/4–5 子集。选题未使用模型输出；种子 20261031。100 题 SHA256 为 `c4dc85fa10b9123a07749a45a1b9031ec1c6765166497d1795954b8af6b0e40c`。
- Apple 源固定 revision `93b5b3758d9d9841ffe81d6cd2ae2b030685b078`，H4 MATH-500 固定 revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`。原始池、数据卡、许可证及 Git blob/文件哈希保存于 `data/boundary_v1/raw/` 和 manifest。GSM 三组配对 20 个家族/实例索引，排除原 200 测试家族与已使用文本重合；不宣称各版本数值严格相同。MATH 排除 42 条图形题及 109 条非标量答案，剩余 349 条中按 6/7/7/10/10 的 level 配额抽样。所选 20 道 P2 的答案已独立算术复核；不冒称全部 100 题均经人工盲审。
- 新增 `prepare_boundary_data.py`、`evaluate_boundary.py`、`analyze_boundary.py`。原始 2B Base 内容指纹锁定 `807045c0c4d4abf41d3644c7587894102f9d60e219367a1ef45b30c60a05c181`；不加载 adapter。提示只含题目，gold/source solution/canary 不进入提示。4 次独立采样，共 400 回答；温度 0.6、top-p 0.95、top-k 0，BF16，max_new_tokens 1024。与旧 512-token 主实验不同，不能混称同协议成绩。
- 用户提供运行图：13 CPU 配额，训练阶段约 60%–70%、评测约 20%；56 GiB 内存总量，约 8–11 GiB 使用；GPU 平均利用率 24.4%、最大 40%，显存最大 4.63 GiB。基于这些图可以尝试双进程；它们不是本次并行实测。单卡 CUDA 多上下文有时间片竞争，不能承诺 2 倍提速。
- 新增独立 `run_boundary_parallel.sh`、`scripts/run_boundary_parallel.py`、`scripts/merge_boundary.py`。两进程共享一张 GPU，各处理 50 题，且每组各 10 题，保持每题4次与原种子。CPU 每进程最多4线程、interop1；依据 affinity/cgroup CPU 配额限流。双进程保守门槛为空闲显存14 GiB、容器可用RAM28 GiB；不保证运行峰值。任一 worker 失败会回收兄弟进程，保留分片，不发布不完整分析。
- 各分片按回答 fsync 并校验身份后续跑；`--sequential` 能依次恢复同样两分片，降低并发内存要求。完整缓存先严格校验，可在不初始化CUDA的情况下完成合并/分析。合并逐条验证归属、源数据、模型/协议、完整网格及判分重放，原子发布；结果保存 `outputs/boundary_v1_parallel/`，总吞吐与启动资源记入 invocation 文件。
- 全部24个CPU套件通过，28个线性生产脚本通过语法检查。包括21数据、59评测/分片/缓存、31统计、43合并/真实脚本接口、8入口和10调度场景。调度测试启动真实CPU fixture子进程，模型/GPU生成使用模拟；真实 evaluator→merge→analyzer 与串行结果对齐（耗时除外）。报告位于 `outputs/validation/boundary_v1/test_report.json`。真实4090双进程速度、峰值和新数据正确率尚未实测。
- 独立上传包为 `boundary_probe_v1/`，部署到原服务器 `/root/code/boundary_probe_v1/`，读取 `../.venv/bin/python` 与 `../models/Qwen3.5-2B-Base`。不覆盖旧训练源码、判分器、检查点及历史输出。源文件/数据/许可全包哈希、验证报告、README/READM/memory随包提供；完整原始池仅保留本地。
- 分组报告给出 pass@1、95%题目家族bootstrap区间、pass@4、截断与提取失败诊断，按距离45%排列来源组。20题/组仅用于探索，4次回答不是80道独立题。后续必须用新题、对GSM排除整个摸底家族，确认难度并再比较Base/RL；不按个别失败或RL收益挑题。本次未调用远程模型、未上传、未启动服务器任务。

## 2026-10-31 · SSH connection update

- User updated the server SSH port from 42258 to 42260: `ssh -p 42260 root@xj-member.bitahub.com`.
- Upload/download instructions updated. Remote project remains `/root/code/boundary_probe_v1/`; no runtime code or data changed.

## 2026-10-31 · boundary-1.0 云端结果复核 · 难度候选与提取边界

- 用户通过更新后的 SSH 端口42172下载评测包至 `project/server_results/boundary_20261031/boundary_probe_v1/`。本次读取真实下载结果，原48个归档文件保持不变；不是模拟生成。
- 验证包26项文件哈希、两个worker完成状态和样本哈希、100×4完整无重复网格、同一原始2B Base指纹/无adapter、相同生成配置及源码/数据身份。模型权重未随包下载，本地未重新散列全量权重。原始题池及固定抽样复核成功。
- 本地使用归档代码重合并、重放判分并重算2000次bootstrap；8个合并/分析输出逐字节一致。验证文件在 `outputs/validation/boundary_review_20261031/`；报告 `review_report.md`，2×2图 `boundary_overview.png/.pdf`。
- 原定五组固定评分 pass@1 / pass@4：GSM main51.25%/85%，P1 30%/55%，P2 23.75%/60%，MATH L1–3 68.75%/95%，MATH L4–5 25%/55%。每组20题、每题4次；main的95%按家族bootstrap区间38.75%–63.75%。四次全错不等于绝对不会做，pass@4不能替代pass@16。
- 事后按来源等级拆分MATH：L1为18/24=75%，L2为22/28=78.57%，L3为15/28=53.57%，L4为16/40=40%，L5为4/40=10%。L4只有10题，探索性2000次按题bootstrap区间25%–55%，截断7/40=17.5%；L5截断16/40=40%。下一轮候选是main和MATH L4，均需新题确认。细分来自源等级，不是按具体题答对/错选择；区间未作多重选组校正。
- 运行实际完成：单RTX4090双独立进程，400回答、135788输出tokens，生成墙钟1880.2966秒（31分20秒），聚合72.216tokens/s。worker分别1867.17/1734.40秒，PyTorch allocated峰值3.559/3.581GiB。不是整卡全局峰值，也没有同批题串行对照，不能声称2倍加速。
- 复核全部32条未截断的提取失败/歧义输出，确认其中15条有清楚的gold一致最终数值：main4、P1 3、P2 3、MATH低难5。问题包括思考区段引用占位符、未覆盖的“final numerical answer”标签、空标记/跨行boxed和货币格式。属于助手定性复核，不是独立人工盲审；36条已截断提取失败未全部完成语义裁定。
- 另对全部400条统一做事后分段诊断：仅当恰有一个</think>且后缀非空时对后缀用原判分器，其余保留全文；10条满足、8条状态改变、6条恢复正确（main/P2/MATH低难各2），无原正确降级。该诊断不读gold来选择跨度，仍未冻结为生产规则。与15个定性漏判计数重叠，不能相加；原结果和生产判分器没有修改。
- 截断抽查至少6例已给出明确正确答案，随后反复讨论格式直到上限（main1、MATH5），与15个未截断案例不重合；未自动补分。更多tokens可能只延长重复，因此须区分未解完、错误计算、正确答案后的重复。MATH高难23条截断与24条提取失败有21条重叠，不能相加。
- 个别GSM比例/植株间距措辞有歧义风险，未确认gold错误。下一步应统一提取规范和模板审查，再用未见家族做main确认及更多MATH L4新题校准；P1/P2作加难，L5暂作压力测试。当前三版本共同未用GSM家族剩24个，不能把同家族换数值当大量独立新题。此轮仅Base，尚不能判断哪组RL收益最大。


## 2026-11-01 · boundary-confirm-2.0 · 200 题原生上下文复核包

- 用户要求 GSM-Symbolic main 与 MATH Level 4 各 100 题，并明确选择原生 262,144 减实际输入 token 数。新增独立 `boundary_confirm_v2/`；仅 Base、BF16、每题 4 次、默认同卡两进程，共 800 回答。原训练代码、旧包及服务器归档保持原样，81 个受保护文件哈希复核无变化。
- 冻结题集 SHA256 `5eb5775c988682af461100a64bf8de0e6189b592c98b64ed5a52e73bce598964`。GSM 覆盖 64 个新家族（36×2 + 28×1）；MATH-500 剩余合格 Level4 仅77题，改用固定 EleutherAI/hendrycks_math revision `21a5633873b6a120296cce3e2df9d5550074f4a3` 的完整原始 test 池，从832道未用标量无图Level4题随机取100。数据选题 seed20261101，无个题模型结果参与筛选。
- 新判分器仅用于独立包，源保留为 `scripts/verify_answers_v2.py`；支持 numerical answer 标签、成对think隔离、空末尾标记、多行boxed展示及Unicode货币，保留矛盾/非法候选拒绝。旧400条离线回放17条恢复正确、1条百分号候选变为解析失败，仅诊断，未重写旧分数。
- 原生额度保存为每回答generation_limit，cache/merge/analyze核对；完整回答可续跑，中断的单条回答从同seed重跑。启动GPU空闲门槛双进程20GiB/单进程11GiB，系统内存32/16GiB；静态KV估算原生长度每进程3GiB，不能等同GPU实测。
- 验证：数据19、判分114、生成/续跑66、合并/分析44、调度10项CPU检查全部通过，双/单并发plan通过，按实际rsync排除规则复制后的包哈希/plan通过。没有本地GPU推理；原生长度速度/峰值尚待云端验证，重复输出可能运行很久，不能承诺无截断。
- 上传约0.55MB独立包，复用 `/root/code/.venv` 和模型；SSH端口42172；目标 `/root/code/boundary_confirm_v2`。具体上传/nohup命令保存在包内 `COMMANDS.md`，版本记录和README随包冻结。


## 2026-11-01 · boundary-guarded-3.0 · 串行实时监控与限时保护

- 用户明确改为串行、单次回答 300 秒未完成计错误；保持原生 262,144 减实际输入 token 数。新增独立 `boundary_guarded_v3/`，复用 v2 同一 200 题、800 回答、模型和判分器，旧 v2 文件哈希复核无变化，不将旧 49 条结果混入新停止协议。
- 每 5 秒保存 token 计数、速率、剩余时间及含特殊 tokens 的输出；保留完整 token 轨迹。每增加 32 tokens 检查重复周期 1–256、至少 4 次、覆盖至少 1,024 tokens 的精确重复。
- `answer_timeout` 记 correct=false/reward=0；重复、原生上限、120 秒无新 token 单列数学正确性未知，主要成功率以全部请求为分母，附未知上下界和完成率。普通 OOM/模型异常作为基础设施失败，不造答错样本。
- 停止请求最多留 15 秒清理；已完成保护记录先验证、fsync，再退出75。父进程校验缓存后最多重启2次；完整分片无需重启。保存记录可以恢复，尚未保存的那次重新生成。
- 测试覆盖真实 CPU 线程队列、子进程硬退出/缓存恢复、严格串行、超时竞态、评分统计和实时查看；并修正最后一条保护记录完成分片时错误消耗重启预算的边界。未执行实际 GPU 推理。具体日志和发布检查在包内 validation/。
- 入口 `bash run_guarded.sh`，实时查看 `python3 -S scripts/monitor_guarded.py --follow`；使用已有 ../.venv 和模型，无新增依赖。远端路径 `/root/code/boundary_guarded_v3`，SSH端口42172。


## 2026-11-01 · guarded-review-20261101 · 真实 GPU 结果复核

- 已核对 Downloads/boundary_guarded_v3_outputs/guarded_v3 的825个文件；800条样本、800份token轨迹与733,721个tokens完整一致。重新合并及评分通过；样本、报告、逐题指标和候选排序逐字节一致，manifest/summary差异仅为路径来源及其哈希。未改下载数据或冻结代码。
- Base成功率：GSM main256/400=64.00%，MATH Level4标量无图子集175/400=43.75%；家族bootstrap95%区间分别55.24%–72.66%、36.99%–50.75%。MATH更接近40%–50%选材目标，但不能据此断言真实数学正确率严格落在目标区间。
- MATH EOS338、超时33、重复29；自然结束中正确率51.78%，四次至少成功一次68/100；各题成功0至4次分布32/17/14/18/19。GSM EOS398、重复2、无超时。
- 实际单卡4090串行运行：两个worker无重叠、均exit0、重启0；总墙钟约5h7m37s。33次超时在300.110–300.146秒结束；64条保护终止占回答8%却占逐回答时间64.00%。强制退出及自动重启分支本轮未触发。
- 审查全部45条EOS提取失败/歧义记录：GSM5、MATH4个明确格式漏判。仅补计这类的事后敏感性为65.25%、44.75%；旧官方分数不变。另有7个MATH自我纠正冲突、1个近似小数，需与格式bug分开。GSM1207-41存在来源措辞/标准答案隐含假设问题。
- 建议继续2B，以MATH Level4作为主实验候选，固定科目与过滤分布、抽取新训练/开发/测试题并排除校准集；先统一格式判分规则，再做同协议Base/RL比较。当前仅baseline校准，不能证明RL效果。
- 报告：outputs/validation/guarded_review_20261101/report_zh.md；四图overview.png/pdf；核验integrity_summary.json/runtime_audit.json；完整质量审查quality_audit.json/md。


## 2026-11-01 · math-l4-1.0 · 独立训练与配对评测

- 用户选择MATH Level4与2B，并明确保留每题16次正式Base/RL采样（6400回答、可能数天）。新增独立math_l4_v1/，不改变旧训练/复核包；旧v2/v3共86个冻结文件哈希无变化。
- 官方train准备512训练+128验证，官方test准备200测试；32engineering⊂128pilot⊂train。固定来源revision21a5633873b6a120296cce3e2df9d5550074f4a3，14parquet哈希，排除46历史文件9712规范化题目。训练与验证、测试互不重叠；按合格池科目比例抽样，不用答题正确率选题。
- 数据manifest0d6dc496cc6766a910452a10de90c7ee26af68f9613ef8a9d388266b225546d7。新scorer3.0SHAf133f1b99257d0c4677b11c44145577a87edefcc55a3d2cb42bcebbe9aa7944f；恢复旧EOS中的4MATH+3GSM格式漏判，无原正确答案降级。此前审查的另外2GSM百分号案例属于语义约定，保守不自动归一化；7自我纠正矛盾与1近似小数仍不放宽。
- GRPO保持LoRA16/32、lr5e-6、8回答/题、32回答/更新、3epochs；训练cap2048，非EOS奖励零。工程2步、pilot96步、正式384步；正式从Base重新初始化。每16步恢复checkpoint，epoch128/256/384全部在128dev×4上比较，同分选最早，冻结选择后才测试。
- 评测Base/RL同提示/seed/BF16/SDPA/判分/原生262144减输入/300秒限时与重复保护；同一时间一个模型进程，含全局运行锁、实时日志、受控终止后保存及有限重启。配对分析pass@1/4/16、bootstrap、覆盖与未知上下界，单seed有限覆盖不等于穷尽能力。
- 11CPU套件658项场景全部通过（73.80秒）；实际数据全部阶段计划、bash语法、rsync过滤后异地目录完整计划通过。新2048-token MATH GPU训练仍未实跑，工程与pilot闸门保留。包78个冻结文件，约9.03MB。
- 云端/root/code/math_l4_v1，SSH42172；复用../.venv及../models/Qwen3.5-2B-Base，无新依赖。运行bash run.sh all，准备bash prepare.sh，实时python3 -S scripts/monitor.py --follow；同命令断点续跑。完整COMMANDS.md、READM.md、memory.md与validation/随包。

## 2026-11-01 · math-l4-1.0.1 · 云端自检修复

- 云端CPU自检失败可在本地设置RLVR_CPU_THREADS=4后复现：挂起线程测试的模拟Torch遗漏set_num_threads和set_num_interop_threads，先前直接测试没有覆盖启动器设置的分支。
- 补齐两个模拟接口并断言参数，覆盖正常生成、超时保存后exit75及基础设施故障exit1；本地总测默认启用同一线程设置。生产脚本、数据、配置、模型和判分器不变。
- 更新冻结包并保留旧失败记录；重启用bash run.sh all --run-dir outputs/math_l4_v1_fix1，日志experiment-fix1.log。不可直接混用旧run_state的包哈希。详见math_l4_v1/COMMANDS.md与validation/thread_mock_fix/。

## 2026-11-03 · environment-export-1 · 配置与环境快照

- 新增独立 maintenance/export_gpu_snapshot.py 和 download_gpu_environment.sh，Mac 一条命令通过 SSH42172 调用云端实际 .venv Python，导出并下载配置与环境 ZIP；不修改冻结 math_l4_v1 包或当前运行结果。
- 包含 Base/tokenizer 配置、所有 adapter/训练参数元数据、冻结实验文件、依赖锁与实际安装版本、CUDA/GPU/CPU/RAM/cgroup、实际 runner/child 进程的环境白名单及最新进度。配置中的权重身份记录保留，不重新读取大权重；不复制可执行 .venv，不导出完整环境或认证包源 URL。
- 云端写 /root/code/diagnostics 下带时间目录与 latest ZIP；Mac 下载至 Downloads/math_l4_environment_时间，逐文件 SHA-256 验证后安全解压。按当前阶段完整回答的实测平均耗时打印 ETA，旧状态明确标记为快照估计。
- 本地真实下载快照烟雾验证通过：281 个归档条目，约6.43MB（本地未包含实际 Base tokenizer 文件，云端体积会不同）；验证与解压使用真实下载脚本内的同一段代码。当前样本210/512对应4.61小时估计；原下载810文件保持不变。缺失GPU/nvcc等命令记录为环境项错误，不影响导出。云端实际采集与 SSH 下载尚未在本次工具会话运行。


## 2026-11-04 — math-l4-rl-split-1.0

- Added `math_l4_rl_split_v1/` as an independent execution extension; original frozen MATH files are unchanged.
- Two disjoint 100-question / 1,600-answer RL shards use the existing checkpoint-384 selection, with identical scientific generation and scoring settings.
- Added download/verification/upload commands, a Base-completion handoff on the existing server, pinned setup for a second RTX 4090, durable shard recovery and strict merge provenance.
- CPU validation: 98 fixture checks plus real frozen-data/source plans passed; no GPU inference, network transfer or real process termination was executed locally.
- The initial local snapshot lacks six candidate-128/256 evidence files; download_current.sh fetches them before building the upload payload. Active Base outputs are excluded from uploads.
- Existing scoring/admission issues remain separate from migration; raw answers and historical selection are preserved.


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

## 2026-11-04 — recover slow dependency downloads from the old server cache

- User's second worker 42186 had only about 39 MiB cached and grew roughly 1 MiB per several seconds. Dependency download had not reached RL inference; no bandwidth estimate is asserted from completed-part progress alone.
- Added independent maintenance/reuse_wheels_from_source.sh and stop_wheel_download.py. The script first checks the old server's manifest/lock equality and all 84 cached wheel hashes. Only a complete matching source permits stopping the exact new-server downloader via pidfd; Base/evaluator/training processes are not targeted.
- Source wheels transfer directly into the new worker with a cache lock, ignore-existing behavior and a separate rsync partial directory. A full local hash verification precedes restarting setup_worker.sh and the requested shard under nohup. Partial public-download files are retained. The interrupted-transfer retry checks its saved stop receipt and refuses active setup/pip/evaluation processes.
- Validation: 47 mocked process-control CPU checks, bash/Python syntax and real local rsync fixtures for repeated copying and preserved partial files. No live SSH transfer, real process signals or GPU inference were executed by the assistant. Existing frozen experiment files remain unchanged.
