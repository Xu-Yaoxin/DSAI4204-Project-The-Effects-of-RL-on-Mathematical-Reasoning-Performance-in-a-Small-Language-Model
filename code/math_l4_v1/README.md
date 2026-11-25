# MATH Level 4 · Qwen3.5-2B-Base · LoRA-GRPO v1

本轮使用独立的新题研究：可验证奖励强化学习（RLVR）能否提高单次成功率，以及多次采样的成功覆盖是否同步提高。模型固定为原始 Qwen3.5-2B-Base；从 Base 新建 LoRA 适配器进行 GRPO，不使用旧 GSM 适配器、SFT 或教师解题过程。

这是待执行的实验代码。已有 CPU 回归、数据来源与流程检查，不代表新的 2,048-token 训练配置已在 GPU 上通过。完整命令见 [COMMANDS.md](COMMANDS.md)。

## 数据 / Data

| 用途 | 数量 | 来源 |
|---|---:|---|
| 训练 train | 512 | 官方 MATH train |
| 验证 dev | 128 | 官方 MATH train，独立于训练 |
| 测试 test | 200 | 官方 MATH test |
| pilot 验收 | 128 | 训练集内嵌套子集 |
| engineering 工程检查 | 32 | pilot 内嵌套子集 |

三个正式集合互不重叠，按固定 seed 20261102 和各科目合格池比例抽样。排除此前 46 个历史数据/结果文件所记录的 9,712 个规范化题目文本，包括前几轮校准题。没有使用正确率挑题。MATH 不提供可信的完整语义模板家族，因此这里的 family_id 表示规范化题目文本身份；不能把检查过文本重复说成已经证明不存在所有语义近似题。

仅保留 Level 4、可提取精确标量答案、不依赖图形的题目。训练池/测试池分别有 1,168/662 道合格未使用题。过滤改变了原始 MATH 的覆盖范围，结论只适用于这个子集。具体科目配额、源文件哈希、历史排除清单在 `data/processed/manifest.json`、`data/README.md` 和 `provenance/`。包内已包含实际 JSONL 数据和 14 个已验证的原始 parquet；运行时不再访问网络。

来源固定为 [EleutherAI/hendrycks_math revision 21a5633](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3)，保留原始 MATH MIT 许可。源码树中的 LFS SHA-256、原始行定位与抽样代码一同保存。数据过滤的 gold parser 固定，不随本轮表现调整。

## 判分 / Scoring

训练奖励和 Base/RL 评测共用 `scripts/verify_answers.py` 的标量判分器 3.0.0。只比较最终数值是否正确，不给格式奖励，不验证整段推理。

修复了内联数学标记、有限的单位后缀、结论等式，以及 “The final answer is correct/given in ... form” 这类说明句误触发答案提取的问题。全部修复有错误金答案、追加矛盾答案和格式破损等负例测试。对上一轮已保存的自然结束回答复核，仅恢复 4 条 MATH 与 3 条 GSM 明确格式漏判，没有让原本正确的回答变错。

仍保留严格边界：前后公开数值矛盾计歧义；有限小数与分数按精确值比较，不新增近似容差；百分号含义不靠 gold 猜测。此前 GSM 审查中两个 `60%` 案例实际涉及百分比/百分点约定，本轮没有把它们当成普通单位自动修复。详见 `validation/scorer_policy.md`。

完成政策与数值提取分开：训练中任何未以 EOS 结束的 rollout 都计零，即使中途出现了正确数字。评测中五分钟超时计零；重复、无 token 等保护停止单列未知，不获得已确认成功分。已有历史分数保持原样，本轮是新评分版本的新实验。

## 训练 / Training

- BF16、LoRA rank 16 / alpha 32，只更新语言模型 attention 与 MLP 投影；冻结 Base 权重。
- GRPO：每题 8 个回答，每次更新 32 个回答，即 4 道题；microbatch 1，梯度累积 32。
- learning rate 5e-6，3 epochs，`loss_type=grpo`、`beta=0`、组内奖励标准化、clip epsilon 0.2、每批 rollout 只优化一次。保留现有固定版本的梯度归一化兼容处理。
- **训练回答上限 2,048 tokens**，输入上限 2,048；输入超过上限直接报错，不静默截断。该训练预算与正式评测的原生长度政策不同，已明确记录。
- 工程阶段 2 次更新；pilot 96 次更新；正式训练 384 次更新。正式训练重新从 Base 初始化，不延续 pilot 的适配器。
- 每 16 个 optimizer steps 封存恢复 checkpoint；epoch 结束的 128、256、384 步作为验证候选。恢复需 optimizer、scheduler、RNG、TRL 采样状态、奖励审计和文件哈希均完整。

保留全部恢复 checkpoint 和逐组奖励审计，便于中断续跑与复核。训练控制台输出阶段、进度、奖励、梯度、长度、步耗时与保存位置；`reward_audit.rank-0.jsonl` 记录已生成的回答及奖励。训练使用 TRL 的批量 rollout，实时逐 token 预览用于串行评测；训练阶段另有进程心跳。

依赖沿用服务器已经安装的 Torch 2.10.0+cu128、Transformers 5.3.0、TRL 1.16.0、PEFT 0.18.1、Accelerate 1.10.1。相关接口按[固定 TRL 源码](https://raw.githubusercontent.com/huggingface/trl/v1.16.0/trl/trainer/grpo_trainer.py)核对。本包不自动升级依赖或安装新的 CUDA 内核。

## 评测与选点 / Evaluation and selection

全程串行使用一张 GPU。验证集对每个 epoch 候选、每题采样 4 次，共 1,536 个回答。按“确认正确回答 / 全部请求”选最佳 checkpoint，同分选更早的步数。候选全集、开发样本、训练报告、适配器和代码哈希绑定后写入不可覆盖的 `selection.json`。测试结果不参与选点。

测试集每题 **Base 与 RL 各采样 16 次**，共 6,400 个回答。用户已确认保留这个采样预算。两模型使用同一问题、提示词、采样 seed、温度 0.6、top-p 0.95、BF16、SDPA、判分器和停止规则；RL 唯一预期模型差异是开发集选定的 LoRA 适配器。

评测长度额度仍为 **262,144 减实际输入 tokens**。每次回答最多 300 秒；仍未结束就记 `answer_timeout`、0 分，保存部分输出后继续。每新增 32 tokens 检查精确周期重复，周期 1–256、至少 4 次、覆盖至少 1,024 tokens 才停止。120 秒无新 token 单列 `no_token_timeout`。正常停止请求最多给 15 秒清理；若线程不返回，先持久保存记录再退出，父进程校验缓存后最多恢复两次。普通 OOM/模型报错不伪装成答错，也不自动无穷重试。

评测每五秒写当前文本、token 数、速率及剩余时间。特殊 tokens 在 raw_text 与预览中保留，完整 token 轨迹可审查。已保存回答可续跑；被强制中断而未提交的当前回答需要重做。

最终分析报告 pass@1、pass@4、pass@16，RL−Base 增益及配对 bootstrap 区间、覆盖的 both/base_only/rl_only/neither，并报告 Δpass@1−Δpass@16。所有指标以固定请求数为基础，超时为预设零分；保护停止的未知结果另给上下界。置信区间以题目家族为重采样单元，不能把 16 次回答当 16 道独立题。pass@16 是有限采样覆盖，不证明穷尽模型能力；单训练 seed 也不能证明普遍的 RL 效应。

## 一键执行 / Execution

```bash
cd /root/code/math_l4_v1
nohup bash run.sh all >> experiment.log 2>&1 < /dev/null &
echo $! > experiment.pid
```

`all` 自动检查 bundle、CPU 回归、数据、已安装包、真实 BF16 CUDA 运算、模型哈希与实际 tokenizer 输入长度。然后依次执行工程 Base、工程训练、pilot、正式训练、三轮 dev、冻结选择、测试 Base、测试 RL、配对统计。任何验收失败都停止，不悄悄缩数据或换模型。

```bash
python3 -S scripts/monitor.py --run-dir outputs/math_l4_v1 --follow
tail -n 80 -f experiment.log
```

`bash prepare.sh` 仅检查准备；`bash run.sh plan` 展示预算而不加载模型；中断后再次运行同一 `all` 命令即可续跑。配置和源码变化会被拒绝混用，应创建新的冻结实验版本。

运行前至少需要 12 GiB 空闲显存、16 GiB 可用主机 RAM；这只是启动门槛，不是新训练配置的显存实测。实际适配以工程/pilot 验收为准。已完成的上一轮推理不能代替本轮训练验证。

按上一轮 MATH 400 个回答约 4.5 小时粗略线性估算，6,400 次正式评测可能约 72 小时，验证候选约 17 小时，训练耗时另计。数据已更换，RL 的长度和超时率也可能不同，所以这些只是预算估算，不能当完成时间承诺。后台任务与 checkpoint 可跨 SSH 断线继续；远端开发机必须保持运行。

## 结果 / Outputs

主目录 `outputs/math_l4_v1/`：

- `run_state.json`：当前阶段及实时状态。
- `engineering/`、`pilot/`、`formal/`：训练报告、奖励审计、适配器与封存 checkpoint。
- `dev/checkpoint-N/`、`selection.json`：开发集选点证据。
- `test_base/`、`test_rl/`：完整回答、token 轨迹、运行与身份清单。
- `comparison/report.md`、`summary.json`、`per_question.jsonl`：最终 Base/RL 配对结果。

上传包不含模型、环境和历史 outputs。所有命令使用包内相对路径，入口会切换到自身目录。CPU 测试无需模型、GPU 或网络：`python3 -S tests/run_all.py`。测试使用线程/子进程 fixture 证明控制逻辑，不应描述为 GPU 训练已完成。发布验证记录在 `validation/`。

1.0.1 修复后总测：**11 个 CPU 套件、661 项场景全部通过**，使用启动器的线程环境。旧 1.0 直接测试没有覆盖 `RLVR_CPU_THREADS=4`，云端因挂起线程测试替身缺少接口而中止。已修复两个模拟接口并验证正常生成、超时与错误退出；训练前失败记录保留，新运行使用 `--run-dir outputs/math_l4_v1_fix1`。完整更新、重启和监控命令见 `COMMANDS.md`；真实 GPU 训练未在本机执行。
