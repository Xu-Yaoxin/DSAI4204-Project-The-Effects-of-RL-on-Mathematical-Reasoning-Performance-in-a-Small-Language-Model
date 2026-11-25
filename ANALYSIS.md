# MATH Level 4 实验分析

## 1. 研究问题与范围

本实验围绕 Yue et al.（2025）的[原论文](docs/papers/2504.13837v5_RLVR_reasoning_capacity.pdf)，考察可验证奖励强化学习（RLVR）对单次答对概率和多次采样覆盖率的影响。模型为 Qwen3.5-2B-Base，采用低秩适配（LoRA）与组相对策略优化（GRPO）；数据为筛选后的 MATH Level 4 子集，训练、开发、测试分别为 512、128、200 题。模型、训练规模和采样次数均有改动，因此这是围绕原论文问题的小规模改编复现。

## 2. 训练与模型选择

[配置](code/math_l4_v1/configs/math_l4.json)使用一个训练随机种子 42，LoRA 秩 16、学习率 0.000005、`loss_type="grpo"`、`beta=0`，每组生成 8 个答案。[正式训练报告](archive/math_l4/experiment/outputs/math_l4_v1_fix1/formal/report.json)记录完成 3 个 epoch、384 个优化器更新。

[检查点选择记录](archive/math_l4/experiment/outputs/math_l4_v1_fix1/selection.json)显示，第 128、256、384 步在开发集每题采样 4 次，分别答对 255、299、309 次，分母均为 512。按开发集“答对次数／请求次数”选中第 384 步，未使用测试结果选择模型；对应[适配器权重](models/rl_checkpoint384/adapter_model.safetensors)和[配置](models/rl_checkpoint384/adapter_config.json)已保留。

训练奖励与开发集选择使用[评分器 v3](code/math_l4_v1/scripts/verify_answers.py)，[选中检查点的开发集输出](archive/math_l4/experiment/outputs/math_l4_v1_fix1/dev/checkpoint-384/samples.jsonl)保留原评分；最终统计使用 v4 对原有答案事后重新评分（post-hoc rescoring）。这次评分仅更新自然结束的答案，未重新训练、生成或选择检查点。原始生成见 [Base](results/raw/baseline.jsonl) 与 [RL](results/raw/rl.jsonl)。

## 3. 评测口径

两模型各对 200 题采样 16 次，共 6,400 个答案；使用相同提示、温度 0.6、top-p 0.95 和评测种子 314159。全部请求保留在分母中：300 秒超时计失败，重复停止等未知结果在主估计中按未答对处理。

设每题已确认正确的答案数为下式中的计数。pass@1 是单次采样成功率的估计；pass@4 是从 16 次中抽取 4 次、至少一次正确的概率估计；pass@16 是 16 次中至少答对一次的题目比例。

$$
N=200,\quad n=16,\quad c_i=\text{第 }i\text{ 题已确认正确的答案数},\quad k\in\{1,4,16\},
$$

$$
\widehat{\mathrm{pass@}k}
=\frac{1}{N}\sum_{i=1}^{N}\left[1-\frac{\binom{n-c_i}{k}}{\binom{n}{k}}\right],
\qquad \binom{a}{k}=0\ \text{当 }a<k.
$$

例如[逐题结果](results/final/per_question.jsonl)中的 `math-l4-test-algebra-0029`，Base 在 16 次中答对 8 次：

$$
\widehat{\mathrm{pass@1}}=\frac{8}{16}=50\%,\qquad
\widehat{\mathrm{pass@4}}=1-\frac{\binom84}{\binom{16}4}
=1-\frac{70}{1820}\approx96.15\%,\qquad
\widehat{\mathrm{pass@16}}=100\%.
$$

因此，多次采样覆盖率不能当作单次回答准确率。95% 置信区间（CI）按题目家族配对 bootstrap 计算，共 2,000 次重采样、种子 42；本测试集有 200 个家族，重复答案不作为独立题目。

## 4. 主要结果

[最终汇总](results/final/summary.json)如下。模型列为百分比，差值列为百分点（pp）；括号内均为 95% CI。

| 指标 | Base | RL | RL − Base |
|---|---:|---:|---:|
| pass@1 | 43.53 [38.78, 48.06] | 61.50 [56.47, 66.28] | +17.97 [15.09, 20.91] |
| pass@4 | 69.63 [64.63, 74.75] | 81.93 [77.55, 85.96] | +12.29 [8.94, 15.70] |
| pass@16 | 85.50 [80.50, 90.01] | 94.00 [90.50, 97.00] | +8.50 [4.00, 13.00] |

两者均覆盖 167 题，仅 RL 覆盖 21 题，仅 Base 覆盖 4 题，两者均未覆盖 8 题。RL 的单次成功率增益大于 16 次覆盖率增益：

$$
\Delta_k=\widehat{\mathrm{pass@}k}_{\mathrm{RL}}-\widehat{\mathrm{pass@}k}_{\mathrm{Base}},\qquad
D=\Delta_1-\Delta_{16}=9.46875\ \mathrm{pp},\qquad
\mathrm{CI}_{95\%}(D)=[4.3742,14.75]\ \mathrm{pp}.
$$

结果支持当前评分与停止规则下，RL 提高答对概率，并增加有限采样预算内的题目覆盖。Base 在 16 次中未答对，不代表其成功概率为零；21 道 RL 独有覆盖题不能直接证明模型获得了基础模型完全不具备的推理能力。该比较也未测得完整的推理能力边界。

## 5. 评分与未知结果

v4 使 RL 的 25 个答案评分字段发生变化，其中 18 个由未答对变为答对；Base 主指标未变化。仍有 176 个 Base 答案抽取失败、41 个存在歧义，合计 217 个；RL 有 67 个抽取失败。评分器尚不能保证所有答案的语义判定准确，最终答案等价也不证明每一步推理正确。[评分变化与诊断](results/final/summary.json)

未知结果数分别为 Base 211、RL 138。若允许这些结果在全错与全对之间变化，pass@16 差值的未知结果界限（unknown-outcome bounds）为 −2.5 至 +14 个百分点。这是敏感性范围，**不是置信区间**，且只改变标记为 unknown 的结果，不涵盖抽取失败、歧义或超时答案的潜在评分误差。因此，主估计的正增益对当前保守计分成立，但覆盖率增益的符号对未知结果的极端赋值并不稳健。

## 6. 停止规则与运行条件

Base、RL 的 300 秒超时分别为 219、320 次，重复停止分别为 211、138 次。RL 减少了重复停止，但超时更多。设备记录均为 RTX 4090，实际吞吐不同：[Base](archive/math_l4/experiment/outputs/math_l4_v1_fix1/test_base/manifest.json)约 40.53 token/s，RL [分片 0](archive/math_l4/experiment/outputs/math_l4_v1_fix1/rl_shards/shard_0/manifest.json)与[分片 1](archive/math_l4/experiment/outputs/math_l4_v1_fix1/rl_shards/shard_1/manifest.json)约 20.00、28.44 token/s。相同秒数限制并不保证相同生成机会，结果包含模型与运行条件共同作用，不能解释为等计算预算下的纯推理能力差异。

## 7. 旧实验与结论范围

[旧 GSM8K 结果](archive/gsm8k/rescore_v1_4/main_analysis/summary.json)在其旧评分口径下，Base／RL 的 pass@1 为 64.8125%／79.0625%，pass@16 为 95.0%／95.5%；这些结果不支持“单次准确率 98%”的说法，也不能与当前 MATH v4 结果混合统计。

当前证据来自一个训练种子、一个经过筛选的小规模子集和每题 16 次采样。bootstrap 区间反映题目抽样变动，不反映不同训练种子的波动。可报告的结论是：在本次固定协议下，RL 同时改善单次成功率和有限采样覆盖率，前者的增益更大；对完整 MATH、其他模型以及基础模型的完整能力边界，仍缺少直接证据。

原论文的完整 ASA 参考文献与版本信息见[论文来源记录](docs/papers/README.md)。
