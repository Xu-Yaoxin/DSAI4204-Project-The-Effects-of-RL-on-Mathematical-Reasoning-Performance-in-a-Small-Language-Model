# 原始回答与最终评分

当前实验为 MATH Level 4，200 题，每题 16 次；Base/RL 各 3,200 个回答。

| 模型 | 原始生成 | 当前最终评分 |
|---|---|---|
| Base | [raw/baseline.jsonl](raw/baseline.jsonl) | [final/base/samples.jsonl](final/base/samples.jsonl) |
| RL | [raw/rl.jsonl](raw/rl.jsonl)，已合并两个分片 | [final/rl/samples.jsonl](final/rl/samples.jsonl) |

最终评分文件仍包含完整回答，`correct`、`status`、`reward` 是 v4 评分；`old_score` 保留生成时的评分。题目、标准答案、回答文本、输出 token、采样种子和终止原因未改变。

[summary.json](final/summary.json) 保存指标与区间，[per_question.jsonl](final/per_question.jsonl) 保存逐题结果，[changes.jsonl](final/changes.jsonl) 保存 25 条评分变化。分析见 [ANALYSIS.md](../ANALYSIS.md)。

`final/` 从原 `code/outputs/rescore_20261106/` 整体移动，内容和 ready 身份未改写。其元数据中的历史绝对路径通过 [relocation.json](../maintenance/relocation.json) 解释；不要根据旧地址重新寻找现行文件，也不要在 `final/` 额外写入分析文件。
