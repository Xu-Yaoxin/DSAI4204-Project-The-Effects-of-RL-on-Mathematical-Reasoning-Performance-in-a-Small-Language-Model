# 实验档案

| 目录 | 保留内容 |
|---|---|
| [math_l4/experiment](math_l4/experiment/) | 124 个冻结文件：原训练包、三个 epoch 候选、开发集选点、原始 Base/RL 回答、原评分与训练审计。内部字节和路径结构不变 |
| [math_l4/supplementary](math_l4/supplementary/) | 从 Downloads 补存的工程测试、pilot 训练、环境与其他训练记录 |
| [gsm8k/run_20261031/code](gsm8k/run_20261031/code/) | 首轮 GSM8K 原训练代码、数据、模型参数、回答与训练／评估结果；旧续训的优化器、调度器和 RNG 状态已删除 |
| [gsm8k/rescore_v1_4](gsm8k/rescore_v1_4/) | 首轮 GSM8K 的离线重评分结果；Base/RL pass@16 为 95.0% / 95.5%，不是当前 MATH 的评分 |
| [difficulty_probes](difficulty_probes/) | 中途的难度摸底代码、数据和已保存回答，用于说明如何选到当前题目范围 |
| [validation_history](validation_history/) | 已有回归、环境及阶段复核记录；其中部分是历史口径，不作为最终结果引用 |
| [notes](notes/) | 旧交接记录和整理前的代码说明，只供追溯 |

当前正式结论统一引用 [results/final](../results/final/) 和 [ANALYSIS.md](../ANALYSIS.md)。历史脚本或说明可能包含原机器、旧目录与 Downloads 地址，它们不是当前入口。根目录 README 中的命令与 `maintenance/verify_project.py` 使用迁移后的本项目路径。

原 manifest/ready/source inventory 保留历史字节以维持审计链；[relocation.json](../maintenance/relocation.json) 记录旧路径到现路径的映射。原始 RL 分片、合并结果和直接使用的导出副本各有用途，因此未按相同文本内容任意删减冻结证据。
