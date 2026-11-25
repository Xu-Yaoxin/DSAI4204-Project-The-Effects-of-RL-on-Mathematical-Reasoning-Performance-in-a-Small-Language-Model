# 训练与评分代码

项目总入口与最终路径见 [../README.md](../README.md)，实验分析见 [../ANALYSIS.md](../ANALYSIS.md)。

| 代码 | 用途 |
|---|---|
| [math_l4_v1/](math_l4_v1/) | 当前 MATH Level 4 的完整冻结训练包，含原评分器 v3、配置、数据、测试与断点恢复代码 |
| [math_l4_rescore_v4/](math_l4_rescore_v4/) | 最终事后评分器 v4 与回归测试；使用项目根目录 rescore_math.sh 传入整理后的来源路径 |
| [math_l4_rl_split_v1/](math_l4_rl_split_v1/) | 原 RL 双机分片评测与合并工具；重复部署 payload 已删除，可按工具文档重建 |
| scripts/、configs/、data/、tests/、run.sh | 首轮 GSM8K 的可运行代码及共用环境／模型下载支持；原云端代码另存 archive/gsm8k |
| packages/、models/ | 完整依赖锁、公开 Base 模型固定版本清单及下载位置；可重新下载的大文件缓存不归档 |

在项目根目录使用 `bash setup_environment.sh`、`bash train.sh` 和 `python3 -B -S maintenance/verify_project.py`。训练包内 81 项文件受清单保护，不要为整理而删改它们。

原 GSM8K 训练可从此目录执行 `bash run.sh all --run-dir outputs/gsm8k_reproduction`，需要先准备环境。旧报告与模型保存在 `../archive/gsm8k/`。首轮离线重评分使用项目根目录 `rescore_gsm8k.sh`，不要使用旧默认归档路径。

整理前的长说明保存在 `../archive/notes/legacy_code_READM.md`，仅作版本历史。原训练、评分与模型源码未在本次整理中修改。
