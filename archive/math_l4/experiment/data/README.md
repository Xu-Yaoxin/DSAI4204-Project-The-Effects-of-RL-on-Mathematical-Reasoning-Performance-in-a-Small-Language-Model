# MATH Level 4 data / 数据说明

本包冻结 512 道训练题、128 道验证题、200 道测试题。训练和验证来自官方 train，测试只来自官方 test。32 道工程检查题包含在 128 道 pilot 题内，pilot 又包含在训练集内；它们不属于独立测试集。

| Split | Official source | Questions | Use |
|---|---|---:|---|
| engineering | train | 32 | GPU and reward preflight |
| pilot | train | 128 | Training viability gate |
| train | train | 512 | Formal LoRA-GRPO training |
| dev | train | 128 | Checkpoint selection only |
| test | test | 200 | Frozen Base/RL comparison |

原始来源为 [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math/tree/21a5633873b6a120296cce3e2df9d5550074f4a3)，固定 revision `21a5633873b6a120296cce3e2df9d5550074f4a3`。14 个 parquet 文件覆盖 7 个科目、官方 train 7,500 题和 test 5,000 题，原始文件约 4.9 MB，逐文件验证 SHA-256。原始 MATH 的 MIT 许可证保存在 `raw/math_full/ORIGINAL_MATH_LICENSE`。

只保留 Level 4、最终答案可解析为精确有理数、题目不带图片代码或明确图示引用的记录。代码排除了 248 条图示/视觉引用记录、606 条不支持的非有理数或非标量答案、220 条已出现过的 Level 4 题；余下官方 train 1,168 题、test 662 题。本包不是完整 MATH Level 4 榜单分数。

排除历史通过题目文本 casefold 后折叠空白，再计算 SHA-256。冻结账本覆盖 46 个旧数据/评测文件，共 9,712 个不同题目文本，并保守排除了整份旧 MATH-500 原始池。抽样只使用问题、来源、难度和答案类型，未读取正确率用于挑题。每个科目按剩余可用池比例以 Hamilton 最大余数法分配名额，固定种子 20261102；完整配额在 `processed/manifest.json`。

| Subject | Train | Dev | Test |
|---|---:|---:|---:|
| Algebra | 139 | 35 | 55 |
| Counting and probability | 64 | 16 | 22 |
| Geometry | 28 | 7 | 9 |
| Intermediate algebra | 96 | 24 | 40 |
| Number theory | 73 | 18 | 30 |
| Prealgebra | 82 | 20 | 33 |
| Precalculus | 30 | 8 | 11 |

`family_id` 表示完全相同规范化题目，不是人工确认的语义模板。完全相同文本在各分片中隔离，但这不证明语义相似题或预训练污染不存在。来源标准答案经过精确数值解析，并未逐题人工验证所有推导。

运行时仅把 `question` 填入提示词。`answer` 用于奖励和评分，`source_answer` 与来源信息用于审计；不得把这些字段或原始解答传入生成提示。

离线完整性检查无需模型、GPU 或 pyarrow：

```bash
python3 -S scripts/prepare_data.py --verify
```

重新解码固定 parquet 并重现划分需要已安装的 pyarrow，不需要下载或调用模型。相同文件只验证，不覆盖已有不同内容：

```bash
python3 scripts/prepare_data.py --offline
```
