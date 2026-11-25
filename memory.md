# 项目记录

## 实验过程

1. 首轮使用 Qwen3.5-2B-Base 与 LoRA-GRPO 完成 GSM8K 训练及 Base/RL 对比。已保留训练代码、数据、参数、原回答和重评分结果；高 pass@16 覆盖率不等于同样高的单次正确率。
2. 通过难度摸底和更长回答预算检查，选择 Base 单次成功率处于目标区间的 MATH Level 4 子集。增加重复停止与 300 秒时限，区分超时、unknown 和数学判错。
3. MATH 使用 512/128/200 题训练、开发、测试，正式训练 384 步。按开发集选择 checkpoint-384，Base/RL 各生成 3,200 条测试回答；RL 测试分两台机器完成后合并。
4. 2026-11-06：用户运行评分器 v4 统一重评分。Base 正确 1,393，RL 正确 1,968；18 条 RL 提取误判恢复为正确。原始回答、终止规则和检查点选择未变。

## 2026-11-06：项目整理

按用户确认保留当前 MATH 实验、首轮 GSM8K 的代码／回答／训练评估、原论文和 proposal。原论文 v5 从官方 arXiv 获取并记录哈希。最终评分移至 `results/final/`，原始回答直接导出至 `results/raw/`，选中的参数放在 `models/rl_checkpoint384/`。冻结来源完整迁移至 `archive/math_l4/experiment/`，外部映射记录解释历史绝对路径。

删除旧 GSM8K optimizer/scheduler/RNG 续训状态、重复部署 payload、可重新下载的 wheel/partial 缓存、临时文件及过时远端清理工具；不删除旧实验模型参数和答案。完整删除列表和数量见 `maintenance/cleanup.json`。

用户将删除 Downloads 中的备份，因此已将当前 MATH 最终 checkpoint 全部 13 个文件，以及工程／pilot／环境等补充记录复制入项目并校验哈希。Base 是公开模型，未复制其完整权重；固定 revision、模型清单和下载脚本保留。`requirement.txt` 与原 84 包完整哈希锁一致。

本次只整理文件、读取数据并核验哈希／统计，不执行训练、模型生成或判分器。保留科学代码及旧身份元数据；新增根目录环境、训练、重评分入口和只读项目核验脚本。GPU 安装和从导出检查点恢复训练未在当前 Mac 实测，不作位级重复性承诺。

## 2026-11-07 — English group presentation

Created `presentation/output/DSAI4207_Group_Presentation.pptx` (13 slides, five native editable data charts) and `presentation/SPEAKER_NOTES.md` (12 timed slides, 600 seconds, three presenter roles). Final Canva: https://www.canva.com/d/LOOPQ54pGMZKgqb. Canva uses editable vector charts because its PowerPoint importer lost native chart labels and could not display the scatter interval plot. The corrected Canva charts were visually checked. Source metrics, answer files and model artifacts were unchanged. Earlier Canva drafts are superseded.
