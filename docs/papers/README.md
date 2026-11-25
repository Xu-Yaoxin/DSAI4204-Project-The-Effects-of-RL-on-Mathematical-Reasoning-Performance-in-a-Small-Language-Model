# 复现参考论文

本目录保留项目参考的原始论文 PDF。文件来自 arXiv 官方 v5 链接，下载后未修改内容。

- 题目：*Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?*
- 作者：Yang Yue、Zhiqi Chen、Rui Lu、Andrew Zhao、Zhaokai Wang、Yang Yue、Shiji Song、Gao Huang。第一作者为乐洋，第六作者为乐阳；论文首页脚注说明两人的英文姓名相同。
- 版本：arXiv:2504.13837v5，共 31 页。首版提交日期为 2025-04-18，v5 提交日期为 2025-11-24；PDF 首页印刷日期为 2025-11-25。
- 官方来源：[版本页面](https://arxiv.org/abs/2504.13837v5) · [PDF](https://arxiv.org/pdf/2504.13837v5)
- 本地文件：[原始论文 PDF](2504.13837v5_RLVR_reasoning_capacity.pdf)；下载时间、文件大小和 SHA-256 见 [source.json](source.json)。

项目使用 Qwen3.5-2B-Base、组相对策略优化（GRPO）和低秩适配（LoRA），主评测每题采样 16 次，属于围绕论文问题开展的小规模改编复现。模型、训练方式、数据量和采样范围与原论文有差异，不能据此声称完整复现原论文全部设置或结论。

ASA 参考文献：

Yue, Yang, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. “Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?” *arXiv preprint*, arXiv:2504.13837v5. Retrieved November 6, 2026 (https://arxiv.org/abs/2504.13837v5).
