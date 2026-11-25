# MATH 判分器 v4

当前原始回答、最终评分和分析入口见项目 [README.md](../../README.md)。已完成结果在 `../../results/final/`，来源包在 `../../archive/math_l4/experiment/`。

`scorer.py` 和 `rescore.py` 保持原始字节，与已完成结果中的 SHA-256 一致。训练奖励及开发选点仍采用冻结训练包里的 v3；这里的 v4 仅用于已生成答案的事后重评分。

在项目根目录检查已有结果，不执行评分：

```bash
python3 -B -S maintenance/verify_project.py
```

如确实需要重新评分，在项目根目录由用户执行：

```bash
bash rescore_math.sh
```

新结果写入 `results/recomputed_v4/`。原 `run.sh` / `rescore.py` 无参数默认地址是历史地址；迁移后使用上述根目录入口传入实际来源和新输出，不向正式 `results/final/` 写入。`review_results.py` 已改成只读项目检查入口，不再依赖 Downloads。

回归测试保留在 `tests/`。本次目录整理没有重新运行判分器或回归测试；迁移和依赖检查通过文件哈希及记录算术完成。原始成功测试证据保留在 `validation/`。完整分析见 [ANALYSIS.md](../../ANALYSIS.md)，变化记录见 [memory.md](memory.md)。
