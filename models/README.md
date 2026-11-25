# 已训练参数

[rl_checkpoint384/](rl_checkpoint384/) 保存最终选中的 MATH Level 4 LoRA 检查点，已直接从 Downloads 复制完整 13 个文件。参数约 43.7 MB；含 tokenizer、优化器、训练状态和恢复审计的完整目录约 182.9 MB。逐文件大小和 SHA-256 见 [导入记录](../maintenance/downloads_import.json)，检查点自身封存清单为 `resume_state.json`。

推理需要 `adapter_model.safetensors` 与 `adapter_config.json`，并加载固定 revision 的 Qwen3.5-2B-Base。项目不包含 Base 完整权重；安装脚本会下载到 `code/models/Qwen3.5-2B-Base/`。这是低秩适配器，不是独立完整的 2B 模型，也不能加载到不同 Base 版本后直接视为同一实验。

checkpoint-384 的选择来自 128 道开发题每题 4 次采样：255/512、299/512、309/512，最终选择 384。原选择记录、三个候选参数与开发输出位于 [冻结实验档案](../archive/math_l4/experiment/outputs/math_l4_v1_fix1/)。

保留最终恢复状态可以避免丢失训练内部状态，但继续训练仍须恢复正确的运行目录、配置与 Base；本次整理只做文件封存核验，未实测从此导出目录直接恢复 GPU 训练。旧 GSM8K 的续训状态已按确认删除，其推理参数仍保存在对应旧实验档案。
