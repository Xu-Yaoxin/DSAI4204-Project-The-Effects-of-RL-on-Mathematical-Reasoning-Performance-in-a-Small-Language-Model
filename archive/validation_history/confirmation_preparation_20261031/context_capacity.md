# 原生上下文容量核对 / Native-context capacity audit

模型为固定版本 Qwen3.5-2B-Base。原生上下文是 **262,144 tokens，包含输入和输出**。实际输入为 200 tokens 时，最大新增输出为 261,944 tokens；遇到 EOS 会提前结束。提高上限不会强制每题生成到上限，也不能保证没有截断。

固定模型配置共有 24 个文本层，其中 6 个全注意力层、18 个线性注意力层。每个全注意力层有 2 个 KV heads，head dimension 为 256；BF16 每元素 2 bytes，单进程 batch=1。

$$
\mathrm{KV\ bytes/token}=6\times2_{(K,V)}\times2_{heads}\times256\times2_{bytes}=12{,}288
$$

$$
\mathrm{KV\ GiB\ at\ native\ context}=12{,}288\times262{,}144/2^{30}=3
$$

| 总上下文 tokens | 单进程全注意力 KV | 双进程全注意力 KV |
|---:|---:|---:|
| 32,768 | 0.375 GiB | 0.750 GiB |
| 65,536 | 0.750 GiB | 1.500 GiB |
| 131,072 | 1.500 GiB | 3.000 GiB |
| 262,144 | 3.000 GiB | 6.000 GiB |

这些数字仅计算全注意力 KV。权重、线性注意力状态、临时张量、CUDA 上下文和内存分配器仍需额外显存。前次 1024-token 上限测试中，两个 worker 的 PyTorch allocated 峰值分别为 3.559 和 3.581 GiB；它们不代表整卡峰值或原生长度峰值。

新启动器设置单进程至少 11 GiB 空闲显存与 16 GiB 可用 RAM，双进程至少 20 GiB 空闲显存与 32 GiB 可用 RAM。这是保守启动门槛，尚无跑满原生长度的 GPU 实测保证。串行 fallback 保留相同的每题长度预算。

只读检查未发现原生长度协议的阻塞问题：评估器核对加载模型的上下文长度，按实际输入扣除输出预算，并记录每个回答的 generation_limit；续跑、合并和分析均检查该预算。碰到上限且未生成 EOS 时保留 truncated 标记。已完成回答可以续跑，正在生成但被中断的单个回答需要从原随机种子重跑。

本次预算比旧筛查的 1024 tokens 更长，应作为新协议报告。极长回答的实际耗时和临时显存需求取决于运行后端，不能按短回答吞吐保证运行时间。

配置证据：本地 2908-byte config.json 的 git blob SHA-1 为 `d30a15be1bbfd610f58ad80f1fbeb6778f58a80f`，与固定下载 manifest 一致；SHA-256 为 `ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4`。原生长度未执行 GPU 测试。

[固定版本官方模型配置](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/config.json)。本次使用经过 blob 校验的本地副本。
