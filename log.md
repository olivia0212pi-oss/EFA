# 项目进度日志

本文档记录 EFA 项目的实际部署状态、验证结果和待办事项。账号密码、SSH 密码、API Key 和访问令牌不会写入仓库。

## 2026-07-19：首次部署完成

### GitHub

- 仓库：<https://github.com/olivia0212pi-oss/EFA>
- 默认分支：`main`
- AutoDL 项目目录：`/root/autodl-tmp/EFA`
- 本次 GPU 验证对应代码提交：`65072dd`（`Limit vLLM context length for 24GB GPUs`）
- 仓库历史作者仅有：`Olivia <olivia0212pi-oss@users.noreply.github.com>`
- 本地和 AutoDL 工作区在验证结束时均无未提交改动。
- AutoDL 当时无法直连 GitHub，更新通过只读加速地址拉取，并核对完整提交哈希 `65072dd9118440beaa0a4eafa7b5f1a2e7f1cc7e` 后快进合并。

### AutoDL 实例

| 项目 | 实际配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 数量 | 1 |
| 显存 | 23.6 GiB |
| 内存 | 90GB |
| 系统盘 | 30GB |
| 数据盘 | 50GB，挂载于 `/root/autodl-tmp` |
| 项目目录 | `/root/autodl-tmp/EFA` |
| 模型缓存 | `/root/autodl-tmp/huggingface` |

当前 50GB 数据盘能够运行已下载的单个 7B 模型。后续若同时缓存 DeepSeek 和 Qwen 两个 7B 模型，需要持续检查磁盘；新建长期实例仍建议选择至少 80GB 数据盘。

### Python 和 GPU 环境

- Conda 环境：`reason`
- Python：3.10
- PyTorch：`2.5.1+cu121`
- CUDA runtime：12.1
- vLLM：0.7.3
- `torch.cuda.is_available()`：`True`
- BF16：支持
- 环境安装脚本：`bash scripts/setup_autodl.sh`
- Hugging Face 环境变量已通过 Conda 环境配置持久保存：

```text
HF_HOME=/root/autodl-tmp/huggingface
HF_ENDPOINT=https://hf-mirror.com
```

### 安装与存储

- 安装完成后清理 pip 缓存 748 个文件，共 4432.5MB。
- 清理后数据盘约有 50GB 可用。
- 下载 DeepSeek 7B 后，Hugging Face 缓存约 15GB。
- 最终数据盘使用约 15GB，剩余约 36GB。
- 最终系统盘使用约 7.9GB，剩余约 23GB。

### 代码调整

为避免 DeepSeek 模型元数据中的超长上下文让 24GB GPU 预留过大的 KV Cache，完成以下调整：

- 在 `configs/base.yaml` 设置 `max_model_len: 8192`。
- `generation/run_model.py` 将 `max_model_len` 传给 vLLM。
- `generation/generate_dataset.py` 将 `max_model_len` 传给 vLLM。
- 调整已作为提交 `65072dd` 推送到 GitHub，作者仅为 Olivia。

### 自动化检查

在 AutoDL 的 `reason` 环境中执行：

```bash
python -m pytest
python -m ruff check .
```

结果：

- Pytest：`11 passed in 0.60s`
- Ruff：`All checks passed!`

### GPU 冒烟测试

执行命令：

```bash
python -m generation.run_model --config configs/smoke.yaml
```

首次运行下载模型 `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`，两块权重文件合计约 15.2GB，下载耗时约 22 分钟。

vLLM 启动实测：

- dtype：BF16
- 最大上下文：8192 tokens
- 模型权重显存：14.27 GiB
- PyTorch activation 峰值：1.44 GiB
- KV Cache：5.02 GiB
- CUDA graph：捕获成功，占用约 0.22 GiB
- 8192-token 请求最大并发估算：11.46x
- 未发生显存溢出

测试问题为 `What is 12 times 15?`，模型在推理中得到正确答案 `180`。本次按 smoke 配置生成 1024 tokens，平均 chosen-token logprob 为 `-0.1649`。

进程退出时出现 PyTorch 的 ProcessGroupNCCL 未显式销毁警告，但命令正常返回、结果完整、显存正常释放，不影响本次单卡验证结论。

## 当前结论

- RTX 4090 24GB 可以运行本项目的 DeepSeek 7B BF16 推理。
- `max_model_len: 8192` 可以避免默认超长上下文导致的 KV Cache 启动问题。
- 安装、单元测试、静态检查、模型下载、GPU 加载和单题推理均已完成。
- 当前不需要 Hugging Face Token：已使用的模型和数据集均为公开资源。
- OpenRouter 尚未接入当前本地 vLLM 流程，也不是下一步 MATH-500 实验的必需项。

## 下一步

### 1. 生成 5 道 MATH-500

```bash
cd /root/autodl-tmp/EFA
conda activate reason
python -m generation.generate_dataset --config configs/smoke.yaml
```

### 2. 自动评分

```bash
python -m evaluation.score_results results/math500_smoke.jsonl \
  --output results/math500_smoke_scored.jsonl
```

### 3. 人工抽查

检查 `results/math500_smoke_scored.jsonl` 中的问题、标准答案、模型最终答案和自动判分是否一致。5 道题全部正常后，再运行 `configs/math500_100.yaml`，不要直接开始 500 道完整实验。

### 4. 使用完毕后关机

模型已经缓存在数据盘，下次开机无需重新下载。不运行实验时应在 AutoDL 控制台关机，避免继续计费；不要删除数据盘或实例中的 `/root/autodl-tmp/huggingface`。

## 2026-07-19（续）：MATH-500 5 题冒烟测试

### 执行记录

远程（AutoDL，代码版本 `4d4bd1a`，已与 GitHub `origin/main` 同步）执行：

```bash
python -m generation.generate_dataset --config configs/smoke.yaml
python -m evaluation.score_results results/math500_smoke.jsonl \
  --output results/math500_smoke_scored.jsonl
```

生成 5 题总耗时约 79 秒（单题 7.9~17.8 秒）。评分结果：

```json
{"samples": 5, "correct": 1, "accuracy": 0.2, "average_tokens": 910.0, "average_runtime_seconds": 15.785}
```

### 人工抽查发现的问题

- 5 题中有 4 题 `total_tokens` 打满 1024 上限，`final_answer` 为 `null`——模型在给出 `\boxed{}` 最终答案之前推理就被截断。
- 唯一在 454 tokens 内自然结束推理的题目（sample_id=2）正确输出 `\boxed{\dfrac{14}{3}}`，与标准答案 `\frac{14}{3}` 判定为等价，`correct: true`。
- 根因：`configs/smoke.yaml` 把 `generation.max_tokens` 从 `base.yaml` 的 `4096` 覆盖为 `1024`；DeepSeek-R1-Distill-Qwen-7B 在 MATH 题目上的思维链通常远超 1024 tokens，因此绝大多数题目在完成推理前就被截断。
- 结论：本次 `accuracy: 0.2` **不代表模型真实能力**，而是评测配置 `max_tokens` 过小导致的截断假阴性。`evaluation.answers`（`extract_answer` / `is_correct`）本身工作正常——sample 2 完整验证了抽取和判分逻辑没问题。

### 待处理

- 需要把 `configs/smoke.yaml` 的 `generation.max_tokens` 调大（例如恢复到与 `base.yaml` 一致的 4096，或先试 2048+ 观察是否够用）后重新跑 5 题冒烟测试，才能做出"人工抽查通过、可以上 100 题"的真实判断。当前不建议在此结果基础上切换到 `configs/math500_100.yaml`。
- 本次验证结束后已通过 SSH 在实例内执行 `shutdown` 关闭 AutoDL 实例，避免继续计费（模型缓存和数据盘均未删除）。
