# 高效推理研究项目

本项目把《推理效率研究_零基础实操版.pdf》落成了一套可复现的实验骨架：

- 共享流水线：MATH-500 生成、逐 token logprob、耗时记录、数学答案判分。
- 同学 A：256-token 检查点、oracle 节省上限、逻辑回归停止分类器。
- 同学 B：固定/自适应 Power Sampling、MH 接受率、废弃 token 与接受率记录。

默认配置只跑 5 道题。确认完整流程无误后，再切换到 100 和 500 道配置。

## 当前状态

截至 2026-07-19（AutoDL 服务器时间），项目已经完成首次部署和单题 GPU 冒烟测试。

- GitHub 仓库：<https://github.com/olivia0212pi-oss/EFA>
- 本次已验证代码提交：`65072dd`（`Limit vLLM context length for 24GB GPUs`）
- Git 作者：仓库历史中只有 `Olivia <olivia0212pi-oss@users.noreply.github.com>`
- AutoDL：单卡 NVIDIA GeForce RTX 4090，23.6 GiB 显存，90GB 内存
- 存储：30GB 系统盘、50GB 数据盘；模型下载后数据盘剩余约 36GB
- Conda 环境：`reason`，Python 3.10
- PyTorch：2.5.1+cu121；CUDA 12.1；BF16 可用
- 推理框架：vLLM 0.7.3
- 模型缓存：`/root/autodl-tmp/huggingface`，约 15GB
- 已下载模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- 项目检查：`11 passed`，Ruff 全部通过
- GPU 冒烟测试：`12 x 15` 正确生成答案 `180`，共生成 1024 tokens
- vLLM 实测：模型权重占用 14.27 GiB，KV Cache 约 5.02 GiB，8192 token 上下文正常启动

当前服务器的 `reason` 环境已经持久设置：

```text
HF_HOME=/root/autodl-tmp/huggingface
HF_ENDPOINT=https://hf-mirror.com
```

详细部署过程和实测数据见 [`log.md`](log.md)。下一步是生成并评分 5 道 MATH-500：

```bash
cd /root/autodl-tmp/EFA
conda activate reason
python -m generation.generate_dataset --config configs/smoke.yaml
python -m evaluation.score_results results/math500_smoke.jsonl \
  --output results/math500_smoke_scored.jsonl
```

## AutoDL 机器怎么选

首选 **单卡 RTX 4090 24GB**。两个 7B 模型都能以 BF16 分别运行，4090 的速度和租用价格通常最适合本项目。不要在同一个进程里同时加载两个模型。

建议创建实例时选择：

- GPU：RTX 4090 24GB x 1。
- 镜像：PyTorch 2.5.1、Python 3.10、CUDA 12.1；没有完全相同镜像时选 Miniconda + CUDA 12.1，安装脚本会装固定版本。
- 数据盘：建议至少 80GB，模型缓存放 `/root/autodl-tmp`，不要放较小的系统盘。当前实例的 50GB 数据盘足够完成单模型冒烟测试，但同时缓存更多模型时需要留意空间。
- CPU/内存：8 vCPU、32GB RAM 足够起步。

备选顺序：RTX 3090 24GB（更慢但便宜）→ A5000 24GB（更慢）→ L40S 48GB（更稳，适合后期长序列或增大批量）。16GB 卡不适合本文档里的 BF16 7B + 4096 token 配置。A100 40/80GB 可以用，但对当前单卡 7B 起步阶段通常不划算。

如果 4090 在 4096 token 时偶发 OOM，先把 `configs/base.yaml` 中 `gpu_memory_utilization` 调到 `0.82`，或把 `max_tokens` 调到 `2048`。后期想提高吞吐、扩大 batch，直接换 L40S 48GB。

## 1. 在 AutoDL 安装

上传项目到数据盘后执行：

```bash
cd /root/autodl-tmp/EFA
bash scripts/setup_autodl.sh
conda activate reason
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
```

如果 Hugging Face 官方站可直接访问，可以不设置 `HF_ENDPOINT`。私有或受限模型需要另行执行 `huggingface-cli login`，不要把 token 写进仓库。

验证 GPU：

```bash
python -m generation.check_gpu
```

## 2. 先跑最小测试

第一次会下载约 15GB 模型：

```bash
python -m generation.run_model --config configs/smoke.yaml
```

生成 5 道 MATH-500：

```bash
python -m generation.generate_dataset --config configs/smoke.yaml
python -m evaluation.score_results results/math500_smoke.jsonl \
  --output results/math500_smoke_scored.jsonl
```

任务中断后可原路径续跑：

```bash
python -m generation.generate_dataset --config configs/smoke.yaml --resume
```

输出的每行都包含问题、标准答案、完整推理、最终答案、token ID、逐 token logprob、token 总数、耗时和完整实验参数。

## 3. 扩到 100 / 500 道

```bash
# 第一周目标
python -m generation.generate_dataset --config configs/math500_100.yaml
python -m evaluation.score_results results/math500_100.jsonl \
  --output results/math500_100_scored.jsonl

# 流程和人工判分抽查都通过后再运行
python -m generation.generate_dataset --config configs/math500_full.yaml
```

至少人工检查几十条 `*_scored.jsonl`。自动判分错误会直接污染后续结论。

## 4. 同学 A：提前停止

先基于完整生成轨迹建立检查点。脚本每 256 token 截取前缀，让模型只给当前答案：

```bash
python -m early_exit.build_checkpoints results/math500_100.jsonl \
  --config configs/base.yaml \
  --output results/checkpoints_100.jsonl

python -m early_exit.oracle results/checkpoints_100.jsonl

python -m early_exit.train_classifier results/checkpoints_100.jsonl \
  --output checkpoints/early_exit.joblib \
  --max-stop-error 0.05
```

数据切分按题目分组，避免同一道题的不同检查点同时进入训练集和校准集。分类器阈值在校准集上按“停错率不超过 5%”选择。

当前检查点实现是离线研究版：先生成完整轨迹，再分析理论停止位置。真正节省在线计算时，需要把已训练策略接到分段生成循环中。

## 5. 同学 B：Power Sampling

固定答案前半段，并对后半段做 8 次后缀提议：

```bash
python -m sampling.power_sampling \
  --config configs/base.yaml \
  --problem "What is 12 times 15?" \
  --output results/power_fixed.json
```

连续多次没有 likelihood 增益时自适应停止：

```bash
python -m sampling.power_sampling \
  --config configs/base.yaml \
  --adaptive \
  --problem "What is 12 times 15?" \
  --output results/power_adaptive.json
```

结果记录初始 token、固定前缀位置、所有提议 token、被拒绝 token、接受率和每一步 likelihood。实现关闭了 `top_k/top_p` 截断，并使用适用于温度提议分布的 Metropolis-Hastings 修正，保证代码中计算的提议概率与实际采样一致。后续“优先挑难点”会改变位置提议概率，加入前必须在接受率中继续计入正反向位置提议比，并先做玩具分布验证。

## 项目结构

```text
configs/       三档实验参数
common/        配置、JSONL、随机种子
generation/    GPU 检查、单题和数据集生成
evaluation/    答案抽取与数学等价判分
early_exit/    检查点、oracle、停止分类器
sampling/      Power Sampling 与自适应停止
results/       实验结果（Git 忽略）
checkpoints/   小分类器权重（Git 忽略）
figures/       图表（Git 忽略）
tests/         不需要 GPU 的单元测试
```

## 本地检查

CPU 或 macOS 上不需要安装 vLLM，也可以检查纯逻辑：

```bash
python -m pytest
python -m ruff check .
```

模型推理必须在 Linux + NVIDIA CUDA 环境运行。
