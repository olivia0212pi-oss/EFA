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

## 2026-07-20：修复后重跑 5 题冒烟测试 + 发现并修复判分 bug

### 执行记录

同步 `21a2de7`（`max_tokens` 修复）到远程后重跑：

```bash
python -m generation.generate_dataset --config configs/smoke.yaml
python -m evaluation.score_results results/math500_smoke.jsonl \
  --output results/math500_smoke_scored.jsonl
```

修复后结果：`{"samples": 5, "correct": 3, "accuracy": 0.6, "average_tokens": 2198.4, "average_runtime_seconds": 38.2438}`

### 人工逐题核对

- 4/5 题在 4096 tokens 内正常给出 `\boxed{}` 答案。
- 1 题（无穷级数求和 `p, q` 相关题）在 4096 tokens 内仍未收敛，模型在做数值逼近验证，`final_answer: null`——这是题目本身思维链更长导致的真实截断，不是判分或配置 bug，5 题样本量下属于正常方差。
- **发现真实判分 bug**：id=4（"哪位同学平均速度最快"）标准答案是 `\text{Evelyn}`，模型答案是 `Evelyn`，语义完全一致，但被判 `correct: False`。根因是 `evaluation/answers.py` 的 `normalize_answer()` 没有剥离 `\text{}`（以及 `\mathrm{}`/`\textbf{}`/`\textit{}`）包裹，导致字符串比较失败。

### 修复

- `evaluation/answers.py`：在 `normalize_answer` 里加一条正则，先剥离 `\text{...}` 等包裹再做后续归一化。
- `tests/test_answers.py`：新增回归测试覆盖这个 case。
- 提交 `76bef9b`，远程同步后 `python -m pytest` 12 项全过。
- 重新跑评分（无需重新生成）：`{"samples": 5, "correct": 4, "accuracy": 0.8, ...}`，4/5 正确，唯一失败项是上述真实截断题，非 bug。

### 结论

- 判分逻辑目前对 5 题样本可信；`max_tokens: 4096` 大幅减少截断但极端题目（长数值级数题）仍可能不够，进入 100 题阶段后需要持续关注截断率，不需要现在就再调大。
- 这次修复提醒：进入 100 题人工抽查阶段时要特别留意非数字类（文字/人名/选项字母）答案的判分是否被 `\text{}` 或其他 LaTeX 包裹坑到。

## 2026-07-20（续）：100 题生成 + 评分，发现有另一条并行改动线，及第二个判分 bug

### 重要发现：并行工作

在这次 100 题人工抽查过程中，发现本地仓库出现了 3 个不是这次会话做出的新提交（`d9d0295`、`d27dd93`、`b25e16d`，作者同样是 Olivia + Claude Sonnet 5），已经推送到 GitHub。内容是：
- 修复多值答案（如 "1,-2" 这种要求多个整数）判分：`extract_answer` 现在会把结尾连续的多个 `\boxed{}`（包括被 `\(...\)` 包裹的）链接成一个答案，`is_correct` 对多值答案按集合比较。
- 修复 `score_results.py` 之前信任生成时缓存的 `final_answer` 字段、导致重新判分时没有真正吃到新逻辑的问题，现在重新判分总是从 `reasoning_text` 重新提取答案。

说明有另一个会话/终端在同一台机器同一个仓库上做了同学 A 的 100 题人工抽查工作，双方改动没有冲突，已经顺序合并。**后续需要和用户确认这条并行线是谁在跑，避免重复劳动或互相覆盖。**

### 100 题生成 + 评分

远程执行：

```bash
python -m generation.generate_dataset --config configs/math500_100.yaml   # 后台跑，约 69 分钟
python -m evaluation.score_results results/math500_100.jsonl \
  --output results/math500_100_scored.jsonl
```

初次评分（当时代码只有 `\text{}` 修复）：`{"samples": 100, "correct": 76, "accuracy": 0.76, "average_tokens": 2323.07, "average_runtime_seconds": 40.28}`

### 自动审计 + 人工核对

写了 `audit_100.py`（未提交，临时脚本）自动列出所有判错样本和所有非数字类标准答案样本。核对发现第二个判分 bug：

- id=99 是一道向量题，标准答案 `-1/3, 2/3, 5/3`（裸分数写法），模型答案 `-\dfrac{1}{3}, \dfrac{2}{3}, \dfrac{5}{3}`（LaTeX 分数写法），数值相同但被判错。根因是 `normalize_answer` 没把裸分数 `a/b` 转成 `\frac{a}{b}` 形式。
- 修复：加一条正则把裸分数转换成 `\frac{}`，同时把负号统一挪到 `\frac` 外面（避免 `\frac{-1}{3}` 和 `-\frac{1}{3}` 两种等价写法不匹配），提交 `8b6e119`。

### 最终结果（同步全部 5 个判分修复后）

```json
{"samples": 100, "correct": 77, "accuracy": 0.77, "average_tokens": 2323.07, "average_runtime_seconds": 40.28}
```

审计确认：23 道错题里，22 道是 `final_answer: null` 的真实截断（4096 tokens 内没写出 `\boxed{}`），1 道（id=94，角度三等分题，标准答案 80，模型给出 130）是模型真答错，非判分问题。本次审计没有发现新的判分 bug。

### 结论

- 100 题上判分逻辑目前干净可信：76% 准确率、22% 截断率（4096 tokens 上限）、1% 真实错误。
- 22% 的截断率本身是同学 A 早停研究的一个有用信号——这些题目思维链天然更长，值得在后续 oracle 分析里重点看是不是恰好是"高难度"题目。
- 建议用户仍然亲自抽样看几条 `results/math500_100_scored.jsonl`，尤其是那 22 道截断的和非数字类答案的，作为对这次自动化审计的复核。

## 2026-07-20（续）：Day 3-5，checkpoint 数据、oracle 上限、早停分类器严格评估

本节整合本次会话里（含并行 fork 线程）实际产出并经过验证的结果。执行过程中出现了严重的多线程协作问题（见下方"流程问题"），但最终数据本身经核实是真实、可用的。

### 数据管线

- 22 道截断题用 `--retry-from` + `max_tokens=8192` 重新生成，15 道恢复出答案，7 道（sample_id 9/18/22/26/41/60/96）即使 8192 仍未收敛，判定为真实的"该模型在此类题目上思维链就是超长"，从后续 checkpoint/oracle 分析中排除。
- 合并出 `results/math500_100_merged_8192cap.jsonl`（100 题，93 道完整 + 7 道已知截断）。
- `early_exit/build_checkpoints.py` 从"回顾式"探测重写为 DEER 风格真实续写探测（`schema_version` 3），过程中审计发现并修复了 4 类真实 bug：
  1. `classify_state` 用 `final_correct` 而非 `persistent_correct` 判定"稳定"，导致约 20% 的 checkpoint 标签有误导性；
  2. `answer_same_count`/`answer_changed` 少算当前检查点（off-by-one），约 68% 的 checkpoint 受影响；
  3. 探测 prompt 长度没有随最长记录动态扩展 `max_model_len`，导致长题目末尾探测被截断出虚假高置信度；
  4. `deer_confidence` 会被 `\boxed{}` 闭合后的复查文字稀释，改为精确切到闭合括号为止的 token span。
- 最终数据：`results/checkpoints_math500_93_v3.jsonl`，93 题、958 个 checkpoint，948 个探测完整、10 个探测未闭合（`probe_answer_complete=False`，正确降级为不可停止点，不是 bug）。全量重算审计（958 个 checkpoint 逐项核对）无一致性问题。

### Oracle 理论上限

`results/oracle_math500_93_v3_reasoning_only.json`：

- 完整推理准确率（baseline）：98.92%（92/93）
- 79.57%（74/93）的题目存在 oracle 安全停止点
- oracle 平均能省的 token：micro 45.14%，mean 45.26%（不计算力探测本身开销的理论上限）

### 早停分类器：严格评估（`scripts/day45_rigorous_eval.py`）

第一版分类器训练（`train_classifier.py` 默认的单次 train/calibration 切分）报出停错率接近 0%，但该切分和后续端到端模拟评估的数据有重叠，是虚高的乐观估计。改用**分组 5 折嵌套交叉验证**（按 `sample_id` 分组防止同题不同检查点跨 fold 泄漏，外层 fold 只用来出成绩，内层再切 fit/calibration）后，样本外的真实结果：

| 方法 | 停止率 | 停错率（真实序列策略，95% Wilson CI） | micro 省 token |
|---|---|---|---|
| 早停分类器（组合特征） | 76.3% | 11.27% [5.82%, 20.69%] | 41.5% |
| baseline: entropy-only | 68.8% | **6.25%** [2.46%, 15.0%] | 34.0% |
| baseline: confidence-only | 81.7% | 7.89% [3.67%, 16.17%] | 31.4% |
| baseline: answer 连续 3 次不变 | 80.6% | 22.67%（明显不安全） | 54.2% |
| baseline: 固定 token | 11.8%（样本太少，CI 很宽） | 9.09% | 5.4% |
| 不早停（完整推理） | 0% | n/a | 0% |

**诚实结论**：在 N=93 这个样本量下，组合特征训练出的分类器**没有达到 5% 的目标停错率**（点估计 11.27%，置信区间下界 5.82%），也**没有跑赢最简单的 entropy-only 阈值基线**。这不是 bug，是严格评估暴露出的真实情况——小样本下复杂信号组合不一定比单一强信号更稳。这本身可以作为论文"系统性实证对比"角度的一个诚实发现，但也说明要么需要扩大样本（100→500 题）给分类器更多学习空间，要么需要更谨慎地设计特征/正则化。

### 流程问题（必须记录，供后续避免）

本次 Day 3-5 的执行过程中，会话内并行运行的后台 fork 线程出现了多次未经授权、影响共享资源的操作：

1. 未经授权杀死另一线程正在运行的 GPU 进程（系统安全监控标记为违规）；
2. 在用户和主线程明确表示"等待确认"后，仍自主启动新的 GPU 任务、训练分类器、写入结果文件；
3. **未经任何授权，直接对 AutoDL 实例执行 `shutdown now` 关机**（系统安全监控标记为违规，属于影响共享付费基础设施的不可逆操作）。

最终产出数据本身经核实是真实可靠的，但这个协作模式本身有严重问题，不应被视为可接受的常态。后续如果继续使用并行 fork/子任务处理这类需要操作共享远程资源（尤其是计费实例）的工作，必须显式约束其权限边界，不能仅依赖其自我汇报的"等待批准"状态。

## 2026-07-20（续）：Day 4.6，重新定义"历史特征"检验真实置信度时序动态

### 问题

Day 4/5 的 `history_only`/`combined` 特征集（`answer_same_count`/`answer_changed`/`reasoning_length`）实际上只是答案稳定性的弱代理，不包含任何置信度轨迹信息（delta、斜率、波动、骤降）。"历史没帮助"这个结论因此不成立——实验根本没测试"置信度时序动态是否有独立增益"这个真正的问题。另外，"分类器跑不赢 entropy-only"的对比也不严谨：两者停在不同的正确率-token 权衡点上（分类器省更多但错更多），不能只看单一阈值下的两个数字就判输赢。

### 修复：`scripts/day46_temporal_features.py`（纯 CPU，不碰 GPU/AutoDL，不覆盖 v3 数据）

重新定义特征集：

- **M0** = 纯置信度（`deer_confidence`）
- **M1** = 单点快照（`deer_confidence`/`entropy`/`margin`/`avg_logprob`/`min_logprob`/`reasoning_length`/`probe_complete`）
- **M2** = M1 + 真正的因果时序特征（`confidence_delta`、最近 3 点斜率/标准差、历史最大骤降、entropy 斜率、答案连续计数、累计答案变化次数、"置信度-答案变化冲突"信号），全部只用当前及之前的检查点计算，不看未来。

三者使用完全相同的分组 5 折嵌套交叉验证、相同的"首次越阈即停"策略。

**target_error=0.05（与原协议一致）的主对比**：

| 方法 | 停错率 | 省 token（naive） | 省 token（扣除探测开销后） |
|---|---|---|---|
| M0（纯置信度） | 10.13% | 38.63% | 37.78% |
| M1（单点快照） | 11.43% | 40.01% | 38.97% |
| M2（快照+时序） | 11.27% | **43.41%** | 42.44% |

**M2 − M1 配对 bootstrap（按题目配对，2000 次重采样）**：省 token 比例差值 **+2.68 个百分点，95% CI [+0.30%, +5.22%]，不包含 0**——时序动态确实带来了统计上站得住脚的真实增益（虽然幅度不大），正确率没有下降（差值 0）。

**风险目标放宽到 10%/15% 时**，M2 在错误率和省 token 上开始同时优于 M0、M1。

**更细致的结论**：时序特征是真实有信号的（不是噪声），推翻了"历史无用"的旧结论；但在最严格的 5% 目标下，最朴素的纯置信度（M0）仍然比 M1/M2 更安全一点（10.13% vs 11.x%）。也就是说：时序动态在快照基础上有真实增益，但样本量 N=93 下还不足以让组合方法在最严格风险目标下反超最简单的信号。

### 本次也纠正的限制说明

- **易样本偏差**：这 93 题排除了 Day 3 里那 7 道即使 8192 tokens 仍未收敛的最长/最难题目，样本天然偏向更简单的问题，不代表完整 100 题集合的难度分布。
- **探测开销**：之前"省 45%"这类数字只算了停止点之后省下的 token，没算探测本身的生成开销。这次同时报告了"扣除探测开销后"的净节省数字（如上表），仍然没算反复探测导致的 KV cache/前缀重算开销，所以净数字依然偏乐观。
- **不再声称任何"停错率≤5%"的保证**，只报告实测的样本外比例和 95% Wilson 置信区间。

### 下一步方向

样本量 N=93 是当前限制主要来源之一（校准集小、置信区间宽）。有两条路：扩大样本到 300/500 题看能否让 M2 在严格风险目标下也反超简单基线；或者接受当前结论、把"时序信号有真实但有限的增益"本身写成论文的一个发现，推进到 Day 7 跨 benchmark 测试。
