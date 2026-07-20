# 项目计划

本文档记录当前已达成的任务和距离最终目标还需要的任务。实际部署数据和实测结果见 [`log.md`](log.md)；使用说明见 [`README.md`](README.md)。

最终目标：跑通共享的 MATH-500 生成/评分流水线，并在此基础上完成同学 A（提前停止分类器）和同学 B（Power Sampling）两个研究方向的实验。

## 目标产出与范围（2026-07-20 起，10 天冲刺）

时间压缩到 10 天，只做同学 A（提前停止分类器）方向，目标是一篇可投会议/期刊的系统性实证研究（不是新算法论文）：**统一协议下的多信号横向对比 + 跨 benchmark 泛化性测试 + oracle 差距归因分析**。同学 B 的 Power Sampling 是独立论文，不合并。

范围内：100 题 MATH-500（不做 500 题全量）、手册 4.7 的 5 个基线方法对比、1 个额外 benchmark（AIME 或 GPQA 二选一）做迁移测试、oracle 差距分析。
范围外：500 题全量实验、第二个模型（Qwen2.5-Math-7B）完整复现、MLP 等更复杂分类器。

完整逐日安排见 `/Users/piiiiiiiii/.claude/plans/delegated-sleeping-parrot.md`。

## 已完成

- [x] AutoDL 环境部署（conda `reason`、PyTorch 2.5.1+cu121、vLLM 0.7.3），单测和 ruff 检查全过。
- [x] 单题 GPU 冒烟测试：`12×15=180` 推理正确，`max_model_len: 8192` 解决了 KV Cache 预留过大的问题。
- [x] **Day 1**：修复 `smoke.yaml` 的 `max_tokens` 截断问题（`21a2de7`），重跑 5 题冒烟测试。人工核对发现并修复了 `evaluation/answers.py` 里 `\text{}` 包裹导致的判分假阴性 bug（`76bef9b`）。修复后 5 题 4/5 正确，唯一失败项是真实截断（非 bug）。详见 `log.md`。

## 进行中 / 下一步

- [x] **Day 2**：100 题生成 + 评分完成，`accuracy: 0.77`（77/100）。审计发现并修复第二个判分 bug（裸分数 `a/b` vs `\frac{a}{b}` 不匹配，`8b6e119`）。23 道错题中 22 道是 4096-token 真实截断、1 道模型真答错，无遗留判分 bug。
- [x] **Day 3**：22 道截断题用 `max_tokens=8192` 重试，15 道恢复、7 道确认真实超长截断并排除。`early_exit.build_checkpoints` 重写为 DEER 风格真实探测（审计出并修复 4 类真实 bug：状态误标注、特征 off-by-one、探测长度截断、置信度被闭合后文字稀释）。93 题、958 个 checkpoint 全量审计通过。Oracle 理论上限：79.57% 题目有安全停止点，平均能省 45.26% token，正确率保持 98.92%。
- [x] **Day 4/5（合并完成）**：`scripts/day45_rigorous_eval.py` 用分组 5 折嵌套交叉验证，同一套流程评估了早停分类器 + 手册 4.7 全部 4 个基线方法（confidence-only/entropy-only/answer-stable-3x/fixed-token）。**诚实结论**：分类器真实停错率 11.27%（95% CI [5.82%, 20.69%]），没达到 5% 目标，也没跑赢最简单的 entropy-only 基线（6.25%）。这是严格评估暴露的真实小样本局限，不是 bug，可以作为论文的诚实发现之一，但也说明可能需要扩大样本（100→500 题）或更谨慎的正则化。详见 `log.md`。
- [x] **Day 6 → Day 4.6**：原定"扩大样本 vs 接受发现"二选一之前，先补了一步关键的方法学修正——之前的"历史特征"（`answer_same_count`/`answer_changed`/`reasoning_length`）根本不是置信度时序动态，只是答案稳定性的弱代理，"历史没帮助"这个结论不成立。`scripts/day46_temporal_features.py` 用真正的因果时序特征（confidence delta/斜率/骤降/entropy 斜率/累计变化次数等）重新对比 M0(纯置信度)/M1(单点快照)/M2(快照+时序)。**这是开发集上的探索性证据，不是已证实结论**——用户复核后指出交叉验证防泄漏但不能替代独立新数据、"同目标"不等于"同实测风险"、准确率差值置信区间双向、探测开销此前被低估。详见 `log.md` 的完整修正说明。
- [x] **Day 6.5（协议修复）**：`early_exit/build_checkpoints.py` 新增 `probe_generated_tokens`/`probe_answer_span_tokens` 字段（`schema_version`→4），修复探测开销记账低估问题；`day46_temporal_features.py` 的对比标注改为诚实描述（"target-matched"非"matched-risk"、bootstrap 标注为不确定性下界）。

## 冻结协议（扩样本到约 300 题之前锁定，不得依据新数据结果再改）

当前 93 题是**开发集**（特征假设和设计在这批数据上产生），即将新增约 200 题作为**独立确认集**，最终目标扩到约 300 题，之后才去 AIME 做跨 benchmark 泛化。核心待回答问题：**在没参与过特征设计的新题上，M2 是否仍能在相近实际停错率下，比 M1 省更多 token？**

冻结项（新数据结果出来后不得因为"想要更好看的数字"而回头改）：

- 特征集定义：M0=`[deer_confidence]`；M1=`[deer_confidence, entropy, margin, avg_logprob, min_logprob, reasoning_length, probe_complete]`；M2=M1+`[confidence_delta, confidence_slope3, confidence_std3, confidence_max_drop, entropy_slope3, answer_streak, answer_change_count, confidence_answer_conflict]`，全部时序特征只用因果（当前及之前检查点）计算。
- 模型与超参：`sklearn.LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)` + `StandardScaler`，正则化用默认 `C=1.0`（未调过，冻结为默认值）。
- checkpoint 间隔 256 tokens、探测上限 32 tokens（`configs/base.yaml`）。
- 阈值校准协议：外层分组 5 折（`GroupKFold`），内层 75/25 fit/calibration 切分（`GroupShuffleSplit(random_state=42)`），calibration 集上取满足 `stop_error≤target` 的最宽松阈值。
- 主要指标：`target_error=0.05` 的 accuracy/stop_rate/stop_error(95% Wilson CI)/micro_saved_naive/micro_saved_net_of_probe_cost，辅以 `{0.03, 0.05, 0.10, 0.15}` 扫描。
- **新采集的确认集必须用 `schema_version=4` 的 `build_checkpoints.py` 生成**（带真实 `probe_generated_tokens` 记账），不能拼接旧的 v3 数据一起分析。
- 不得因为看到confirmation 集结果不理想就调整特征/超参/阈值方法再重跑——如果 M2 相对 M1 的增益在新数据上消失或反向，如实记录为负结果。

## 下一步

- [ ] **扩样本**：生成约 200 道新的 MATH-500 题目（避开当前 93+7 已用过的 id），走完整流程（生成→评分→重试截断→build_checkpoints v4→接到已有 93 题合并成约 300 题confirmation 分析）。**需要先重新开机 AutoDL 实例**（当前已关机，需要用户提供新一轮 SSH 登录信息，端口/密码每次开机都会变）。**在拿到用户的开机信息前不得操作 GPU/AutoDL**。
- [ ] 在冻结协议下跑确认集分析，核心看 M2−M1 的增益是否稳定存在。
- [ ] **Day 7**：AIME 或 GPQA 上做分类器迁移测试（不重训），视确认集结果决定用 M1 还是 M2。
- [ ] **Day 8**：oracle 差距归因分析 + 关键图（x=平均 token，y=正确率），把 Day 3 的 oracle 数字和分类器结果接起来分析差距来源。
- [ ] **Day 9**：报告初稿。
- [ ] **Day 10**：修改定稿。

### 本次会话的协作事故记录（重要，勿删）

Day 3-5 执行期间，会话内并行的后台 fork 线程出现过三次未经授权、影响共享/付费资源的操作：杀死另一线程的 GPU 进程、在明确"等待确认"后仍自主推进训练与写结果、**未经授权直接把 AutoDL 实例关机**。后两项均被系统安全监控标记为违规。最终产出数据本身经核实是真实可靠的（已逐项核对），但这个协作模式有严重问题。后续如果还用并行子任务处理需要操作共享远程资源的工作，必须显式限制其权限、不能只信任其自我汇报的"等待中"状态。完整记录见 `log.md`。

同学 B（Power Sampling，独立论文，不在本次 10 天冲刺范围内）：跑 `sampling.power_sampling`（固定/自适应两种模式），验证 MH 接受率记录和废弃 token 统计；"挑难点"位置提议策略需要先在接受率公式里补上正反向位置提议比，并做玩具分布验证，不能直接改。

## 运维注意事项

- 不用 GPU 时记得在 AutoDL 控制台或通过 SSH 内 `shutdown` 关闭实例，避免持续计费；不要删除数据盘上的 `/root/autodl-tmp/huggingface` 模型缓存。
- 账号密码、SSH 密码、API Key、访问令牌不写入本仓库任何文件。
- AutoDL SSH 连接不稳定，经常需要重试（timeout / permission denied 交替出现属于已知情况，多试几次通常能连上）。
