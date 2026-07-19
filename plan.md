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

- [x] **Day 2**：100 题生成 + 评分完成，`accuracy: 0.77`（77/100）。审计发现并修复第二个判分 bug（裸分数 `a/b` vs `\frac{a}{b}` 不匹配，`8b6e119`）。23 道错题中 22 道是 4096-token 真实截断、1 道模型真答错，无遗留判分 bug。**注意**：发现有另一条并行工作线（3 个非本会话提交的 commit `d9d0295`/`d27dd93`/`b25e16d`）在同时做同学 A 的人工抽查和判分修复，需要和用户确认是谁在跑，避免重复劳动。详见 `log.md`。
- [ ] 用户亲自抽样复核 `results/math500_100_scored.jsonl`（尤其 22 道截断题和非数字类答案），作为对自动化审计的复核。
- [ ] **Day 3**：`early_exit.build_checkpoints` + `early_exit.oracle`，算出理论最大节省比例。
- [ ] **Day 4**：训练早停分类器（逻辑回归），按停错率≤5%在校准集调阈值，跑通完整早停流程，得到 MVP 结果（省 token / 正确率）。
- [ ] **Day 5**：实现手册 4.7 其余基线方法，统一停错率协议下横向对比。
- [ ] **Day 6**：缓冲日。
- [ ] **Day 7**：AIME 或 GPQA 上做分类器迁移测试（不重训）。
- [ ] **Day 8**：oracle 差距归因分析 + 关键图（x=平均 token，y=正确率）。
- [ ] **Day 9**：报告初稿。
- [ ] **Day 10**：修改定稿。

同学 B（Power Sampling，独立论文，不在本次 10 天冲刺范围内）：跑 `sampling.power_sampling`（固定/自适应两种模式），验证 MH 接受率记录和废弃 token 统计；"挑难点"位置提议策略需要先在接受率公式里补上正反向位置提议比，并做玩具分布验证，不能直接改。

## 运维注意事项

- 不用 GPU 时记得在 AutoDL 控制台或通过 SSH 内 `shutdown` 关闭实例，避免持续计费；不要删除数据盘上的 `/root/autodl-tmp/huggingface` 模型缓存。
- 账号密码、SSH 密码、API Key、访问令牌不写入本仓库任何文件。
- AutoDL SSH 连接不稳定，经常需要重试（timeout / permission denied 交替出现属于已知情况，多试几次通常能连上）。
