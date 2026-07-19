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

- [ ] **Day 2**：跑 `configs/math500_100.yaml`（100 题）生成 + 评分，人工抽查至少 20-30 条，特别注意非数字类答案（文字/字母/人名）是否被 LaTeX 包裹坑到判分。
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
