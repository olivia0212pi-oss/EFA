# 项目计划

本文档记录当前已达成的任务和距离最终目标还需要的任务。实际部署数据和实测结果见 [`log.md`](log.md)；使用说明见 [`README.md`](README.md)。

最终目标：跑通共享的 MATH-500 生成/评分流水线，并在此基础上完成同学 A（提前停止分类器）和同学 B（Power Sampling）两个研究方向的实验。

## 已完成

- [x] AutoDL 环境部署（conda `reason`、PyTorch 2.5.1+cu121、vLLM 0.7.3），单测（`11 passed`）和 ruff 检查全过。
- [x] 单题 GPU 冒烟测试：`12×15=180` 推理正确，`max_model_len: 8192` 解决了 KV Cache 预留过大的问题。
- [x] MATH-500 5 题生成 + 评分脚本本身跑通（流程无报错），但**结果不可用**——见下方问题。

## 进行中 / 阻塞

- [ ] **修复 `configs/smoke.yaml` 的 `max_tokens` 过小问题**：当前 `1024` 导致 5 题中 4 题在给出 `\boxed{}` 答案前被截断，`accuracy: 0.2` 是截断假阴性，不是真实模型能力。需要调大（如恢复到 `base.yaml` 的 `4096`）后重新跑 5 题冒烟测试。
- [ ] 人工抽查修复后的 5 题结果（问题、标准答案、模型最终答案、自动判分是否一致），确认无误后才能进入下一步。

## 待办（按 README.md 顺序）

1. 5 题冒烟测试人工抽查通过。
2. 运行 `configs/math500_100.yaml`（100 题），评分。
3. 人工抽查 100 题评分结果（至少几十条），确认自动判分逻辑在更大样本上稳定可靠。
4. 运行 `configs/math500_full.yaml`（500 题完整实验）。
5. 同学 A：基于完整轨迹跑 `early_exit.build_checkpoints` → `early_exit.oracle` → `early_exit.train_classifier`，产出提前停止分类器。
6. 同学 B：跑 `sampling.power_sampling`（固定/自适应两种模式），验证 MH 接受率记录和废弃 token 统计。
7. 视情况扩展"优先挑难点"的位置提议策略——README.md 已标注这需要先在接受率公式里补上正反向位置提议比，并做玩具分布验证，不能直接改。

## 运维注意事项

- 不用 GPU 时记得在 AutoDL 控制台或通过 SSH 内 `shutdown` 关闭实例，避免持续计费；不要删除数据盘上的 `/root/autodl-tmp/huggingface` 模型缓存。
- 账号密码、SSH 密码、API Key、访问令牌不写入本仓库任何文件。
