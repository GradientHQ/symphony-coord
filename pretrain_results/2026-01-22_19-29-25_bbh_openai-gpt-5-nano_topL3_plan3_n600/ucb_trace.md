# UCB 参数变化记录

本文件对应 pretrain 阶段的 UCB 参数轨迹，逐步追加到 ucb_trace.jsonl。

字段说明：
- i: 全局 step
- phase: 阶段（pretrain）
- t: UCB 迭代步数
- alpha/l2/delta/S/d: UCB 超参数
- A_inv: A 的逆矩阵（列表）
- b: 向量 b
- theta_hat: 估计参数向量
- beta: 置信半径项
