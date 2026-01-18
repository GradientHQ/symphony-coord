# Exp1 真实实验设置说明

根据你提供的详细 setting，我已经创建了配置文件 `config_exp1.yaml`。这个文档说明如何将这些设置应用到代码中。

## 当前状态

✅ **已完成**：
1. 创建了 `config_exp1.yaml` 配置文件，包含所有实验设置
2. 配置文件包含：
   - 7个agents的完整配置（A-G）
   - 数据流设置（HumanEval 164 + GSM8K 200）
   - 解码参数（temperature=0.2, top_p=0.95）
   - Reward设计（r = correct - 0.3*cost_norm - 0.05*lat_norm）
   - LinUCB参数（L=3, λ=1.0, α=1.0, δ=0.1）
   - Baselines设置（SOP-Strong, SOP-Rule）

## 需要更新的代码部分

### 1. Agent Pool（7个模型）

**当前代码**：`agent_id_map` 只有6个agents（A-F）

**需要更新**：
- 添加第7个agent（G: Qwen3-4B）
- 从 `config_exp1.yaml` 读取agent配置
- 支持真实美元成本计算

**位置**：`experiments/exp1_real_openrouter/exp1_real_openrouter.py` line 233-240

### 2. 任务生成（HumanEval + GSM8K）

**当前代码**：使用简单的placeholder prompts

**需要更新**：
- 从实际数据集加载HumanEval（164个任务）和GSM8K（200个任务）
- 实现with replacement采样
- 支持code/math混合（1:1比例）
- 支持easy/hard混合（80/20比例）

**位置**：`experiments/exp1_real_openrouter/exp1_real_openrouter.py` line 187-218

### 3. Reward设计

**当前代码**：`reward = 1(success) - latency_penalty*sqrt(lat_norm) - cost_lambda*cost_norm`

**需要更新**：
- 改为：`r = 1[correct] - λ_c * c_norm - λ_t * t_norm`
- λ_c = 0.3, λ_t = 0.05
- 从配置文件读取参数

**位置**：`experiments/exp1_real_openrouter/exp1_real_openrouter.py` line 306-315

### 4. 解码参数

**当前代码**：从agent配置读取

**需要更新**：
- 统一设置：temperature=0.2, top_p=0.95
- GSM8K: max_output_tokens=256
- HumanEval: max_output_tokens=512
- 统一系统提示

**位置**：需要在 `RealAgentWrapper.execute()` 中应用

### 5. Cost计算

**当前代码**：使用简化的 `call_cost` 估算

**需要更新**：
- 支持真实美元成本（API tokens × 单价）
- 支持本地模型GPU-hour成本
- 从配置文件读取价格信息

**位置**：需要创建新的cost计算函数

### 6. Baselines

**当前代码**：`always_A`, `static_rule`, `random`, `linucb`

**需要更新**：
- `always_A` → `SOP-Strong`（使用配置中的strong_model）
- `static_rule` → `SOP-Rule`（easy→cheap, hard→strong）
- 从配置文件读取cheap_model和strong_model

**位置**：`experiments/exp1_real_openrouter/exp1_real_openrouter.py` run_policy函数

### 7. LinUCB参数

**当前代码**：默认值（alpha=1.0, l2=1.0, delta=0.05, S=1.0）

**需要更新**：
- delta = 0.1（按setting要求）
- 从配置文件读取所有参数

**位置**：`experiments/exp1_real_openrouter/exp1_real_openrouter.py` line 348

### 8. Context特征

**当前代码**：使用简单的6维特征（match_score, load, latency, reputation, available）

**需要更新**：
- 添加task特征（is_code, is_math, is_easy, is_hard）
- 添加agent特征（model_id_onehot, log_price）
- 添加动态状态特征（ema_success, ema_latency, ema_cost）
- 更新 `build_x` 函数或创建新的特征构建函数

**位置**：需要更新特征构建逻辑

## 配置文件结构

配置文件 `config_exp1.yaml` 包含：

1. **agents**: 7个agents的完整配置
2. **data**: 数据流设置（任务池、采样、比例）
3. **decoding**: 解码参数和系统提示
4. **cost**: 成本计算方式
5. **reward**: Reward设计参数
6. **routing**: Symphony 2.0路由设置
7. **baselines**: Baselines配置
8. **output**: 输出指标和图表设置

## 下一步

建议按以下顺序更新代码：

1. ✅ 创建配置文件（已完成）
2. ⏳ 添加配置加载函数
3. ⏳ 更新agent_id_map支持7个agents
4. ⏳ 实现真实数据集加载（HumanEval + GSM8K）
5. ⏳ 更新reward设计
6. ⏳ 更新cost计算
7. ⏳ 更新baselines
8. ⏳ 更新context特征
9. ⏳ 更新解码参数应用

## 注意事项

1. **数据集加载**：需要确保可以访问HumanEval和GSM8K数据集（可能需要通过datasets库或本地文件）
2. **成本计算**：真实美元成本需要从API响应中获取token数量
3. **特征维度**：更新context特征后，需要更新LinUCB的维度参数（d）
4. **向后兼容**：更新时尽量保持与现有代码的兼容性

