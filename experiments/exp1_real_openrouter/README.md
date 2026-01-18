# Exp1 Real OpenRouter: Efficiency & Cost

这是 Exp1 的真实版本，使用真实的 OpenRouter agents（从配置文件加载）而不是模拟 agents。所有其他逻辑（多策略对比、reward shaping、fallback、drift 等）与模拟版本保持一致。

## 🚀 环境设置（必须先完成）

在运行实验之前，**必须先设置 Python 环境**：

### 1. 创建并激活虚拟环境

```bash
# 进入项目根目录
cd /Users/caohuixi/symphony2.0

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # macOS/Linux
# Windows: venv\Scripts\activate
```

### 2. 安装依赖

```bash
# 升级 pip
pip install --upgrade pip

# 安装主项目依赖
pip install -r requirements.txt

# 安装 symphony-data-generator 依赖（用于加载 HumanEval/GSM8K）
pip install -r symphony-data-generator/requirements.txt
```

### 3. 设置 OpenRouter API Key

```bash
# 临时设置（当前终端会话）
export OPENROUTER_API_KEY="sk-or-v1-..."
# 或永久设置（添加到 ~/.zshrc 或 ~/.bashrc）
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.zshrc
source ~/.zshrc
```

### 4. 验证环境

```bash
# 检查关键依赖
python3 -c "import yaml, numpy, pandas, json; print('✅ Core libraries OK')"
python3 -c "from datasets import load_dataset; print('✅ datasets OK')"
python3 -c "import requests; print('✅ requests OK')"

# 检查 API Key
python3 -c "import os; key = os.getenv('OPENROUTER_API_KEY'); print('✅ API Key set' if key else '❌ API Key not set')"
```

### 5. 准备数据集（如果需要）

```bash
# 如果还没有生成数据集，运行：
cd symphony-data-generator
python src/quick_start.py
cd ..
```

**详细的环境设置说明请参考：[SETUP_ENV.md](SETUP_ENV.md)**

## 前置要求

1. **Python 环境**：已设置虚拟环境并安装所有依赖（见上方）
2. **API Key**：已设置 `OPENROUTER_API_KEY` 环境变量
3. **配置文件**：确保以下配置文件存在于 `runtime/` 目录：
   - `config_agent_openrouter_1.yaml` (Agent A)
   - `config_agent_openrouter_2.yaml` (Agent B)
   - `config_agent_openrouter_3.yaml` (Agent C)
   - `config_agent_openrouter_4.yaml` (Agent D)
   - `config_agent_openrouter_5.yaml` (Agent E)
   - `config_agent_openrouter_6.yaml` (Agent F)
   - `config_agent_openrouter_7.yaml` (Agent G)

## 快速开始

### 基本运行（使用默认值）

```bash
# 确保虚拟环境已激活
source venv/bin/activate  # 如果使用 venv

# 使用所有默认值运行（推荐）
cd /Users/caohuixi/symphony2.0
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py

# 等价于：
# --n 2000 --seed 1 --topL 3 --agents 1,2,3,4,5,6,7 --benchmarks humaneval,gsm8k
```

### 小规模测试

```bash
# 运行 20 个任务的小规模测试
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --n 20 \
  --agents 1,2,3
```

### 完整实验（使用配置文件设置）

```bash
# 运行 2000 个任务的完整实验（使用默认值）
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py

# 或指定参数
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --n 2000 \
  --seed 1 \
  --topL 3 \
  --agents 1,2,3,4,5,6,7 \
  --benchmarks humaneval,gsm8k
```

### 只使用单个 Benchmark

```bash
# 只使用 HumanEval
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --benchmarks humaneval

# 只使用 GSM8K
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --benchmarks gsm8k
```

### 带 drift 的实验

```bash
cd /Users/caohuixi/symphony2.0

# 运行包含 drift 的实验
python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --n 1000 \
  --p-hard 0.2 \
  --seed 123 \
  --topL 2 \
  --fallback \
  --drift \
  --agents 1,2,3,4,5,6
```

python3 experiments/exp1_real_openrouter/exp1_real_openrouter.py \
  --n 100 \
  --p-hard 0.2 \
  --seed 123 \
  --topL 2 \
  --fallback \
  --drift \
  --agents 1,2,3,4,5,6,7 \
  --benchmarks humaneval,gsm8k

## 参数说明

### 基本参数
- `--n`: 任务数量（默认：`2000`，从 config_exp1.yaml）
- `--p-hard`: 困难任务概率（默认：`0.2`，80/20 分割）
- `--seed`: 随机种子（默认：`1`）
- `--topL`: Top-L 候选选择（默认：`3`，从 config_exp1.yaml）
- `--benchmarks`: 使用的 benchmark 列表，逗号分隔（默认：`humaneval,gsm8k`）
  - 可选值：`humaneval`, `gsm8k`, 或 `humaneval,gsm8k`
- `--outdir`: 输出目录（默认：`experiments/exp1_real_openrouter/results`）
- `--no-plots`: 不生成图表
- `--config-dir`: 配置文件目录（默认：`runtime`）
- `--agents`: 要使用的 agent ID 列表，逗号分隔（默认：`1,2,3,4,5,6,7`，所有 7 个 agents）

### LinUCB 参数
- `--alpha`: LinUCB 探索尺度（默认：`1.0`，从 config_exp1.yaml）
- `--l2`: LinUCB L2 正则化（默认：`1.0`，从 config_exp1.yaml）
- `--delta`: LinUCB 置信度（默认：`0.1`，从 config_exp1.yaml）
- `--S`: ||theta*|| 的上界（默认：`1.0`，从 config_exp1.yaml）

### Reward Shaping 参数
- `--latency-scale-ms`: 延迟归一化尺度（默认：`2000.0`）
- `--latency-penalty`: 延迟惩罚乘数（默认：`0.05`，从 config_exp1.yaml）
- `--cost-lambda`: 成本惩罚乘数（默认：`0.3`，从 config_exp1.yaml）

### 真实度开关
- `--fallback`: 如果选择的 agent 失败，fallback 到强 agent A
- `--drift`: 启用非平稳 drift（t>=500 后，目前对真实 agents 忽略）

## 输出

结果保存在 `experiments/exp1_real_openrouter/results/YYYY-MM-DD_HH-MM-SS/` 目录下：

- `summary.csv`: 各策略的汇总统计（cost / success / latency / fallback 等）
- `trajectory_always_A.csv`: Always-A 策略的逐步轨迹
- `trajectory_static_rule.csv`: Static rule 策略的逐步轨迹
- `trajectory_random.csv`: Random 策略的逐步轨迹
- `trajectory_linucb.csv`: LinUCB 策略的逐步轨迹
- `plot_cost_success_pretty.png`: 成本和成功率的柱状图
- `plot_all_cum_cost_clean.png`: 累积平均成本曲线
- `plot_all_cum_success_clean.png`: 累积平均成功率曲线

## 示例输出

运行时会看到类似以下输出：

```
============================================================
Loading OpenRouter Agents
============================================================
✅ Loaded agent A: agent-openrouter-001 (openrouter:google/gemini-2.5-flash-lite)
✅ Loaded agent B: agent-openrouter-002 (openrouter:openai/gpt-5-nano)
✅ Loaded agent C: agent-openrouter-003 (openrouter:openai/gpt-4o-mini)
...

✅ Loaded 6 agents

============================================================
Generating Tasks
============================================================
✅ Generated 1000 tasks

============================================================
Running Policies
============================================================

[always_A] Starting...
[always_A] Task 10/1000, Success: 9/10 (90.0%)
[always_A] Task 20/1000, Success: 18/20 (90.0%)
...
[always_A] success=0.950 total_cost=150.0 avg_cost=0.150 avg_lat=900.0ms fallback=0.000 choose(A,B,C,D,E)=(1000,0,0,0,0)

[static_rule] Starting...
...
[linucb] Starting...
...
```

## 与模拟版本的对比

- **相同点**：
  - 所有策略逻辑完全一致
  - Reward shaping 参数一致
  - LinUCB 参数一致
  - 状态更新逻辑一致

- **不同点**：
  - 模拟版本使用 `SimAgent`（模拟执行）
  - 真实版本使用 `RealAgentWrapper`（调用真实 OpenRouter API）
  - 真实版本需要 API key 和网络连接
  - 真实版本的执行时间取决于 API 响应时间

## 注意事项

1. **API 成本**：真实版本会调用 OpenRouter API，会产生实际成本。建议先用小规模测试（`--n 20`）验证设置。

2. **网络连接**：确保网络连接正常，能够访问 `openrouter.ai`。

3. **API Key**：确保 `OPENROUTER_API_KEY` 环境变量已设置。

4. **执行时间**：真实版本的执行时间取决于 API 响应时间，1000 个任务可能需要较长时间。

5. **错误处理**：如果某个 agent 调用失败，会打印错误信息并继续执行（失败的任务会被标记为失败）。

## 故障排除

如果遇到问题：

1. **检查 API Key**：
   ```bash
   echo $OPENROUTER_API_KEY
   ```

2. **测试配置**：
   ```bash
   python3 test_openrouter_setup.py
   ```

3. **查看错误信息**：运行时会打印详细的错误信息，包括失败的 API 调用。

