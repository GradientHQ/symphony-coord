# OpenRouter 配置指南

## 目录

1. [API Key 设置](#api-key-设置)
2. [Agent 配置](#agent-配置)
3. [使用方式](#使用方式)
4. [故障排除](#故障排除)

---

## API Key 设置

### ⚠️ 重要：API Key 安全

**API Key 目前没有存储在代码文件中**，这是正确的做法！

当前情况：
- ✅ API key **不在**配置文件中
- ✅ API key **不在**代码中硬编码
- ✅ API key 只在环境变量中存储

**安全规则：**
- ❌ **不要**将 API key 提交到 Git 仓库
- ❌ **不要**在配置文件中硬编码 API key
- ✅ **使用**环境变量或 `.env` 文件
- ✅ **确保** `.env` 在 `.gitignore` 中
- ✅ **不要**在公共场合分享 API key

### 方法1: 临时设置（当前会话有效）

在终端中运行：

```bash
export OPENROUTER_API_KEY="sk-or-v1-[...]"
```

### 方法2: 持久化设置（推荐用于个人开发）

添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
# 添加到文件末尾
echo 'export OPENROUTER_API_KEY="sk-or-v1-[...]"' >> ~/.zshrc

# 重新加载配置
source ~/.zshrc
```

### 方法3: 使用 .env 文件（推荐用于开发环境）

1. 在项目根目录创建 `.env` 文件：

```bash
cd /Users/caohuixi/symphony2.0
cat > .env << 'EOF'
OPENROUTER_API_KEY=sk-or-v1-[...]
EOF
```

2. 确保 `.env` 在 `.gitignore` 中（避免提交到 Git）：

```bash
# 检查 .gitignore
if ! grep -q "^\.env$" .gitignore; then
    echo ".env" >> .gitignore
    echo "✅ Added .env to .gitignore"
fi
```

3. 代码会自动读取环境变量，**不需要**额外加载 .env 文件（见下方说明）。

### 验证 API Key 是否设置

在终端中验证：

```bash
# 检查环境变量
echo $OPENROUTER_API_KEY

# 或者在 Python 中验证
python3 -c "import os; print('✅ API Key set' if os.getenv('OPENROUTER_API_KEY') else '❌ API Key not set')"
```

### 代码中的实现

在 `agents/agent.py` 中，API key 通过以下方式读取：

```python
api_key = os.getenv('OPENROUTER_API_KEY', os.getenv('OPENAI_API_KEY', 'EMPTY'))
```

这意味着：
- 优先使用 `OPENROUTER_API_KEY` 环境变量
- 如果没有，回退到 `OPENAI_API_KEY`
- 如果都没有，使用 `'EMPTY'`（会导致 API 调用失败）

**注意**：Symphony 2.0 的代码已经使用 `os.getenv()`，所以如果你设置了环境变量，代码会自动读取。如果使用 `.env` 文件，你可能需要安装 `python-dotenv` 包并在代码中调用 `load_dotenv()`，但对于大多数情况，直接设置环境变量就足够了。

---

## Agent 配置

### Agent 1: Gemini 2.5 Flash-Lite

创建 `runtime/config_agent_openrouter_1.yaml`:

```yaml
# Agent 1: Gemini 2.5 Flash-Lite
debug: true
role: "agent"
node_id: "agent-openrouter-001"
base_model: "openrouter:google/gemini-2.5-flash-lite"
sys_prompt: "You are a helpful AI assistant specialized in fast responses."
capabilities:
  - math
  - reasoning
  - general
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
```

### Agent 2: GPT-5 Nano

创建 `runtime/config_agent_openrouter_2.yaml`:

```yaml
# Agent 2: GPT-5 Nano
debug: true
role: "agent"
node_id: "agent-openrouter-002"
base_model: "openrouter:openai/gpt-5-nano"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
```

### Agent 3: GPT-4o Mini

创建 `runtime/config_agent_openrouter_3.yaml`:

```yaml
# Agent 3: GPT-4o Mini
debug: true
role: "agent"
node_id: "agent-openrouter-003"
base_model: "openrouter:openai/gpt-4o-mini"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
```

### Agent 4: GPT-5.1 Codex Mini

创建 `runtime/config_agent_openrouter_4.yaml`:

```yaml
# Agent 4: GPT-5.1 Codex Mini
debug: true
role: "agent"
node_id: "agent-openrouter-004"
base_model: "openrouter:openai/gpt-5.1-codex-mini"
sys_prompt: "You are a helpful AI assistant specialized in code generation."
capabilities:
  - code
  - debugging
  - code-implementation
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
```

### Agent 5: Qwen2.5-7B-Instruct

创建 `runtime/config_agent_openrouter_5.yaml`:

```yaml
# Agent 5: Qwen2.5-7B-Instruct
debug: true
role: "agent"
node_id: "agent-openrouter-005"
base_model: "openrouter:qwen/qwen-2.5-7b-instruct"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
```

### Agent 6: Qwen3-8B

创建 `runtime/config_agent_openrouter_6.yaml`:

```yaml
# Agent 6: Qwen3-8B
debug: true
role: "agent"
node_id: "agent-openrouter-006"
base_model: "openrouter:qwen/qwen-3-8b"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
```

### 快速创建所有配置文件

使用以下命令快速创建所有配置文件：

```bash
cd /Users/caohuixi/symphony2.0/runtime

# Agent 1: Gemini 2.5 Flash-Lite
cat > config_agent_openrouter_1.yaml << 'EOF'
debug: true
role: "agent"
node_id: "agent-openrouter-001"
base_model: "openrouter:google/gemini-2.5-flash-lite"
sys_prompt: "You are a helpful AI assistant specialized in fast responses."
capabilities:
  - math
  - reasoning
  - general
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
EOF

# Agent 2: GPT-5 Nano
cat > config_agent_openrouter_2.yaml << 'EOF'
debug: true
role: "agent"
node_id: "agent-openrouter-002"
base_model: "openrouter:openai/gpt-5-nano"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
EOF

# Agent 3: GPT-4o Mini
cat > config_agent_openrouter_3.yaml << 'EOF'
debug: true
role: "agent"
node_id: "agent-openrouter-003"
base_model: "openrouter:openai/gpt-4o-mini"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
EOF

# Agent 4: GPT-5.1 Codex Mini
cat > config_agent_openrouter_4.yaml << 'EOF'
debug: true
role: "agent"
node_id: "agent-openrouter-004"
base_model: "openrouter:openai/gpt-5.1-codex-mini"
sys_prompt: "You are a helpful AI assistant specialized in code generation."
capabilities:
  - code
  - debugging
  - code-implementation
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
EOF

# Agent 5: Qwen2.5-7B-Instruct
cat > config_agent_openrouter_5.yaml << 'EOF'
debug: true
role: "agent"
node_id: "agent-openrouter-005"
base_model: "openrouter:qwen/qwen-2.5-7b-instruct"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
EOF

# Agent 6: Qwen3-8B
cat > config_agent_openrouter_6.yaml << 'EOF'
debug: true
role: "agent"
node_id: "agent-openrouter-006"
base_model: "openrouter:qwen/qwen-3-8b"
sys_prompt: "You are a helpful AI assistant."
capabilities:
  - math
  - code
  - reasoning
max_tokens: 512
temperature: 0.2
top_p: 0.9
gpu_id: 0
EOF
```

### 模型名称映射

| 显示名称 | OpenRouter 格式 | 完整配置值 |
|---------|----------------|-----------|
| Gemini 2.5 Flash-Lite | `google/gemini-2.5-flash-lite` | `openrouter:google/gemini-2.5-flash-lite` |
| GPT-5 Nano | `openai/gpt-5-nano` | `openrouter:openai/gpt-5-nano` |
| GPT-4o Mini | `openai/gpt-4o-mini` | `openrouter:openai/gpt-4o-mini` |
| GPT-5.1 Codex Mini | `openai/gpt-5.1-codex-mini` | `openrouter:openai/gpt-5.1-codex-mini` |
| Qwen2.5-7B-Instruct | `qwen/qwen-2.5-7b-instruct` | `openrouter:qwen/qwen-2.5-7b-instruct` |
| Qwen3-8B | `qwen/qwen-3-8b` | `openrouter:qwen/qwen-3-8b` |

---

## 使用方式

### 1. 设置 API Key

按照 [API Key 设置](#api-key-设置) 部分的说明设置环境变量。

### 2. 加载配置创建 Agent

```python
from agents.agent import Agent
import yaml

# 加载配置
with open('runtime/config_agent_openrouter_1.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建 Agent
agent = Agent(config=config)
```

### 3. 验证配置

```python
# 验证 agent 是否正确初始化
print(f"Agent ID: {agent.agent_id}")
print(f"Base Model: {config['base_model']}")
print(f"Capabilities: {agent.capabilities}")
```

---

## 故障排除

### 如果遇到 "Invalid API Key" 错误

1. 检查环境变量是否设置：`echo $OPENROUTER_API_KEY`
2. 确认 API key 格式正确（以 `sk-or-v1-` 开头）
3. 检查 API key 是否过期或被撤销
4. 验证 API key 是否生效：`python3 -c "import os; print('✅ API Key set' if os.getenv('OPENROUTER_API_KEY') else '❌ API Key not set')"`

### 如果遇到 "Model not found" 错误

1. 确认模型名称正确（参考 OpenRouter 文档）
2. 检查模型是否可用（有些模型可能需要特殊权限）
3. 尝试使用其他可用模型
4. 参考上方的[模型名称映射表](#模型名称映射)确认格式正确

### 如果遇到网络错误

1. 检查网络连接
2. 检查防火墙设置
3. 尝试使用 VPN（如果在中国大陆）
4. 检查 OpenRouter 服务状态

### 其他常见问题

- **API key 未生效**：确保在运行代码的终端中设置了环境变量，或重启终端/IDE
- **配置文件格式错误**：检查 YAML 语法，确保缩进正确
- **成本控制**：注意 API 调用成本，建议在 OpenRouter 控制台设置预算限制

---

## 总结

**API key 应该放在：**
1. ✅ 环境变量（`export OPENROUTER_API_KEY="..."`）- **推荐用于生产环境**
2. ✅ `.env` 文件（在项目根目录）- **推荐用于开发环境**
3. ✅ `~/.bashrc` 或 `~/.zshrc`（持久化）- **推荐用于个人开发**

**API key 不应该放在：**
- ❌ 配置文件中（`runtime/config_*.yaml`）
- ❌ 代码文件中（`*.py`）
- ❌ Git 仓库中（应该忽略）
