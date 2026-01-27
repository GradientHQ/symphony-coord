# OpenRouter Configuration Guide

## Table of Contents

1. [API Key Setup](#api-key-setup)
2. [Agent Configuration](#agent-configuration)
3. [Usage](#usage)
4. [Troubleshooting](#troubleshooting)

---

## API Key Setup

### ⚠️ Important: API Key Security

**API Keys are NOT stored in code files** - this is the correct approach!

Current status:
- ✅ API key is **not** in configuration files
- ✅ API key is **not** hardcoded in source code
- ✅ API key is only stored in environment variables

**Security Rules:**
- ❌ **Do not** commit API keys to Git repository
- ❌ **Do not** hardcode API keys in configuration files
- ✅ **Use** environment variables or `.env` files
- ✅ **Ensure** `.env` is in `.gitignore`
- ✅ **Do not** share API keys publicly

### Method 1: Temporary Setup (Current Session Only)

Run in terminal:

```bash
export OPENROUTER_API_KEY="sk-or-v1-[...]"
```

### Method 2: Persistent Setup (Recommended for Personal Development)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# Add to end of file
echo 'export OPENROUTER_API_KEY="sk-or-v1-[...]"' >> ~/.zshrc

# Reload configuration
source ~/.zshrc
```

### Method 3: Using .env File (Recommended for Development)

1. Create `.env` file in project root:

```bash
cd /path/to/symphony
cat > .env << 'EOF'
OPENROUTER_API_KEY=sk-or-v1-[...]
EOF
```

2. Ensure `.env` is in `.gitignore` (to prevent committing to Git):

```bash
# Check .gitignore
if ! grep -q "^\.env$" .gitignore; then
    echo ".env" >> .gitignore
    echo "✅ Added .env to .gitignore"
fi
```

3. Code will automatically read environment variables; **no need** to load .env file separately (see below).

### Verify API Key is Set

Verify in terminal:

```bash
# Check environment variable
echo $OPENROUTER_API_KEY

# Or verify in Python
python3 -c "import os; print('✅ API Key set' if os.getenv('OPENROUTER_API_KEY') else '❌ API Key not set')"
```

### Implementation in Code

In `agents/agent.py`, API key is read as follows:

```python
api_key = os.getenv('OPENROUTER_API_KEY', os.getenv('OPENAI_API_KEY', 'EMPTY'))
```

This means:
- Prefers `OPENROUTER_API_KEY` environment variable
- Falls back to `OPENAI_API_KEY` if not available
- Uses `'EMPTY'` if neither is available (will cause API calls to fail)

**Note**: Symphony 2.0 code already uses `os.getenv()`, so if you set the environment variable, code will read it automatically. If using `.env` file, you may need to install `python-dotenv` package and call `load_dotenv()` in code, but for most cases, directly setting environment variables is sufficient.

---

## Agent Configuration

> ✅ Recommended structure: organize files by model/agent name:
> `runtime/configs/openrouter/<agent-name>/config_agent_openrouter_<id>.yaml`

### Agent 1: Gemini 2.5 Flash-Lite

Create `runtime/configs/openrouter/<agent-name>/config_agent_openrouter_1.yaml`:

```yaml
# Agent 1: Gemini 2.5 Flash-Lite
debug: false
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

Create `runtime/configs/openrouter/config_agent_openrouter_2.yaml`:

```yaml
# Agent 2: GPT-5 Nano
debug: false
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

Create `runtime/configs/openrouter/config_agent_openrouter_3.yaml`:

```yaml
# Agent 3: GPT-4o Mini
debug: false
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

### Model Name Mapping

| Display Name | OpenRouter Format | Full Config Value |
|---------|----------------|-----------|
| Gemini 2.5 Flash-Lite | `google/gemini-2.5-flash-lite` | `openrouter:google/gemini-2.5-flash-lite` |
| GPT-5 Nano | `openai/gpt-5-nano` | `openrouter:openai/gpt-5-nano` |
| GPT-4o Mini | `openai/gpt-4o-mini` | `openrouter:openai/gpt-4o-mini` |
| GPT-5.1 Codex Mini | `openai/gpt-5.1-codex-mini` | `openrouter:openai/gpt-5.1-codex-mini` |
| Qwen2.5-7B-Instruct | `qwen/qwen-2.5-7b-instruct` | `openrouter:qwen/qwen-2.5-7b-instruct` |
| Qwen3-8B | `qwen/qwen-3-8b` | `openrouter:qwen/qwen-3-8b` |

---

## Usage

### 1. Set API Key

Follow the [API Key Setup](#api-key-setup) section instructions to set environment variables.

### 2. Load Configuration and Create Agent

```python
from agents.agent import Agent
import yaml

# Load configuration
with open('runtime/configs/openrouter/config_agent_openrouter_1.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create Agent
agent = Agent(config=config)
```

### 3. Verify Configuration

```python
# Verify agent is correctly initialized
print(f"Agent ID: {agent.agent_id}")
print(f"Base Model: {config['base_model']}")
print(f"Capabilities: {agent.capabilities}")
```

---

## Troubleshooting

### If you encounter "Invalid API Key" error

1. Check if environment variable is set: `echo $OPENROUTER_API_KEY`
2. Confirm API key format is correct (starts with `sk-or-v1-`)
3. Check if API key has expired or been revoked
4. Verify API key is working: `python3 -c "import os; print('✅ API Key set' if os.getenv('OPENROUTER_API_KEY') else '❌ API Key not set')"`

### If you encounter "Model not found" error

1. Confirm model name is correct (refer to OpenRouter documentation)
2. Check if model is available (some models may require special permissions)
3. Try using another available model
4. Refer to [Model Name Mapping](#model-name-mapping) table above for correct format

### If you encounter network errors

1. Check network connection
2. Check firewall settings
3. Try using VPN (if in restricted regions)
4. Check OpenRouter service status

### Other Common Issues

- **API key not working**: Ensure environment variable is set in terminal running the code, or restart terminal/IDE
- **Configuration file format error**: Check YAML syntax, ensure indentation is correct
- **Cost control**: Be aware of API call costs, recommend setting budget limits in OpenRouter console

---

## Summary

**API key should be stored in:**
1. ✅ Environment variables (`export OPENROUTER_API_KEY="..."`) - **Recommended for production**
2. ✅ `.env` file (in project root) - **Recommended for development**
3. ✅ `~/.bashrc` or `~/.zshrc` (persistent) - **Recommended for personal development**

**API key should NOT be in:**
- ❌ Configuration files (`runtime/config_*.yaml`)
- ❌ Source code files (`*.py`)
- ❌ Git repository (should be ignored)
