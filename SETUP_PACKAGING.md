# Packaging Setup Instructions

按照路线 A 设置的最小 packaging 配置已经完成。

## 已创建的文件

1. ✅ `pyproject.toml` - 包配置文件
2. ✅ `core/__init__.py` - core 包的初始化文件

## 下一步操作

### 1. 安装包（在项目根目录执行）

```bash
cd /Users/caohuixi/symphony2.0
pip install -e .
```

### 2. 验证安装

```bash
python3 -c "import core; from core.linucb_selector import GlobalLinUCB; print('✅ Import successful!')"
```

### 3. 运行实验

安装完成后，可以直接运行实验：

```bash
# Exp1
python3 experiments/exp1_sim_efficiency_cost/sim_efficiency_cost.py --n 1000 --seed 123 --topL 3 --fallback

# Exp3 Sim
bash experiments/exp3/run_exp3_both.sh

# Exp3 Real (需要先生成数据)
cd symphony-data-generator && python src/quick_start.py && cd ..
python3 experiments/exp3/real/exp3_real.py \
  --tasks symphony-data-generator/data/exp3/task_pool.jsonl \
  --benchmark gsm8k --n 1500 --sample-with-replacement \
  --shock A_unavailable --shock-point 750 --seed 42 \
  --outdir experiments/exp3/real/results/gsm8k_replacement
```

## 说明

- ✅ 不需要修改任何实验代码
- ✅ 不需要修改任何 import 语句
- ✅ 实验代码已经使用 try-except 支持两种导入方式（local 和 package）
- ✅ 安装后，实验代码会自动使用 package 模式的导入

