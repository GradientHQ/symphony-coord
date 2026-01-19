# Run BBH, Medical, GSM8K in one terminal

This script runs GSM8K and Medical sequentially across:
- `deepseek-v3`
- `deepseek-v3-0324`
- `openai-gpt-5-nano`
- `openai-gpt-4-1-nano`
- `openai-gpt-oss-120b`

Each dataset appends to its own logfile.
This setup runs cold start, pretrain, and a separate test stage.
There is no validation stage. Split = cold_start:pretrain:test = 1:7:2.

```bash
set -euo pipefail

ROOT="/Users/caohuixi/symphony2.0"
RUNTIME_DIR="runtime/configs/openrouter"
RUN_ID="$(date '+%F_%H-%M-%S')"
LOG_DIR="${ROOT}/logs/${RUN_ID}"

# Split: cold_start:pretrain:test = 1:7:2
# Optional: cap tasks per benchmark (leave empty for full set)
TOTAL_N_CAP=""

get_counts() {
  local meta_path="$1"
  python3 - <<'PY' "${meta_path}"
import json
import sys

meta_path = sys.argv[1]
with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)
total_full = int(meta.get("n_tasks_full") or 0)
import os

cap = os.environ.get("TOTAL_N_CAP", "").strip()
if cap:
    total = min(total_full, int(cap))
else:
    total = total_full
cold = total * 1 // 10
pretrain = total * 7 // 10
test = total - cold - pretrain
print(total, cold, pretrain, test)
PY
}

mkdir -p "${LOG_DIR}"

MODELS=(
  "deepseek-v3"
  "deepseek-v3-0324"
  "openai-gpt-5-nano"
  "openai-gpt-4-1-nano"
  "openai-gpt-oss-120b"
)

run_dataset() {
  local name="$1"
  local pool="$2"
  local benchmark="$3"
  local meta="$4"
  local log_file="$5"

  read -r TOTAL_N COLD_N PRETRAIN_N TEST_N < <(get_counts "${meta}")
  echo "[$(date '+%F %T')] ${name} split: total=${TOTAL_N}, cold=${COLD_N}, pretrain=${PRETRAIN_N}, test=${TEST_N}" | tee -a "${log_file}"

  for model in "${MODELS[@]}"; do
    echo "[$(date '+%F %T')] ${name} | ${model} start" | tee -a "${log_file}"
    python3 "${ROOT}/Pre-train.py" \
      --task-pool "${pool}" \
      --benchmark "${benchmark}" \
      --agents "${model}" \
      --runtime-dir "${RUNTIME_DIR}" \
      --n "${TOTAL_N}" \
      --plan-k 1 \
      --topL 1 \
      --cot-count 1 \
      --cold-n "${COLD_N}" \
      --pretrain-n "${PRETRAIN_N}" \
      --test-n "${TEST_N}" \
      --seed 42 \
      --plot-acc \
      --save-selector ucb_state.json \
      --print-each-step \
      --outdir "pretrain_results/${RUN_ID}/${benchmark}/${model}" \
      2>&1 | tee -a "${log_file}"
  done
}

run_dataset "GSM8K" \
  "${ROOT}/symphony-data-generator/data/benchmarks/full/gsm8k_full.jsonl" \
  "gsm8k" \
  "${ROOT}/symphony-data-generator/data/benchmarks/full/gsm8k_full_meta.json" \
  "${LOG_DIR}/gsm8k.log"

run_dataset "Medical" \
  "${ROOT}/symphony-data-generator/data/benchmarks/full/medical_qa_full.jsonl" \
  "medical_qa" \
  "${ROOT}/symphony-data-generator/data/benchmarks/full/medical_qa_full_meta.json" \
  "${LOG_DIR}/medical.log"

# BBH is intentionally skipped for now.
```
