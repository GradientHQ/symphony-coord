#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import argparse
from typing import Any, Dict, List, Tuple


def _canonical_num(num_str: str) -> str:
    try:
        val = float(num_str)
    except Exception:
        return num_str.strip()
    if abs(val - round(val)) < 1e-9:
        return str(int(round(val)))
    text = f"{val:.10f}".rstrip("0").rstrip(".")
    return text


def _extract_last_number(text: str) -> str:
    if not text:
        return ""
    clean = text.replace(",", "").replace("$", "").replace("%", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", clean)
    return _canonical_num(nums[-1]) if nums else ""


def _normalize_answer(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        return str(x).strip().lower()
    s = str(x).strip().lower()
    for ch in ["\n", "\t", ",", ".", ":", ";", "!", "?", "\"", "'", "(", ")", "[", "]"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def strip_code_fences(text: str) -> str:
    """Remove surrounding ``` or ```json fences if present."""
    if not text:
        return ""
    t = text.strip()
    if not t.startswith("```"):
        return t

    lines = t.splitlines()
    # drop first fence line
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # drop last fence line if it's ```
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def try_json_final_answer(text: str) -> str:
    if not text:
        return ""
    t = strip_code_fences(text)

    # Some models wrap JSON inside extra text; try to locate first {...} block.
    # This is conservative: we first try whole string, then fallback to regex.
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            for key in ["final_answer", "answer", "final", "output"]:
                if key in obj and obj[key] is not None:
                    return str(obj[key]).strip()
    except Exception:
        pass

    # Fallback: find first JSON object substring
    m = re.search(r"\{.*\}", t, flags=re.DOTALL)
    if m:
        s = m.group(0)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for key in ["final_answer", "answer", "final", "output"]:
                    if key in obj and obj[key] is not None:
                        return str(obj[key]).strip()
        except Exception:
            pass

    return ""


def extract_pred_fixed(text: str) -> str:
    if not text:
        return ""
    json_ans = try_json_final_answer(text)
    if json_ans:
        return json_ans

    low = text.lower()
    key = "final answer:"
    if key in low:
        idx = low.rfind(key)
        return text[idx + len(key):].strip()
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def extract_gold_text(log_row: Dict[str, Any]) -> str:
    # Your existing logs store gold as "gold"
    return str(log_row.get("gold", "") or "")


def is_correct_gsm8k(pred: str, gold_text: str) -> int:
    pnum = _extract_last_number(pred)
    gnum = _extract_last_number(gold_text)
    if pnum and gnum:
        return 1 if _canonical_num(pnum) == _canonical_num(gnum) else 0
    # fallback string normalize
    return 1 if _normalize_answer(pred) and _normalize_answer(pred) == _normalize_answer(gold_text) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Path to pretrain_*.jsonl")
    ap.add_argument("--outfile", required=True, help="Path to write updated jsonl")
    ap.add_argument("--benchmark", default="gsm8k", help="Only recalc for this benchmark (default gsm8k)")
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    bench = args.benchmark.strip().lower()

    old_correct = 0
    new_correct = 0
    total = 0

    for r in rows:
        b = str(r.get("benchmark", "") or "").strip().lower()
        if bench and b != bench:
            continue

        total += 1
        old_acc = int(r.get("acc", 0) or 0)
        old_correct += old_acc

        pred_text = str(r.get("pred", "") or "")
        gold_text = extract_gold_text(r)

        new_pred = extract_pred_fixed(pred_text)
        new_acc = is_correct_gsm8k(new_pred, gold_text)

        # ✅ Update pred and acc with fixed values (clean format)
        r["pred"] = new_pred
        r["acc"] = new_acc
        # Keep pred_fixed and acc_fixed for reference (if needed for comparison)
        r["pred_fixed"] = new_pred
        r["acc_fixed"] = new_acc

        new_correct += new_acc

    with open(args.outfile, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def pct(x: int, n: int) -> float:
        return (x / n * 100.0) if n > 0 else 0.0

    print(f"[DONE] benchmark={bench} total={total}")
    print(f"  old_acc: {old_correct}/{total} = {pct(old_correct, total):.2f}%")
    print(f"  new_acc: {new_correct}/{total} = {pct(new_correct, total):.2f}%")
    print(f"[WROTE] {args.outfile}")


if __name__ == "__main__":
    main()

