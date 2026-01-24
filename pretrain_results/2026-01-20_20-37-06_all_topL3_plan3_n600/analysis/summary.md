# Routing Match Analysis Summary

This report analyzes whether Symphony's routing matched the most suitable agent for each task type.

## Methodology

- **Oracle Best Agent**: The agent with highest empirical accuracy within each task type (minimum samples required: configurable)
- **Router Accuracy**: Actual accuracy achieved by the routing system
- **Regret**: `oracle_acc - router_acc` (how much better the oracle would have been)
- **Match Rate**: Fraction of tasks where the router selected the oracle-best agent

---

## Overall Statistics by Phase & Benchmark

| Phase | Benchmark | Total Tasks | Router Acc | Oracle Acc | Avg Regret | Match Rate |
|-------|-----------|-------------|------------|------------|------------|------------|
| cold_start | bbh | 40 | 0.425 | 0.636 | 0.211 | 27.5% |
| cold_start | gsm8k | 35 | 0.400 | 0.333 | -0.067 | 34.3% |
| cold_start | humaneval | 42 | 0.190 | 0.273 | 0.082 | 26.2% |
| cold_start | medical_qa | 44 | 0.341 | 0.500 | 0.159 | 22.7% |
| pretrain | amc | 73 | 0.000 | 0.000 | 0.000 | 100.0% |
| pretrain | bbh | 60 | 0.450 | 0.457 | 0.007 | 58.3% |
| pretrain | gsm8k | 56 | 0.446 | 0.550 | 0.104 | 35.7% |
| pretrain | humaneval | 58 | 0.241 | 0.333 | 0.092 | 25.9% |
| pretrain | medical_qa | 53 | 0.585 | 0.765 | 0.180 | 32.1% |
| test | amc | 18 | 0.000 | 0.000 | 0.000 | 100.0% |
| test | bbh | 24 | 0.375 | 0.333 | -0.042 | 62.5% |

## Agent Selection Statistics

| Phase | Benchmark | Agent | Chosen Count | Acc When Chosen |
|-------|-----------|-------|---------------|------------------|
| cold_start | amc | agent-openrouter-gemma-3-4b-it-005 | 9 | 0.000000 |
| cold_start | amc | agent-openrouter-qwen-2.5-7b-instruct-004 | 9 | 0.000000 |
| cold_start | amc | agent-openrouter-deepseek-v3-006 | 8 | 0.000000 |
| cold_start | amc | agent-openrouter-openai-gpt-5-nano-002 | 8 | 0.000000 |
| cold_start | amc | agent-openrouter-openai-gpt-oss-120b-003 | 5 | 0.000000 |
| cold_start | gsm8k | agent-openrouter-openai-gpt-5-nano-002 | 12 | 0.333333 |
| cold_start | gsm8k | agent-openrouter-deepseek-v3-006 | 8 | 0.500000 |
| cold_start | gsm8k | agent-openrouter-gemma-3-4b-it-005 | 7 | 0.142857 |
| cold_start | gsm8k | agent-openrouter-openai-gpt-oss-120b-003 | 5 | 0.800000 |
| cold_start | gsm8k | agent-openrouter-qwen-2.5-7b-instruct-004 | 3 | 0.333333 |
| cold_start | medical_qa | agent-openrouter-qwen-2.5-7b-instruct-004 | 11 | 0.454545 |
| cold_start | medical_qa | agent-openrouter-openai-gpt-oss-120b-003 | 10 | 0.500000 |
| cold_start | medical_qa | agent-openrouter-gemma-3-4b-it-005 | 8 | 0.250000 |
| cold_start | medical_qa | agent-openrouter-openai-gpt-5-nano-002 | 8 | 0.000000 |
| cold_start | medical_qa | agent-openrouter-deepseek-v3-006 | 7 | 0.428571 |
| cold_start | bbh | agent-openrouter-openai-gpt-oss-120b-003 | 11 | 0.636364 |
| cold_start | bbh | agent-openrouter-deepseek-v3-006 | 9 | 0.222222 |
| cold_start | bbh | agent-openrouter-qwen-2.5-7b-instruct-004 | 8 | 0.625000 |
| cold_start | bbh | agent-openrouter-openai-gpt-5-nano-002 | 7 | 0.000000 |
| cold_start | bbh | agent-openrouter-gemma-3-4b-it-005 | 5 | 0.600000 |
| cold_start | humaneval | agent-openrouter-gemma-3-4b-it-005 | 11 | 0.272727 |
| cold_start | humaneval | agent-openrouter-openai-gpt-oss-120b-003 | 9 | 0.111111 |
| cold_start | humaneval | agent-openrouter-qwen-2.5-7b-instruct-004 | 9 | 0.111111 |
| cold_start | humaneval | agent-openrouter-deepseek-v3-006 | 8 | 0.375000 |
| cold_start | humaneval | agent-openrouter-openai-gpt-5-nano-002 | 5 | 0.000000 |
| pretrain | medical_qa | agent-openrouter-qwen-2.5-7b-instruct-004 | 17 | 0.764706 |
| pretrain | medical_qa | agent-openrouter-openai-gpt-5-nano-002 | 15 | 0.333333 |
| pretrain | medical_qa | agent-openrouter-openai-gpt-oss-120b-003 | 13 | 0.692308 |
| pretrain | medical_qa | agent-openrouter-gemma-3-4b-it-005 | 5 | 0.600000 |
| pretrain | medical_qa | agent-openrouter-deepseek-v3-006 | 3 | 0.333333 |
| pretrain | humaneval | agent-openrouter-deepseek-v3-006 | 20 | 0.200000 |
| pretrain | humaneval | agent-openrouter-openai-gpt-5-nano-002 | 15 | 0.333333 |
| pretrain | humaneval | agent-openrouter-qwen-2.5-7b-instruct-004 | 14 | 0.214286 |
| pretrain | humaneval | agent-openrouter-gemma-3-4b-it-005 | 6 | 0.333333 |
| pretrain | humaneval | agent-openrouter-openai-gpt-oss-120b-003 | 3 | 0.000000 |
| pretrain | amc | agent-openrouter-openai-gpt-oss-120b-003 | 73 | 0.000000 |
| pretrain | bbh | agent-openrouter-openai-gpt-oss-120b-003 | 35 | 0.457143 |
| pretrain | bbh | agent-openrouter-deepseek-v3-006 | 9 | 0.444444 |
| pretrain | bbh | agent-openrouter-openai-gpt-5-nano-002 | 8 | 0.500000 |
| pretrain | bbh | agent-openrouter-qwen-2.5-7b-instruct-004 | 5 | 0.400000 |
| pretrain | bbh | agent-openrouter-gemma-3-4b-it-005 | 3 | 0.333333 |
| pretrain | gsm8k | agent-openrouter-openai-gpt-5-nano-002 | 20 | 0.550000 |
| pretrain | gsm8k | agent-openrouter-qwen-2.5-7b-instruct-004 | 16 | 0.500000 |
| pretrain | gsm8k | agent-openrouter-gemma-3-4b-it-005 | 11 | 0.181818 |
| pretrain | gsm8k | agent-openrouter-openai-gpt-oss-120b-003 | 7 | 0.571429 |
| pretrain | gsm8k | agent-openrouter-deepseek-v3-006 | 2 | 0.000000 |
| test | gsm8k | agent-openrouter-gemma-3-4b-it-005 | 8 | 0.125000 |
| test | gsm8k | agent-openrouter-openai-gpt-oss-120b-003 | 8 | 0.750000 |
| test | gsm8k | agent-openrouter-openai-gpt-5-nano-002 | 4 | 0.750000 |
| test | gsm8k | agent-openrouter-deepseek-v3-006 | 3 | 1.000000 |
| test | gsm8k | agent-openrouter-qwen-2.5-7b-instruct-004 | 2 | 0.000000 |
| test | bbh | agent-openrouter-openai-gpt-oss-120b-003 | 15 | 0.333333 |
| test | bbh | agent-openrouter-deepseek-v3-006 | 4 | 0.500000 |
| test | bbh | agent-openrouter-openai-gpt-5-nano-002 | 3 | 0.666667 |
| test | bbh | agent-openrouter-gemma-3-4b-it-005 | 1 | 0.000000 |
| test | bbh | agent-openrouter-qwen-2.5-7b-instruct-004 | 1 | 0.000000 |
| test | medical_qa | agent-openrouter-openai-gpt-5-nano-002 | 7 | 0.714286 |
| test | medical_qa | agent-openrouter-qwen-2.5-7b-instruct-004 | 6 | 0.666667 |
| test | medical_qa | agent-openrouter-deepseek-v3-006 | 1 | 1.000000 |
| test | medical_qa | agent-openrouter-gemma-3-4b-it-005 | 1 | 0.000000 |
| test | medical_qa | agent-openrouter-openai-gpt-oss-120b-003 | 1 | 0.000000 |
| test | amc | agent-openrouter-openai-gpt-oss-120b-003 | 18 | 0.000000 |
| test | humaneval | agent-openrouter-deepseek-v3-006 | 6 | 0.500000 |
| test | humaneval | agent-openrouter-openai-gpt-5-nano-002 | 5 | 0.000000 |
| test | humaneval | agent-openrouter-qwen-2.5-7b-instruct-004 | 3 | 0.000000 |
| test | humaneval | agent-openrouter-gemma-3-4b-it-005 | 2 | 1.000000 |
| test | humaneval | agent-openrouter-openai-gpt-oss-120b-003 | 1 | 0.000000 |

## Top 20 Worst-Regret Task Types

These are task types where the router's choice deviated most from the oracle-best agent.

| Phase | Benchmark | Task Type | Difficulty | N | Router Acc | Oracle Agent | Oracle Acc | Regret | Match Rate |
|-------|-----------|-----------|------------|---|------------|--------------|------------|--------|------------|
| cold_start | bbh | unknown | __all__ | 40 | 0.425000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.636364 | 0.211364 | 0.275000 |
| pretrain | medical_qa | unknown | __all__ | 53 | 0.584906 | agent-openrouter-qwen-2.5-7b-instruct-004 | 0.764706 | 0.179800 | 0.320755 |
| cold_start | medical_qa | unknown | __all__ | 44 | 0.340909 | agent-openrouter-openai-gpt-oss-120b-003 | 0.500000 | 0.159091 | 0.227273 |
| pretrain | gsm8k | unknown | __all__ | 56 | 0.446429 | agent-openrouter-openai-gpt-5-nano-002 | 0.550000 | 0.103571 | 0.357143 |
| pretrain | humaneval | unknown | __all__ | 58 | 0.241379 | agent-openrouter-openai-gpt-5-nano-002 | 0.333333 | 0.091954 | 0.258621 |
| cold_start | humaneval | unknown | __all__ | 42 | 0.190476 | agent-openrouter-gemma-3-4b-it-005 | 0.272727 | 0.082251 | 0.261905 |
| pretrain | bbh | unknown | __all__ | 60 | 0.450000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.457143 | 0.007143 | 0.583333 |
| pretrain | amc | unknown | __all__ | 73 | 0.000000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.000000 | 0.000000 | 1.000000 |
| test | amc | unknown | __all__ | 18 | 0.000000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.000000 | 0.000000 | 1.000000 |
| test | bbh | unknown | __all__ | 24 | 0.375000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.333333 | -0.041667 | 0.625000 |
| cold_start | gsm8k | unknown | __all__ | 35 | 0.400000 | agent-openrouter-openai-gpt-5-nano-002 | 0.333333 | -0.066667 | 0.342857 |

## Best-Matched Task Types (Top 15 by Match Rate)

Task types where the router frequently selected the oracle-best agent.

| Phase | Benchmark | Task Type | Difficulty | N | Router Acc | Oracle Agent | Oracle Acc | Regret | Match Rate |
|-------|-----------|-----------|------------|---|------------|--------------|------------|--------|------------|
| pretrain | amc | unknown | __all__ | 73 | 0.000000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.000000 | 0.000000 | 1.000000 |
| test | amc | unknown | __all__ | 18 | 0.000000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.000000 | 0.000000 | 1.000000 |
| test | bbh | unknown | __all__ | 24 | 0.375000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.333333 | -0.041667 | 0.625000 |
| pretrain | bbh | unknown | __all__ | 60 | 0.450000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.457143 | 0.007143 | 0.583333 |
| pretrain | gsm8k | unknown | __all__ | 56 | 0.446429 | agent-openrouter-openai-gpt-5-nano-002 | 0.550000 | 0.103571 | 0.357143 |
| cold_start | gsm8k | unknown | __all__ | 35 | 0.400000 | agent-openrouter-openai-gpt-5-nano-002 | 0.333333 | -0.066667 | 0.342857 |
| pretrain | medical_qa | unknown | __all__ | 53 | 0.584906 | agent-openrouter-qwen-2.5-7b-instruct-004 | 0.764706 | 0.179800 | 0.320755 |
| cold_start | bbh | unknown | __all__ | 40 | 0.425000 | agent-openrouter-openai-gpt-oss-120b-003 | 0.636364 | 0.211364 | 0.275000 |
| cold_start | humaneval | unknown | __all__ | 42 | 0.190476 | agent-openrouter-gemma-3-4b-it-005 | 0.272727 | 0.082251 | 0.261905 |
| pretrain | humaneval | unknown | __all__ | 58 | 0.241379 | agent-openrouter-openai-gpt-5-nano-002 | 0.333333 | 0.091954 | 0.258621 |
| cold_start | medical_qa | unknown | __all__ | 44 | 0.340909 | agent-openrouter-openai-gpt-oss-120b-003 | 0.500000 | 0.159091 | 0.227273 |

## Oracle Best Agents by Task Type

| Phase | Benchmark | Task Type | Difficulty | Oracle Best Agent | Oracle Acc | N |
|-------|-----------|-----------|------------|-------------------|------------|---|
| cold_start | bbh | unknown | __all__ | agent-openrouter-openai-gpt-oss-120b-003 | 0.636364 | 11 |
| cold_start | gsm8k | unknown | __all__ | agent-openrouter-openai-gpt-5-nano-002 | 0.333333 | 12 |
| cold_start | humaneval | unknown | __all__ | agent-openrouter-gemma-3-4b-it-005 | 0.272727 | 11 |
| cold_start | medical_qa | unknown | __all__ | agent-openrouter-openai-gpt-oss-120b-003 | 0.500000 | 10 |
| pretrain | amc | unknown | __all__ | agent-openrouter-openai-gpt-oss-120b-003 | 0.000000 | 73 |
| pretrain | bbh | unknown | __all__ | agent-openrouter-openai-gpt-oss-120b-003 | 0.457143 | 35 |
| pretrain | gsm8k | unknown | __all__ | agent-openrouter-openai-gpt-5-nano-002 | 0.550000 | 20 |
| pretrain | humaneval | unknown | __all__ | agent-openrouter-openai-gpt-5-nano-002 | 0.333333 | 15 |
| pretrain | medical_qa | unknown | __all__ | agent-openrouter-qwen-2.5-7b-instruct-004 | 0.764706 | 17 |
| test | amc | unknown | __all__ | agent-openrouter-openai-gpt-oss-120b-003 | 0.000000 | 18 |
| test | bbh | unknown | __all__ | agent-openrouter-openai-gpt-oss-120b-003 | 0.333333 | 15 |

---

*Generated by analyze_routing.py*
