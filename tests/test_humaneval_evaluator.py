"""HumanEval execution checks that do not need a model or API connection."""

import ast
from pathlib import Path
import signal
from typing import Any, Dict, Optional

import pytest


@pytest.fixture(scope="module")
def evaluate_humaneval():
    # Load the actual evaluator definitions without importing the model stack.
    path = Path(__file__).resolve().parents[1] / "experiments" / "pretrain.py"
    names = {"_clean_code_block", "_exec_with_timeout", "_format_humaneval_code", "_eval_humaneval"}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in definitions} == names
    namespace = {"signal": signal, "Any": Any, "Dict": Dict, "Optional": Optional}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_eval_humaneval"]


def score(evaluate, candidate):
    return evaluate(
        candidate,
        {
            "raw_data": {
                "prompt": "def solve(n):\n    pass\n",
                "test": "def check(candidate):\n    assert candidate(5) == 120\n",
                "entry_point": "solve",
            }
        },
    )


@pytest.mark.parametrize(
    "candidate",
    [
        "import math\ndef solve(n):\n    return math.factorial(n)",
        "def helper(n):\n    return 120\ndef solve(n):\n    return helper(n)",
        "def solve(n):\n    return 1 if n == 0 else n * solve(n - 1)",
        "VALUE = 120\ndef solve(n):\n    return VALUE",
        "VALUE = 24\ndef solve(n):\n    return sum(VALUE for _ in range(n))",
        "class Answer:\n    value = 120\ndef solve(n):\n    return Answer.value",
    ],
    ids=["import", "helper", "recursion", "constant", "comprehension", "class"],
)
def test_module_names_visible_to_candidate(evaluate_humaneval, candidate):
    assert score(evaluate_humaneval, candidate) == 1


def test_simple_candidate(evaluate_humaneval):
    assert score(evaluate_humaneval, "def solve(n):\n    return 120") == 1


def test_wrong_answer_still_rejected(evaluate_humaneval):
    assert score(evaluate_humaneval, "def solve(n):\n    return 121") == 0


def test_syntax_error_still_rejected(evaluate_humaneval):
    assert score(evaluate_humaneval, "def solve(n)\n    return 120") == 0


def test_candidates_have_separate_namespaces(evaluate_humaneval):
    assert score(evaluate_humaneval, "VALUE = 120\ndef solve(n):\n    return 120") == 1
    assert score(evaluate_humaneval, "def solve(n):\n    return VALUE") == 0
