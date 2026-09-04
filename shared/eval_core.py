"""Scoring logic, light enough to run inside any framework's venv.

Imports nothing heavy - just the eval set and the shared RunResult. A framework's
run_all.py calls run_over_cases() with its own agent.run and prints the rows as
JSON; the orchestrator (shared/evaluate.py) reads that back and aggregates.
"""

import json
from dataclasses import asdict

from .config import EVAL_PATH
from .runner import RunResult


def score_answer(result: RunResult, case: dict) -> dict:
    """Grade one answer against its gold data."""
    answer_lower = result.answer.lower()
    hits = [s for s in case["must_include"] if s.lower() in answer_lower]
    answer_ok = len(hits) == len(case["must_include"])
    cited_gold = bool(set(result.citations) & set(case["gold_doc_ids"]))
    return {
        "id": case["id"],
        "answer_ok": answer_ok,
        "missing_facts": [s for s in case["must_include"] if s.lower() not in answer_lower],
        "citation_ok": cited_gold,
        "crashed": result.error is not None,
    }


def run_over_cases(run_fn) -> list[dict]:
    """Run one framework's agent over every eval case and return scored rows."""
    cases = json.loads(EVAL_PATH.read_text())
    rows = []
    for case in cases:
        result: RunResult = run_fn(case["question"])
        rows.append(
            {
                "grade": score_answer(result, case),
                "result": asdict(result),
                "cost": result.cost_usd,
            }
        )
    return rows
