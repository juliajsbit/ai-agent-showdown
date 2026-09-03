"""The single eval harness. Every framework is scored here, the same way.

Usage:
    ./venv/bin/python -m shared.evaluate langgraph

It runs the named framework's agent over every question in the eval set, scores
each answer, prints a scorecard, and saves results/<framework>.json so the final
comparison table can read them all back.
"""

import importlib
import json
import statistics
import sys
import time
from dataclasses import asdict

from .config import EVAL_PATH, ROOT
from .runner import RunResult


def load_framework(name: str):
    """Import frameworks/<name>/agent.py and return its run(question) function."""
    module = importlib.import_module(f"frameworks.{name}.agent")
    return module.run


def score_answer(result: RunResult, case: dict) -> dict:
    """Grade one answer against its gold data."""
    answer_lower = result.answer.lower()
    # Answer is correct if it contains every required fact.
    hits = [s for s in case["must_include"] if s.lower() in answer_lower]
    answer_ok = len(hits) == len(case["must_include"])
    # Citations are correct if the agent cited at least one of the gold docs.
    cited_gold = bool(set(result.citations) & set(case["gold_doc_ids"]))
    return {
        "id": case["id"],
        "answer_ok": answer_ok,
        "missing_facts": [s for s in case["must_include"] if s.lower() not in answer_lower],
        "citation_ok": cited_gold,
        "crashed": result.error is not None,
    }


def evaluate(name: str) -> dict:
    run = load_framework(name)
    cases = json.loads(EVAL_PATH.read_text())

    rows = []
    print(f"\nrunning {name} over {len(cases)} questions...\n")
    for case in cases:
        result = run(case["question"])
        grade = score_answer(result, case)
        rows.append({"grade": grade, "result": asdict(result), "cost": result.cost_usd})

        mark = "ok " if grade["answer_ok"] else "MISS"
        cite = "cite:ok" if grade["citation_ok"] else "cite:-- "
        note = "" if grade["answer_ok"] else f"  missing {grade['missing_facts']}"
        if grade["crashed"]:
            mark, note = "ERR ", f"  {result.error[:60]}"
        print(f"  {mark} {case['id']:18s} {result.latency_s:5.1f}s  {cite}{note}")

    n = len(rows)
    passed = sum(r["grade"]["answer_ok"] for r in rows)
    cited = sum(r["grade"]["citation_ok"] for r in rows)
    crashed = sum(r["grade"]["crashed"] for r in rows)
    latencies = [r["result"]["latency_s"] for r in rows]
    summary = {
        "framework": name,
        "task_success": passed / n,
        "citation_accuracy": cited / n,
        "reliability": (n - crashed) / n,
        "avg_latency_s": statistics.mean(latencies),
        "p95_latency_s": max(latencies),
        "avg_tool_calls": statistics.mean(r["result"]["tool_calls"] for r in rows),
        "total_cost_usd": sum(r["cost"] for r in rows),
        "n": n,
    }

    print("\n  SCORECARD")
    print(f"    task success      {passed}/{n}   {summary['task_success']*100:.0f}%")
    print(f"    citation accuracy {cited}/{n}   {summary['citation_accuracy']*100:.0f}%")
    print(f"    reliability       {n-crashed}/{n}   (no crashes)")
    print(f"    avg latency       {summary['avg_latency_s']:.2f}s  (p95 {summary['p95_latency_s']:.2f}s)")
    print(f"    avg tool calls    {summary['avg_tool_calls']:.1f}")
    print(f"    total cost        ${summary['total_cost_usd']:.4f}  ({n} questions)")

    out = ROOT / "results" / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\n  saved {out.relative_to(ROOT)}\n")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m shared.evaluate <framework>")
        sys.exit(1)
    evaluate(sys.argv[1])
