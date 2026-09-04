"""Eval orchestrator. Runs each framework in its OWN isolated venv.

Every framework has conflicting dependencies, so they can't share one venv. This
runs frameworks/<name>/venv/bin/python -m shared.run_all <name> as a subprocess,
reads back the scored rows, aggregates them, prints a scorecard, and saves
results/<name>.json. Retrieval and reranker run as services, so the framework
venv only needs the framework itself.

Usage:
    ./venv/bin/python -m shared.evaluate langgraph
"""

import json
import os
import re
import statistics
import subprocess
import sys

from .config import ROOT


def _venv_python(name: str) -> str:
    py = ROOT / "frameworks" / name / "venv" / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"no venv for {name} - create frameworks/{name}/venv first")
    return str(py)


def run_framework(name: str) -> list[dict]:
    """Subprocess into the framework's venv and get its scored rows back."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CREWAI_TRACING_ENABLED": "false"}
    proc = subprocess.run(
        [_venv_python(name), "-m", "shared.run_all", name],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    match = re.search(r"<<<ROWS>>>(.*)<<<END>>>", proc.stdout, re.DOTALL)
    if not match:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-2000:] + "\n")
        raise SystemExit(f"{name}: no rows returned (see output above)")
    return json.loads(match.group(1))


def summarize(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    passed = sum(r["grade"]["answer_ok"] for r in rows)
    cited = sum(r["grade"]["citation_ok"] for r in rows)
    crashed = sum(r["grade"]["crashed"] for r in rows)
    latencies = [r["result"]["latency_s"] for r in rows]
    return {
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


def print_scorecard(rows: list[dict], s: dict) -> None:
    print(f"\nran {s['framework']} over {s['n']} questions (isolated venv)\n")
    for r in rows:
        g = r["grade"]
        mark = "ok " if g["answer_ok"] else "MISS"
        if g["crashed"]:
            mark = "ERR "
        cite = "cite:ok" if g["citation_ok"] else "cite:--"
        note = "" if g["answer_ok"] or g["crashed"] else f"  missing {g['missing_facts']}"
        print(f"  {mark} {g['id']:18s} {r['result']['latency_s']:5.1f}s  {cite}{note}")
    n, passed = s["n"], round(s["task_success"] * s["n"])
    print("\n  SCORECARD")
    print(f"    task success      {passed}/{n}   {s['task_success']*100:.0f}%")
    print(f"    citation accuracy {round(s['citation_accuracy']*n)}/{n}   {s['citation_accuracy']*100:.0f}%")
    print(f"    reliability       {round(s['reliability']*n)}/{n}")
    print(f"    avg latency       {s['avg_latency_s']:.2f}s  (p95 {s['p95_latency_s']:.2f}s)")
    print(f"    avg tool calls    {s['avg_tool_calls']:.1f}")
    print(f"    total cost        ${s['total_cost_usd']:.4f}  ({n} questions)")


def evaluate(name: str) -> dict:
    rows = run_framework(name)
    summary = summarize(name, rows)
    print_scorecard(rows, summary)
    out = ROOT / "results" / f"{name}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\n  saved {out.relative_to(ROOT)}\n")
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m shared.evaluate <framework>")
        sys.exit(1)
    evaluate(sys.argv[1])
