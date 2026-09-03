"""Read every results/<framework>.json and print one comparison table.

Usage:
    ./venv/bin/python -m shared.compare          # print the table
    ./venv/bin/python -m shared.compare --readme # also write it into README.md
"""

import json
import sys

from .config import ROOT

RESULTS_DIR = ROOT / "results"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def load_summaries() -> list[dict]:
    summaries = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        summaries.append(json.loads(path.read_text())["summary"])
    return summaries


def build_table(summaries: list[dict]) -> str:
    header = (
        "| Framework | Task success | Citations | Reliability | Avg latency | Avg tool calls | Cost / 10 Q |\n"
        "|---|---|---|---|---|---|---|"
    )
    lines = [header]
    # Best task success, then cheapest, ranks first.
    for s in sorted(summaries, key=lambda x: (-x["task_success"], x["total_cost_usd"])):
        lines.append(
            f"| {s['framework']} "
            f"| {s['task_success']*100:.0f}% "
            f"| {s['citation_accuracy']*100:.0f}% "
            f"| {s['reliability']*100:.0f}% "
            f"| {s['avg_latency_s']:.2f}s "
            f"| {s['avg_tool_calls']:.1f} "
            f"| ${s['total_cost_usd']:.4f} |"
        )
    return "\n".join(lines)


def write_readme(table: str) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START in text and END in text:
        pre = text.split(START)[0]
        post = text.split(END)[1]
        text = f"{pre}{START}\n{table}\n{END}{post}"
    else:  # append a Results section if the markers aren't there yet
        text = text.rstrip() + f"\n\n## Results\n\n{START}\n{table}\n{END}\n"
    readme.write_text(text)
    print("updated README.md")


if __name__ == "__main__":
    summaries = load_summaries()
    if not summaries:
        print("no results yet - run shared.evaluate first")
        sys.exit(0)
    table = build_table(summaries)
    print("\n" + table + "\n")
    if "--readme" in sys.argv:
        write_readme(table)
