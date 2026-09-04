"""Run one framework's agent over the whole eval set and print rows as JSON.

Invoked by the orchestrator inside the framework's OWN venv:
    frameworks/<name>/venv/bin/python -m shared.run_all <name>

It imports only that one framework, so isolated venvs never clash.
"""

import importlib
import json
import sys


def main() -> None:
    name = sys.argv[1]
    agent = importlib.import_module(f"frameworks.{name}.agent")
    from shared.eval_core import run_over_cases

    rows = run_over_cases(agent.run)
    # Marker-wrapped so the orchestrator can find the JSON even if the framework
    # prints banners/logs to stdout.
    print("<<<ROWS>>>" + json.dumps(rows) + "<<<END>>>")


if __name__ == "__main__":
    main()
