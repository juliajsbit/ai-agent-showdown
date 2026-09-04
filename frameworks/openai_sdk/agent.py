"""OpenAI Agents SDK implementation of the same support agent.

The SDK is built for OpenAI models; Claude is reached through its LiteLLM model
adapter. Same task, tool, prompt, model and Answer schema as the other
frameworks. Retrieval comes from the shared HTTP service, so this framework's
venv stays isolated.
"""

import json
import os
import re
import time

from agents import Agent, ModelSettings, Runner, function_tool
from agents.extensions.models.litellm_model import LitellmModel

from shared.config import MODEL, anthropic_headers
from shared.contract import SYSTEM_PROMPT, Answer
from shared.runner import RunResult, format_hits
from shared.search_client import search_documents

# The SDK's structured output over LiteLLM+Anthropic raises "Invalid JSON when
# parsing model output", so we ask for JSON in the prompt and parse it ourselves.
_JSON_SUFFIX = """

When you have the answer, reply with ONLY a JSON object, nothing else:
{"answer": "<your answer>", "citations": ["<doc id>", ...]}"""


def _parse_answer(text: str) -> Answer:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Answer(answer=text, citations=[])
    try:
        data = json.loads(match.group(0))
        return Answer(answer=data.get("answer", ""), citations=data.get("citations", []))
    except (json.JSONDecodeError, TypeError):
        return Answer(answer=text, citations=[])


@function_tool
def search_knowledge_base(query: str) -> str:
    """Search the Meridian Robotics knowledge base and return the most relevant
    documents. Input is a natural-language query. Returns documents with their id."""
    return format_hits(search_documents(query, k=3))


def _build_agent() -> Agent:
    model = LitellmModel(model=f"anthropic/{MODEL}", api_key=os.getenv("ANTHROPIC_API_KEY"))
    return Agent(
        name="support_assistant",
        instructions=SYSTEM_PROMPT + _JSON_SUFFIX,
        tools=[search_knowledge_base],
        model=model,
        model_settings=ModelSettings(extra_headers=anthropic_headers()),  # workspace id
    )


def run(question: str) -> RunResult:
    agent = _build_agent()
    start = time.perf_counter()
    try:
        result = Runner.run_sync(agent, question)
    except Exception as e:  # noqa: BLE001 - reliability is a metric
        return RunResult(answer="", error=f"{type(e).__name__}: {e}", latency_s=time.perf_counter() - start)
    latency = time.perf_counter() - start

    # Sum token usage across model calls; count tool calls from the run items.
    in_tok = out_tok = 0
    for resp in result.raw_responses:
        usage = getattr(resp, "usage", None)
        if usage:
            in_tok += getattr(usage, "input_tokens", 0)
            out_tok += getattr(usage, "output_tokens", 0)
    tool_calls = sum(1 for item in result.new_items if item.__class__.__name__ == "ToolCallItem")

    answer = _parse_answer(str(result.final_output))
    return RunResult(
        answer=answer.answer,
        citations=answer.citations,
        latency_s=latency,
        input_tokens=in_tok,
        output_tokens=out_tok,
        tool_calls=tool_calls,
    )


if __name__ == "__main__":
    r = run("How many vacation days do employees get, and how many roll over?")
    print("answer:", r.answer)
    print("citations:", r.citations)
    print(f"latency: {r.latency_s:.2f}s  tokens: {r.input_tokens}+{r.output_tokens}  "
          f"tool calls: {r.tool_calls}  cost: ${r.cost_usd:.5f}")
