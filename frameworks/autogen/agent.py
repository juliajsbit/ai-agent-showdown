"""AutoGen implementation of the same support agent.

AutoGen (v0.4, the autogen-agentchat rewrite) is async and message-driven: you
make a model client, wrap it in an AssistantAgent with tools, and call run().
Same task, tool, prompt, model and Answer schema as the other frameworks.
"""

import asyncio
import json
import os
import re
import time

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from shared.anthropic_compat import install as _install_compat
from shared.config import MODEL, anthropic_headers
from shared.contract import SYSTEM_PROMPT, Answer
from shared.search_client import search_documents
from shared.runner import RunResult, format_hits

# AutoGen hardcodes temperature=1.0; the installed Anthropic SDK rejects it.
_install_compat()

# AutoGen's Anthropic client has no native structured output, so we ask for JSON
# in the prompt and parse it. This suffix is AutoGen-only - the shared prompt
# stays clean for the frameworks that support structured output natively.
_JSON_SUFFIX = """

When you have the answer, reply with ONLY a JSON object, nothing else:
{"answer": "<your answer>", "citations": ["<doc id>", ...]}"""


def _parse_answer(text: str) -> Answer:
    """Pull the JSON object out of the model's final message."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return Answer(answer=text, citations=[])
    try:
        data = json.loads(match.group(0))
        return Answer(answer=data.get("answer", ""), citations=data.get("citations", []))
    except (json.JSONDecodeError, TypeError):
        return Answer(answer=text, citations=[])


def search_knowledge_base(query: str) -> str:
    """Search the Meridian Robotics knowledge base and return the most relevant
    documents. Input is a natural-language query. Returns documents with their id."""
    return format_hits(search_documents(query, k=3))


def _build_agent() -> AssistantAgent:
    model_client = AnthropicChatCompletionClient(
        model=MODEL,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        default_headers=anthropic_headers(),  # carries the workspace id
        # AutoGen doesn't know this model id, so it assumes no tool support.
        # Tell it the real capabilities.
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "claude-3-5-haiku",
        },
    )
    return AssistantAgent(
        name="support_assistant",
        model_client=model_client,
        tools=[search_knowledge_base],
        system_message=SYSTEM_PROMPT + _JSON_SUFFIX,
        reflect_on_tool_use=True,  # produce a final text answer after the tool call
    )


async def _arun(question: str) -> RunResult:
    agent = _build_agent()
    start = time.perf_counter()
    try:
        result = await agent.run(task=question)
    except Exception as e:  # noqa: BLE001 - reliability is a metric
        return RunResult(answer="", error=f"{type(e).__name__}: {e}", latency_s=time.perf_counter() - start)
    latency = time.perf_counter() - start

    in_tok = out_tok = tool_calls = 0
    final_text = ""
    for msg in result.messages:
        usage = getattr(msg, "models_usage", None)
        if usage:
            in_tok += usage.prompt_tokens
            out_tok += usage.completion_tokens
        if isinstance(msg, ToolCallExecutionEvent):
            tool_calls += len(msg.content)
        if isinstance(msg, TextMessage) and msg.source == "support_assistant":
            final_text = msg.content  # last one wins = the final answer

    answer_obj = _parse_answer(final_text)
    return RunResult(
        answer=answer_obj.answer,
        citations=answer_obj.citations,
        latency_s=latency,
        input_tokens=in_tok,
        output_tokens=out_tok,
        tool_calls=tool_calls,
    )


def run(question: str) -> RunResult:
    return asyncio.run(_arun(question))


if __name__ == "__main__":
    r = run("How many vacation days do employees get, and how many roll over?")
    print("answer:", r.answer)
    print("citations:", r.citations)
    print(f"latency: {r.latency_s:.2f}s  tokens: {r.input_tokens}+{r.output_tokens}  "
          f"tool calls: {r.tool_calls}  cost: ${r.cost_usd:.5f}")
