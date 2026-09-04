"""LangGraph implementation of the support agent - the baseline.

Uses LangGraph's prebuilt ReAct agent: bind one tool (document search), let the
model decide when to call it, then force a structured Answer at the end. Every
other framework will reproduce this same behaviour; only this file changes.
"""

import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared.config import MODEL, anthropic_headers
from shared.contract import SYSTEM_PROMPT, Answer
from shared.search_client import search_documents
from shared.runner import RunResult, format_hits

load_dotenv()


@tool
def search_knowledge_base(query: str) -> str:
    """Search the Meridian Robotics knowledge base and return the most relevant
    documents. Input is a natural-language query. Returns documents with their id."""
    return format_hits(search_documents(query, k=3))


def _build():
    llm = ChatAnthropic(model=MODEL, max_tokens=1024, temperature=0, default_headers=anthropic_headers())
    return create_react_agent(
        llm,
        tools=[search_knowledge_base],
        prompt=SYSTEM_PROMPT,
        response_format=Answer,  # last step coerces the reply into our schema
    )


_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = _build()
    return _agent


def run(question: str) -> RunResult:
    """Answer one question and report how the run went."""
    agent = _get_agent()
    start = time.perf_counter()
    try:
        state = agent.invoke({"messages": [("user", question)]})
    except Exception as e:  # noqa: BLE001 - reliability is a metric; record, don't raise
        return RunResult(answer="", error=f"{type(e).__name__}: {e}", latency_s=time.perf_counter() - start)
    latency = time.perf_counter() - start

    # Add up token usage and tool calls across every model turn.
    in_tok = out_tok = tool_calls = 0
    for m in state["messages"]:
        if isinstance(m, AIMessage):
            usage = m.usage_metadata or {}
            in_tok += usage.get("input_tokens", 0)
            out_tok += usage.get("output_tokens", 0)
            tool_calls += len(m.tool_calls or [])

    answer: Answer = state["structured_response"]
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
