"""CrewAI implementation of the same support agent.

Same task, tool, prompt, model and Answer schema as the LangGraph version - only
the orchestration is CrewAI's. CrewAI thinks in terms of an Agent (a role) that
runs a Task inside a Crew, so the shape is heavier than LangGraph's one function.
"""

import time

from crewai import LLM, Agent, Crew, Task
from crewai.tools import tool

from shared.config import MODEL, anthropic_headers
from shared.contract import SYSTEM_PROMPT, Answer
from shared.search_client import search_documents
from shared.runner import RunResult, format_hits

# CrewAI gives no easy per-run tool-call count, so we count calls ourselves.
_tool_calls = {"n": 0}


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search the Meridian Robotics knowledge base and return the most relevant
    documents. Input is a natural-language query. Returns documents with their id."""
    _tool_calls["n"] += 1
    return format_hits(search_documents(query, k=3))


def _build_llm() -> LLM:
    return LLM(
        model=f"anthropic/{MODEL}",
        temperature=0,
        max_tokens=1024,
        # Force the LiteLLM path. CrewAI's native Anthropic path (crewai 1.15)
        # crashes passing `temperature` to Messages.create and doesn't forward our
        # workspace header. LiteLLM handles both.
        is_litellm=True,
        # litellm forwards extra_headers to Anthropic - this carries the workspace id.
        additional_params={"extra_headers": anthropic_headers()},
    )


def _build_agent(llm: LLM) -> Agent:
    return Agent(
        role="Meridian Robotics support assistant",
        goal="Answer employee and customer questions using the company knowledge base.",
        backstory=SYSTEM_PROMPT,
        tools=[search_knowledge_base],
        llm=llm,
        verbose=False,
    )


def run(question: str) -> RunResult:
    _tool_calls["n"] = 0
    agent = _build_agent(_build_llm())
    task = Task(
        description=question,
        expected_output="A short answer grounded in the docs, plus the ids of the documents used.",
        agent=agent,
        output_pydantic=Answer,
    )
    crew = Crew(agents=[agent], tasks=[task], verbose=False)

    start = time.perf_counter()
    try:
        result = crew.kickoff()
    except Exception as e:  # noqa: BLE001 - reliability is a metric
        return RunResult(answer="", error=f"{type(e).__name__}: {e}", latency_s=time.perf_counter() - start)
    latency = time.perf_counter() - start

    answer: Answer = result.pydantic
    usage = result.token_usage  # UsageMetrics
    return RunResult(
        answer=answer.answer if answer else str(result),
        citations=answer.citations if answer else [],
        latency_s=latency,
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
        tool_calls=_tool_calls["n"],
    )


if __name__ == "__main__":
    r = run("How many vacation days do employees get, and how many roll over?")
    print("answer:", r.answer)
    print("citations:", r.citations)
    print(f"latency: {r.latency_s:.2f}s  tokens: {r.input_tokens}+{r.output_tokens}  "
          f"tool calls: {r.tool_calls}  cost: ${r.cost_usd:.5f}")
