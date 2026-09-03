# LangGraph - field notes

First framework, so this is the baseline the others get compared against.

## Setup
- Install: `langgraph` + `langchain-anthropic`. No extra services.
- Time to a working agent: ~15 minutes. The prebuilt `create_react_agent` does
  the whole agent loop, so there was almost nothing to write.

## What the code looks like
- The agent is 4 arguments to one function: model, tools, prompt, response_format.
- Tools are plain Python functions with an `@tool` decorator.
- `response_format=Answer` (a Pydantic model) adds a final step that coerces the
  reply into structured output with citations. Nice - no manual JSON parsing.

## Gotchas hit
- `create_react_agent` moved packages in v1.0 - it now lives in `langchain.agents`
  as `create_agent`, and the old import warns as deprecated. Still works for now.
- Token usage lives on each `AIMessage.usage_metadata`; had to sum it across all
  messages in the final state to get a total.
- Identity-linked Anthropic API key needs an `anthropic-workspace-id` header. Had
  to pass it via `default_headers`. Not a LangGraph thing, but worth noting.

## Scorecard (10 questions, Haiku 4.5)
- task success: 9/10 (90%)
- citation accuracy: 9/10
- reliability: 10/10, no crashes
- avg latency: 3.5s
- avg tool calls: 1.1
- total cost: $0.024

## Honest take
Easiest possible start. The prebuilt agent hides the graph entirely - you don't
see any of the "graph" that gives LangGraph its name until you need custom
control flow. For a simple tool-using agent it's almost too easy; the power
(explicit nodes and edges) only shows up on harder tasks.

## One real miss to dig into later
q-office-days failed: the agent didn't cite the remote-work doc and missed the
"Tuesday/Wednesday" facts. Same retrieval feeds every framework, so this is a
shared-tool issue, not a LangGraph one - but worth a look before drawing
conclusions.
