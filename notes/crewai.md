# CrewAI - field notes

Second framework. Same task, tool, prompt, model and Answer schema as LangGraph.

## Setup
- Install: `crewai`. Heavy - pulls a large dependency tree.
- Mental model is different: you don't build an agent, you build an **Agent** (a
  role with a goal and backstory) that runs a **Task** inside a **Crew**. More
  ceremony than LangGraph's single function.

## Gotchas hit (this took real time)
- CrewAI 1.15 has a "native Anthropic" path that crashed: it passes `temperature`
  to `Messages.create()` in a way the Anthropic SDK rejects
  (`unexpected keyword argument 'temperature'`), and it didn't forward our custom
  `anthropic-workspace-id` header. Both broke the run silently - the agent just
  returned an empty answer.
- Fix: force the LiteLLM path with `is_litellm=True` and pass the workspace header
  through `additional_params={"extra_headers": ...}`.
- But in 1.15 LiteLLM is no longer installed by default - had to `pip install
  litellm` on top of crewai. So the working setup is crewai + litellm.
- Structured output via `Task(output_pydantic=Answer)` worked cleanly once the
  LLM path was fixed. Token usage is on `result.token_usage`.
- No built-in per-run tool-call count - had to wrap the tool in a counter myself.

## Scorecard (10 questions, Haiku 4.5)
- task success: 10/10
- citation accuracy: 10/10
- reliability: 10/10
- avg latency: 3.40s
- avg tool calls: 1.1
- total cost: $0.034 (vs LangGraph $0.026)

## Honest take
Same answers as LangGraph, but ~30% more expensive per run. CrewAI wraps the
agent in a lot of role/goal/backstory scaffolding, which inflates the input
tokens. For a single-agent tool task that scaffolding buys you nothing here - it
would start to pay off with multiple collaborating agents, which is what CrewAI
is actually built for. The Anthropic-path breakage cost the most time; on a
first read the docs point you at the native path that doesn't work.
