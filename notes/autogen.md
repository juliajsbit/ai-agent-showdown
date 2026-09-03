# AutoGen - field notes

Third framework (Microsoft, v0.4 / autogen-agentchat rewrite). Same task, tool,
prompt, model and Answer schema as the others.

## Setup
- Install: `autogen-agentchat` + `autogen-ext[anthropic]`.
- Mental model: async and message-driven. Make a model client, wrap it in an
  `AssistantAgent` with tools, `await agent.run(task=...)`. The whole run is a
  list of typed messages you walk afterward.

## Gotchas hit (three real ones)
1. **Unknown model = no tools.** AutoGen keeps a registry of known models and
   defaults unknown ones to "no function calling". Our `claude-haiku-4-5` wasn't
   in it, so tools were silently refused (`model does not support function
   calling`). Fix: pass `model_info={..., "function_calling": True, ...}`.
2. **No native structured output for Anthropic.** `output_content_type=Answer`
   raises `Structured output is currently not supported for Anthropic models`.
   LangGraph and CrewAI both did this out of the box. Workaround: ask for JSON in
   the prompt and parse it myself (AutoGen-only prompt suffix + a regex/json
   parser). It worked - citations still landed 10/10 - but it's manual.
3. **Hardcoded temperature.** AutoGen injects `temperature=1.0` into every
   request and gives no way to omit it. The installed Anthropic SDK (1.3.0)
   removed sampling params, so every call crashed with `unexpected keyword
   argument 'temperature'`. Had to shim the SDK to drop it. Same root cause broke
   CrewAI's native Anthropic path.

## Scorecard (10 questions, Haiku 4.5)
- task success: 10/10
- citation accuracy: 10/10 (via manual JSON parsing)
- reliability: 10/10
- avg latency: 2.84s (fastest so far)
- avg tool calls: 1.2
- total cost: $0.027

## Honest take
Fastest of the three and cheap, but it took the most work to get running - three
separate breakages before the first successful answer. Two of them (unknown-model
tool refusal, no Anthropic structured output) are real rough edges for anyone
using AutoGen with Claude rather than OpenAI; the temperature one is an SDK-version
mismatch. Once past the setup, the async message model is clean and easy to read.
