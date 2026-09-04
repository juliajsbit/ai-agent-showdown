# OpenAI Agents SDK - field notes

Fourth framework. Built for OpenAI models; Claude is reached through the SDK's
LiteLLM model adapter. Same task, tool, prompt, model and Answer schema.

## Setup
- Install: `openai-agents[litellm]`.
- Shape is clean: `@function_tool` on a plain function, an `Agent` with
  instructions + tools + model, `Runner.run_sync(agent, question)`.
- Claude goes through `LitellmModel(model="anthropic/claude-haiku-4-5")`.

## Gotchas hit
- **Installing it forced the whole dependency split.** openai-agents pulled
  `openai` 3.x, which broke CrewAI (needs `openai` <3) and bumped protobuf until
  CrewAI wouldn't even import. That is what pushed the project to one venv per
  framework plus retrieval as a service. Real dependency hell, not hypothetical.
- **No reliable structured output for Claude.** `output_type=Answer` raised
  `ModelBehaviorError: Invalid JSON when parsing model output` over LiteLLM +
  Anthropic. Same workaround as AutoGen: ask for JSON in the prompt and parse it.
  That is two of four frameworks where native structured output doesn't hold for
  Claude.
- Workspace header goes through `ModelSettings(extra_headers=...)`. Clean.
- A harmless `OPENAI_API_KEY is not set, skipping trace export` line prints on
  every run - tracing wants an OpenAI key even when the model is Claude.

## Scorecard (10 questions, Haiku 4.5)
- task success: 10/10
- citation accuracy: 10/10 (via manual JSON parsing)
- reliability: 10/10
- avg latency: 3.9s
- avg tool calls: 1.1
- total cost: $0.027

## Honest take
Nice ergonomics once running - the cleanest API of the four after LangGraph. But
it clearly assumes OpenAI: structured output and tracing both degrade with Claude,
and it was the framework whose install broke everything else. Good choice if you
are on OpenAI models; more friction on Claude than the numbers suggest.
