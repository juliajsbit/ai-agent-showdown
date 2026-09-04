# Post 1 - Four frameworks, dependency hell

Image: `01-four-frameworks.png` (attach to the post).
Repo link goes in the first comment, not the body.

---

Two of these agent frameworks refuse to live in the same Python environment.

OpenAI Agents SDK wants openai 3.x. CrewAI wants 2.x. Installing one broke the other, down to a protobuf clash that stopped CrewAI from even importing.

That was the real lesson from building the same agent in four frameworks this week.

The task was identical: a support agent answering from a document knowledge base, one model (Claude Haiku 4.5), one eval over 10 questions. All four scored 100%. So accuracy told me nothing.

What did:

Dependency isolation. The fix was one venv per framework, plus moving retrieval into a shared service so the framework envs stay thin. That is the gap between a demo and something you can ship and maintain.

Structured output. LangGraph and CrewAI return a typed object out of the box. AutoGen and OpenAI SDK do not, at least not reliably with Claude, so I parsed JSON by hand for both.

Setup pain. LangGraph: an agent in one function, about 15 minutes. The other three each cost an afternoon of debugging before the first correct answer.

Cost and speed were close (numbers in the image). CrewAI ran about 30% pricier for identical output.

For a single tool-using agent, they converge on quality. What you actually pick is dependency sanity, cost, and how hard the framework fights you on day one.

4 down, 2 to go: Pydantic AI and LlamaIndex. Repo in the comments.

Which of those two do you expect to behave?

#AIEngineering #LLM #Agents
