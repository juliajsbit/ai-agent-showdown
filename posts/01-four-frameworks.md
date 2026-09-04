# Post 1 - Four frameworks, dependency hell

Image: `01-four-frameworks.png` (attach to the post).
Repo link goes in the first comment, not the body.

---

Hey folks. I've been teaching myself how AI agents actually work under the hood, and I got curious about one thing: if I build the exact same agent in different frameworks, which one actually holds up?

So I made a little project to find out.

Here's the setup. It's a support agent that answers questions from a company knowledge base. It searches the documents (vector search plus a reranker), then answers with sources. Everyone gets the same model (Claude Haiku 4.5) and the same test set of 10 questions. The only thing I swap is the framework.

So far I've done four: LangGraph, CrewAI, AutoGen, and the OpenAI Agents SDK.

Here's what surprised me.

All four got the answers right. 100% each. I really thought accuracy would be the interesting part, and it just... wasn't.

The real differences were the unglamorous, practical stuff:

- Two of them refuse to share a Python environment. OpenAI's SDK wants one version of a library, CrewAI wants another, and installing one quietly broke the other. I ended up giving each framework its own isolated setup.
- Only half of them return a clean structured answer with Claude. For the other two I had to parse it by hand.
- LangGraph was running in about 15 minutes. The other three each cost me an afternoon of debugging before the first correct answer.
- Cost was close, but CrewAI came out about 30% pricier for the same answers.

My honest takeaway so far: for a simple agent, the framework barely changes the quality. What it changes is how much it fights you, what it costs, and whether it plays nice with the rest of your stack.

Two more to go (Pydantic AI and LlamaIndex). It's all public, repo in the comments, and I'm genuinely figuring this out as I go. If you've shipped with any of these, I'd love to hear what tripped you up.

#AIEngineering #LLM #Agents
