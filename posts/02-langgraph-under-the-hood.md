# Post 2 - LangGraph under the hood

Image: `02-langgraph-under-the-hood.png` (attach to the post).
Repo link goes in the first comment, not the body.
Post Tue-Thu, ~8am PST.

---

Hey folks. Last week I mentioned LangGraph got me a working agent in about 15 minutes, in one function call. A couple of people asked what that function actually does. Fair question, so I opened it up.

This is what I wrote:

agent = create_react_agent(llm, tools=[search_knowledge_base], prompt=SYSTEM_PROMPT, response_format=Answer)

And this is what it built. I pulled the graph straight out of the running agent - LangGraph can draw its own structure, which is a nice touch.

Five nodes. That's the whole thing.

start hands the question to agent. agent is the model thinking. If it decides it needs information, it goes to tools, my search runs, and the result comes straight back to agent. That little back and forth is the entire ReAct loop - the model keeps going around until it has enough. When it's done, it drops into generate_structured_response, which forces the answer into the shape I asked for, and then end.

The part that clicked for me was the dashed arrows. Those are conditional edges, and they're the only real decisions in the graph. My code never chooses whether to search. The model does. Everything else is fixed plumbing.

That's why one function is enough here. There's exactly one loop and one decision.

It stops being enough the moment you need a second decision. A step that needs human approval before it runs. A second agent taking over part of the job. A retry with a different strategy when verification fails. Each of those is a new node and a new edge, and that's the point where you stop calling the helper and build the graph yourself.

So the one-liner isn't magic. It's a sensible default for the most common shape of agent. Knowing what it generated is what tells you when you've outgrown it.

All the code is public, repo in the comments. Two frameworks left in the series.

Has anyone here already dropped from create_react_agent down to a hand-built graph? Curious what pushed you over.

#AIEngineering #LLM #AIAgents #LangGraph #LangChain
