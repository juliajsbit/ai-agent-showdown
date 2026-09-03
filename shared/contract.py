"""The shared agent contract: same task, same tool, same output shape, same
system prompt for all six frameworks. Only the orchestration code differs.
"""

from pydantic import BaseModel, Field


class Answer(BaseModel):
    """Structured final answer every framework must produce."""

    answer: str = Field(description="Direct answer to the user's question, grounded in the retrieved docs.")
    citations: list[str] = Field(
        default_factory=list,
        description="IDs of the documents the answer relied on, e.g. ['hr-pto'].",
    )


SYSTEM_PROMPT = """You are a support assistant for Meridian Robotics employees and customers.

Answer questions using ONLY the company knowledge base. To read it, call the
search_documents tool with a focused query. You may search more than once if the
question has multiple parts.

ALWAYS search the knowledge base before answering, on every question, even if the
question sounds personal ("how many days do I get", "which days am I in the
office"). These are questions about company policy - the answer is in the docs.
Never refuse or defer to HR without searching first.

Rules:
- Ground every fact in retrieved documents. Do not use outside knowledge.
- If the documents do not contain the answer, say you don't have that information.
- Keep answers short and specific. Include the exact numbers from the docs.
- Return the document ids you used as citations.
"""

# The task the eval harness and every agent share.
TOOL_NAME = "search_documents"
TOOL_DESCRIPTION = (
    "Search the Meridian Robotics knowledge base and return the most relevant "
    "documents. Input: a natural-language query string. Returns a list of "
    "documents with id, title, and text."
)
