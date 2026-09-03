"""The common result shape every framework fills in.

Each framework implements a run(question) -> RunResult. The eval harness only
knows about RunResult, so it scores all six frameworks the exact same way. That
is what makes the comparison fair and visible.
"""

from dataclasses import dataclass, field

from .config import cost_usd


@dataclass
class RunResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    # raw performance signals, filled by each framework
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    error: str | None = None  # set if the run crashed

    @property
    def cost_usd(self) -> float:
        return cost_usd(self.input_tokens, self.output_tokens)


def format_hits(hits) -> str:
    """Turn retrieved docs into the string the agent's search tool returns.

    Shared so every framework's tool hands the model identically shaped text -
    including the doc id, so the model can cite it.
    """
    if not hits:
        return "No documents found."
    blocks = []
    for h in hits:
        blocks.append(f"[{h.id}] {h.title}\n{h.text}")
    return "\n\n".join(blocks)
