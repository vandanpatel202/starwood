"""Tool entrypoints exposed to the agent loop.

The implementations stay in focused modules; this file defines the small,
explicit surface the model-facing dispatcher can select from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from src.categorization import CategorizationResult, LLMClient, categorize_missing
from src.ingestion import Transaction
from src.totals import Totals, compute_totals


class ToolExecutionError(RuntimeError):
    """Raised when a selected tool cannot produce a valid observation."""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Metadata and callable implementation for one agent tool."""

    name: str
    description: str
    run: Callable[..., object]


def categorize_tool(
    transactions: Sequence[Transaction],
    client: LLMClient,
    examples: Sequence[Transaction] = (),
) -> CategorizationResult:
    """Categorize missing expense labels through the configured LLM client."""

    result = categorize_missing(transactions, client, examples=examples)
    if result.error:
        raise ToolExecutionError(result.error)
    return result


def totals_tool(transactions: Sequence[Transaction]) -> Totals:
    """Compute report totals using deterministic transaction arithmetic."""

    return compute_totals(transactions)


TOOL_REGISTRY: Mapping[str, ToolSpec] = {
    "categorize_transactions": ToolSpec(
        name="categorize_transactions",
        description="Assign approved expense categories to uncategorized expenses.",
        run=categorize_tool,
    ),
    "compute_totals": ToolSpec(
        name="compute_totals",
        description="Compute income, expenses, net, savings rate, and category sums.",
        run=totals_tool,
    ),
}


def run_tool(name: str, **kwargs: object) -> object:
    """Dispatch one named tool and return its observation."""

    try:
        tool = TOOL_REGISTRY[name]
    except KeyError as error:
        raise ToolExecutionError(f"Unknown tool: {name}") from error
    return tool.run(**kwargs)
