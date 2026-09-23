"""Minimal tool-using agent orchestration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Sequence

from pydantic import BaseModel, ConfigDict, TypeAdapter

from src.anomalies import find_duplicate_charges
from src.categorization import LLMClient
from src.ingestion import Transaction
from src.tools import TOOL_REGISTRY, ToolExecutionError, run_tool
from src.totals import Totals
from src.settings import REPORT_TRANSACTION_SOURCE


logger = logging.getLogger(__name__)


class AgentDecision(BaseModel):
    """The only model output the agent loop accepts for tool selection."""

    model_config = ConfigDict(extra="forbid", strict=True)
    tool: str


_DECISION_ADAPTER = TypeAdapter(AgentDecision)


@dataclass(frozen=True, slots=True)
class AgentResult:
    """The report and execution trace produced by one bounded run."""

    report: dict[str, object]
    transactions: tuple[Transaction, ...]
    totals: Totals
    steps: int
    errors: tuple[str, ...]


def run_agent(
    transactions: Sequence[Transaction],
    client: LLMClient,
    *,
    max_steps: int = 6,
) -> AgentResult:
    """Run a model-selected tool loop and always return a report.

    A malformed decision, unknown tool, or tool exception becomes an
    observation for the next step.  When the model cannot finish before the
    step limit, totals are computed directly as a safe deterministic fallback.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    all_transactions = tuple(transactions)
    reference_transactions = tuple(
        transaction
        for transaction in all_transactions
        if transaction.source == "expenses.csv" and transaction.category is not None
    )
    report_transactions = tuple(
        transaction
        for transaction in all_transactions
        if transaction.source == REPORT_TRANSACTION_SOURCE
    )
    # Keep the fallback useful for focused callers/tests that provide only
    # synthetic transactions without the canonical CSV source.
    current_transactions = report_transactions or all_transactions
    totals: Totals | None = None
    errors: list[str] = []
    observations: list[str] = []
    categorization_done = not _needs_categorization(current_transactions)

    for step in range(1, max_steps + 1):
        decision_prompt = _decision_prompt(
            observations,
            totals is not None,
            categorization_required=not categorization_done,
        )
        logger.info("Agent step %d/%d: requesting tool decision", step, max_steps)
        try:
            decision_text = client.complete(decision_prompt)
            decision = _parse_decision(decision_text)
            if decision.tool not in TOOL_REGISTRY:
                raise ToolExecutionError(f"Unknown tool: {decision.tool}")
            logger.info("Agent selected tool: %s", decision.tool)

            if decision.tool == "categorize_transactions":
                result = run_tool(
                    decision.tool,
                    transactions=current_transactions,
                    client=client,
                    examples=reference_transactions,
                )
                current_transactions = tuple(result.transactions)
                categorization_done = True
                observation = "categorize_transactions completed"
            else:
                if not categorization_done:
                    raise ToolExecutionError(
                        "categorization is required before compute_totals"
                    )
                totals = run_tool(
                    decision.tool,
                    transactions=current_transactions,
                )
                observation = "compute_totals completed"
            observations.append(observation)
            if totals is not None:
                break
        except Exception as error:
            message = f"step {step}: {error}"
            logger.warning("Agent step failed; retrying: %s", message)
            errors.append(message)
            observations.append(message)

    if totals is None:
        # This path does not trust any model response and guarantees output.
        totals = run_tool("compute_totals", transactions=current_transactions)
        logger.warning("Agent reached step limit; used deterministic totals fallback")
        observations.append("compute_totals fallback completed")

    report = build_report(current_transactions, totals)
    report["agent"] = {
        "steps": min(max_steps, len(observations)),
        "errors": errors,
    }
    return AgentResult(
        report=report,
        transactions=current_transactions,
        totals=totals,
        steps=min(max_steps, len(observations)),
        errors=tuple(errors),
    )


def build_report(
    transactions: Sequence[Transaction], totals: Totals
) -> dict[str, object]:
    """Build the report contract from code-owned values."""

    totals_dict = totals.as_dict()
    return {
        "totals": {
            key: totals_dict[key]
            for key in ("income", "expenses", "net", "savings_rate")
        },
        "by_category": totals_dict["by_category"],
        "flagged": find_duplicate_charges(transactions),
        "transactions": [
            {
                "date": transaction.date,
                "description": transaction.description,
                "amount": float(transaction.amount),
                "category": transaction.category or "Uncategorized",
            }
            for transaction in transactions
        ],
        "summary": _summary(totals),
    }


def _parse_decision(response: str) -> AgentDecision:
    """Validate the model's tool-selection response."""

    return _DECISION_ADAPTER.validate_json(response)


def _decision_prompt(
    observations: Sequence[str],
    totals_done: bool,
    *,
    categorization_required: bool,
) -> str:
    state = "\n".join(observations[-6:]) or "No tools have run yet."
    return (
        "You are a finance agent. Select exactly one available tool and return "
        "JSON only in the form {\"tool\": \"name\"}.\n"
        "Available tools: "
        + json.dumps(
            [
                {"name": spec.name, "description": spec.description}
                for spec in TOOL_REGISTRY.values()
            ]
        )
        + f"\nTotals already computed: {totals_done}."
        + f"\nCategorization required before totals: {categorization_required}."
        + f"\nObservations:\n{state}"
    )


def _needs_categorization(transactions: Sequence[Transaction]) -> bool:
    return any(
        transaction.category is None
        and (transaction.amount < 0 or transaction.is_refund)
        and transaction.included_in_totals
        for transaction in transactions
    )


def _summary(totals: Totals) -> str:
    return (
        f"Income was ${totals.income:.2f}, expenses were ${totals.expenses:.2f}, "
        f"and net savings were ${totals.net:.2f} "
        f"({totals.savings_rate:.1%} savings rate)."
    )
