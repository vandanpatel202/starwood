"""LLM-facing categorization with strict validation of untrusted responses."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Protocol, Sequence

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, field_validator

from src.ingestion import Transaction
from src.settings import EXPENSE_CATEGORIES


class LLMClient(Protocol):
    """The small interface needed by categorization and easy to fake in tests."""

    def complete(self, prompt: str) -> str:
        """Return the model's text response for ``prompt``."""


class CategorizationItem(BaseModel):
    """One model-produced category assignment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    category: str

    @field_validator("category")
    @classmethod
    def category_must_be_known(cls, value: str) -> str:
        if value not in EXPENSE_CATEGORIES:
            allowed = ", ".join(EXPENSE_CATEGORIES)
            raise ValueError(f"must be one of: {allowed}")
        return value


_RESPONSE_ADAPTER = TypeAdapter(list[CategorizationItem])


@dataclass(frozen=True, slots=True)
class CategorizationResult:
    """Categorized transactions and any rows the model failed to categorize."""

    transactions: tuple[Transaction, ...]
    uncategorized_rows: tuple[str, ...]
    raw_response: str
    error: str | None = None


def categorize_missing(
    transactions: Sequence[Transaction],
    client: LLMClient,
    *,
    examples: Sequence[Transaction] = (),
) -> CategorizationResult:
    """Ask an LLM to categorize blank categories without trusting its output.

    Only posted expense rows are sent to the model.  Income, transfers, and
    pending authorizations do not belong to the expense-category vocabulary.
    A bad response leaves a row's category as ``None`` rather than inventing a
    category outside the approved list.
    """

    targets = tuple(
        transaction
        for transaction in transactions
        if transaction.category is None
        and (transaction.amount < 0 or transaction.is_refund)
        and transaction.included_in_totals
    )
    if not targets:
        return CategorizationResult(tuple(transactions), (), "")

    prompt = build_categorization_prompt(targets, examples=examples)
    try:
        raw_response = client.complete(prompt)
    except Exception as error:  # The agent can continue and produce a report.
        return CategorizationResult(
            tuple(transactions),
            tuple(_transaction_id(transaction) for transaction in targets),
            "",
            f"LLM request failed: {error}",
        )

    categories, error = parse_categorization_response(raw_response, targets)
    if error:
        repair_prompt = build_repair_prompt(prompt, raw_response, error)
        try:
            repaired_response = client.complete(repair_prompt)
        except Exception as repair_error:  # Preserve the original validation error.
            return CategorizationResult(
                tuple(transactions),
                tuple(_transaction_id(transaction) for transaction in targets),
                raw_response,
                f"{error}; repair request failed: {repair_error}",
            )
        repaired_categories, repair_error = parse_categorization_response(
            repaired_response, targets
        )
        if repair_error:
            return CategorizationResult(
                tuple(transactions),
                tuple(_transaction_id(transaction) for transaction in targets),
                repaired_response,
                f"{error}; repaired response also invalid: {repair_error}",
            )
        categories = repaired_categories
        raw_response = repaired_response

    categorized = tuple(
        replace(transaction, category=categories[_transaction_id(transaction)])
        if _transaction_id(transaction) in categories
        else transaction
        for transaction in transactions
    )
    return CategorizationResult(categorized, (), raw_response)


def build_categorization_prompt(
    transactions: Sequence[Transaction],
    *,
    examples: Sequence[Transaction] = (),
) -> str:
    """Create a constrained, batch prompt for the categorization model."""

    rows = [
        {
            "id": _transaction_id(transaction),
            "description": transaction.description,
        }
        for transaction in transactions
    ]
    categories = ", ".join(EXPENSE_CATEGORIES)
    reference_rows = [
        {
            "date": transaction.date,
            "description": transaction.description,
            "amount": str(transaction.amount),
            "category": transaction.category,
        }
        for transaction in examples
        if transaction.category is not None
    ]
    reference_text = (
        "Labeled examples from expenses.csv (use them as reference, not as output rows): "
        f"{json.dumps(reference_rows)}\n"
        if reference_rows
        else ""
    )
    return (
        "Categorize each bank-card expense using only one of these exact labels: "
        f"{categories}.\n"
        "Return JSON only, with this exact shape: "
        '[{"id": "transactions_uncategorized.csv:2", "category": "Food"}].\n'
        f"{reference_text}"
        f"Transactions: {json.dumps(rows)}"
    )


def build_repair_prompt(original_prompt: str, response: str, error: str) -> str:
    """Ask the model to correct a response using concrete validation feedback."""

    return (
        "Your previous categorization response failed validation. Return a corrected "
        "JSON list only, preserving every requested id.\n"
        f"Validation errors: {error}\n"
        f"Invalid response: {response}\n"
        f"Original request:\n{original_prompt}"
    )


def parse_categorization_response(
    response: str, transactions: Sequence[Transaction]
) -> tuple[dict[str, str], str | None]:
    """Validate that a model response covers every requested row exactly once."""

    try:
        items = _RESPONSE_ADAPTER.validate_json(response)
    except ValidationError as error:
        details = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}"
            for issue in error.errors()
        )
        return {}, f"Pydantic validation failed: {details}"

    expected_rows = {_transaction_id(transaction) for transaction in transactions}
    categories: dict[str, str] = {}
    for item in items:
        row = item.id
        category = item.category
        if row not in expected_rows or row in categories:
            return {}, "LLM response has an unexpected or duplicate row"
        categories[row] = category

    if set(categories) != expected_rows:
        return {}, "LLM response did not categorize every requested row"
    return categories, None


def _transaction_id(transaction: Transaction) -> str:
    """Return a stable identifier that remains unique across CSV files."""

    return f"{transaction.source}:{transaction.source_row}"
