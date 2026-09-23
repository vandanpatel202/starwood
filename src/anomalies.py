"""Deterministic optional anomaly checks."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from decimal import Decimal
from typing import Iterable

from src.ingestion import Transaction


def find_duplicate_charges(transactions: Iterable[Transaction]) -> list[dict[str, str]]:
    """Flag equal merchant/amount charges occurring within 31 days."""

    groups: defaultdict[tuple[str, Decimal], list[Transaction]] = defaultdict(list)
    for transaction in transactions:
        if transaction.included_in_totals and transaction.amount < 0:
            groups[(transaction.description.casefold(), transaction.amount)].append(transaction)

    flagged: list[dict[str, str]] = []
    for matching in groups.values():
        if len(matching) < 2:
            continue
        matching = sorted(matching, key=lambda transaction: transaction.date)
        for previous, current in zip(matching, matching[1:]):
            days = (date.fromisoformat(current.date) - date.fromisoformat(previous.date)).days
            if days <= 31:
                flagged.append({
                    "description": current.description,
                    "reason": (
                        f"Same merchant and amount as {previous.date}; "
                        f"charges are {days} days apart"
                    ),
                })
    return flagged
