"""Deterministic financial totals.

This module deliberately accepts normalized transactions rather than model
text.  Every value here is derived from signed transaction amounts in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable

from src.ingestion import Transaction


UNCATEGORIZED = "Uncategorized"


@dataclass(frozen=True, slots=True)
class Totals:
    """Computed totals and expense sums by category.

    ``savings_rate`` is a ratio (for example, ``0.25`` means 25 percent).
    Monetary fields remain Decimal until ``as_dict`` converts them for JSON.
    """

    income: Decimal
    expenses: Decimal
    net: Decimal
    savings_rate: Decimal
    by_category: dict[str, Decimal]

    def as_dict(self) -> dict[str, object]:
        """Return the report-shaped, JSON-compatible representation."""

        return {
            "income": float(self.income),
            "expenses": float(self.expenses),
            "net": float(self.net),
            "savings_rate": float(self.savings_rate),
            "by_category": {
                category: float(amount)
                for category, amount in self.by_category.items()
            },
        }


def compute_totals(transactions: Iterable[Transaction]) -> Totals:
    """Compute income, spending, net, savings rate, and category sums.

    Pending rows and transfers are excluded.  Positive refunds are reductions
    of spending, never income.  Debit category sums use positive magnitudes so
    a report can say that Food cost ``120.50``; refunds subtract from the
    corresponding category when one is available.
    """

    income = Decimal("0")
    expenses = Decimal("0")
    by_category: dict[str, Decimal] = {}

    for transaction in transactions:
        if not transaction.included_in_totals:
            continue

        category = transaction.category or UNCATEGORIZED
        if transaction.is_refund:
            expenses -= transaction.amount
            by_category[category] = by_category.get(category, Decimal("0")) - transaction.amount
        elif transaction.amount > 0:
            income += transaction.amount
        elif transaction.amount < 0:
            spending = -transaction.amount
            expenses += spending
            by_category[category] = by_category.get(category, Decimal("0")) + spending

    net = income - expenses
    savings_rate = net / income if income else Decimal("0")
    return Totals(
        income=income,
        expenses=expenses,
        net=net,
        savings_rate=savings_rate,
        by_category=dict(sorted(by_category.items())),
    )

