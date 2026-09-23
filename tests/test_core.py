import json
from decimal import Decimal

from src.agent import run_agent
from src.anomalies import find_duplicate_charges
from src.categorization import categorize_missing
from src.ingestion import Transaction, ingest_directory
from src.totals import compute_totals


def test_loads_all_exports_and_normalizes_messy_rows():
    result = ingest_directory()
    assert len(result.transactions) == 69
    rows = [row for row in result.transactions if row.source == "transactions_uncategorized.csv"]
    assert rows[1].date == "2024-01-03"
    assert rows[3].description == "AMZN MKTP US*AB12C3D4E"
    assert next(row for row in rows if row.is_transfer).is_transfer
    assert next(row for row in rows if row.is_refund).is_refund
    assert next(row for row in rows if row.is_pending).is_pending


def test_invalid_categorization_response_is_repaired_once():
    class RepairClient:
        calls = 0

        def complete(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return '[{"id": "test.csv:2", "category": "Other"}]'
            return json.dumps([{"id": "test.csv:2", "category": "Food"}])

    expense = Transaction(
        date="2024-01-01", description="Coffee", amount=Decimal("-5"),
        category=None, source="test.csv", source_row=2, raw_description="Coffee",
    )
    client = RepairClient()
    result = categorize_missing((expense,), client)
    assert client.calls == 2
    assert result.error is None
    assert result.transactions[0].category == "Food"


def test_totals_are_deterministic_and_exclude_transfers_pending():
    common = dict(date="2024-01-01", source="test.csv", source_row=2, raw_description="x")
    rows = (
        Transaction(description="Salary", amount=Decimal("100"), category=None, **common),
        Transaction(description="Food", amount=Decimal("-30"), category="Food", **common),
        Transaction(description="Refund", amount=Decimal("5"), category="Food", is_refund=True, **common),
        Transaction(description="Transfer", amount=Decimal("-40"), category=None, is_transfer=True, **common),
        Transaction(description="Pending", amount=Decimal("0"), category=None, is_pending=True, **common),
    )
    totals = compute_totals(rows)
    assert totals.income == Decimal("100")
    assert totals.expenses == Decimal("25")
    assert totals.net == Decimal("75")
    assert totals.savings_rate == Decimal("0.75")
    assert totals.by_category == {"Food": Decimal("25")}


def test_bad_decisions_hit_step_limit_and_still_report():
    class BrokenClient:
        def complete(self, prompt):
            return '{"tool": 42}'

    expense = Transaction(
        date="2024-01-01", description="Unknown", amount=Decimal("-10"),
        category=None, source="transactions_uncategorized.csv", source_row=2,
        raw_description="Unknown",
    )
    result = run_agent((expense,), BrokenClient(), max_steps=2)
    assert result.steps == 2
    assert len(result.errors) == 2
    assert "totals" in result.report
    assert "summary" in result.report
    assert "flagged" in result.report
    assert all(
        isinstance(transaction["category"], str)
        for transaction in result.report["transactions"]
    )
    assert result.report["transactions"][0]["category"] == "Uncategorized"


def test_agent_requires_categorization_before_totals():
    class OutOfOrderClient:
        def __init__(self):
            self.decisions = 0

        def complete(self, prompt):
            if prompt.startswith("You are"):
                self.decisions += 1
                return json.dumps({
                    "tool": "compute_totals" if self.decisions == 1 else "categorize_transactions"
                })
            rows = json.loads(prompt.split("Transactions: ", 1)[1])
            return json.dumps([{"id": row["id"], "category": "Food"} for row in rows])

    result = run_agent((Transaction(
        date="2024-01-01", description="Coffee", amount=Decimal("-5"),
        category=None, source="transactions_uncategorized.csv", source_row=2,
        raw_description="Coffee",
    ),), OutOfOrderClient(), max_steps=4)

    assert result.errors
    assert "categorization is required before compute_totals" in result.errors[0]


def test_duplicate_charge_is_flagged():
    rows = [row for row in ingest_directory().transactions
            if row.source == "transactions_uncategorized.csv"]
    flagged = find_duplicate_charges(rows)
    assert len(flagged) == 1
    assert "WHOLEFDS" in flagged[0]["description"]
