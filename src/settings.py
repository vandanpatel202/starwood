"""Application configuration shared by the finance-agent modules."""

from pathlib import Path
from typing import Final, Mapping


PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR: Final = PROJECT_ROOT / "sample_data"
REPORT_TRANSACTION_SOURCE: Final = "transactions_uncategorized.csv"

# These are the labels already used by the labeled expense export.  The model
# must choose exactly one of them for a previously uncategorized expense.
EXPENSE_CATEGORIES: Final[tuple[str, ...]] = (
    "Food",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Healthcare",
    "Home",
    "Health",
    "Insurance",
)

# File order is stable so output and replay recordings are reproducible.
CSV_SCHEMAS: Final[Mapping[str, frozenset[str]]] = {
    "income.csv": frozenset({"Date", "Description", "Amount", "Source"}),
    "expenses.csv": frozenset({"Date", "Description", "Amount", "Category"}),
    "bank_statement.csv": frozenset(
        {"Date", "Description", "Amount", "Balance", "Type"}
    ),
    "transactions_uncategorized.csv": frozenset(
        {"Date", "Description", "Amount", "Category"}
    ),
}
