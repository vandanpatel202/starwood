"""CSV ingestion and normalization for the personal-finance report.

This module intentionally does no categorization and no arithmetic.  It turns
the four known exports into one consistent transaction shape, retaining enough
metadata for later stages to explain why a row was or was not included in
financial totals.

Cleaning rules:

* Dates accept ISO dates (including single-digit month/day) and ``MM/DD/YYYY``
  and are emitted as ``YYYY-MM-DD``.
* Descriptions have leading/trailing whitespace removed and internal runs of
  whitespace collapsed.  The original CSV value remains in ``raw_description``.
* Empty expense categories become ``None`` for LLM categorization later.
  Positive income, explicit transfers, and pending authorizations receive
  deterministic ``Income``, ``Transfer``, and ``Pending`` labels.
* A zero-value pending authorization is retained, marked ``is_pending``, and
  excluded from totals.  It is not silently discarded.
* Rows explicitly described as a transfer are retained but marked
  ``is_transfer``.  Transfers move money between accounts and are excluded
  from income and expense totals by the reporting layer.
* A positive row whose description begins with ``Refund`` is retained and
  marked ``is_refund``.  The totals layer can use its signed amount to reduce
  spending rather than treating it as earnings.

No cross-file de-duplication happens here.  The exports overlap but their
descriptions are not consistently identical (for example, "Salary" versus
"Salary Deposit"), so a fuzzy match at ingestion time could silently lose real
transactions.  Reconciliation is a separate, explicit reporting decision.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Mapping

from src.settings import CSV_SCHEMAS, DEFAULT_DATA_DIR


class IngestionError(ValueError):
    """Raised when an input file cannot be safely interpreted."""


@dataclass(frozen=True, slots=True)
class Transaction:
    """A normalized row from one of the supported CSV exports."""

    date: str
    description: str
    amount: Decimal
    category: str | None
    source: str
    source_row: int
    raw_description: str
    is_pending: bool = False
    is_transfer: bool = False
    is_refund: bool = False

    @property
    def included_in_totals(self) -> bool:
        """Whether the row is a posted transaction rather than an account move."""

        return not self.is_pending and not self.is_transfer


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """All rows plus the input files that produced them."""

    transactions: tuple[Transaction, ...]
    files: tuple[Path, ...]


def normalize_date(value: str) -> str:
    """Parse a supported date and return its canonical ISO representation."""

    cleaned = value.strip()
    for pattern in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(cleaned, pattern).date().isoformat()
        except ValueError:
            pass
    raise IngestionError(
        f"Unsupported date {value!r}; expected YYYY-MM-DD or MM/DD/YYYY"
    )


def parse_amount(value: str) -> Decimal:
    """Parse a decimal amount exactly, rejecting blanks and non-numeric text."""

    cleaned = value.strip().replace(",", "")
    if not cleaned:
        raise IngestionError("Amount is blank")
    try:
        amount = Decimal(cleaned)
    except InvalidOperation as error:
        raise IngestionError(f"Invalid amount {value!r}") from error
    if not amount.is_finite():
        raise IngestionError(f"Invalid amount {value!r}")
    return amount


def normalize_description(value: str) -> str:
    """Make merchant text readable while preserving its original value elsewhere."""

    return " ".join(value.split())


def load_csv(path: Path) -> list[Transaction]:
    """Load one supported CSV file into normalized transactions.

    ``path.name`` identifies the export schema.  This is safer than guessing
    from similar headers, which could assign the wrong semantics to an export.
    """

    expected_headers = CSV_SCHEMAS.get(path.name)
    if expected_headers is None:
        supported = ", ".join(sorted(CSV_SCHEMAS))
        raise IngestionError(f"Unsupported file {path.name!r}; expected one of {supported}")
    if not path.is_file():
        raise IngestionError(f"CSV file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            headers = frozenset(reader.fieldnames or ())
            missing_headers = expected_headers - headers
            if missing_headers:
                missing = ", ".join(sorted(missing_headers))
                raise IngestionError(f"{path.name} is missing required column(s): {missing}")
            return list(_parse_rows(reader, path.name))
    except UnicodeDecodeError as error:
        raise IngestionError(f"{path.name} is not valid UTF-8 CSV") from error


def ingest_directory(data_dir: str | Path = DEFAULT_DATA_DIR) -> IngestionResult:
    """Load every required export from ``data_dir`` in a stable file order."""

    directory = Path(data_dir)
    if not directory.is_dir():
        raise IngestionError(f"Data directory does not exist: {directory}")

    files = tuple(directory / filename for filename in CSV_SCHEMAS)
    missing_files = [file.name for file in files if not file.is_file()]
    if missing_files:
        raise IngestionError("Missing required CSV file(s): " + ", ".join(missing_files))

    transactions = tuple(transaction for file in files for transaction in load_csv(file))
    return IngestionResult(transactions=transactions, files=files)


def _parse_rows(
    rows: Iterable[dict[str, str | None]], source: str
) -> Iterable[Transaction]:
    for source_row, row in enumerate(rows, start=2):
        try:
            raw_description = _required_value(row, "Description")
            description = normalize_description(raw_description)
            if not description:
                raise IngestionError("Description is blank")
            amount = parse_amount(_required_value(row, "Amount"))
            normalized_date = normalize_date(_required_value(row, "Date"))
        except IngestionError as error:
            raise IngestionError(f"{source}:{source_row}: {error}") from error

        category = _optional_value(row, "Category")
        upper_description = description.upper()
        is_pending = amount == 0 and "PENDING" in upper_description
        is_transfer = "TRANSFER" in upper_description
        is_refund = amount > 0 and upper_description.startswith("REFUND")
        if category is None:
            if is_pending:
                category = "Pending"
            elif is_transfer:
                category = "Transfer"
            elif amount > 0 and not is_refund:
                category = "Income"

        yield Transaction(
            date=normalized_date,
            description=description,
            amount=amount,
            category=category,
            source=source,
            source_row=source_row,
            raw_description=raw_description,
            is_pending=is_pending,
            is_transfer=is_transfer,
            is_refund=is_refund,
        )


def _required_value(row: Mapping[str, str | None], column: str) -> str:
    value = row.get(column)
    if value is None or not value.strip():
        raise IngestionError(f"{column} is blank")
    return value


def _optional_value(row: Mapping[str, str | None], column: str) -> str | None:
    value = row.get(column)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
