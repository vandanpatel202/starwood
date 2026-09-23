# Personal Finance Agent

This project ingests the four CSV exports in `sample_data/`, categorizes missing
expense labels with an OpenAI model, computes financial totals in Python, and
writes a report.

## Setup and commands

`pixi.lock` currently only locks the `linux-64` platform, so `pixi install`
will fail on macOS or Windows. If you have Pixi on Linux:

```bash
pixi install
pixi run test
pixi run run-online   # requires OPENAI_API_KEY, see below
pixi run run-offline
```

On any other platform, use the equivalent `uv` path instead (works on macOS,
Windows, and Linux):

```bash
uv sync --extra dev
uv run pytest
uv run python -m src.cli --data sample_data --out report.json \
  --online --record llm_recording.json      # requires OPENAI_API_KEY
uv run python -m src.cli --data sample_data --out report.json \
  --offline --replay llm_recording.json
```

Both paths write `report.json` and read/write `llm_recording.json` at the
repository root; a different recording can be selected with `--replay`.

A recorded run is already committed at `llm_recording.json`, with the
`report.json` it produced also committed, so the offline command above
reproduces that exact report with no network access and no API key.

The brief recommends temperature 0. The selected model (`gpt-5.6-terra`)
rejects the `temperature` parameter, so the request omits it and uses the
model's fixed sampling behavior.

Rough time spent: approximately two hours of implementation and review.

## Data and accounting rules

Ingestion reads and validates all four CSV files. Dates are normalized to
`YYYY-MM-DD`, descriptions have whitespace normalized, and amounts use
`Decimal`. Pending zero-value authorizations are retained but excluded from
totals. Explicit transfers are retained but excluded from income and expenses.
Positive refunds reduce expenses rather than counting as income. Original
merchant text and source row numbers are retained for traceability. No fuzzy
cross-file deduplication occurs because the exports lack stable transaction
identifiers; the repeated Whole Foods charge is retained, counted, and flagged
for review.
Blank positive transactions receive the code-owned `Income` label; explicit
transfers and pending authorizations receive `Transfer` and `Pending`. This
keeps the report's category field a string while reserving the LLM for semantic
expense categorization. If the model cannot categorize an expense before the
step limit, its report label is `Uncategorized`.

The report uses `transactions_uncategorized.csv` as its canonical transaction
set because the other exports overlap. The labeled rows in `expenses.csv` are
provided to the categorization prompt as reference examples containing date,
description, amount, and category. Existing labels are preserved. Missing
expense and refund labels are the only rows sent for categorization; income
rows do not receive expense categories.

The model selects tools, but code validates its JSON with Pydantic. Code owns
all arithmetic: income, expenses, net, savings rate, and category sums. A
failed or malformed model response is retried once with validation errors. The
agent has a step limit and computes a fallback report if the model cannot
finish. Duplicate merchant-and-amount charges within 31 days are listed in
`flagged`. Duplicate detection is the one optional extra selected from the
brief.

**Known limitation:** `EXPENSE_CATEGORIES` is drawn directly from the labels
already present in `expenses.csv` (Food, Transportation, Shopping,
Entertainment, Healthcare, Home, Health, Insurance). There is no "Housing"
category, so the committed run forces `ZELLE PAYMENT TO LANDLORD` (-$1,800,
the largest single expense) into "Home". `VENMO *JOHN SMITH` (P2P transfer)
and `ATM WITHDRAWAL` similarly get forced into the closest available label
("Shopping") rather than a more accurate "Transfer" or "Cash". With more
time, the category list would be extended (or a "Transfer/Other" bucket
added) rather than left tied to the reference export's existing labels.

## Output

`report.json` contains `totals`, `by_category`, `flagged`, `transactions`, and a
deterministic summary based on the computed totals. It also includes agent
execution information.

## AI usage and decisions

OpenAI Codex was used to draft the initial Python modules, pytest coverage,
README, and iterations of the agent and replay flow. The generated work was
reviewed and revised throughout. In particular, the response validation was
rewritten around strict Pydantic models, overlapping exports were changed to a
single canonical report source, model-produced arithmetic was rejected, and
categorization accuracy was removed because the brief allows at most one
optional extra. The final design decisions are:

- Keep semantic categorization and tool selection in the model, with strict
  Pydantic validation and a repair attempt for malformed output.
- Keep all monetary arithmetic, date normalization, duplicate detection, and
  report construction deterministic.
- Use the labeled expense export as few-shot context without allowing those
  examples to become output transactions.
- Treat model responses as untrusted and persist them for reproducible replay.

## Deliberate scope

The implementation does not reconcile every overlapping export into a full
account ledger, call an external anomaly service, or provide a UI. It keeps the
canonical report focused on the deliberately messy transaction export and
leaves deeper cross-account reconciliation for future work. With more time, I
would measure categorization accuracy on held-out labeled rows, add stronger
duplicate heuristics, and make reconciliation use explicit source-specific
matching rules.

## Walkthrough notes

- Run online once, then replay the exact recording offline.
- Trace a noisy merchant from CSV normalization through validated categorization.
- Show that totals are computed from `Decimal` transaction values, never model text.
- Demonstrate a malformed tool decision and the step-limit fallback report.
- Show deterministic duplicate detection for the repeated Whole Foods charge.
