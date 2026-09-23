"""Command-line entrypoint for the personal finance agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from openai import OpenAI

from src.agent import run_agent
from src.ingestion import ingest_directory
from src.recording import RecordingClient, ReplayClient

DEFAULT_RECORDING_PATH = Path("llm_recording.json")


OPENAI_MODEL = "gpt-5.6-terra"
logger = logging.getLogger(__name__)


class OpenAIClient:
    """Small OpenAI Responses API client for live runs."""

    def __init__(self) -> None:
        self.model = OPENAI_MODEL
        self.client = OpenAI()

    def complete(self, prompt: str) -> str:
        logger.info("Calling OpenAI model %s", self.model)
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )
        logger.info("OpenAI response received (response_chars=%d)", len(response.output_text))
        return response.output_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the personal finance agent")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing the four CSV files")
    parser.add_argument("--out", type=Path, required=True, help="Path for report.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--online", action="store_true", help="Call OpenAI and optionally record responses")
    mode.add_argument("--offline", action="store_true", help="Replay recorded responses without network access")
    parser.add_argument("--record", type=Path, help="Save every live LLM call to this JSON file")
    parser.add_argument("--replay", type=Path, help="Replay LLM calls from this JSON file")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting %s run", "online" if args.online else "offline")
    if args.online and args.replay:
        print("--replay requires --offline", file=sys.stderr)
        return 2
    if args.offline and args.record:
        print("--record requires --online", file=sys.stderr)
        return 2
    if args.online and not args.record:
        print("--online requires --record so the run can be replayed", file=sys.stderr)
        return 2
    try:
        transactions = ingest_directory(args.data).transactions
        replay_path = args.replay or DEFAULT_RECORDING_PATH
        replay_client = ReplayClient(replay_path) if args.offline else None
        client = replay_client or OpenAIClient()
        recording_client = RecordingClient(client) if args.record else None
        run_client = recording_client or client
        result = run_agent(transactions, run_client, max_steps=args.max_steps)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.report, indent=2) + "\n", encoding="utf-8")
        if recording_client:
            recording_client.save(args.record)
        if replay_client:
            replay_client.assert_consumed()
        logger.info("Wrote report to %s", args.out)
        return 0
    except Exception as error:
        # The CLI boundary converts ingestion, replay, I/O, and SDK failures
        # (including a missing OPENAI_API_KEY) into a non-zero exit code.
        print(f"finance agent failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
