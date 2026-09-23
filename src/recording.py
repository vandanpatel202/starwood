"""Recording and offline replay for LLM calls."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


logger = logging.getLogger(__name__)


class LLMCallable(Protocol):
    def complete(self, prompt: str) -> str:
        """Return one model response."""


class ReplayError(RuntimeError):
    """Raised when a replay does not match the calls made by the agent."""


@dataclass
class RecordingClient:
    """Wrap an LLM client and retain every prompt/response pair."""

    client: LLMCallable

    def __post_init__(self) -> None:
        self.entries: list[dict[str, str]] = []

    def complete(self, prompt: str) -> str:
        logger.info("LLM request %d started (prompt_chars=%d)", len(self.entries) + 1, len(prompt))
        try:
            response = self.client.complete(prompt)
        except Exception as error:
            # Save failed calls too, so an online run explains why no usable
            # response was produced. Replay will reproduce the failure.
            self.entries.append({"prompt": prompt, "response": "", "error": str(error)})
            logger.exception("LLM request %d failed", len(self.entries))
            raise
        self.entries.append({"prompt": prompt, "response": response})
        logger.info("LLM request %d completed (response_chars=%d)", len(self.entries), len(response))
        return response

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.entries, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Saved %d LLM recording entries to %s", len(self.entries), destination)


class ReplayClient:
    """Replay recorded responses in order and verify prompt alignment."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ReplayError(f"Could not read LLM recording {self.path}: {error}") from error
        if not isinstance(loaded, list) or not all(
            isinstance(entry, dict)
            and isinstance(entry.get("prompt"), str)
            and isinstance(entry.get("response"), str)
            for entry in loaded
        ):
            raise ReplayError("LLM recording must be a list of prompt/response objects")
        if not loaded:
            raise ReplayError("LLM recording is empty; run online mode with --record first")
        self.entries: list[dict[str, str]] = loaded
        self.position = 0

    def complete(self, prompt: str) -> str:
        if self.position >= len(self.entries):
            raise ReplayError("LLM replay exhausted before the agent finished")
        entry = self.entries[self.position]
        self.position += 1
        if entry["prompt"] != prompt:
            raise ReplayError(
                f"LLM replay prompt mismatch at response {self.position}: "
                "the data, agent steps, or recording do not match"
            )
        if entry.get("error"):
            raise ReplayError(f"Recorded LLM request failed: {entry['error']}")
        logger.info("Replayed LLM response %d", self.position)
        return entry["response"]

    def assert_consumed(self) -> None:
        if self.position != len(self.entries):
            raise ReplayError(
                f"LLM replay has {len(self.entries) - self.position} unused response(s)"
            )
