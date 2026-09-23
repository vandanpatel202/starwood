import json

from src.agent import run_agent
from src.cli import main
from src.ingestion import ingest_directory
from src.recording import RecordingClient, ReplayClient


class ScriptedLLM:
    def __init__(self):
        self.decision_count = 0

    def complete(self, prompt):
        if prompt.startswith("You are"):
            self.decision_count += 1
            tool = "categorize_transactions" if self.decision_count == 1 else "compute_totals"
            return json.dumps({"tool": tool})
        rows = json.loads(prompt.split("Transactions: ", 1)[1])
        return json.dumps([{"id": row["id"], "category": "Food"} for row in rows])


def test_recording_replays_and_cli_writes_report(tmp_path):
    recording_path = tmp_path / "llm.json"
    output_path = tmp_path / "report.json"
    recorder = RecordingClient(ScriptedLLM())
    run_agent(ingest_directory().transactions, recorder)
    recorder.save(recording_path)

    replay_result = run_agent(ingest_directory().transactions, ReplayClient(recording_path))
    assert replay_result.errors == ()

    exit_code = main([
        "--data", "sample_data",
        "--out", str(output_path),
        "--offline",
        "--replay", str(recording_path),
    ])
    assert exit_code == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert "totals" in report
    assert "flagged" in report
