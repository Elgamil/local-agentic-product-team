import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

class EventLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.transcript_path = self.run_dir / "transcript.md"

    def log_event(self, event_type: str, payload: dict[str, Any]):
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            **payload
        }
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def append_transcript(self, markdown_text: str):
        with open(self.transcript_path, "a", encoding="utf-8") as f:
            f.write(markdown_text + "\n\n")

    def log_workflow_start(self, run_id: str, feature_request: str):
        self.log_event("WorkflowStarted", {"run_id": run_id, "feature_request": feature_request})
        self.append_transcript(f"# Workflow Run {run_id}\n\n**Feature Request:**\n\n```text\n{feature_request}\n```\n")

    def log_workflow_end(self, status: str, duration_ms: float):
        self.log_event("WorkflowFinished", {"status": status, "duration_ms": duration_ms})
        self.append_transcript(f"## Workflow Finished\n\nStatus: {status}\nDuration: {duration_ms:.0f} ms\n")

    def log_agent_start(self, agent_role: str, task: str):
        self.log_event("AgentStarted", {"agent_role": agent_role, "task": task})
        self.append_transcript(f"## Agent: {agent_role}\n\n**Task / Instructions:**\n{task}\n")

    def log_agent_end(self, agent_role: str, duration_ms: float, success: bool, artifact_path: str = ""):
        self.log_event("AgentCompleted", {
            "agent_role": agent_role,
            "duration_ms": duration_ms,
            "success": success,
            "artifact_path": artifact_path,
        })
        self.append_transcript(f"**Agent Finished:** {success} in {duration_ms:.0f} ms\nArtifact: `{artifact_path}`")

    def log_model_request(self, model: str, instruction: str, user_message: str):
        self.append_transcript(f"### Model Request ({model})\n\n**System Instruction:**\n```text\n{instruction}\n```\n\n**User Message:**\n```text\n{user_message}\n```\n")

    def log_model_response(self, text: str, duration_ms: float, events: list[dict]):
        self.log_event("ModelRequestCompleted", {
            "duration_ms": duration_ms,
            "adk_events": events,
            "success": True
        })
        self.append_transcript(f"### Model Response (took {duration_ms:.0f} ms)\n\n**Raw Output:**\n```text\n{text}\n```\n")
        if events:
            self.append_transcript(f"**ADK Events / Tool Calls:**\n```json\n{json.dumps(events, indent=2)}\n```\n")

    def log_validation_failed(
        self,
        schema_name: str,
        attempt: int,
        max_attempts: int,
        error_message: str,
        raw_snippet: str = "",
    ):
        self.log_event("StructuredValidationFailed", {
            "schema": schema_name,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "error": error_message,
            "raw_snippet": raw_snippet[:300],
        })
        self.append_transcript(
            f"⚠️ **Validation Failed for `{schema_name}` (Attempt {attempt}/{max_attempts})**\n\n"
            f"**Error:**\n```text\n{error_message}\n```\n"
        )

    def log_retry_attempt(
        self,
        schema_name: str,
        attempt: int,
        max_attempts: int,
        correction_message: str,
    ):
        self.log_event("StructuredRetryInitiated", {
            "schema": schema_name,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "correction_message": correction_message[:300],
        })
        self.append_transcript(
            f"🔄 **Retrying Structured Output for `{schema_name}` (Attempt {attempt}/{max_attempts})**\n\n"
            f"**Correction Prompt sent to model:**\n```text\n{correction_message}\n```\n"
        )

    def log_stage_error(self, agent_role: str, stage_name: str, error: str):
        self.log_event("StageFailed", {
            "agent_role": agent_role,
            "stage_name": stage_name,
            "error": error,
        })
        self.append_transcript(
            f"❌ **Stage Failed: {agent_role} (`{stage_name}`)**\n\n"
            f"**Error Details:**\n```text\n{error}\n```\n"
        )

    def log_test_run(self, command: str, exit_code: int, passed: int, failed: int, stdout: str, stderr: str):
        self.log_event("TestCommandFinished", {
            "command": command,
            "exit_code": exit_code,
            "passed_count": passed,
            "failed_count": failed,
            "stdout": stdout,
            "stderr": stderr
        })
        self.append_transcript(f"## Test Run\n\nCommand: `{command}`\nExit Code: {exit_code}\nPassed: {passed}, Failed: {failed}\n\n**Stdout:**\n```text\n{stdout}\n```\n")
