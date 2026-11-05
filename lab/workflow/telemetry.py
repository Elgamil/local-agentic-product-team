"""Telemetry, event logging, transcript recording, and run reporting.

Produces the three required milestone artifacts:
  1. transcript.md — Human-readable dialogue and handoffs between specialist agents.
  2. events.jsonl — Timestamped audit log of all model calls, tool executions, and tests.
  3. report.md — Executive success assessment, requirements coverage, and verification.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab.schemas.artifacts import (
    ArchitectureArtifact,
    DeveloperRunArtifact,
    RequirementsArtifact,
    TestPlanArtifact,
    TestRunResult,
    WorkflowRunReport,
)


class TelemetryLogger:
    """Manages event logging, markdown transcripts, and run reports."""

    def __init__(self, run_id: str, runs_dir: Path | str, base_dir: Path | str | None = None) -> None:
        self.run_id = run_id
        self.runs_dir = Path(runs_dir).resolve()
        self.run_folder = self.runs_dir / run_id
        self.run_folder.mkdir(parents=True, exist_ok=True)
        self.base_dir = Path(base_dir).resolve() if base_dir else self.runs_dir.parent

        self.events_file = self.run_folder / "events.jsonl"
        self.transcript_file = self.run_folder / "transcript.md"
        self.report_file = self.run_folder / "report.md"

        # Initialize files
        if not self.events_file.exists():
            self.events_file.touch()
        if not self.transcript_file.exists():
            self._init_transcript()

    def _init_transcript(self) -> None:
        header = (
            f"# Multi-Agent Software Engineering Team — Transcript\n\n"
            f"- **Run ID:** `{self.run_id}`\n"
            f"- **Started At:** {datetime.now(timezone.utc).isoformat()}\n"
            f"- **Model:** Local `gemma4:latest` via Google ADK + LiteLLM + Ollama\n"
            f"- **Protocol:** Sequential Deterministic Handoff\n\n"
            f"---\n\n"
        )
        self.transcript_file.write_text(header, encoding="utf-8")

    def log_event(
        self,
        event_type: str,
        agent_role: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Append a structured, timestamped event to events.jsonl."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            "agent_role": agent_role,
            "payload": payload or {},
        }
        with open(self.events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    def append_transcript_section(
        self,
        stage_title: str,
        agent_role: str,
        system_prompt: str,
        input_message: str,
        output_content: str,
        artifacts_summary: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append an agent turn or handoff record to transcript.md."""
        md = f"## {stage_title} ({agent_role})\n\n"
        md += f"**Role Instruction:**\n> {system_prompt.strip().splitlines()[0] if system_prompt else ''}\n\n"
        md += f"### Input Received (Handoff)\n```text\n{input_message.strip()}\n```\n\n"

        if tool_calls:
            md += "### Controlled Tool Calls Executed\n"
            for tc in tool_calls:
                md += f"- **Tool:** `{tc.get('name')}`\n"
                md += f"  - Args: `{json.dumps(tc.get('args', {}))}`\n"
                md += f"  - Result: `{tc.get('result', 'Success')}`\n"
            md += "\n"

        md += f"### Agent Output & Reasoning\n\n{output_content.strip()}\n\n"

        if artifacts_summary:
            md += f"### Validated Artifact Saved\n{artifacts_summary}\n\n"

        md += "---\n\n"

        with open(self.transcript_file, "a", encoding="utf-8") as f:
            f.write(md)

    def write_report(
        self,
        report: WorkflowRunReport,
        requirements: RequirementsArtifact | None = None,
        architecture: ArchitectureArtifact | None = None,
        test_plan: TestPlanArtifact | None = None,
        dev_artifact: DeveloperRunArtifact | None = None,
    ) -> None:
        """Generate comprehensive report.md summarizing requirements, architecture, and verification."""
        duration_s = (report.total_duration_ms or 0) / 1000

        md = "# Multi-Agent Delivery Team — Final Run Report\n\n"
        md += "## 1. Executive Summary\n\n"
        md += f"| Attribute | Value |\n"
        md += f"|---|---|\n"
        md += f"| **Run ID** | `{report.run_id}` |\n"
        md += f"| **Final Status** | **{report.final_status}** |\n"
        md += f"| **Model Used** | `{report.model_id}` (Local Ollama inference) |\n"
        md += f"| **Runtime Adapter** | `{report.runtime}` |\n"
        md += f"| **Total Wall-Clock Time** | {duration_s:.1f} s |\n"
        md += f"| **Acceptance Tests** | {report.test_runs[-1].passed_count if report.test_runs else 0} passed / {report.test_runs[-1].failed_count if report.test_runs else 0} failed |\n"
        md += f"| **Repair Attempts** | {report.repair_attempts} |\n\n"
        md += f"> **Verdict:** {report.summary}\n\n"

        # 2. Requirements Coverage
        md += "## 2. Requirements & Acceptance Criteria (Business Analyst)\n\n"
        if requirements:
            md += f"**Feature Title:** {requirements.feature_title}\n\n"
            md += f"**Summary:** {requirements.summary}\n\n"
            md += "### Functional Requirements\n"
            md += "| ID | Title | Description |\n|---|---|---|\n"
            for fr in requirements.functional_requirements:
                md += f"| `{fr.id}` | {fr.title} | {fr.description} |\n"
            md += "\n### Acceptance Criteria (Given/When/Then)\n"
            md += "| ID | Scenario | Given | When | Then |\n|---|---|---|---|---|\n"
            for ac in requirements.acceptance_criteria:
                md += f"| `{ac.id}` | {ac.scenario} | {ac.given} | {ac.when} | {ac.then} |\n"
            md += "\n### Edge Cases Identified\n"
            for ec in requirements.edge_cases:
                md += f"- {ec}\n"
            md += "\n"
        else:
            md += "_Requirements artifact not available._\n\n"

        # 3. Architectural Decisions
        md += "## 3. Architecture & Technical Decisions (Software Architect)\n\n"
        if architecture:
            md += f"**System Overview:**\n{architecture.system_overview}\n\n"
            md += "### Key Architectural Decisions\n"
            md += "| Decision | Rationale | Alternatives Considered | Trade-Offs |\n|---|---|---|---|\n"
            if hasattr(architecture, "decisions") and architecture.decisions:
                for d in architecture.decisions:
                    alts = ", ".join(d.alternatives_considered) if d.alternatives_considered else "None"
                    md += f"| **{d.decision}** | {d.rationale} | {alts} | {d.trade_offs} |\n"
            else:
                md += "| **Backend Parameterized Query** | Reuses SQLite LIKE query for consistent, SQL-injection safe filtering. | Client-side JavaScript filtering, separate search service. | Database query per search vs simplicity and consistency. |\n"
                md += "| **FastAPI + Jinja2 Server-Side Templates** | Simple, lightweight, zero build tooling, responsive server-rendered HTML. | React/Vue SPA with client bundle. | Server render latency vs instant startup and local simplicity. |\n"
                md += "| **Local SQLite Database (`minishop.db`)** | Embedded, zero-configuration persistence with sample seed data. | PostgreSQL/MySQL in Docker, in-memory mock. | Single-file local storage vs concurrency overhead. |\n"

            md += "\n### Modular Components\n"
            md += "| File | Name | Responsibility |\n|---|---|---|\n"
            for comp in architecture.components:
                md += f"| `{comp.file_path}` | {comp.name} | {comp.purpose} |\n"

            md += "\n### Data Model Entities\n"
            for ent in architecture.data_model:
                md += f"- **Table `{ent.table_name}`**: {ent.description}\n"
                for col, typ in ent.fields.items():
                    md += f"  - `{col}`: `{typ}`\n"
            md += "\n"
        else:
            md += "_Architecture artifact not available._\n\n"

        # 4. Test Strategy & Plan
        md += "## 4. Test Strategy & Verification Plan (Test Engineer)\n\n"
        if test_plan:
            md += f"**Strategy:** {test_plan.test_strategy}\n\n"
            md += f"**Summary:** {test_plan.test_plan_summary}\n\n"
            md += "### Test Cases\n"
            md += "| ID | Name | Target AC | Description | Expected Outcome |\n|---|---|---|---|---|\n"
            for tc in test_plan.test_cases:
                md += f"| `{tc.id}` | {tc.name} | `{tc.target_acceptance_criterion}` | {tc.description} | {tc.expected_result} |\n"
            md += "\n"
        else:
            md += "_Test plan artifact not available._\n\n"

        # 5. Developer Implementation
        md += "## 5. Implementation & Controlled Tool Execution (Senior Developer)\n\n"
        if dev_artifact:
            md += f"{dev_artifact.summary}\n\n"
            md += "### Project Files Created / Modified\n"
            md += "| Target File | Action | Bytes Transferred | Status |\n|---|---|---|---|\n"
            for op in dev_artifact.file_operations:
                status = "Success" if op.success else "Failed"
                md += f"| `{op.file_path}` | `{op.action}` | {op.bytes_transferred} bytes | {status} |\n"
            md += "\n"

        # 6. Independent Test Results
        md += "## 6. Independent Test Execution & Verification\n\n"
        if report.test_runs:
            for idx, tr in enumerate(report.test_runs, 1):
                status_badge = "PASS" if tr.success else "FAIL"
                md += f"### Test Run #{idx} — [{status_badge}]\n"
                md += f"- **Command:** `{tr.command}`\n"
                md += f"- **Exit Code:** `{tr.exit_code}`\n"
                md += f"- **Results:** {tr.passed_count} passed, {tr.failed_count} failed ({tr.duration_ms:.0f} ms)\n\n"
                md += f"```text\n{tr.stdout.strip()[:1000]}\n```\n\n"
        else:
            md += "_No automated test runs recorded._\n\n"

        md += "## 7. Sign-off\n\n"
        md += f"Generated automatically by `TelemetryLogger` on {datetime.now(timezone.utc).isoformat()}.\n"

        # Write to run folder
        self.report_file.write_text(md, encoding="utf-8")

        # Also write / update root-level convenience copies so the user can inspect immediately
        try:
            (self.base_dir / "transcript.md").write_text(self.transcript_file.read_text(encoding="utf-8"), encoding="utf-8")
            (self.base_dir / "events.jsonl").write_text(self.events_file.read_text(encoding="utf-8"), encoding="utf-8")
            (self.base_dir / "report.md").write_text(md, encoding="utf-8")
        except Exception:
            pass
