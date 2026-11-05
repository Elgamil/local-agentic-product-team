"""Unit tests for the Milestone 2 Workflow Orchestrator (mocked runtime)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from lab.schemas.artifacts import (
    AcceptanceCriterion,
    ApiEndpoint,
    ArchitectureArtifact,
    ComponentSpec,
    DataModelEntity,
    FunctionalRequirement,
    RequirementsArtifact,
    TestCase,
    TestPlanArtifact,
    TestRunResult,
    WorkflowRunReport,
)
from lab.workflow.orchestrator import WorkflowOrchestrator


@pytest.fixture
def mock_runtime():
    runtime = MagicMock()
    runtime.model_id = "gemma4:latest"
    runtime.runtime_label = "google-adk+litellm+ollama"

    dummy_req = RequirementsArtifact(
        feature_title="MiniShop Lite",
        summary="Catalog Feature",
        functional_requirements=[
            FunctionalRequirement(id="FR-1", title="List", description="List products")
        ],
        acceptance_criteria=[
            AcceptanceCriterion(
                id="AC-1",
                scenario="Catalog",
                given="Items exist",
                when="Visiting /",
                then="Render items",
            )
        ],
        edge_cases=["No results"],
    )

    dummy_arch = ArchitectureArtifact(
        system_overview="FastAPI + SQLite",
        decisions=[
            {
                "id": "ADR-1",
                "decision": "Use SQLite",
                "rationale": "Simple and local",
                "alternatives_considered": [],
                "trade_offs": "",
                "assumptions": [],
                "risks": [],
                "requirements_addressed": ["FR-1"]
            }
        ],
        components=[
            ComponentSpec(name="DB", file_path="database.py", purpose="Data layer")
        ],
        data_model=[
            DataModelEntity(table_name="products", fields={"id": "INTEGER PRIMARY KEY"}, description="Products")
        ],
        api_endpoints=[
            ApiEndpoint(path="/", method="GET", description="Catalog", query_params=["q"])
        ],
        implementation_steps=["Step 1"],
    )

    dummy_plan = TestPlanArtifact(
        test_plan_summary="Plan",
        test_cases=[
            TestCase(
                id="TC-1",
                name="Catalog Test",
                target_acceptance_criterion="AC-1",
                description="Desc",
                steps=["Step 1"],
                expected_result="Ok",
            )
        ],
        test_strategy="Acceptance",
    )

    async def fake_complete_structured(prompt, schema, instruction=None, session_id="", max_retries=2):
        if schema == RequirementsArtifact:
            return dummy_req
        if schema == ArchitectureArtifact:
            return dummy_arch
        if schema == TestPlanArtifact:
            return dummy_plan
        raise ValueError(f"Unknown schema: {schema}")

    runtime.complete_structured_schema = AsyncMock(side_effect=fake_complete_structured)
    runtime.run_agent = AsyncMock(return_value=("Done", []))

    return runtime


@pytest.mark.asyncio
async def test_orchestrator_stage_flow_success(tmp_path, mock_runtime):
    orchestrator = WorkflowOrchestrator(
        runtime=mock_runtime,
        workspace_dir=tmp_path / "minishop",
        runs_dir=tmp_path / "runs",
    )

    mock_test_result = TestRunResult(
        command="pytest",
        exit_code=0,
        passed_count=6,
        failed_count=0,
        duration_ms=100.0,
        stdout="6 passed",
        stderr="",
        success=True,
    )

    with patch("lab.workflow.orchestrator.run_controlled_tests", return_value=mock_test_result):
        report = await orchestrator.execute_workflow(feature_request="Build product catalog")

    assert isinstance(report, WorkflowRunReport)
    assert report.final_status == "SUCCESS"
    assert len(report.stages) == 4
    assert report.stages[0].stage_name == "business_analyst"
    assert report.stages[1].stage_name == "software_architect"
    assert report.stages[2].stage_name == "test_engineer"
    assert report.stages[3].stage_name == "developer"
    assert len(report.test_runs) == 1
    assert report.test_runs[0].success is True
    assert report.repair_attempts == 0


@pytest.mark.asyncio
async def test_orchestrator_repair_flow(tmp_path, mock_runtime):
    orchestrator = WorkflowOrchestrator(
        runtime=mock_runtime,
        workspace_dir=tmp_path / "minishop",
        runs_dir=tmp_path / "runs",
    )

    failing_result = TestRunResult(
        command="pytest",
        exit_code=1,
        passed_count=4,
        failed_count=2,
        duration_ms=100.0,
        stdout="2 failed",
        stderr="",
        success=False,
    )
    passing_result = TestRunResult(
        command="pytest",
        exit_code=0,
        passed_count=6,
        failed_count=0,
        duration_ms=100.0,
        stdout="6 passed",
        stderr="",
        success=True,
    )

    with patch(
        "lab.workflow.orchestrator.run_controlled_tests",
        side_effect=[failing_result, passing_result],
    ):
        report = await orchestrator.execute_workflow(feature_request="Build product catalog")

    assert report.final_status == "SUCCESS"
    assert len(report.test_runs) == 2
    assert report.repair_attempts == 1


@pytest.mark.asyncio
async def test_orchestrator_stage_failure_diagnostics_and_partial_history(tmp_path, mock_runtime):
    # Simulate Software Architect failing with a ValueError after exhausting retries
    async def failing_complete_structured(prompt, schema, instruction=None, session_id="", max_retries=2):
        if schema == RequirementsArtifact:
            return RequirementsArtifact(
                feature_title="MiniShop Lite",
                summary="Catalog",
                functional_requirements=[
                    FunctionalRequirement(id="FR-1", title="FR1", description="Desc1")
                ],
                acceptance_criteria=[
                    AcceptanceCriterion(id="AC-1", scenario="Scen", given="G", when="W", then="T")
                ],
            )
        if schema == ArchitectureArtifact:
            raise ValueError("Model response could not be parsed into ArchitectureArtifact: 3 validation errors for ArchitectureArtifact\ncomponents\n  Field required")
        raise ValueError(f"Unknown schema: {schema}")

    mock_runtime.complete_structured_schema = AsyncMock(side_effect=failing_complete_structured)

    runs_dir = tmp_path / "runs"
    orchestrator = WorkflowOrchestrator(
        runtime=mock_runtime,
        workspace_dir=tmp_path / "minishop",
        runs_dir=runs_dir,
    )

    report = await orchestrator.execute_workflow(feature_request="Build product catalog")

    assert report.final_status == "FAILED"
    assert "Software Architect" in report.summary or "ArchitectureArtifact" in report.summary

    # Ensure both stages are recorded: Business Analyst (PASS) and Software Architect (FAIL)
    assert len(report.stages) == 2
    assert report.stages[0].stage_name == "business_analyst"
    assert report.stages[0].success is True
    assert report.stages[1].stage_name == "software_architect"
    assert report.stages[1].success is False
    assert "ArchitectureArtifact" in report.stages[1].error

    # Downstream stages should NOT have run
    assert report.test_runs == []
    assert report.repair_attempts == 0

    # Ensure run files were generated and contain diagnostics
    run_dir = runs_dir / report.run_id
    report_json_path = run_dir / "report.json"
    report_md_path = run_dir / "report.md"
    events_path = run_dir / "events.jsonl"
    transcript_path = run_dir / "transcript.md"

    assert report_json_path.exists()
    assert report_md_path.exists()
    assert events_path.exists()
    assert transcript_path.exists()

    report_md_content = report_md_path.read_text(encoding="utf-8")
    assert "Stage Failure Diagnostics" in report_md_content
    assert "software_architect" in report_md_content
    # Preserved requirements must still appear in report.md
    assert "Requirements & Acceptance Criteria" in report_md_content
    assert "FR-1" in report_md_content
