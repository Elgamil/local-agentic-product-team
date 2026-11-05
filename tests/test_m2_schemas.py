"""Unit tests for Milestone 2 artifact schemas."""

from __future__ import annotations

from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from lab.schemas.artifacts import (
    AcceptanceCriterion,
    ApiEndpoint,
    ArchitectureArtifact,
    ArchitecturalDecision,
    ComponentSpec,
    DataModelEntity,
    DeveloperRunArtifact,
    FileOperationRecord,
    FunctionalRequirement,
    RequirementsArtifact,
    TestCase,
    TestPlanArtifact,
    TestRunResult,
    WorkflowRunReport,
    WorkflowStageResult,
)


class TestRequirementsArtifact:
    def test_valid_requirements(self):
        req = RequirementsArtifact(
            feature_title="Catalog Feature",
            summary="Product catalog specification",
            functional_requirements=[
                FunctionalRequirement(id="FR-1", title="List Products", description="Show all products"),
            ],
            acceptance_criteria=[
                AcceptanceCriterion(
                    id="AC-1",
                    scenario="View Catalog",
                    given="Catalog has items",
                    when="User opens catalog",
                    then="Products are rendered",
                )
            ],
            edge_cases=["Empty search query", "Zero results"],
        )
        assert req.feature_title == "Catalog Feature"
        assert len(req.functional_requirements) == 1
        assert len(req.acceptance_criteria) == 1
        assert len(req.edge_cases) == 2

    def test_empty_requirements_rejected(self):
        with pytest.raises(ValidationError):
            RequirementsArtifact(
                feature_title="Invalid",
                summary="Missing requirements",
                functional_requirements=[],
                acceptance_criteria=[],
            )

    def test_alias_normalization(self):
        # Verify that 'title' and 'requirements' map to canonical field names
        data = {
            "title": "Normalized Feature",
            "summary": "Summary",
            "requirements": [{"id": "FR-1", "title": "Req", "description": "Desc"}],
            "acceptance_criteria": [{"id": "AC-1", "scenario": "Scen", "given": "G", "when": "W", "then": "T"}],
        }
        req = RequirementsArtifact.model_validate(data)
        assert req.feature_title == "Normalized Feature"
        assert len(req.functional_requirements) == 1


class TestArchitectureArtifact:
    def test_valid_architecture(self):
        arch = ArchitectureArtifact(
            system_overview="FastAPI + SQLite architecture",
            components=[
                ComponentSpec(name="App", file_path="app.py", purpose="FastAPI main application"),
            ],
            data_model=[
                DataModelEntity(
                    table_name="products",
                    fields={"id": "INTEGER PRIMARY KEY", "name": "TEXT"},
                    description="Products table",
                )
            ],
            api_endpoints=[
                ApiEndpoint(path="/", method="GET", description="Catalog home", query_params=["q"])
            ],
            implementation_steps=["Step 1", "Step 2"],
            decisions=[
                ArchitecturalDecision(
                    id="ADR-1",
                    decision="Use SQLite",
                    rationale="Simple and local",
                    alternatives_considered=["PostgreSQL"],
                    trade_offs="Not scalable for multiple nodes",
                    assumptions=["Low traffic"],
                    risks=["Database locking"],
                    requirements_addressed=["FR-1"],
                )
            ],
        )
        assert len(arch.components) == 1
        assert arch.data_model[0].table_name == "products"
        assert len(arch.decisions) == 1

    def test_missing_mandatory_fields_rejected_no_fabrication(self):
        # Verify that missing components/data_model/api_endpoints/implementation_steps are NOT silently fabricated
        with pytest.raises(ValidationError) as exc_info:
            ArchitectureArtifact.model_validate({
                "system_overview": "Only overview provided",
            })
        err_str = str(exc_info.value)
        assert "components" in err_str
        assert "data_model" in err_str
        assert "api_endpoints" in err_str
        assert "implementation_steps" in err_str

    def test_alias_normalization_without_fabrication(self):
        # Verify that aliases like 'architectural_decisions' and 'steps' are mapped
        data = {
            "system_overview": "Overview",
            "architectural_decisions": [
                {
                    "id": "ADR-1",
                    "title": "Use SQLite",  # alias for decision
                    "reason": "Local persistence",  # alias for rationale
                }
            ],
            "components": [
                {
                    "name": "DB",
                    "path": "database.py",  # alias for file_path
                    "description": "Database layer",  # alias for purpose
                }
            ],
            "data_model": [
                {
                    "name": "products",  # alias for table_name
                    "columns": {"id": "INTEGER PRIMARY KEY"},  # alias for fields
                    "description": "Products table",
                }
            ],
            "api_endpoints": [
                {
                    "route": "/",  # alias for path
                    "method": "GET",
                    "description": "Index route",
                }
            ],
            "steps": ["Step 1: DB", "Step 2: App"],  # alias for implementation_steps
        }
        arch = ArchitectureArtifact.model_validate(data)
        assert len(arch.decisions) == 1
        assert arch.decisions[0].decision == "Use SQLite"
        assert arch.decisions[0].rationale == "Local persistence"
        assert arch.components[0].file_path == "database.py"
        assert arch.components[0].purpose == "Database layer"
        assert arch.data_model[0].table_name == "products"
        assert "id" in arch.data_model[0].fields
        assert arch.api_endpoints[0].path == "/"
        assert arch.implementation_steps == ["Step 1: DB", "Step 2: App"]


class TestTestPlanArtifact:
    def test_valid_test_plan(self):
        plan = TestPlanArtifact(
            test_plan_summary="Acceptance test plan",
            test_cases=[
                TestCase(
                    id="TC-1",
                    name="Test Seed Products",
                    target_acceptance_criterion="AC-1",
                    description="Verifies seed products display",
                    steps=["Open /", "Inspect HTML"],
                    expected_result="All items present",
                )
            ],
            test_strategy="Acceptance testing via TestClient",
        )
        assert len(plan.test_cases) == 1
        assert plan.test_cases[0].id == "TC-1"

    def test_test_case_alias_normalization(self):
        data = {
            "summary": "Plan summary",  # alias for test_plan_summary
            "test_cases": [
                {
                    "id": "TC-1",
                    "title": "Test Search",  # alias for name
                    "ac_id": "AC-2",  # alias for target_acceptance_criterion
                    "description": "Tests search matching",
                    "steps": ["Search query", "Check result"],
                    "expected": "Matching items shown",  # alias for expected_result
                }
            ],
        }
        plan = TestPlanArtifact.model_validate(data)
        assert plan.test_plan_summary == "Plan summary"
        assert plan.test_cases[0].name == "Test Search"
        assert plan.test_cases[0].target_acceptance_criterion == "AC-2"
        assert plan.test_cases[0].expected_result == "Matching items shown"


class TestDeveloperAndWorkflowReport:
    def test_workflow_run_report_round_trip(self):
        now = datetime.now(timezone.utc)
        report = WorkflowRunReport(
            run_id="20261019T000000Z",
            model_id="gemma4:latest",
            runtime="google-adk+litellm+ollama",
            feature_request="Build product catalog",
            started_at=now,
            finished_at=now,
            total_duration_ms=1500.0,
            stages=[
                WorkflowStageResult(
                    stage_name="business_analyst",
                    agent_role="Business Analyst",
                    started_at=now,
                    finished_at=now,
                    duration_ms=500.0,
                    success=True,
                    artifact_path="artifacts/requirements.json",
                )
            ],
            test_runs=[
                TestRunResult(
                    command="pytest tests/acceptance",
                    exit_code=0,
                    passed_count=6,
                    failed_count=0,
                    duration_ms=450.0,
                    stdout="6 passed",
                    stderr="",
                    success=True,
                )
            ],
            repair_attempts=0,
            final_status="SUCCESS",
            summary="All stages completed successfully.",
        )
        json_data = report.model_dump_json()
        recovered = WorkflowRunReport.model_validate_json(json_data)
        assert recovered.run_id == report.run_id
        assert recovered.final_status == "SUCCESS"
        assert len(recovered.stages) == 1
        assert recovered.test_runs[0].passed_count == 6
