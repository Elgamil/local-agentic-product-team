"""Pydantic v2 data contracts for Milestone 2 Multi-Agent Software Delivery.

These schemas govern the structured artifacts exchanged between:
  1. Business Analyst (RequirementsArtifact)
  2. Software Architect (ArchitectureArtifact)
  3. Test Engineer (TestPlanArtifact)
  4. Developer (DeveloperRunArtifact)
  5. Test Runner & Orchestrator (WorkflowRunReport)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Business Analyst Schemas
# ---------------------------------------------------------------------------


class FunctionalRequirement(BaseModel):
    """A discrete functional requirement."""

    id: str = Field(description="Unique requirement ID, e.g. 'FR-1'")
    title: str = Field(description="Short descriptive title of the requirement")
    description: str = Field(description="Detailed statement of required functionality")


class AcceptanceCriterion(BaseModel):
    """An acceptance criterion expressed in standard Given/When/Then style."""

    id: str = Field(description="Unique acceptance criterion ID, e.g. 'AC-1'")
    scenario: str = Field(description="Scenario name or title")
    given: str = Field(description="Preconditions / initial context")
    when: str = Field(description="Trigger action or event")
    then: str = Field(description="Expected observable outcome")


class RequirementsArtifact(BaseModel):
    """Structured requirements artifact produced by the Business Analyst."""

    feature_title: str = Field(description="Title of the feature under specification")
    summary: str = Field(description="Executive summary of the feature")
    functional_requirements: list[FunctionalRequirement] = Field(
        min_length=1,
        description="List of functional requirements",
    )
    acceptance_criteria: list[AcceptanceCriterion] = Field(
        min_length=1,
        description="List of verifiable acceptance criteria",
    )
    edge_cases: list[str] = Field(
        default_factory=list,
        description="Identified edge cases and boundary conditions",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "feature_title" not in data and "title" in data:
                data["feature_title"] = data["title"]
            if "functional_requirements" not in data and "requirements" in data:
                data["functional_requirements"] = data["requirements"]
        return data


# ---------------------------------------------------------------------------
# Software Architect Schemas
# ---------------------------------------------------------------------------


class ComponentSpec(BaseModel):
    """Specification of a software component / file."""

    name: str = Field(description="Component name, e.g. 'Database Layer' or 'Core Engine'")
    file_path: str = Field(description="Relative file path within project workspace, e.g. 'database.py' or 'engine.py'")
    purpose: str = Field(description="Responsibility and purpose of this component")

    @model_validator(mode="before")
    @classmethod
    def normalize_component_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "file_path" not in data and "path" in data:
                data["file_path"] = data["path"]
            if "purpose" not in data and "description" in data:
                data["purpose"] = data["description"]
        return data


class DataModelEntity(BaseModel):
    """Specification of an entity / data structure / database table."""

    table_name: str = Field(description="Name of the entity or table, e.g. 'products' or 'game_state'")
    fields: dict[str, str] = Field(
        description="Mapping of field name to data type/constraints, e.g. {'id': 'INTEGER PRIMARY KEY'}"
    )
    description: str = Field(description="Description of the entity")

    @model_validator(mode="before")
    @classmethod
    def normalize_entity_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "table_name" not in data and "name" in data:
                data["table_name"] = data["name"]
            if "fields" not in data and "columns" in data:
                data["fields"] = data["columns"]
        return data


class ApiEndpoint(BaseModel):
    """Specification of an HTTP endpoint."""

    path: str = Field(description="URL path, e.g. '/' or '/health'")
    method: Literal["GET", "POST", "PUT", "DELETE"] = Field(default="GET", description="HTTP method")
    description: str = Field(description="Purpose of endpoint")
    query_params: list[str] = Field(default_factory=list, description="Query parameter names, e.g. ['q']")

    @model_validator(mode="before")
    @classmethod
    def normalize_endpoint_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "path" not in data:
                if "route" in data:
                    data["path"] = data["route"]
                elif "url" in data:
                    data["path"] = data["url"]
        return data


class ArchitecturalDecision(BaseModel):
    """Structured architectural decision with rationale, alternatives, and trade-offs."""

    id: str = Field(description="Unique decision ID, e.g. 'ADR-1'")
    decision: str = Field(description="The architectural decision made")
    rationale: str = Field(description="Why this choice was selected")
    alternatives_considered: list[str] = Field(
        default_factory=list,
        description="Alternative approaches evaluated",
    )
    trade_offs: str = Field(
        default="",
        description="Trade-offs and architectural implications",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Assumptions made when making this decision",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Risks associated with this decision",
    )
    requirements_addressed: list[str] = Field(
        default_factory=list,
        description="Requirements (e.g., FR-1) that motivated this decision",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_decision_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "decision" not in data and "title" in data:
                data["decision"] = data["title"]
            if "rationale" not in data and "reason" in data:
                data["rationale"] = data["reason"]
        return data


class ArchitectureArtifact(BaseModel):
    """Structured architecture artifact produced by the Software Architect."""

    system_overview: str = Field(description="Overview of technical architecture")
    decisions: list[ArchitecturalDecision] = Field(
        default_factory=list,
        description="Key architectural decisions, rationales, alternatives, and trade-offs",
    )
    components: list[ComponentSpec] = Field(
        min_length=1,
        description="Components and files to be implemented",
    )
    data_model: list[DataModelEntity] = Field(
        min_length=1,
        description="Database entities and schemas",
    )
    api_endpoints: list[ApiEndpoint] = Field(
        min_length=1,
        description="API and web route specifications",
    )
    implementation_steps: list[str] = Field(
        min_length=1,
        description="Ordered sequence of implementation steps for the Developer",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Normalize decision aliases without fabricating data
            if "decisions" not in data and "architectural_decisions" in data:
                data["decisions"] = data["architectural_decisions"]
            # Normalize implementation step aliases without fabricating data
            if "implementation_steps" not in data:
                if "steps" in data:
                    data["implementation_steps"] = data["steps"]
                elif "implementation_plan" in data:
                    data["implementation_steps"] = data["implementation_plan"]
                elif "plan" in data:
                    data["implementation_steps"] = data["plan"]
        return data


# ---------------------------------------------------------------------------
# Test Engineer Schemas
# ---------------------------------------------------------------------------


class TestCase(BaseModel):
    """An individual test case specification."""

    __test__ = False
    id: str = Field(description="Unique test ID, e.g. 'TC-1'")
    name: str = Field(description="Test case name")
    target_acceptance_criterion: str = Field(description="Associated AC ID, e.g. 'AC-1'")
    description: str = Field(description="What this test verifies")
    steps: list[str] = Field(description="Step-by-step test execution actions")
    expected_result: str = Field(description="Expected outcome or assertion")

    @model_validator(mode="before")
    @classmethod
    def normalize_test_case_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "name" not in data and "title" in data:
                data["name"] = data["title"]
            if "target_acceptance_criterion" not in data:
                if "ac_id" in data:
                    data["target_acceptance_criterion"] = data["ac_id"]
                elif "acceptance_criterion" in data:
                    data["target_acceptance_criterion"] = data["acceptance_criterion"]
            if "expected_result" not in data and "expected" in data:
                data["expected_result"] = data["expected"]
        return data


class TestPlanArtifact(BaseModel):
    """Structured test plan produced by the Test Engineer."""

    __test__ = False
    test_plan_summary: str = Field(description="Summary of the testing approach")
    test_cases: list[TestCase] = Field(
        min_length=1,
        description="Collection of authoritative test cases",
    )
    test_strategy: str = Field(
        default="Automated acceptance testing via pytest",
        description="Testing strategy (unit, integration, acceptance)",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "test_plan_summary" not in data and "summary" in data:
                data["test_plan_summary"] = data["summary"]
            if "test_strategy" not in data:
                if "strategy" in data:
                    data["test_strategy"] = data["strategy"]
                else:
                    data["test_strategy"] = "Automated acceptance testing via pytest"
        return data


# ---------------------------------------------------------------------------
# Developer & Tool Execution Schemas
# ---------------------------------------------------------------------------


class FileOperationRecord(BaseModel):
    """Record of a safe file operation performed by the Developer."""

    file_path: str = Field(description="Relative path of the target file")
    action: Literal["write", "read", "list"] = Field(description="Operation type")
    success: bool
    bytes_transferred: int = Field(default=0, ge=0)
    message: str = Field(description="Outcome description or error message")


class DeveloperRunArtifact(BaseModel):
    """Summary artifact produced by the Developer agent stage."""

    implemented_files: list[str] = Field(description="List of files created or modified")
    file_operations: list[FileOperationRecord] = Field(default_factory=list)
    summary: str = Field(description="Developer's summary of the implementation")


# ---------------------------------------------------------------------------
# Test Runner & Orchestration Schemas
# ---------------------------------------------------------------------------


class TestRunResult(BaseModel):
    """Result of running the automated test suite."""

    __test__ = False
    command: str = Field(description="Exact command executed")
    exit_code: int = Field(description="Process exit code (0 for pass)")
    passed_count: int = Field(ge=0, description="Number of passed tests")
    failed_count: int = Field(ge=0, description="Number of failed tests")
    duration_ms: float = Field(ge=0, description="Execution duration in milliseconds")
    stdout: str = Field(description="Process standard output")
    stderr: str = Field(description="Process standard error")
    success: bool = Field(description="True if exit_code == 0 and failed_count == 0")


class WorkflowStageResult(BaseModel):
    """Result of one stage in the multi-agent workflow."""

    stage_name: str = Field(description="Name of the stage, e.g. 'business_analyst'")
    agent_role: str = Field(description="Role title, e.g. 'Business Analyst'")
    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0)
    success: bool
    artifact_path: str | None = None
    summary: str | None = None
    error: str | None = None


class WorkflowRunReport(BaseModel):
    """Comprehensive report of a Milestone 2 multi-agent workflow run."""

    run_id: str = Field(description="Unique ISO-8601 run identifier")
    model_id: str = Field(description="Model used for all agents, e.g. 'gemma4:latest'")
    runtime: str = Field(description="Runtime stack, e.g. 'google-adk+litellm+ollama'")
    feature_request: str = Field(description="The input feature request")
    started_at: datetime
    finished_at: datetime | None = None
    total_duration_ms: float | None = None
    stages: list[WorkflowStageResult] = Field(default_factory=list)
    test_runs: list[TestRunResult] = Field(default_factory=list)
    repair_attempts: int = Field(default=0, ge=0, le=2)
    final_status: Literal["SUCCESS", "FAILED", "INCOMPLETE"] = Field(default="INCOMPLETE")
    summary: str = Field(description="High-level narrative of workflow execution")

    def add_stage(self, stage: WorkflowStageResult) -> None:
        self.stages.append(stage)
