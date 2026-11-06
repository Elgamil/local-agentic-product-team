"""Pydantic v2 data contracts for the Local AI Product Development Team.

Three groups of contracts live here:

* **Model-facing artifacts** — what the specialist agents must return.  These are
  validated strictly; a model that cannot satisfy them is retried, then failed
  over, and never accommodated by loosening a rule.
* **Durable workflow records** — the checkpoints, handoffs, attempts and evidence
  written to ``products_log/<slug>/``.
* **The phase machine** — ``ProductPhase`` plus ``ALLOWED_TRANSITIONS``, the only
  route by which workflow state may change.  Models never see these.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints, field_validator, model_validator

from lab.schemas.artifacts import (
    AcceptanceCriterion,
    ArchitecturalDecision,
    ComponentSpec,
    DataModelEntity,
    FunctionalRequirement,
    TestCase,
    TestRunResult,
)

SCHEMA_VERSION = 1

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def safe_relative_path(value: str) -> str:
    """Normalise a model-supplied path, rejecting anything that could escape."""
    raw = value.strip().replace("\\", "/")
    if not raw:
        raise ValueError("path must not be empty")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw.startswith("~") or ".." in path.parts:
        raise ValueError(f"path must be relative and stay inside its root: {value!r}")
    return str(path)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AgentRole(str, Enum):
    BUSINESS_ANALYST = "business_analyst"
    SOFTWARE_ARCHITECT = "software_architect"
    DEVELOPER = "developer"
    TEST_ENGINEER = "test_engineer"


class ProductPhase(str, Enum):
    DISCOVERY = "discovery"
    BRIEF_DRAFTING = "brief_drafting"
    AWAITING_NAME = "awaiting_name"
    ROADMAP_DRAFTING = "roadmap_drafting"
    AWAITING_ROADMAP_APPROVAL = "awaiting_roadmap_approval"
    AWAITING_STEP_START = "awaiting_step_start"
    STEP_PREPARING = "step_preparing"
    STEP_ARCHITECT = "step_architect"
    STEP_DEVELOPER_PLAN = "step_developer_plan"
    STEP_TEST_PLAN = "step_test_plan"
    STEP_PROPOSAL = "step_proposal"
    AWAITING_STEP_APPROVAL = "awaiting_step_approval"
    IMPLEMENTING = "implementing"
    VERIFYING_TESTS = "verifying_tests"
    REVIEWING = "reviewing"
    REPAIRING = "repairing"
    STEP_CLOSING = "step_closing"
    PRODUCT_COMPLETE = "product_complete"
    BLOCKED = "blocked"
    HISTORICAL = "historical"


#: Phases that stop and wait for the human.  No code path may leave one of these
#: except a method that received a ``HumanDecision``, or, at the gates in
#: ``AUTONOMOUS_GATES`` only, a decision citing a recorded human authorization.
HUMAN_GATES = frozenset(
    {
        ProductPhase.AWAITING_NAME,
        ProductPhase.AWAITING_ROADMAP_APPROVAL,
        ProductPhase.AWAITING_STEP_START,
        ProductPhase.AWAITING_STEP_APPROVAL,
    }
)

#: The gates a human may authorize in advance.  The name and the roadmap are not here:
#: an authorization is a decision about the roadmap, so it cannot also approve it.
AUTONOMOUS_GATES = frozenset(
    {ProductPhase.AWAITING_STEP_START, ProductPhase.AWAITING_STEP_APPROVAL}
)

_RESUMABLE_FROM_BLOCKED = {
    ProductPhase.DISCOVERY,
    ProductPhase.BRIEF_DRAFTING,
    ProductPhase.ROADMAP_DRAFTING,
    ProductPhase.AWAITING_STEP_START,
    ProductPhase.STEP_PREPARING,
    ProductPhase.STEP_ARCHITECT,
    ProductPhase.STEP_DEVELOPER_PLAN,
    ProductPhase.STEP_TEST_PLAN,
    ProductPhase.STEP_PROPOSAL,
    ProductPhase.IMPLEMENTING,
    ProductPhase.VERIFYING_TESTS,
    ProductPhase.REVIEWING,
    ProductPhase.REPAIRING,
    ProductPhase.STEP_CLOSING,
    ProductPhase.PRODUCT_COMPLETE,
}

ALLOWED_TRANSITIONS: dict[ProductPhase, frozenset[ProductPhase]] = {
    ProductPhase.DISCOVERY: frozenset({ProductPhase.BRIEF_DRAFTING, ProductPhase.BLOCKED}),
    ProductPhase.BRIEF_DRAFTING: frozenset({ProductPhase.AWAITING_NAME, ProductPhase.BLOCKED}),
    ProductPhase.AWAITING_NAME: frozenset({ProductPhase.ROADMAP_DRAFTING}),
    ProductPhase.ROADMAP_DRAFTING: frozenset(
        {ProductPhase.AWAITING_ROADMAP_APPROVAL, ProductPhase.BLOCKED}
    ),
    ProductPhase.AWAITING_ROADMAP_APPROVAL: frozenset(
        {ProductPhase.AWAITING_STEP_START, ProductPhase.ROADMAP_DRAFTING}
    ),
    ProductPhase.AWAITING_STEP_START: frozenset(
        {ProductPhase.STEP_PREPARING, ProductPhase.ROADMAP_DRAFTING}
    ),
    ProductPhase.STEP_PREPARING: frozenset({ProductPhase.STEP_ARCHITECT, ProductPhase.BLOCKED}),
    ProductPhase.STEP_ARCHITECT: frozenset(
        {ProductPhase.STEP_DEVELOPER_PLAN, ProductPhase.BLOCKED}
    ),
    ProductPhase.STEP_DEVELOPER_PLAN: frozenset(
        {ProductPhase.STEP_TEST_PLAN, ProductPhase.BLOCKED}
    ),
    ProductPhase.STEP_TEST_PLAN: frozenset({ProductPhase.STEP_PROPOSAL, ProductPhase.BLOCKED}),
    ProductPhase.STEP_PROPOSAL: frozenset(
        {ProductPhase.AWAITING_STEP_APPROVAL, ProductPhase.BLOCKED}
    ),
    ProductPhase.AWAITING_STEP_APPROVAL: frozenset(
        {
            ProductPhase.IMPLEMENTING,
            ProductPhase.STEP_ARCHITECT,
            ProductPhase.AWAITING_STEP_START,
        }
    ),
    ProductPhase.IMPLEMENTING: frozenset({ProductPhase.VERIFYING_TESTS, ProductPhase.BLOCKED}),
    ProductPhase.VERIFYING_TESTS: frozenset(
        {
            ProductPhase.REVIEWING,
            ProductPhase.REPAIRING,
            ProductPhase.STEP_CLOSING,
            ProductPhase.BLOCKED,
        }
    ),
    ProductPhase.REVIEWING: frozenset(
        {ProductPhase.REPAIRING, ProductPhase.STEP_CLOSING, ProductPhase.BLOCKED}
    ),
    ProductPhase.REPAIRING: frozenset({ProductPhase.VERIFYING_TESTS, ProductPhase.BLOCKED}),
    ProductPhase.STEP_CLOSING: frozenset(
        {
            ProductPhase.AWAITING_STEP_START,
            ProductPhase.PRODUCT_COMPLETE,
            ProductPhase.BLOCKED,
        }
    ),
    ProductPhase.BLOCKED: frozenset(_RESUMABLE_FROM_BLOCKED),
    ProductPhase.PRODUCT_COMPLETE: frozenset({ProductPhase.ROADMAP_DRAFTING}),
    ProductPhase.HISTORICAL: frozenset(),
}


class ApprovalKind(str, Enum):
    NAME = "name"
    ROADMAP = "roadmap"
    STEP_START = "step_start"
    STEP_PROPOSAL = "step_proposal"
    STEP_ACCEPTANCE = "step_acceptance"
    AUTONOMY = "autonomy"
    TEST_REVISION = "test_revision"


class StepStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    REJECTED = "rejected"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED_VERIFICATION = "failed_verification"
    REMOVED = "removed"


class FailureClass(str, Enum):
    """Why a task failed.  The class alone decides retry and failover behaviour."""

    MODEL_CONNECTION = "model_connection"
    INFERENCE_TIMEOUT = "inference_timeout"
    INVALID_RESPONSE = "invalid_response"
    INVALID_JSON = "invalid_json"
    SCHEMA_VALIDATION = "schema_validation"
    TOOL_EXECUTION = "tool_execution"
    FILESYSTEM = "filesystem"
    DEPENDENCY = "dependency"
    ACCEPTANCE_TEST = "acceptance_test"


#: Deterministic failures: another attempt or another model would hit the same wall.
NON_RETRYABLE_FAILURES = frozenset({FailureClass.FILESYSTEM, FailureClass.DEPENDENCY})


class BlockKind(str, Enum):
    MODEL_FAILURE = "model_failure"
    DETERMINISTIC_FAILURE = "deterministic_failure"
    VERIFICATION_FAILED = "verification_failed"
    CONDITIONS_UNMET = "conditions_unmet"
    INTEGRITY = "integrity"


class ExecutionMode(str, Enum):
    """Whether a person is asked at the step gates.  The workflow itself never differs."""

    MANUAL = "manual"
    AUTONOMOUS_PRODUCT = "autonomous_product"
    AUTONOMOUS_CHECKPOINT = "autonomous_checkpoint"


#: Who wrote a task result that came from a person's manual fix, not from a model.
MANUAL_AUTHOR = "human"


# ---------------------------------------------------------------------------
# Business Analyst artifacts (model-facing)
# ---------------------------------------------------------------------------


class DiscoveryTurn(BaseModel):
    """One Business Analyst turn during product discovery.

    ``ready_for_brief`` is a recommendation.  The orchestrator decides what
    actually happens next.
    """

    reply: Text = Field(description="What to say to the user, in plain, friendly language")
    understanding: Text = Field(description="One or two sentences restating the product so far")
    target_users: str = Field(default="", description="Who the product is for, as understood so far")
    category: Text = Field(description="Id of the most suitable product category")
    questions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Only the questions still needed, at most three",
    )
    ready_for_brief: bool = Field(
        description="True when enough is known to write the product brief"
    )


class ProductBrief(BaseModel):
    """Business-level definition of the product."""

    product_name: Text = Field(description="Short, descriptive product name")
    one_line_summary: Text = Field(description="One sentence describing the product")
    description: Text = Field(description="A short paragraph describing the product")
    category: Text = Field(description="Id of the product category")
    target_users: list[Text] = Field(min_length=1, description="Intended users")
    problem_statement: Text = Field(description="The need the product addresses")
    goals: list[Text] = Field(min_length=1, description="What success looks like")
    key_features: list[Text] = Field(min_length=1, description="Main capabilities in scope")
    out_of_scope: list[str] = Field(default_factory=list, description="Explicitly excluded items")
    constraints: list[str] = Field(default_factory=list, description="Known constraints")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made")


class RoadmapStepDraft(BaseModel):
    """One proposed roadmap item in business language."""

    number: int = Field(ge=1, description="Position in the roadmap, starting at the given number")
    name: Text = Field(description="Business-friendly step name")
    objective: Text = Field(description="What this step is for")
    user_visible_outcome: Text = Field(description="What the user can see or do afterwards")
    deliverables: list[Text] = Field(min_length=1, description="Main deliverables")
    dependencies: list[int] = Field(
        default_factory=list, description="Numbers of earlier steps this one needs"
    )
    completion_conditions: list[Text] = Field(
        min_length=1, description="Observable conditions proving the step is done"
    )
    milestone: str = Field(
        default="",
        description="Short name of the larger stage this step belongs to, such as Backend",
    )


class RoadmapDraft(BaseModel):
    """A complete proposed roadmap as returned by the Business Analyst."""

    summary: Text = Field(description="One or two sentences describing the plan")
    steps: list[RoadmapStepDraft] = Field(min_length=1, description="The ordered steps")

    @model_validator(mode="after")
    def _steps_are_ordered(self) -> "RoadmapDraft":
        numbers = [step.number for step in self.steps]
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            raise ValueError(f"step numbers must be consecutive and ascending, got {numbers}")
        names = [step.name.casefold() for step in self.steps]
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique")
        for step in self.steps:
            if any(dep >= step.number for dep in step.dependencies):
                raise ValueError(
                    f"step {step.number} may only depend on earlier steps, "
                    f"got {step.dependencies}"
                )
        return self


class StepBrief(BaseModel):
    """Phase A: the Business Analyst's preparation of one roadmap step."""

    objective: Text = Field(description="The approved objective of this step")
    business_outcome: Text = Field(description="The expected business outcome")
    requirements: list[FunctionalRequirement] = Field(
        min_length=1, description="What this step must deliver"
    )
    relevant_existing_work: list[str] = Field(
        default_factory=list, description="Existing implementation this step builds on"
    )
    dependencies: list[str] = Field(default_factory=list, description="What this step relies on")
    assumptions: list[str] = Field(default_factory=list, description="Assumptions made")


class ModificationRouting(BaseModel):
    """Advisory: which specialists a change request touches."""

    restated_request: Text = Field(description="The change request restated precisely")
    affected_agents: list[Literal["software_architect", "developer", "test_engineer"]] = Field(
        min_length=1, description="Specialists whose work the change affects"
    )


class StepProposal(BaseModel):
    """Phase C: the Business Analyst's human-readable consolidation."""

    what_this_step_delivers: Text
    why_this_design: Text = Field(description="Why the Architect recommends this design")
    technology_choices_explained: list[Text] = Field(
        min_length=1, description="Each technology choice explained simply"
    )
    what_the_developer_will_do: Text
    how_it_will_be_verified: Text
    completion_conditions: list[Text] = Field(min_length=1)


class ConditionCheck(BaseModel):
    condition: Text
    met: bool
    evidence: Text = Field(description="The concrete evidence for this judgement")


class StepClosureReport(BaseModel):
    """Phase F: the Business Analyst's closing report.

    Its opinion can stop a step from completing, but never force one to complete.
    Python decides that from the recorded verification evidence.
    """

    message: Text = Field(description="Plain-language report to the user")
    what_was_delivered: Text
    what_was_tested: Text
    remaining_limitations: list[str] = Field(default_factory=list)
    conditions: list[ConditionCheck] = Field(
        min_length=1, description="One entry for every completion condition of the step"
    )


class TeamAnswer(BaseModel):
    """A specialist's plain-language answer to a human question."""

    answer: Text = Field(description="Answer in plain, simple language")


# ---------------------------------------------------------------------------
# Specialist artifacts (model-facing)
# ---------------------------------------------------------------------------


class TechnologyChoice(BaseModel):
    name: Text = Field(description="The technology or approach")
    purpose: Text = Field(description="What it is used for here")
    reason: Text = Field(description="Why it was selected")


class InterfaceSpec(BaseModel):
    """A public contract, deliberately not tied to HTTP."""

    name: Text = Field(description="Function, route, command, screen or event name")
    kind: Text = Field(description="For example function, http_route, cli_command, or ui_screen")
    location: Text = Field(description="File that provides it, relative to the workspace root")
    contract: Text = Field(description="Inputs, outputs and behaviour")


class StepArchitecture(BaseModel):
    """Phase B: the Architect's proposal for one step."""

    summary: Text = Field(description="Short overview of the technical design")
    design: Text = Field(description="How the parts work together")
    components: list[ComponentSpec] = Field(min_length=1)
    interfaces: list[InterfaceSpec] = Field(
        default_factory=list, description="Public contracts the tests can rely on"
    )
    data_changes: list[DataModelEntity] = Field(default_factory=list)
    technology_choices: list[TechnologyChoice] = Field(min_length=1)
    decisions: list[ArchitecturalDecision] = Field(min_length=1)
    extensibility: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    handoff_summary: Text = Field(description="Two sentences for the Developer and Test Engineer")


class PlannedFileChange(BaseModel):
    path: str = Field(description="File path relative to the workspace root")
    action: Literal["create", "modify"]
    purpose: Text = Field(description="What this file does or what changes in it")
    implementation_notes: str = Field(default="", description="How it will be implemented")

    _safe_path = field_validator("path")(safe_relative_path)


class DeveloperPlan(BaseModel):
    """Phase B: the Developer's plan.  Writing no code at this stage is the point."""

    summary: Text = Field(description="What will be implemented")
    approach: Text = Field(description="How it will be implemented")
    file_changes: list[PlannedFileChange] = Field(min_length=1)
    dependencies: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    handoff_summary: Text = Field(description="Two sentences for the Test Engineer and reviewer")

    @model_validator(mode="after")
    def _paths_are_unique(self) -> "DeveloperPlan":
        paths = [change.path for change in self.file_changes]
        if len(set(paths)) != len(paths):
            raise ValueError(f"each file may be listed once, got {paths}")
        return self


class StepTestCase(TestCase):
    __test__ = False
    scenario_type: Literal["positive", "negative", "edge"] = Field(
        description="positive, negative or edge scenario"
    )


class PlannedTestFile(BaseModel):
    path: str = Field(description="Test file path relative to the test directory")
    purpose: Text = Field(description="Which test cases it automates")

    _safe_path = field_validator("path")(safe_relative_path)


class StepTestPlan(BaseModel):
    """Phase B: the Test Engineer's independent plan."""

    __test__ = False
    summary: Text = Field(description="Summary of the verification approach")
    verification_profile: Text = Field(description="Id of the verification profile to use")
    acceptance_criteria: list[AcceptanceCriterion] = Field(min_length=1)
    test_cases: list[StepTestCase] = Field(min_length=2)
    edge_cases: list[str] = Field(default_factory=list)
    regression_checks: list[str] = Field(default_factory=list)
    test_files: list[PlannedTestFile] = Field(min_length=1, max_length=4)
    handoff_summary: Text = Field(description="Two sentences for the Business Analyst")

    @model_validator(mode="after")
    def _plan_is_traceable_and_balanced(self) -> "StepTestPlan":
        known = {criterion.id for criterion in self.acceptance_criteria}
        orphans = [
            case.id for case in self.test_cases if case.target_acceptance_criterion not in known
        ]
        if orphans:
            raise ValueError(f"test cases target unknown acceptance criteria: {orphans}")
        kinds = {case.scenario_type for case in self.test_cases}
        if "positive" not in kinds or "negative" not in kinds:
            raise ValueError("the plan needs both positive and negative scenarios")
        return self


class ExecutableTestFile(BaseModel):
    """Executable test source.  Python writes it; the model never touches the disk."""

    __test__ = False
    path: str = Field(description="The planned test file path")
    content: Text = Field(description="Complete, runnable test source code")

    _safe_path = field_validator("path")(safe_relative_path)


class TestCaseAssessment(BaseModel):
    __test__ = False
    test_case_id: Text
    status: Literal["covered", "not_covered", "failed"]
    note: str = ""


class TestAssessment(BaseModel):
    """Phase E: the Test Engineer's reading of the real test output.

    It can downgrade a passing run but can never upgrade a failing one.
    """

    __test__ = False
    verdict: Literal["pass", "fail"]
    summary: Text
    plan_coverage: list[TestCaseAssessment] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    severity: Literal["blocker", "major", "minor", "info"]
    file_path: str = ""
    description: Text
    recommendation: Text


class ImplementationReview(BaseModel):
    """Phase E: the Architect's review of the actual files."""

    verdict: Literal["approved", "changes_requested"]
    summary: Text
    consistency_with_design: Text
    findings: list[ReviewFinding] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verdict_matches_findings(self) -> "ImplementationReview":
        serious = [f for f in self.findings if f.severity in ("blocker", "major")]
        if self.verdict == "approved" and any(f.severity == "blocker" for f in self.findings):
            raise ValueError("a review with blocker findings cannot be approved")
        if self.verdict == "changes_requested" and not serious:
            raise ValueError("changes_requested needs at least one blocker or major finding")
        return self


# ---------------------------------------------------------------------------
# Product records
# ---------------------------------------------------------------------------


class ProductMetadata(BaseModel):
    schema_version: int = SCHEMA_VERSION
    product_id: str
    display_name: str
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    category: str | None = None
    category_label: str | None = None
    description: str = ""
    origin: Literal["interactive", "imported"] = "interactive"
    name_confirmed: bool = False
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


class RoadmapStep(BaseModel):
    """A roadmap item with an orchestrator-assigned identity.

    ``step_id`` never changes or gets reused, so reordering a roadmap cannot
    point at another step's directory.  ``number`` is only the current position.
    """

    step_id: str = Field(pattern=r"^step-\d{2,}$")
    number: int = Field(ge=1)
    name: Text
    objective: Text
    user_visible_outcome: Text
    deliverables: list[Text] = Field(min_length=1)
    dependencies: list[int] = Field(default_factory=list)
    completion_conditions: list[Text] = Field(min_length=1)
    milestone: str = ""


class Roadmap(BaseModel):
    schema_version: int = SCHEMA_VERSION
    version: int = Field(ge=1)
    summary: Text
    steps: list[RoadmapStep] = Field(min_length=1)
    created_at: datetime = Field(default_factory=utcnow)
    change_request: str | None = None
    next_step_seq: int = Field(ge=1)

    def step(self, step_id: str) -> RoadmapStep:
        for item in self.steps:
            if item.step_id == step_id:
                return item
        raise KeyError(step_id)


class ImplementationTask(BaseModel):
    """One file of the approved Developer plan."""

    path: str
    action: Literal["create", "modify"]
    purpose: str
    implementation_notes: str = ""
    status: Literal["pending", "running", "done"] = "pending"
    sha256: str | None = None
    model: str | None = None


class StepProgress(BaseModel):
    step_id: str
    status: StepStatus = StepStatus.PENDING
    proposal_version: int = 0
    approved_version: int | None = None
    approved_proposal_sha256: str | None = None
    rerun_agents: list[str] = Field(
        default_factory=lambda: ["software_architect", "developer", "test_engineer"]
    )
    modification_requests: list[str] = Field(default_factory=list)
    question_count: int = 0
    implementation_tasks: list[ImplementationTask] = Field(default_factory=list)
    developer_plan_model: str | None = None
    developer_models: list[str] = Field(default_factory=list)
    verification_cycle: int = 0
    repair_cycles_used: int = 0
    repair_cycle_limit: int = 0
    pending_repair: str | None = None
    verification_passed: bool = False
    evidence_file: str | None = None
    accepted_by_human: bool = False
    completed_at: datetime | None = None
    # Hashes of everything the approval covered, so a later change can be detected.
    approved_documents: dict[str, str] = Field(default_factory=dict)
    sealed_manifest_sha256: str | None = None
    # Set when a person changed or rejected this step's proposal: an autonomous policy
    # authorized the proposals the team would bring, not this new direction.
    human_approval_required: bool = False
    # Set once a person's fix passes verification: a later finding comes back to them,
    # because a model repair could overwrite what they wrote.
    manual_verification: bool = False


class OperationRef(BaseModel):
    op_id: str
    name: str
    at: datetime = Field(default_factory=utcnow)
    detail: str = ""


class ActiveTask(BaseModel):
    """The in-flight logical task.  Present on disk means it has not finished."""

    task_id: str
    kind: str
    role: str
    step_id: str | None = None
    started_at: datetime = Field(default_factory=utcnow)
    model: str | None = None


class BlockedInfo(BaseModel):
    kind: BlockKind
    resume_phase: ProductPhase
    message: str
    failure_class: FailureClass | None = None
    task_id: str | None = None
    at: datetime = Field(default_factory=utcnow)
    task_kind: str | None = None
    role: str | None = None
    # What the failed task should have produced, so a person can supply it by hand.
    artifact: str | None = None
    artifact_fields: list[str] = Field(default_factory=list)
    expected_files: list[str] = Field(default_factory=list)


class AutonomyCheckpoint(BaseModel):
    """Where an authorized run stops.  Always a roadmap step, chosen by a person."""

    step_id: str
    requested: str = Field(description="What the person typed")
    milestone: str = ""


class AutonomyPolicy(BaseModel):
    """What the human authorized in advance.  Manual unless a person said otherwise."""

    mode: ExecutionMode = ExecutionMode.MANUAL
    checkpoint: AutonomyCheckpoint | None = None
    authorization_id: str | None = None
    authorized_at: datetime | None = None


class TestFileChange(BaseModel):
    """One difference between a sealed suite and the disk."""

    __test__ = False
    path: str
    status: Literal["added", "removed", "modified"]
    old_sha256: str | None = None
    new_sha256: str | None = None


class TestRevision(BaseModel):
    """A person's edit to a sealed suite, held until a person approves it."""

    __test__ = False
    revision_id: str
    step_id: str
    changes: list[TestFileChange]
    warnings: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    diff_path: str
    diff_excerpt: str = ""
    old_manifest_sha256: str | None = None


class HumanIntervention(BaseModel):
    """The persistent BLOCKED / HUMAN INTERVENTION REQUIRED checkpoint.

    It adds to ``BlockedInfo`` and replaces none of it: the blocked phase, the resume
    phase and the attempt ledger stay exactly as the failure left them.
    """

    intervention_id: str
    step_id: str | None = None
    task_id: str | None = None
    report_path: str
    report_version: int = 1
    change_sets: int = 0
    verify_requests: int = 0
    pending_revision: TestRevision | None = None
    opened_at: datetime = Field(default_factory=utcnow)


class InterventionBaseline(BaseModel):
    """Hashes at the failure checkpoint.  Later differences are the person's edits."""

    files: dict[str, str | None] = Field(default_factory=dict)
    approved: dict[str, str | None] = Field(default_factory=dict)
    taken_at: datetime = Field(default_factory=utcnow)


class WorkflowState(BaseModel):
    """The checkpoint.  Everything needed to continue lives here or in artifacts."""

    schema_version: int = SCHEMA_VERSION
    revision: int = 0
    phase: ProductPhase = ProductPhase.DISCOVERY
    paused: bool = False
    current_step_id: str | None = None
    current_agent: str | None = None
    active_model: str | None = None
    active_task: ActiveTask | None = None
    last_successful_operation: OperationRef | None = None
    last_failed_operation: OperationRef | None = None
    retry_count: int = 0
    pending_approval: ApprovalKind | None = None
    next_expected_action: str = "Describe the product idea."
    blocked: BlockedInfo | None = None
    pending_input: str | None = None
    discovery_round: int = 0
    roadmap_version: int = 0
    approved_roadmap_version: int | None = None
    roadmap_change_request: str | None = None
    steps: dict[str, StepProgress] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=utcnow)
    autonomy: AutonomyPolicy = Field(default_factory=AutonomyPolicy)
    intervention: HumanIntervention | None = None
    intervention_count: int = 0
    # A task whose result a person supplied by hand, waiting for its handler to adopt it.
    manual_artifact_task: str | None = None


class ProductEvent(BaseModel):
    event_id: str
    op_id: str | None = None
    timestamp: datetime = Field(default_factory=utcnow)
    type: str
    phase: str | None = None
    step_id: str | None = None
    agent: str | None = None
    model: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ApprovalRecord(BaseModel):
    """Proof of a human decision, bound to the exact artifact it covered.

    A record decided by ``autonomy_policy`` is only as good as the human authorization
    it cites, which is why ``authorization_id`` must name a human AUTONOMY record.
    """

    approval_id: str
    kind: ApprovalKind
    decision: Literal[
        "approved", "rejected", "changes_requested", "accepted_with_limitations", "withdrawn"
    ]
    subject: str
    artifact_sha256: str | None = None
    note: str = ""
    decided_by: Literal["human", "autonomy_policy"] = "human"
    authorization_id: str | None = None
    at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Model attempts, failover and handoffs
# ---------------------------------------------------------------------------


class ModelAttempt(BaseModel):
    """One model call within a logical task.  Attempts in one round are bounded."""

    attempt_id: str
    task_id: str
    round: int = 1
    model: str
    model_attempt: int = Field(ge=1, description="Attempt number on this model in this round")
    overall_attempt: int = Field(ge=1, description="Attempt number in this round")
    switched_from: str | None = None
    status: Literal["started", "succeeded", "failed", "interrupted"] = "started"
    failure_class: FailureClass | None = None
    error: str | None = None
    prompt_chars: int = 0
    response_chars: int = 0
    duration_ms: float = 0.0
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None


class TaskResult(BaseModel):
    """Written once, after the task's artifact is safely on disk."""

    task_id: str
    kind: str
    role: str
    step_id: str | None = None
    model: str
    round: int
    attempts: int
    artifact: str | None = None
    artifact_sha256: str | None = None
    objective: str = ""
    response_excerpt: str = ""
    finished_at: datetime = Field(default_factory=utcnow)


class HandoffRecord(BaseModel):
    """A structured record of one agent-to-agent exchange."""

    handoff_id: str
    task_id: str
    sender: str
    receiver: str
    model: str
    step_id: str | None = None
    task_input: str
    relevant_context: list[str] = Field(default_factory=list)
    response_artifact: str | None = None
    response_excerpt: str = ""
    decisions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    handoff_summary: str = ""
    validation_status: Literal["validated", "accepted_text"] = "validated"
    attempts: int = 1
    models_tried: list[str] = Field(default_factory=list)
    failed_over: bool = False
    timestamp: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# File operations and verification evidence
# ---------------------------------------------------------------------------


class JournaledFileOperation(BaseModel):
    """One line of the write-ahead journal for Developer file writes."""

    op_id: str
    stage: Literal["intent", "commit", "recovered"]
    path: str
    task_id: str | None = None
    sha256_before: str | None = None
    sha256_after: str | None = None
    bytes_written: int = 0
    unchanged: bool = False
    at: datetime = Field(default_factory=utcnow)


class FileReconciliation(BaseModel):
    """What a resume found when it compared the journal with the disk."""

    confirmed: list[str] = Field(default_factory=list)
    recovered: list[str] = Field(default_factory=list)
    not_applied: list[str] = Field(default_factory=list)
    drifted: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Phase D outcome, built from the journal rather than from model claims."""

    step_id: str
    proposal_version: int
    planned_files: list[str]
    written_files: dict[str, str] = Field(description="Relative path to SHA-256")
    unplanned_files: list[str] = Field(default_factory=list)
    developer_models: list[str] = Field(default_factory=list)
    summary: str = ""


class VerificationRun(TestRunResult):
    """One real execution of a sealed test suite."""

    __test__ = False
    step_id: str
    scope: Literal["acceptance", "regression"]
    profile_id: str
    cycle: int
    cwd: str
    timed_out: bool = False
    manifest_sha256: str | None = None
    workspace_fingerprint: str | None = None
    stdout_truncated: bool = False
    started_at: datetime = Field(default_factory=utcnow)


class VerificationEvidence(BaseModel):
    """Everything a step's completion rests on."""

    step_id: str
    cycle: int
    proposal_version: int
    tested_workspace_fingerprint: str
    manifest_sha256: str
    runs: list[VerificationRun] = Field(min_length=1)
    passed: bool = False
    failure_summary: str = ""
    recorded_at: datetime = Field(default_factory=utcnow)


class TestManifest(BaseModel):
    """Sealed hashes of the authoritative tests a human approved."""

    __test__ = False
    step_id: str
    profile_id: str
    proposal_version: int
    files: dict[str, str]
    sealed_at: datetime = Field(default_factory=utcnow)
    # Above zero only after a person approved a revised suite.
    revision: int = 0
    previous_manifest_sha256: str | None = None


# ---------------------------------------------------------------------------
# Human intervention reports
# ---------------------------------------------------------------------------


class FailedTest(BaseModel):
    __test__ = False
    name: str
    file: str = ""
    message: str = ""
    expected: str = ""
    actual: str = ""


class SuiteDiagnosis(BaseModel):
    """What one real test run showed.  Built from its output, never from a model."""

    step_id: str
    scope: str
    command: str = ""
    exit_code: int
    passed: int
    failed: int
    timed_out: bool = False
    executable: bool = Field(description="False when the suite could not run its tests")
    failed_tests: list[FailedTest] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)
    static_findings: list[str] = Field(default_factory=list)
    diagnostic_output: str = ""


class RecommendedAction(BaseModel):
    problem: str
    error: str = ""
    why_it_matters: str
    action: str
    files: list[str] = Field(default_factory=list)


class ManualFileChange(BaseModel):
    path: str = Field(description="Relative to the product folder")
    status: Literal["added", "removed", "modified"]
    old_sha256: str | None = None
    new_sha256: str | None = None


class AttemptSummary(BaseModel):
    model: str
    attempt: int
    status: str
    failure_class: str | None = None
    error: str = ""


class InterventionReport(BaseModel):
    """The structured Human Intervention Report.

    ``confirmed_facts``, ``agent_diagnosis`` and ``unconfirmed_causes`` stay separate
    fields so that what was observed can never be mistaken for what was guessed.
    """

    intervention_id: str
    version: int
    created_at: datetime = Field(default_factory=utcnow)

    # A. Context
    product: str
    slug: str
    step_number: int | None = None
    step_name: str | None = None
    phase: str
    block_kind: str
    responsible_agent: str
    preferred_model: str | None = None
    fallback_model: str | None = None
    attempts_used: int = 0
    attempts_allowed: int = 0
    repair_cycles_used: int = 0
    repair_cycles_allowed: int = 0
    task_id: str | None = None
    execution_mode: str = ""
    headline: str

    # B. Exact problem
    files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    attempts: list[AttemptSummary] = Field(default_factory=list)
    failed_tests: list[FailedTest] = Field(default_factory=list)
    suites: list[SuiteDiagnosis] = Field(default_factory=list)
    deliverable_problems: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    acceptance_criteria: list[str] = Field(default_factory=list)
    test_cases: list[str] = Field(default_factory=list)
    preserved_outputs: list[str] = Field(default_factory=list)
    manual_changes: list[ManualFileChange] = Field(default_factory=list)

    # C. Root-cause assessment
    confirmed_facts: list[str] = Field(default_factory=list)
    agent_diagnosis: list[str] = Field(default_factory=list)
    unconfirmed_causes: list[str] = Field(default_factory=list)

    # D. Recommended manual intervention
    actions: list[RecommendedAction] = Field(default_factory=list)
    next_commands: list[str] = Field(default_factory=list)
