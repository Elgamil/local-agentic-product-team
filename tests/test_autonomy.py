"""Autonomous execution: the human's approval is given in advance, nothing else changes.

Only the models are scripted.  Sealing, journals, checkpoints and test runs are real.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os

import pytest

from lab.product.autonomy import (
    USAGE,
    clarification,
    describe_mode,
    parse_autonomy_request,
    resolve_checkpoint,
)
from lab.product.console import ConsoleTerminal
from lab.product.decisions import Intent, parse_input
from lab.product.session import ProductSession, _LeaveProduct
from lab.product.store import ProductStore
from lab.product.workflow import (
    GateViolation,
    HumanDecision,
    PolicyDecision,
    ProductWorkflow,
)
from lab.schemas.product import (
    ALLOWED_TRANSITIONS,
    AUTONOMOUS_GATES,
    HUMAN_GATES,
    ApprovalKind,
    BlockKind,
    ExecutionMode,
    ProductPhase,
    Roadmap,
    RoadmapDraft,
    RoadmapStep,
    StepStatus,
    WorkflowState,
)
from tests.fakes import (
    APPROVE,
    BACKEND_THEN_FRONTEND,
    CALC_BUGGY,
    Fail,
    Harness,
    ScriptedTerminal,
    SimulatedPowerLoss,
    Text,
    complete_first_step,
    enable_autonomy,
    roadmap_of,
    to_name_gate,
    to_proposal_gate,
    to_roadmap_gate,
    to_step_start_gate,
)

STEP_SEQUENCE = [
    "step_brief",
    "architect",
    "developer_plan",
    "test_plan",
    "test_file",
    "proposal",
    "implement",
    "assess",
    "review",
    "closure",
]


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def session_for(harness: Harness, *answers: str) -> tuple[ProductSession, ScriptedTerminal]:
    terminal = ScriptedTerminal(*answers)
    return (
        ProductSession(harness.manager, harness.router, harness.policy, harness.playbooks, terminal),
        terminal,
    )


def three_steps(harness: Harness) -> None:
    harness.team.script("roadmap", Text(roadmap_of(*BACKEND_THEN_FRONTEND)))


def phases_of(workflow: ProductWorkflow) -> list[str]:
    return [
        event.payload["target"]
        for event in workflow.store.read_events()
        if event.type == "PhaseChanged"
    ]


def calls_for_step(harness: Harness, step_id: str) -> list[str]:
    return [call.kind for call in harness.team.calls if call.task_id.startswith(step_id)]


class TestManualRemainsTheDefault:
    async def test_a_new_product_is_manual(self, harness):
        workflow = harness.new_workflow()
        assert workflow.state.autonomy.mode == ExecutionMode.MANUAL
        assert workflow.state.autonomy.authorization_id is None
        assert not workflow.autonomy_active

    async def test_manual_mode_still_stops_at_every_gate(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        for expected in (ProductPhase.AWAITING_STEP_START,):
            await workflow.advance()
            assert workflow.state.phase == expected
        workflow.start_step(APPROVE)
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert harness.team.calls_of("implement") == []

    async def test_each_product_has_its_own_mode(self, harness):
        first = harness.new_workflow()
        await to_step_start_gate(first)
        enable_autonomy(first)
        second = harness.new_workflow()
        await to_step_start_gate(second)
        assert first.autonomy_active
        assert not second.autonomy_active

    def test_a_state_written_before_this_feature_loads_as_manual(self, tmp_path):
        saved = json.loads(WorkflowState().model_dump_json())
        del saved["autonomy"], saved["intervention"], saved["intervention_count"]
        del saved["manual_artifact_task"]
        state = WorkflowState.model_validate(saved)
        assert state.autonomy.mode == ExecutionMode.MANUAL
        assert state.intervention is None


class TestWhatAPolicyMayDo:
    def test_only_the_step_gates_can_be_authorized_in_advance(self):
        assert AUTONOMOUS_GATES == {
            ProductPhase.AWAITING_STEP_START,
            ProductPhase.AWAITING_STEP_APPROVAL,
        }
        assert AUTONOMOUS_GATES < HUMAN_GATES

    def test_autonomy_adds_no_route_around_the_approval_gate(self):
        sources = {
            phase
            for phase, targets in ALLOWED_TRANSITIONS.items()
            if ProductPhase.IMPLEMENTING in targets
        }
        assert sources == {ProductPhase.AWAITING_STEP_APPROVAL, ProductPhase.BLOCKED}

    def test_new_human_actions_require_a_decision_argument(self):
        for name in (
            "change_autonomy",
            "verify_manual_fix",
            "approve_test_revision",
            "reject_test_revision",
        ):
            assert "decision" in inspect.signature(getattr(ProductWorkflow, name)).parameters

    async def test_the_name_gate_is_never_passed_by_policy(self, harness):
        workflow = harness.new_workflow()
        await to_name_gate(workflow)
        workflow.state.autonomy.mode = ExecutionMode.AUTONOMOUS_PRODUCT
        workflow.state.autonomy.authorization_id = "x"
        calls = len(harness.team.calls)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_NAME
        assert len(harness.team.calls) == calls

    async def test_a_policy_needs_a_recorded_authorization_to_act(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.state.autonomy.mode = ExecutionMode.AUTONOMOUS_PRODUCT

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START

    async def test_an_approval_citing_no_real_authorization_stops_implementation(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(PolicyDecision("never-granted", "forged"))

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.kind == BlockKind.INTEGRITY
        assert "never gave" in workflow.state.blocked.message
        assert harness.team.calls_of("implement") == []

    async def test_a_withdrawn_record_is_not_an_authorization(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        enable_autonomy(workflow)
        granted = workflow.state.autonomy.authorization_id
        workflow.change_autonomy(HumanDecision("off"))
        withdrawn = next(
            item for item in workflow.store.read_approvals() if item.decision == "withdrawn"
        )
        assert withdrawn.authorization_id == granted
        workflow.approve_step_proposal(PolicyDecision(withdrawn.approval_id, "forged"))

        await workflow.advance()

        assert workflow.state.blocked.kind == BlockKind.INTEGRITY


class TestEnablingAtTheRoadmapGate:
    async def test_it_arms_the_mode_but_does_not_approve_the_roadmap(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        calls = len(harness.team.calls)

        message = enable_autonomy(workflow)
        await workflow.advance()

        assert "does not approve the roadmap" in message
        assert workflow.autonomy_active
        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL
        assert workflow.state.approved_roadmap_version is None
        assert len(harness.team.calls) == calls

    async def test_approving_the_roadmap_then_starts_the_run(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        enable_autonomy(workflow)

        workflow.approve_roadmap(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_the_roadmap_gate_says_how_the_team_will_run(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        enable_autonomy(workflow)
        assert "Autonomous, complete product" in workflow.gate().question

    @pytest.mark.parametrize("reach", [None, to_name_gate])
    async def test_there_is_nothing_to_authorize_before_a_roadmap(self, harness, reach):
        workflow = harness.new_workflow()
        if reach is not None:
            await reach(workflow)
        with pytest.raises(GateViolation, match="no roadmap"):
            enable_autonomy(workflow)
        assert not workflow.autonomy_active


class TestEnablingAtLaterGates:
    async def test_at_the_step_start_gate(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)

        enable_autonomy(workflow)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_at_a_development_proposal_it_covers_that_proposal(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        message = enable_autonomy(workflow)
        await workflow.advance()

        assert "covers the proposal in front of you" in message
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_after_a_completed_step(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        assert calls_for_step(harness, "step-02") == []

        enable_autonomy(workflow)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE
        assert calls_for_step(harness, "step-02") == STEP_SEQUENCE

    async def test_a_step_finishes_before_autonomy_can_claim_a_finished_roadmap(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.state.autonomy.mode = ExecutionMode.AUTONOMOUS_PRODUCT
        workflow.state.steps["step-01"].status = StepStatus.COMPLETED
        workflow.state.steps["step-02"].status = StepStatus.COMPLETED
        with pytest.raises(GateViolation, match="already complete"):
            enable_autonomy(workflow)


class TestCompleteProduct:
    async def test_it_never_waits_for_a_person(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE
        assert all(
            workflow.state.steps[step].status == StepStatus.COMPLETED
            for step in ("step-01", "step-02")
        )

    async def test_every_agent_works_in_the_same_order_as_manual_mode(self, tmp_path):
        manual = Harness(tmp_path / "manual")
        flow = manual.new_workflow()
        await complete_first_step(flow)
        flow.start_step(APPROVE)
        await flow.advance()
        flow.approve_step_proposal(APPROVE)
        await flow.advance()

        auto = Harness(tmp_path / "auto")
        run = auto.new_workflow()
        await to_step_start_gate(run)
        enable_autonomy(run)
        await run.advance()

        assert auto.team.kinds() == manual.team.kinds()
        assert phases_of(run) == phases_of(flow)
        assert [call.model for call in auto.team.calls] == [
            call.model for call in manual.team.calls
        ]
        assert calls_for_step(auto, "step-01") == STEP_SEQUENCE

    async def test_it_ends_when_the_roadmap_ends(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert not workflow.autonomy_active
        assert "every roadmap step is complete" in workflow.store.read_text("decisions.md")

    async def test_a_later_addition_to_the_roadmap_needs_approval_again(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        addition = json.loads(roadmap_of(("History", "")))
        addition["steps"][0].update(number=3, dependencies=[])
        harness.team.script("roadmap", Text(json.dumps(addition)))

        workflow.request_roadmap_change(HumanDecision("Add a history screen"))
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL

    async def test_proposals_stay_in_the_audit_trail(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        for step in ("step-01", "step-02"):
            for name in ("proposal", "architect", "developer", "test_plan", "step_brief"):
                assert workflow.store.exists(f"steps/{step}/proposals/v1/{name}.json")
            assert workflow.store.exists(f"steps/{step}/proposal.md")
        handoffs = (workflow.store.logs_dir / "handoffs.jsonl").read_text().splitlines()
        assert len(handoffs) >= 2 * len(STEP_SEQUENCE)

    async def test_each_pass_of_a_gate_cites_the_human_authorization(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        authorization = workflow.state.autonomy.authorization_id

        await workflow.advance()

        records = workflow.store.read_approvals()
        granted = [item for item in records if item.kind == ApprovalKind.AUTONOMY]
        assert [(item.approval_id, item.decided_by) for item in granted] == [
            (authorization, "human")
        ]
        passed = [
            item
            for item in records
            if item.kind in (ApprovalKind.STEP_START, ApprovalKind.STEP_PROPOSAL)
        ]
        assert len(passed) == 4
        assert all(item.decided_by == "autonomy_policy" for item in passed)
        assert all(item.authorization_id == authorization for item in passed)

    async def test_a_proposal_approval_is_bound_to_the_exact_proposal(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        record = next(
            item
            for item in workflow.store.read_approvals()
            if item.kind == ApprovalKind.STEP_PROPOSAL
        )
        assert record.artifact_sha256 == workflow.store.sha256_of(
            "steps/step-01/proposals/v1/proposal.json"
        )

    async def test_the_human_can_still_see_the_run_afterwards(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        assert "Test Engineer" in workflow.team_activity(limit=30)


class TestQualityIsNotBypassed:
    async def test_tests_are_sealed_exactly_as_in_manual_mode(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()

        sealed = workflow.store.verification_dir / "step-01" / "test_calc.py"
        assert workflow.verification.is_sealed("step-01")
        assert sealed.stat().st_mode & 0o222 == 0
        assert workflow.verification.verify_seal("step-01").proposal_version == 1

    @staticmethod
    def repairs_that_never_work(harness: Harness) -> None:
        harness.team.sources = {"calc.py": CALC_BUGGY}
        for operator in "*/":
            harness.team.script(
                "repair",
                lambda model, prompt, tools, op=operator: (
                    tools[0]("calc.py", f"def add(a, b):\n    return a {op} b\n"), "Done."
                )[1],
            )

    async def test_failing_tests_stop_the_run(self, harness):
        self.repairs_that_never_work(harness)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.kind == BlockKind.VERIFICATION_FAILED
        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED
        assert calls_for_step(harness, "step-02") == []

    async def test_repairs_stay_bounded(self, harness):
        self.repairs_that_never_work(harness)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        limit = harness.policy.workflow.max_repair_cycles
        assert len(harness.team.calls_of("repair")) == limit

    async def test_changing_a_sealed_test_during_a_run_stops_it(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        def tamper(model, prompt, tools):
            sealed = workflow.store.verification_dir / "step-01" / "test_calc.py"
            os.chmod(sealed, 0o644)
            sealed.write_text("def test_nothing():\n    assert True\n")
            tools[0]("calc.py", "def add(a, b):\n    return 0\n")
            return "Done."

        harness.team.script("implement", tamper)

        await workflow.advance()

        assert workflow.state.blocked.kind == BlockKind.INTEGRITY
        assert harness.team.calls_of("assess") == []

    async def test_a_failed_review_is_not_skipped(self, harness):
        harness.policy.workflow.max_repair_cycles = 0
        review = {
            "verdict": "changes_requested",
            "summary": "No.",
            "consistency_with_design": "Off.",
            "findings": [
                {"severity": "major", "description": "Bad.", "recommendation": "Fix."}
            ],
        }
        harness.team.script("review", Text(json.dumps(review)))
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED

    async def test_unmet_completion_conditions_stop_the_run(self, harness):
        closure = harness.team._default("closure", "step-01")
        closure["conditions"][0]["met"] = False
        harness.team.script("closure", Text(json.dumps(closure)))
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert workflow.state.blocked.kind == BlockKind.CONDITIONS_UNMET
        assert calls_for_step(harness, "step-02") == []

    async def test_a_missing_tool_stops_the_run(self, harness, monkeypatch):
        from lab.product import workflow as module
        from lab.tools.verification import VerificationToolMissing

        def missing(*args, **kwargs):
            raise VerificationToolMissing("'node' is required but is not installed.")

        monkeypatch.setattr(module, "run_profile", missing)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert workflow.state.blocked.kind == BlockKind.DETERMINISTIC_FAILURE

    async def test_model_budgets_are_unchanged(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 8)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert len(harness.team.calls_of("architect")) == 4
        assert workflow.state.blocked.kind == BlockKind.MODEL_FAILURE

    async def test_a_blocked_product_does_no_more_work(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        calls = len(harness.team.calls)

        for _ in range(3):
            await workflow.advance()

        assert len(harness.team.calls) == calls
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_a_block_does_not_disable_the_selected_mode(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 1")
        await workflow.advance()

        assert workflow.state.autonomy.mode == ExecutionMode.AUTONOMOUS_CHECKPOINT

    async def test_only_one_inference_runs_at_a_time(self, harness):
        harness.team.delay = 0.001
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        assert harness.team.max_active == 1
        assert harness.team.max_resident <= 1


class TestCheckpoints:
    async def _armed(self, harness, text: str) -> ProductWorkflow:
        three_steps(harness)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, text)
        return workflow

    @pytest.mark.parametrize(
        "text", ["until backend complete", "until the backend", "until step 2", "until Statistics"]
    )
    async def test_it_stops_at_the_requested_point(self, harness, text):
        workflow = await self._armed(harness, text)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert workflow.state.steps["step-02"].status == StepStatus.COMPLETED
        assert workflow.state.steps["step-03"].status == StepStatus.PENDING
        assert calls_for_step(harness, "step-03") == []

    async def test_a_milestone_runs_every_step_it_contains(self, harness):
        workflow = await self._armed(harness, "until backend complete")

        await workflow.advance()

        assert calls_for_step(harness, "step-01") == STEP_SEQUENCE
        assert calls_for_step(harness, "step-02") == STEP_SEQUENCE

    async def test_reaching_it_returns_to_manual_and_says_so(self, harness):
        terminal = ScriptedTerminal()
        three_steps(harness)
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until backend complete")

        await workflow.advance()

        assert not workflow.autonomy_active
        assert any("Checkpoint reached" in line for line in terminal.said)
        assert any(e.type == "AutonomyCheckpointReached" for e in workflow.store.read_events())

    async def test_the_gate_asks_about_the_next_step(self, harness):
        workflow = await self._armed(harness, "until step 2")
        await workflow.advance()
        assert "Shall we proceed to Step 3" in workflow.gate().question

    async def test_continuing_manually_after_a_checkpoint(self, harness):
        workflow = await self._armed(harness, "until step 2")
        await workflow.advance()

        workflow.start_step(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert harness.team.calls_of("implement")[-1].task_id.startswith("step-02")

    async def test_authorizing_more_after_a_checkpoint(self, harness):
        workflow = await self._armed(harness, "until step 1")
        await workflow.advance()
        assert workflow.state.steps["step-02"].status == StepStatus.PENDING

        enable_autonomy(workflow, "until step 2")
        await workflow.advance()
        assert workflow.state.steps["step-02"].status == StepStatus.COMPLETED
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START

        enable_autonomy(workflow)
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_a_checkpoint_on_the_last_step_completes_the_product(self, harness):
        workflow = await self._armed(harness, "until step 3")

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE
        assert not workflow.autonomy_active

    async def test_a_checkpoint_that_leaves_the_roadmap_returns_to_manual(self, harness):
        three_steps(harness)
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        enable_autonomy(workflow, "until step 3")

        workflow.remove_roadmap_step(3, HumanDecision("remove step 3"))

        assert not workflow.autonomy_active
        assert workflow.state.autonomy.checkpoint is None

    async def test_a_checkpoint_follows_its_step_when_the_roadmap_is_reordered(self, harness):
        three_steps(harness)
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        enable_autonomy(workflow, "until Statistics")
        target = workflow.state.autonomy.checkpoint.step_id

        workflow.remove_roadmap_step(1, HumanDecision("remove step 1"))

        assert workflow.autonomy_active
        assert workflow.state.autonomy.checkpoint.step_id == target
        assert "Step 1: Statistics" in workflow.execution_mode_label()

    async def test_a_milestone_checkpoint_follows_a_shorter_milestone(self, harness):
        three_steps(harness)
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        enable_autonomy(workflow, "until backend")

        workflow.remove_roadmap_step(2, HumanDecision("remove step 2"))

        assert workflow.autonomy_active
        assert workflow.state.autonomy.checkpoint.step_id == "step-01"

    async def test_nothing_starts_when_a_checkpoint_does_not_map(self, harness):
        three_steps(harness)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        calls = len(harness.team.calls)

        with pytest.raises(GateViolation, match="cannot map 'authentication'"):
            enable_autonomy(workflow, "until authentication")
        await workflow.advance()

        assert not workflow.autonomy_active
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert len(harness.team.calls) == calls
        assert not any(
            item.kind == ApprovalKind.AUTONOMY for item in workflow.store.read_approvals()
        )

    async def test_a_completed_step_is_not_a_checkpoint(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        with pytest.raises(GateViolation, match="already complete"):
            enable_autonomy(workflow, "until step 1")


def roadmap(*steps: tuple[str, str]) -> Roadmap:
    draft = RoadmapDraft.model_validate_json(roadmap_of(*steps))
    return Roadmap(
        version=1,
        summary=draft.summary,
        steps=[
            RoadmapStep(step_id=f"step-{item.number:02d}", **item.model_dump())
            for item in draft.steps
        ],
        next_step_seq=len(draft.steps) + 1,
    )


class TestCheckpointResolution:
    PLAN = roadmap(
        ("Database design and validation", "Backend"),
        ("Authentication and authorization", "Backend"),
        ("The API layer", "Backend"),
        ("Product catalog screens", "Frontend"),
        ("Checkout screens", "Frontend"),
    )

    def resolve(self, text: str, plan: Roadmap | None = None, state: WorkflowState | None = None):
        return resolve_checkpoint(text, plan or self.PLAN, state or WorkflowState())

    @pytest.mark.parametrize(
        ("text", "step_id"),
        [
            ("step 3", "step-03"),
            ("Step 3", "step-03"),
            ("step-03", "step-03"),
            ("3", "step-03"),
            ("complete step 3", "step-03"),
            ("step 3 is done", "step-03"),
            ("database design and validation", "step-01"),
            ("Database design and validation complete", "step-01"),
            ("authentication and authorization", "step-02"),
            ("the API layer", "step-03"),
            ("api", "step-03"),
            ("checkout", "step-05"),
        ],
    )
    def test_clear_requests_map_to_one_step(self, text, step_id):
        assert self.resolve(text).step.step_id == step_id

    @pytest.mark.parametrize(
        ("text", "milestone", "step_id"),
        [
            ("backend", "Backend", "step-03"),
            ("backend complete", "Backend", "step-03"),
            ("the backend is finished", "Backend", "step-03"),
            ("frontend", "Frontend", "step-05"),
        ],
    )
    def test_a_milestone_ends_with_its_last_step(self, text, milestone, step_id):
        found = self.resolve(text)
        assert (found.milestone, found.step.step_id) == (milestone, step_id)

    def test_several_matching_steps_are_a_question_not_a_guess(self):
        found = self.resolve("screens")
        assert found.step is None
        assert [item.step_id for item in found.candidates] == ["step-04", "step-05"]

    def test_several_matching_milestones_are_a_question(self):
        plan = roadmap(("A", "Backend API"), ("B", "Backend data"))
        found = self.resolve("backend", plan)
        assert found.step is None
        assert [item.step_id for item in found.candidates] == ["step-01", "step-02"]

    def test_several_steps_of_a_name_without_milestones_are_a_question(self):
        plan = roadmap(("Backend models", ""), ("Backend routes", ""))
        assert self.resolve("backend", plan).step is None

    @pytest.mark.parametrize("text", ["payments", "mobile app", "deployment"])
    def test_something_not_on_the_roadmap_does_not_map(self, text):
        assert self.resolve(text).step is None

    def test_a_word_only_in_a_description_is_suggested_never_assumed(self):
        plan = roadmap(("Login", ""), ("Reports", ""))
        plan.steps[0].objective = "Add authentication for staff."
        found = self.resolve("authentication", plan)
        assert found.step is None
        assert [item.step_id for item in found.candidates] == ["step-01"]

    @pytest.mark.parametrize("text", ["done", "complete", "the"])
    def test_words_about_finishing_name_nothing(self, text):
        found = self.resolve(text)
        assert found.step is None
        assert "does not name" in found.problem

    def test_an_unknown_step_number_says_so(self):
        assert "no step 9" in self.resolve("step 9").problem

    def test_the_current_step(self):
        state = WorkflowState(current_step_id="step-02")
        assert self.resolve("current step", state=state).step.step_id == "step-02"
        assert self.resolve("this step", state=state).step.step_id == "step-02"

    def test_the_next_step_while_one_is_in_progress(self):
        state = WorkflowState(current_step_id="step-02")
        assert self.resolve("next step", state=state).step.step_id == "step-03"

    def test_the_next_step_between_steps_is_the_next_pending_one(self):
        assert self.resolve("next step").step.step_id == "step-01"

    def test_the_current_module_ends_with_its_milestone(self):
        state = WorkflowState(current_step_id="step-02")
        found = self.resolve("current module", state=state)
        assert (found.milestone, found.step.step_id) == ("Backend", "step-03")

    def test_the_current_module_without_milestones_is_a_question(self):
        plan = roadmap(("Login", ""), ("Reports", ""))
        found = self.resolve("current module", plan)
        assert found.step is None
        assert "named milestone" in found.problem

    def test_a_finished_milestone_cannot_be_a_checkpoint(self):
        from lab.schemas.product import StepProgress

        state = WorkflowState(
            steps={"step-03": StepProgress(step_id="step-03", status=StepStatus.COMPLETED)}
        )
        assert "already complete" in self.resolve("backend", state=state).problem

    def test_the_question_lists_the_choices_and_how_to_answer(self):
        question = clarification("screens", self.resolve("screens"), self.PLAN)
        assert "Step 4: Product catalog screens" in question
        assert "Step 5: Checkout screens" in question
        assert "/autonomy until step" in question
        assert "nothing has started" in question

    def test_with_no_candidates_the_question_lists_the_roadmap(self):
        question = clarification("payments", self.resolve("payments"), self.PLAN)
        assert "Step 1: Database design and validation" in question

    def test_a_plural_matches_its_singular(self):
        plan = roadmap(("Report", ""), ("Login", ""))
        assert self.resolve("reports", plan).step.step_id == "step-01"

    def test_part_of_a_word_is_not_a_match(self):
        plan = roadmap(("Reporting", ""), ("Login", ""))
        assert self.resolve("port", plan).step is None


class TestReadingTheCommand:
    @pytest.mark.parametrize("text", ["on", "ON", "full", "enable"])
    def test_on(self, text):
        assert parse_autonomy_request(text).action == "on"

    @pytest.mark.parametrize("text", ["off", "manual", "disable"])
    def test_off(self, text):
        assert parse_autonomy_request(text).action == "off"

    @pytest.mark.parametrize("text", ["", "status", "show"])
    def test_status(self, text):
        assert parse_autonomy_request(text).action == "status"

    @pytest.mark.parametrize(
        ("text", "target"),
        [
            ("until backend complete", "backend complete"),
            ("Until Step 3.", "Step 3"),
            ("through the API layer", "the API layer"),
            ("up to step 2", "step 2"),
        ],
    )
    def test_until(self, text, target):
        request = parse_autonomy_request(text)
        assert (request.action, request.target) == ("until", target)

    @pytest.mark.parametrize("text", ["maybe", "until", "yes please", "on and off"])
    def test_anything_else_explains_the_usage(self, text):
        with pytest.raises(ValueError, match="/autonomy on"):
            parse_autonomy_request(text)

    def test_the_slash_command_reaches_the_parser_with_its_arguments(self):
        parsed = parse_input("/autonomy until backend complete")
        assert (parsed.intent, parsed.text) == (Intent.AUTONOMY, "until backend complete")

    def test_new_commands_are_recognised(self):
        assert parse_input("/verify").intent == Intent.VERIFY
        assert parse_input("/report").intent == Intent.REPORT
        assert parse_input("/stop").intent == Intent.STOP

    def test_plain_words_never_enable_autonomy(self):
        for text in ("autonomy on", "turn on autonomy", "yes run everything"):
            assert parse_input(text).intent != Intent.AUTONOMY

    async def test_a_bad_command_changes_nothing(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        with pytest.raises(GateViolation) as caught:
            enable_autonomy(workflow, "sometimes")
        assert USAGE in str(caught.value)
        assert not workflow.autonomy_active


class TestSwitchingModes:
    async def test_switching_off_restores_the_step_gates(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 2")
        workflow.change_autonomy(HumanDecision("off"))

        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START

        workflow.start_step(APPROVE)
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert harness.team.calls_of("implement") == []

    async def test_switching_off_when_already_manual_says_so(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        assert "already Manual" in workflow.change_autonomy(HumanDecision("off"))
        assert not any(
            item.kind == ApprovalKind.AUTONOMY for item in workflow.store.read_approvals()
        )

    async def test_a_new_authorization_replaces_the_old_checkpoint(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 1")
        first = workflow.state.autonomy.authorization_id

        enable_autonomy(workflow)

        assert workflow.state.autonomy.mode == ExecutionMode.AUTONOMOUS_PRODUCT
        assert workflow.state.autonomy.checkpoint is None
        assert workflow.state.autonomy.authorization_id != first

    async def test_the_mode_can_change_at_a_blocked_gate(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()

        workflow.change_autonomy(HumanDecision("off"))

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert not workflow.autonomy_active

    async def test_every_change_is_audited(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 1")
        workflow.change_autonomy(HumanDecision("off"))

        changes = [e for e in workflow.store.read_events() if e.type == "AutonomyChanged"]
        assert [e.payload["mode"] for e in changes] == ["autonomous_checkpoint", "manual"]
        records = [
            item.decision
            for item in workflow.store.read_approvals()
            if item.kind == ApprovalKind.AUTONOMY
        ]
        assert records == ["approved", "withdrawn"]
        assert "Execution mode changed by the user" in workflow.store.read_text("decisions.md")

    async def test_status_on_its_own_changes_nothing(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        message = workflow.change_autonomy(HumanDecision("status"))
        assert "Manual" in message
        assert not workflow.autonomy_active


class TestChangesAPersonMade:
    """Advance authorization covers what the team proposes, not a new human direction."""

    async def test_a_changed_proposal_comes_back_to_the_person(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.team.repaired_sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        assert workflow.state.blocked.kind == BlockKind.VERIFICATION_FAILED

        await workflow.request_step_modification(HumanDecision("Change the acceptance criteria"))
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert workflow.autonomy_active
        assert len(harness.team.calls_of("implement")) == 1

    async def test_approving_it_lets_the_run_continue(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        await workflow.request_step_modification(HumanDecision("Use subtraction"))
        await workflow.advance()
        harness.team.sources = {"calc.py": "def add(a, b):\n    return a + b\n"}

        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_a_rejected_proposal_is_not_rebuilt_and_approved_by_policy(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.reject_step_proposal(HumanDecision("Not this design"))
        enable_autonomy(workflow)
        calls = len(harness.team.calls)

        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert len(harness.team.calls) == calls

        workflow.start_step(APPROVE)
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    async def test_the_confirmation_does_not_claim_to_cover_a_held_proposal(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        await workflow.request_step_modification(HumanDecision("Use a class"))
        await workflow.advance()

        message = enable_autonomy(workflow)

        assert "covers the proposal" not in message


class TestRestart:
    async def test_the_mode_survives_a_restart(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 1")
        policy = workflow.state.autonomy

        restarted = harness.restart(workflow)

        assert restarted.state.autonomy == policy

    async def test_a_restarted_run_keeps_going_to_its_checkpoint(self, harness):
        three_steps(harness)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until backend")
        restarted = harness.restart(workflow)

        await restarted.advance()

        assert restarted.state.steps["step-02"].status == StepStatus.COMPLETED
        assert restarted.state.steps["step-03"].status == StepStatus.PENDING

    async def test_power_loss_during_a_run_loses_neither_mode_nor_work(self, harness):
        harness.team.script("assess", Fail(SimulatedPowerLoss()))
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()
        implemented = len(harness.team.calls_of("implement"))

        restarted = harness.restart(workflow)
        assert restarted.autonomy_active
        await restarted.advance()

        assert restarted.state.phase == ProductPhase.PRODUCT_COMPLETE
        # Step 2 adds one implementation.  Step 1's was not repeated.
        assert len(harness.team.calls_of("implement")) == implemented + 1

    async def test_the_previous_checkpoint_still_carries_the_mode(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        workflow._save()
        workflow.store.state_path.write_text("{ corrupted")

        restored = ProductStore(workflow.store.root).load_state()

        assert restored.autonomy.mode == ExecutionMode.AUTONOMOUS_PRODUCT

    async def test_a_manual_return_survives_a_restart(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        workflow.change_autonomy(HumanDecision("off"))

        restarted = harness.restart(workflow)
        await restarted.advance()

        assert restarted.state.phase == ProductPhase.AWAITING_STEP_START
        assert not restarted.autonomy_active


class TestShowingTheMode:
    async def test_status_shows_manual(self, harness):
        workflow = harness.new_workflow()
        assert "Execution mode: Manual" in workflow.status_report()

    async def test_status_shows_a_full_run(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        assert "Execution mode: Autonomous, complete product" in workflow.status_report()

    async def test_status_shows_the_checkpoint(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 2")
        assert "Execution mode: Autonomous, until Step 2: Statistics" in workflow.status_report()

    async def test_status_shows_the_milestone(self, harness):
        three_steps(harness)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until backend")
        assert "until the Backend milestone (ends with Step 2: Statistics)" in (
            workflow.status_report()
        )

    async def test_progress_lines_show_the_mode_during_a_run(self, harness):
        terminal = ScriptedTerminal()
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await workflow.advance()

        started = [line for line in terminal.statuses if "Starting Step 1" in line]
        assert started and "Execution mode: Autonomous, complete product" in started[0]
        assert any("steps/step-01/proposal.md" in line for line in terminal.statuses)

    async def test_the_product_list_shows_only_a_non_default_mode(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        assert "Execution mode" not in harness.manager.discover()[0][0].render()

        enable_autonomy(workflow)

        assert "Execution mode: Autonomous" in harness.manager.discover()[0][0].render()

    def test_describing_a_checkpoint_without_its_roadmap_still_works(self):
        from lab.schemas.product import AutonomyCheckpoint, AutonomyPolicy

        policy = AutonomyPolicy(
            mode=ExecutionMode.AUTONOMOUS_CHECKPOINT,
            checkpoint=AutonomyCheckpoint(step_id="step-04", requested="x"),
        )
        assert "step-04" in describe_mode(policy, None)


@pytest.fixture
def fast_controls(monkeypatch):
    monkeypatch.setattr("lab.product.session.CONTROL_POLL_SECONDS", 0.001)


class TestControlDuringARun:
    """The session used to be deaf while the team worked.  These prove it is not."""

    async def _session(self, harness, *answers):
        session, terminal = session_for(harness, *answers)
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_step_start_gate(workflow)
        return session, terminal, workflow

    @staticmethod
    def once(terminal: ScriptedTerminal, when, line: str) -> None:
        sent = []

        def poll() -> str | None:
            if not sent and when():
                sent.append(line)
                return line
            return None

        terminal.on_poll = poll

    async def test_turning_autonomy_off_stops_at_the_next_gate(self, harness):
        session, terminal, workflow = await self._session(harness, "/exit")
        enable_autonomy(workflow)
        self.once(terminal, lambda: harness.team.calls_of("implement"), "/autonomy off")

        with pytest.raises(BaseException) as leaving:
            await session._product_loop(workflow, None)

        assert type(leaving.value).__name__ == "_ExitSession"
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert calls_for_step(harness, "step-02") == []
        assert not workflow.autonomy_active

    async def test_a_running_model_call_is_never_interrupted(self, harness, fast_controls):
        harness.team.delay = 0.05
        session, terminal, workflow = await self._session(harness, "/exit")
        enable_autonomy(workflow)
        self.once(terminal, lambda: harness.team.active > 0, "/autonomy off")

        with pytest.raises(BaseException):
            await session._product_loop(workflow, None)

        outcomes = [
            attempt.status
            for call in harness.team.calls
            for attempt in workflow._attempts_of(call.task_id)
        ]
        assert outcomes and set(outcomes) == {"succeeded"}
        changed = next(
            e for e in workflow.store.read_events()
            if e.type == "AutonomyChanged" and e.payload["mode"] == "manual"
        )
        assert changed.phase == ProductPhase.STEP_PREPARING.value

    async def test_the_change_is_saved_at_once_while_the_model_runs(self, harness, fast_controls):
        harness.team.delay = 0.05
        session, terminal, workflow = await self._session(harness, "/exit")
        enable_autonomy(workflow)
        self.once(terminal, lambda: harness.team.active > 0, "/autonomy off")
        seen: list[str] = []

        task = asyncio.ensure_future(session._advance(workflow))
        while not workflow.store.read_text("decisions.md").count("returned to Manual"):
            await asyncio.sleep(0.001)
        seen.append(ProductStore(workflow.store.root).load_state().autonomy.mode.value)
        await task

        assert seen == ["manual"]

    async def test_narrowing_a_run_to_a_checkpoint(self, harness):
        session, terminal, workflow = await self._session(harness, "/exit")
        enable_autonomy(workflow)
        terminal.typed.append("/autonomy until step 1")

        with pytest.raises(BaseException):
            await session._product_loop(workflow, None)

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert calls_for_step(harness, "step-02") == []

    async def test_a_confusing_checkpoint_during_a_run_leaves_it_alone(self, harness):
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow)
        terminal.typed.append("/autonomy until payments")

        await session._advance(workflow)

        assert any("cannot map 'payments'" in line for line in terminal.warnings)
        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_pause_stops_at_the_next_phase(self, harness):
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow)
        self.once(terminal, lambda: harness.team.calls_of("architect"), "/pause")

        with pytest.raises(_LeaveProduct):
            await session._product_loop(workflow, None)

        assert workflow.state.paused
        assert workflow.state.phase == ProductPhase.STEP_DEVELOPER_PLAN
        assert harness.team.calls_of("developer_plan") == []
        assert workflow.autonomy_active

    async def test_pause_stops_between_two_planned_files(self, harness):
        harness.team.sources = {"calc.py": "def add(a, b):\n    return a + b\n", "extra.py": "x = 1\n"}
        plan = harness.team._default("developer_plan", "step-01")
        plan["file_changes"].append({"path": "extra.py", "action": "create", "purpose": "More."})
        harness.team.script("developer_plan", Text(json.dumps(plan)))
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow, "until step 1")
        self.once(terminal, lambda: harness.team.calls_of("implement"), "/pause")

        with pytest.raises(_LeaveProduct):
            await session._product_loop(workflow, None)

        assert workflow.state.phase == ProductPhase.IMPLEMENTING
        assert len(harness.team.calls_of("implement")) == 1
        tasks = workflow.state.steps["step-01"].implementation_tasks
        assert [item.status for item in tasks] == ["done", "pending"]

        restarted = harness.restart(workflow)
        await restarted.advance()

        assert len(harness.team.calls_of("implement")) == 2
        assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED

    async def test_stop_ends_the_authorization_too(self, harness):
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow)
        self.once(terminal, lambda: harness.team.calls_of("architect"), "/stop")

        with pytest.raises(_LeaveProduct):
            await session._product_loop(workflow, None)

        assert workflow.state.paused
        assert not workflow.autonomy_active

    async def test_exit_leaves_at_a_safe_point(self, harness):
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow)
        self.once(terminal, lambda: harness.team.calls_of("architect"), "/exit")

        with pytest.raises(BaseException) as leaving:
            await session._product_loop(workflow, None)

        assert type(leaving.value).__name__ == "_ExitSession"
        assert harness.team.calls_of("developer_plan") == []
        assert workflow.autonomy_active

    async def test_status_answers_during_a_run(self, harness):
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow)
        terminal.typed.append("/status")

        await session._advance(workflow)

        assert any("Execution mode: Autonomous" in line for line in terminal.said)

    async def test_an_approval_typed_during_a_run_means_nothing(self, harness):
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow)
        terminal.typed.extend(["yes", "/approve", "make it blue"])

        await session._advance(workflow)

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE
        assert sum("The team is working" in line for line in terminal.said) == 3
        assert not any("make it blue" in call.prompt for call in harness.team.calls)

    async def test_the_person_is_told_what_they_can_type(self, harness):
        session, terminal, workflow = await self._session(harness)
        enable_autonomy(workflow)

        await session._advance(workflow)

        assert any("/autonomy off" in line and "/pause" in line for line in terminal.statuses)

    async def test_manual_mode_reads_nothing_during_a_run(self, harness):
        session, terminal, workflow = await self._session(harness)
        terminal.typed.append("/autonomy on")
        workflow.start_step(APPROVE)

        await session._advance(workflow)

        assert terminal.typed == ["/autonomy on"]
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    async def test_a_terminal_without_controls_still_runs(self, harness):
        class Plain(ScriptedTerminal):
            poll_line = None  # type: ignore[assignment]

        terminal = Plain()
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)

        await session._advance(workflow)

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE


class TestPausedRunsNeedAWord:
    async def _paused(self, harness, *answers):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        workflow.pause(HumanDecision("/pause"))
        session, terminal = session_for(harness, *answers)
        return session, terminal, workflow

    async def test_reopening_holds_until_the_person_says_so(self, harness):
        session, terminal, workflow = await self._paused(harness, "/exit")
        calls = len(harness.team.calls)

        await session.run_product(session.open_product(workflow.store.slug))

        assert len(harness.team.calls) == calls
        assert ProductStore(workflow.store.root).load_state().paused
        assert any("You paused this product" in line for line in terminal.prompts)

    async def test_yes_continues_autonomously(self, harness):
        session, terminal, workflow = await self._paused(harness, "yes", "no")

        await session.run_product(session.open_product(workflow.store.slug))

        state = ProductStore(workflow.store.root).load_state()
        assert state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_returning_to_manual_first(self, harness):
        session, terminal, workflow = await self._paused(harness, "/autonomy off", "/exit")

        await session.run_product(session.open_product(workflow.store.slug))

        state = ProductStore(workflow.store.root).load_state()
        assert state.phase == ProductPhase.AWAITING_STEP_START
        assert state.autonomy.mode == ExecutionMode.MANUAL
        assert not state.paused

    async def test_a_crash_is_not_a_pause(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        session, terminal = session_for(harness, "no")

        await session.run_product(session.open_product(workflow.store.slug))

        state = ProductStore(workflow.store.root).load_state()
        assert state.phase == ProductPhase.PRODUCT_COMPLETE

    @staticmethod
    async def press_ctrl_c_during_a_model_call(harness, session, workflow) -> None:
        """run_interruptibly turns the key press into this cancellation of the session."""
        harness.team.delay = 0.05
        running = asyncio.ensure_future(session.open_product(workflow.store.slug))
        while not harness.team.active:
            await asyncio.sleep(0.001)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running

    async def test_ctrl_c_during_a_run_counts_as_stopping_it(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        session, terminal = session_for(harness)

        await self.press_ctrl_c_during_a_model_call(harness, session, workflow)

        state = ProductStore(workflow.store.root).load_state()
        assert state.paused
        assert state.autonomy.mode == ExecutionMode.AUTONOMOUS_PRODUCT
        assert not workflow.store.lock_path.exists()

    async def test_ctrl_c_in_manual_mode_is_unchanged(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        await workflow.request_step_modification(HumanDecision("A change"))
        session, terminal = session_for(harness)

        await self.press_ctrl_c_during_a_model_call(harness, session, workflow)

        state = ProductStore(workflow.store.root).load_state()
        assert not state.paused
        assert state.phase == ProductPhase.STEP_ARCHITECT
        assert not workflow.store.lock_path.exists()


class TestTheConsoleChannel:
    """Polling input for real, through a pipe, because this path has no fake."""

    def console(self) -> tuple[ConsoleTerminal, int]:
        read_end, write_end = os.pipe()
        return ConsoleTerminal(control_fd=read_end, read_line=lambda: "typed"), write_end

    def test_nothing_typed_returns_at_once(self):
        terminal, write_end = self.console()
        assert terminal.poll_line() is None
        os.close(write_end)

    def test_one_line(self):
        terminal, write_end = self.console()
        os.write(write_end, b"/status\n")
        assert terminal.poll_line() == "/status"
        assert terminal.poll_line() is None
        os.close(write_end)

    def test_two_lines_at_once(self):
        terminal, write_end = self.console()
        os.write(write_end, b"/status\n/pause\n")
        assert terminal.poll_line() == "/status"
        assert terminal.poll_line() == "/pause"
        os.close(write_end)

    def test_half_a_line_waits_for_the_rest(self):
        terminal, write_end = self.console()
        os.write(write_end, b"/sta")
        assert terminal.poll_line() is None
        os.write(write_end, b"tus\n")
        assert terminal.poll_line() == "/status"
        os.close(write_end)

    def test_closed_input_is_quiet(self):
        terminal, write_end = self.console()
        os.close(write_end)
        assert terminal.poll_line() is None
        assert terminal.poll_line() is None

    def test_a_line_left_over_reaches_the_next_question(self):
        terminal, write_end = self.console()
        os.write(write_end, b"/status\nyes\n")
        assert terminal.poll_line() == "/status"
        assert terminal.ask("Approve?") == "yes"
        assert terminal.ask("Approve?") == "typed"
        os.close(write_end)

    def test_an_injected_reader_is_not_polled(self):
        terminal = ConsoleTerminal(read_line=lambda: "x")
        assert terminal.poll_line() is None

    def test_polling_starts_no_thread(self):
        import threading

        terminal, write_end = self.console()
        before = threading.active_count()
        terminal.poll_line()
        assert threading.active_count() == before
        os.close(write_end)
