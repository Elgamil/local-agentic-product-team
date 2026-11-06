"""Both features together, driven through the real terminal session.

Each test types what a person would type.  The models are scripted, and everything
else, from checkpoints to test runs to report files, is the real thing.
"""

from __future__ import annotations

import json
import os

import pytest

from lab.product.session import ProductSession, _ExitSession
from lab.product.store import ProductStore
from lab.product.workflow import HumanDecision, VerifyStatus
from lab.schemas.product import (
    ApprovalKind,
    BlockKind,
    ExecutionMode,
    ProductPhase,
    StepStatus,
)
from tests.fakes import (
    APPROVE,
    BACKEND_THEN_FRONTEND,
    CALC_SOURCE,
    CALC_TEST,
    Harness,
    ScriptedTerminal,
    Text,
    enable_autonomy,
    roadmap_of,
    to_step_start_gate,
    to_roadmap_gate,
)

NOT_JSON = "this is not json"
FULL_STEP = [
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
    harness = Harness(tmp_path)
    harness.team.script("roadmap", Text(roadmap_of(*BACKEND_THEN_FRONTEND)))
    return harness


def session_for(harness: Harness, *answers: str) -> tuple[ProductSession, ScriptedTerminal]:
    terminal = ScriptedTerminal(*answers)
    return (
        ProductSession(harness.manager, harness.router, harness.policy, harness.playbooks, terminal),
        terminal,
    )


def step_calls(harness: Harness, step_id: str) -> list[str]:
    return [call.kind for call in harness.team.calls if call.task_id.startswith(step_id)]


def state_of(workflow):
    return ProductStore(workflow.store.root).load_state()


async def leaving(session, workflow) -> None:
    with pytest.raises(_ExitSession):
        await session._product_loop(workflow, None)


def write_test_by_hand(workflow) -> None:
    target = workflow.store.verification_dir / "step-01" / "test_calc.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CALC_TEST)


class TestFailureRecoveryAndContinuationToACheckpoint:
    """The example in the brief, from start to finish."""

    async def test_the_whole_story(self, harness):
        harness.team.script("test_file", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)

        # 1. The Test Engineer fails all four attempts, during the backend milestone.
        session, terminal = session_for(harness, "/autonomy until backend complete", "/exit")
        await session._on_gate(workflow, terminal.answers.pop(0))
        await leaving(session, workflow)

        state = state_of(workflow)
        assert state.phase == ProductPhase.BLOCKED
        assert len(harness.team.calls_of("test_file")) == 4
        assert harness.team.calls_of("implement") == []
        assert any("HUMAN INTERVENTION REQUIRED" in line for line in terminal.said)
        report = workflow.store.read_text("interventions/INT-001/report.md")
        assert "verification/step-01/test_calc.py" in report

        # 2. The mode survived the block, and leaving the application.
        assert state.autonomy.mode == ExecutionMode.AUTONOMOUS_CHECKPOINT
        assert not workflow.store.lock_path.exists()

        # 3. The person writes the file with the application closed, and comes back.
        write_test_by_hand(workflow)
        session, terminal = session_for(harness, "/verify", "/exit")
        await session.run_product(session.open_product(workflow.store.slug))

        # 4. It continued by itself, through both backend steps, and stopped there.
        state = state_of(workflow)
        assert state.steps["step-01"].status == StepStatus.COMPLETED
        assert state.steps["step-02"].status == StepStatus.COMPLETED
        assert state.steps["step-03"].status == StepStatus.PENDING
        assert state.phase == ProductPhase.AWAITING_STEP_START
        assert step_calls(harness, "step-03") == []
        assert any("Checkpoint reached" in line for line in terminal.said)
        assert any("Shall we proceed to Step 3" in line for line in terminal.prompts)
        assert state.autonomy.mode == ExecutionMode.MANUAL

    async def test_every_agent_still_ran_around_the_failure(self, harness):
        harness.team.script("test_file", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until backend")
        await workflow.advance()
        write_test_by_hand(workflow)

        workflow.verify_manual_fix(HumanDecision("/verify"))
        await workflow.advance()

        kinds = [kind for kind in step_calls(harness, "step-01") if kind != "test_file"]
        assert kinds == [kind for kind in FULL_STEP if kind != "test_file"]
        assert step_calls(harness, "step-02") == FULL_STEP

    async def test_a_failure_after_approval_recovers_the_same_way(self, harness):
        harness.team.script("implement", *[Text("I will describe it instead.")] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until backend")
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.resume_phase == ProductPhase.IMPLEMENTING

        (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)
        outcome = workflow.verify_manual_fix(HumanDecision("/verify"))
        await workflow.advance()

        assert outcome.status == VerifyStatus.PASSED
        assert workflow.state.steps["step-02"].status == StepStatus.COMPLETED
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert workflow.state.steps["step-03"].status == StepStatus.PENDING

    async def test_a_failure_across_a_restart_and_a_failed_first_try(self, harness):
        harness.team.script("implement", *[Text("I will describe it instead.")] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 1")
        await workflow.advance()

        restarted = harness.restart(workflow)
        (restarted.store.workspace_dir / "calc.py").write_text("def add(a, b):\n    return 0\n")
        assert restarted.verify_manual_fix(HumanDecision("/verify")).status == VerifyStatus.FAILED
        assert restarted.autonomy_active

        restarted = harness.restart(restarted)
        (restarted.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)
        assert restarted.verify_manual_fix(HumanDecision("/verify")).status == VerifyStatus.PASSED
        await restarted.advance()

        assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED
        assert restarted.state.steps["step-02"].status == StepStatus.PENDING
        assert restarted.state.phase == ProductPhase.AWAITING_STEP_START

    async def test_revised_tests_still_stop_for_a_person_mid_run(self, harness):
        harness.policy.workflow.max_repair_cycles = 0
        harness.team.sources = {"calc.py": "def add(a, b):\n    return 0\n"}
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until backend")
        await workflow.advance()
        assert workflow.state.blocked.kind == BlockKind.VERIFICATION_FAILED

        (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)
        sealed = workflow.store.verification_dir / "step-01" / "test_calc.py"
        os.chmod(sealed, 0o644)
        sealed.write_text(CALC_TEST + "\n# corrected\n")
        session, terminal = session_for(harness, "/verify", "yes", "/exit")
        await leaving(session, workflow)

        assert workflow.state.steps["step-02"].status == StepStatus.COMPLETED
        records = workflow.store.read_approvals()
        revision = next(item for item in records if item.kind == ApprovalKind.TEST_REVISION)
        assert revision.decided_by == "human"
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START


class TestSeveralStepsWithoutAPerson:
    async def test_the_whole_roadmap_from_the_roadmap_gate(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        session, terminal = session_for(harness, "/autonomy on", "yes", "no")

        with pytest.raises(BaseException) as leaving_session:
            await session._product_loop(workflow, None)

        assert type(leaving_session.value).__name__ == "_LeaveProduct"
        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE
        for step in ("step-01", "step-02", "step-03"):
            assert step_calls(harness, step) == FULL_STEP
        prompts = " ".join(terminal.prompts)
        assert "Do you approve this proposal" not in prompts
        assert "Shall we" not in prompts
        assert harness.team.max_active == 1

    async def test_status_while_it_runs_and_afterwards(self, harness):
        session, terminal = session_for(harness, "/autonomy until step 2", "/status", "yes", "/exit")
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_roadmap_gate(workflow)

        await leaving(session, workflow)

        assert any("Autonomous, until Step 2: Statistics" in line for line in terminal.said)
        assert any("Checkpoint reached" in line for line in terminal.said)

    async def test_a_checkpoint_that_does_not_map_asks_first(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        workflow.approve_roadmap(APPROVE)
        calls = len(harness.team.calls)
        session, terminal = session_for(harness, "/autonomy until payments", "/exit")

        await leaving(session, workflow)

        assert any("cannot map 'payments'" in line for line in terminal.warnings)
        assert len(harness.team.calls) == calls


class TestRecoveryKeepsTheChosenMode:
    async def test_manual_recovery_leaves_full_autonomy_on(self, harness):
        harness.team.script("implement", *[Text("I will describe it instead.")] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        authorization = workflow.state.autonomy.authorization_id
        await workflow.advance()

        (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)
        workflow.verify_manual_fix(HumanDecision("/verify"))
        assert workflow.state.autonomy.authorization_id == authorization
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE
        assert step_calls(harness, "step-03") == FULL_STEP

    async def test_an_ai_retry_also_leaves_it_on(self, harness):
        harness.team.script("architect", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow, "until step 1")
        await workflow.advance()

        workflow.retry_blocked(HumanDecision("/retry"))
        await workflow.advance()

        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.steps["step-02"].status == StepStatus.PENDING

    async def test_a_failed_check_does_not_turn_it_off(self, harness):
        harness.team.script("implement", *[Text("I will describe it instead.")] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()

        assert workflow.verify_manual_fix(HumanDecision("/verify")).status == VerifyStatus.FAILED

        assert workflow.state.autonomy.mode == ExecutionMode.AUTONOMOUS_PRODUCT


class TestSwitchingToManualAfterRecovery:
    async def test_before_verifying(self, harness):
        harness.team.script("implement", *[Text("I will describe it instead.")] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)
        session, terminal = session_for(harness, "/autonomy off", "/verify", "/exit")

        await leaving(session, workflow)

        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert step_calls(harness, "step-02") == []
        assert not workflow.autonomy_active

    async def test_after_verifying_it_stops_at_the_next_gate(self, harness):
        harness.team.script("implement", *[Text("I will describe it instead.")] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)
        workflow.verify_manual_fix(HumanDecision("/verify"))

        workflow.change_autonomy(HumanDecision("off"))
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert step_calls(harness, "step-02") == []

    async def test_manual_gates_then_work_as_usual(self, harness):
        harness.team.script("implement", *[Text("I will describe it instead.")] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        enable_autonomy(workflow)
        await workflow.advance()
        workflow.change_autonomy(HumanDecision("off"))
        (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)
        workflow.verify_manual_fix(HumanDecision("/verify"))
        await workflow.advance()

        workflow.start_step(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        records = [
            json.loads(line)
            for line in (workflow.store.logs_dir / "approvals.jsonl").read_text().splitlines()
        ]
        assert records[-1]["kind"] == "step_start"
        assert records[-1]["decided_by"] == "human"
