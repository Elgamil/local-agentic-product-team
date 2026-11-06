"""Approval gates: nothing proceeds without the human, and the LLM cannot move state."""

from __future__ import annotations

import inspect

import pytest

from lab.product.workflow import GateViolation, HumanDecision, ProductWorkflow
from lab.schemas.product import (
    ALLOWED_TRANSITIONS,
    HUMAN_GATES,
    ApprovalKind,
    BlockKind,
    ProductPhase,
    Roadmap,
    StepStatus,
    VerificationEvidence,
)
from tests.fakes import (
    APPROVE,
    CALC_BUGGY,
    CALC_SOURCE,
    GEMMA,
    QWEN,
    Fail,
    Harness,
    Silent,
    Text,
    complete_first_step,
    to_name_gate,
    to_proposal_gate,
    to_roadmap_gate,
    to_step_start_gate,
)


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


class TestStateMachine:
    def test_every_phase_has_a_transition_entry(self):
        assert set(ALLOWED_TRANSITIONS) == set(ProductPhase)

    def test_implementation_is_reachable_only_from_the_approval_gate_or_recovery(self):
        sources = {
            phase
            for phase, targets in ALLOWED_TRANSITIONS.items()
            if ProductPhase.IMPLEMENTING in targets
        }
        assert sources == {ProductPhase.AWAITING_STEP_APPROVAL, ProductPhase.BLOCKED}

    async def test_illegal_transitions_are_refused(self, harness):
        workflow = harness.new_workflow()
        with pytest.raises(GateViolation):
            workflow._transition(ProductPhase.IMPLEMENTING)

    def test_a_historical_product_cannot_change_phase(self):
        assert ALLOWED_TRANSITIONS[ProductPhase.HISTORICAL] == frozenset()

    def test_gate_actions_require_a_human_decision_argument(self):
        for name in (
            "confirm_name",
            "approve_roadmap",
            "start_step",
            "approve_step_proposal",
            "reject_step_proposal",
            "retry_blocked",
            "accept_with_limitations",
        ):
            parameters = inspect.signature(getattr(ProductWorkflow, name)).parameters
            assert "decision" in parameters, name


class TestAdvanceStopsAtGates:
    @pytest.mark.parametrize(
        "reach",
        [to_name_gate, to_roadmap_gate, to_step_start_gate, to_proposal_gate],
    )
    async def test_advance_never_passes_a_human_gate(self, harness, reach):
        workflow = harness.new_workflow()
        await reach(workflow)
        assert workflow.state.phase in HUMAN_GATES
        calls_before = len(harness.team.calls)

        for _ in range(5):
            await workflow.advance()

        assert workflow.state.phase in HUMAN_GATES
        assert len(harness.team.calls) == calls_before

    async def test_a_model_saying_ready_does_not_skip_the_name_gate(self, harness):
        workflow = harness.new_workflow()
        await to_name_gate(workflow)
        assert harness.team.calls_of("roadmap") == []

    async def test_no_implementation_before_approval(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        assert harness.team.calls_of("implement") == []
        assert list(workflow.store.workspace_dir.iterdir()) == []

    async def test_a_completed_step_does_not_roll_into_the_next(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert workflow.state.steps["step-02"].status == StepStatus.PENDING
        question = workflow.gate().question
        assert "Step 1 is completed. Shall we proceed to Step 2" in question


class TestWrongPhase:
    async def test_cannot_approve_a_step_while_the_roadmap_awaits(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        with pytest.raises(GateViolation):
            workflow.approve_step_proposal(APPROVE)

    async def test_cannot_start_a_step_before_the_roadmap_is_approved(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        with pytest.raises(GateViolation):
            workflow.start_step(APPROVE)

    async def test_cannot_approve_the_same_proposal_twice(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        with pytest.raises(GateViolation):
            workflow.approve_step_proposal(APPROVE)


class TestDiscoveryAndNaming:
    async def test_discovery_can_take_several_rounds(self, harness):
        harness.team.discovery_ready = False
        workflow = harness.new_workflow()
        await workflow.discovery_message("An idea.")
        await workflow.discovery_message("More details.")
        assert workflow.state.phase == ProductPhase.DISCOVERY
        assert workflow.state.discovery_round == 2

    async def test_discovery_is_bounded(self, harness):
        harness.team.discovery_ready = False
        harness.policy.workflow.discovery_max_rounds = 2
        workflow = harness.new_workflow()
        await workflow.discovery_message("An idea.")
        await workflow.discovery_message("More.")
        assert workflow.state.phase == ProductPhase.AWAITING_NAME
        assert "asked enough" in harness.team.calls_of("discovery")[-1].prompt

    async def test_the_user_can_end_discovery_early(self, harness):
        harness.team.discovery_ready = False
        workflow = harness.new_workflow()
        await workflow.discovery_message("An idea.")
        workflow.finish_discovery(HumanDecision("/done"))
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_NAME

    async def test_conversation_context_is_bounded(self, harness):
        harness.policy.context.max_conversation_turns = 2
        harness.team.discovery_ready = False
        harness.policy.workflow.discovery_max_rounds = 9
        workflow = harness.new_workflow()
        await workflow.discovery_message("first message marker")
        await workflow.discovery_message("second")
        await workflow.discovery_message("third")
        assert "first message marker" not in harness.team.calls_of("discovery")[-1].prompt


class TestRoadmapApproval:
    async def test_approval_sets_the_baseline(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        assert workflow.state.approved_roadmap_version is None
        assert "not yet approved" in workflow.store.read_text("roadmap.md")

        workflow.approve_roadmap(APPROVE)

        assert workflow.state.approved_roadmap_version == 1
        assert "Approved baseline" in workflow.store.read_text("roadmap.md")
        assert "version 1 approved" in workflow.store.read_text("decisions.md")

    async def test_change_request_creates_a_new_version_and_keeps_the_old(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)

        workflow.request_roadmap_change(HumanDecision("Add a history feature"))
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL
        assert workflow.state.roadmap_version == 2
        assert workflow.store.exists("artifacts/roadmap.v1.json")
        assert workflow.store.exists("artifacts/roadmap.v2.json")
        prompt = harness.team.calls_of("roadmap")[-1].prompt
        assert "Add a history feature" in prompt
        assert "Roadmap change requested" in workflow.store.read_text("decisions.md")

    async def test_remove_step_is_deterministic_and_needs_no_model(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        calls = len(harness.team.calls)

        workflow.remove_roadmap_step(1, HumanDecision("remove step 1"))

        assert len(harness.team.calls) == calls
        assert [step.name for step in workflow.roadmap.steps] == ["Statistics"]
        assert workflow.roadmap.steps[0].number == 1
        assert workflow.roadmap.steps[0].dependencies == []
        assert workflow.state.steps["step-01"].status == StepStatus.REMOVED

    async def test_move_step_respects_dependencies(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        with pytest.raises(GateViolation, match="depends on"):
            workflow.move_roadmap_step(2, 1, HumanDecision("move"))
        assert workflow.state.roadmap_version == 1

    async def test_an_edited_roadmap_still_needs_approval(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        workflow.remove_roadmap_step(2, HumanDecision("remove step 2"))
        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL
        assert workflow.state.approved_roadmap_version is None

    async def test_an_approved_baseline_needs_reapproval_after_a_change(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.request_roadmap_change(HumanDecision("Add a step"))
        await workflow.advance()

        assert workflow.state.approved_roadmap_version == 1
        with pytest.raises(GateViolation):
            workflow.start_step(APPROVE)
        workflow.approve_roadmap(APPROVE)
        assert workflow.state.approved_roadmap_version == 2

    async def test_completed_steps_survive_a_roadmap_change(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)

        draft = {
            "summary": "Revised.",
            "steps": [
                {
                    "number": 2,
                    "name": "Reporting",
                    "objective": "Report.",
                    "user_visible_outcome": "A report.",
                    "deliverables": ["report"],
                    "dependencies": [1],
                    "completion_conditions": ["It reports"],
                }
            ],
        }
        import json

        harness.team.script("roadmap", Text(json.dumps(draft)))
        workflow.request_roadmap_change(HumanDecision("Replace the rest with reporting"))
        await workflow.advance()

        roadmap = workflow.roadmap
        assert [step.name for step in roadmap.steps] == ["Core arithmetic", "Reporting"]
        assert roadmap.steps[0].step_id == "step-01"
        assert roadmap.steps[1].step_id == "step-03", "ids are never reused"
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.steps["step-02"].status == StepStatus.REMOVED
        assert workflow.store.exists("verification/step-01/manifest.json")

    async def test_a_revised_roadmap_cannot_renumber_finished_work(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        workflow.request_roadmap_change(HumanDecision("Change the rest"))
        await workflow.advance()
        # The default script numbers from 1, which is refused twice per model.
        assert workflow.state.phase == ProductPhase.BLOCKED
        assert "starting at 2" in (
            harness.team.calls_of("roadmap")[-1].prompt
        )

    async def test_step_ids_stay_stable_when_steps_are_reordered(self, harness):
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        original = {step.name: step.step_id for step in workflow.roadmap.steps}
        workflow.request_roadmap_change(HumanDecision("Keep the same steps"))
        await workflow.advance()
        assert {step.name: step.step_id for step in workflow.roadmap.steps} == original

    async def test_step_dependencies_are_enforced(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        # Point the roadmap so the next pending step depends on an unfinished one.
        roadmap: Roadmap = workflow.roadmap
        workflow.roadmap = roadmap.model_copy(
            update={"steps": [roadmap.steps[1].model_copy(update={"number": 1, "dependencies": [1]})]}
        )
        with pytest.raises(GateViolation, match="needs step"):
            workflow.start_step(APPROVE)


class TestStepProposalGate:
    async def test_the_human_proposal_covers_each_required_topic(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        message = workflow.gate().message
        for heading in (
            "What this step will deliver",
            "Why the Architect recommends this design",
            "Technology choices",
            "What the Developer intends to implement",
            "How the Test Engineer will verify it",
            "Completion conditions",
        ):
            assert heading in message

    async def test_specialists_run_in_order_with_separate_tasks(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        planning = [
            kind
            for kind in harness.team.kinds()
            if kind in ("step_brief", "architect", "developer_plan", "test_plan", "test_file",
                        "proposal")
        ]
        assert planning == [
            "step_brief", "architect", "developer_plan", "test_plan", "test_file", "proposal",
        ]

    async def test_the_developer_has_no_tools_during_planning(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        assert harness.team.calls_of("developer_plan")[0].tools == []

    async def test_modification_reruns_affected_agents_and_keeps_history(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        first_proposal = workflow.store.read_text("steps/step-01/proposals/v1/proposal.md")

        await workflow.request_step_modification(HumanDecision("Use a different structure"))
        await workflow.advance()

        progress = workflow.state.steps["step-01"]
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert progress.proposal_version == 2
        assert len(harness.team.calls_of("architect")) == 2
        assert len(harness.team.calls_of("developer_plan")) == 2
        assert len(harness.team.calls_of("test_plan")) == 2
        assert "Apply the requested change" in harness.team.calls_of("architect")[-1].prompt
        assert workflow.store.read_text("steps/step-01/proposals/v1/proposal.md") == first_proposal
        assert workflow.store.exists("steps/step-01/proposals/v2/proposal.md")
        assert workflow.store.exists("steps/step-01/proposals/v1/tests/test_calc.py")
        assert "changes_requested" in {r.decision for r in workflow.store.read_approvals()}

    async def test_a_test_only_change_leaves_the_design_alone(self, harness):
        import json

        harness.team.script(
            "route_modification",
            Text(json.dumps({"restated_request": "More tests", "affected_agents": ["test_engineer"]})),
        )
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        await workflow.request_step_modification(HumanDecision("Add more tests"))
        await workflow.advance()

        assert len(harness.team.calls_of("architect")) == 1
        assert len(harness.team.calls_of("developer_plan")) == 1
        assert len(harness.team.calls_of("test_plan")) == 2
        assert workflow.store.exists("steps/step-01/proposals/v2/architect.json")

    async def test_a_developer_only_change_keeps_the_tests(self, harness):
        import json

        harness.team.script(
            "route_modification",
            Text(json.dumps({"restated_request": "Rework it", "affected_agents": ["developer"]})),
        )
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        await workflow.request_step_modification(HumanDecision("Rework it"))
        await workflow.advance()

        assert len(harness.team.calls_of("test_plan")) == 1
        assert len(harness.team.calls_of("developer_plan")) == 2
        assert workflow.store.exists("verification/step-01/test_calc.py")
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START

    async def test_a_failed_routing_call_falls_back_to_everyone(self, harness):
        team = harness.team
        team.script("route_modification", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        await workflow.request_step_modification(HumanDecision("Change it"))
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert len(team.calls_of("architect")) == 2
        assert "Change it" in team.calls_of("architect")[-1].prompt

    async def test_questions_do_not_move_the_workflow(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        answer = await workflow.ask_team(HumanDecision("Why this framework?"))
        assert "Software Architect" in answer
        assert harness.team.calls_of("team_question")[0].model == QWEN
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    @pytest.mark.parametrize(
        ("question", "role"),
        [
            ("How will you test this?", "test_engineer"),
            ("Which files will change?", "developer"),
            ("Why this database?", "software_architect"),
            ("When will it be ready?", "business_analyst"),
        ],
    )
    def test_question_routing_is_deterministic(self, question, role):
        assert ProductWorkflow.route_question(question) == role

    async def test_an_unanswerable_question_leaves_the_gate_intact(self, harness):
        harness.team.script("team_question", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        answer = await workflow.ask_team(HumanDecision("Why this design?"))
        assert "still waiting" in answer
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert workflow.state.active_task is None

    async def test_rejection_returns_to_the_start_gate_and_preserves_the_proposal(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        workflow.reject_step_proposal(HumanDecision("Too complicated"))

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert workflow.state.steps["step-01"].status == StepStatus.REJECTED
        assert workflow.store.exists("steps/step-01/proposals/v1/proposal.md")
        assert not workflow.store.exists("verification/step-01/test_calc.py")

        workflow.start_step(APPROVE)
        await workflow.advance()
        assert workflow.state.steps["step-01"].proposal_version == 2
        assert "Too complicated" in harness.team.calls_of("architect")[-1].prompt


class TestImplementationIntegrity:
    async def test_implementation_refuses_a_proposal_changed_after_approval(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        path = workflow.store.root / "steps/step-01/proposals/v1/proposal.json"
        path.write_text(path.read_text().replace("A working feature.", "Something else."))

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.kind == BlockKind.INTEGRITY
        assert "changed after it was approved" in workflow.state.blocked.message
        assert harness.team.calls_of("implement") == []

    async def test_implementation_refuses_a_forged_approval(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        # A hand-edited checkpoint claiming approval, with no approval record behind it.
        progress = workflow.state.steps["step-01"]
        progress.approved_version = progress.proposal_version
        workflow.state.phase = ProductPhase.IMPLEMENTING
        workflow._save()

        await workflow.advance()

        assert workflow.state.blocked.kind == BlockKind.INTEGRITY
        assert harness.team.calls_of("implement") == []

    async def test_a_developer_that_writes_nothing_cannot_succeed(self, harness):
        harness.team.script("implement", Silent(), Silent(), Silent(), Silent())
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.failure_class.value == "tool_execution"
        assert len(harness.team.calls_of("implement")) == 4
        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED

    async def test_claiming_success_without_a_write_triggers_failover(self, harness):
        harness.team.script("implement", Silent(), Silent())
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert harness.team.models_for("implement") == [QWEN, QWEN, GEMMA]
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.steps["step-01"].developer_models == [GEMMA]
        # Failover in implementation flips the Test Engineer's preferred model.
        assert set(harness.team.models_for("assess")) == {QWEN}


class TestBoundedRepair:
    @staticmethod
    def _bad_writer(body):
        def write(model, prompt, tools):
            tools[0]("calc.py", f"def add(a, b):\n    return {body}\n")
            return "Fixed."

        return write

    def _always_failing(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.team.script(
            "repair", self._bad_writer("a * b"), self._bad_writer("a ** b"),
            self._bad_writer("a // 1"), self._bad_writer("0"),
        )

    async def test_repair_stops_after_the_configured_limit(self, harness):
        self._always_failing(harness)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        progress = workflow.state.steps["step-01"]
        assert len(harness.team.calls_of("repair")) == 2
        assert progress.verification_cycle == 3
        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.kind == BlockKind.VERIFICATION_FAILED
        assert progress.status == StepStatus.FAILED_VERIFICATION

    async def test_a_human_can_grant_more_repair_attempts(self, harness):
        self._always_failing(harness)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        harness.team.scripts["repair"] = [self._bad_writer("a + b")]
        workflow.retry_blocked(HumanDecision("yes"))
        await workflow.advance()

        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert len(harness.team.calls_of("repair")) == 3

    async def test_model_cannot_declare_success_over_failing_tests(self, harness):
        self._always_failing(harness)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        # The scripted closure report claims every condition was met.
        closure = harness.team._default("closure", "step-01-closure-v1-c1")
        assert all(item["met"] for item in closure["conditions"])
        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED
        report = workflow.store.read_text("steps/step-01/closure.md")
        assert "Verification verdict: NOT PASSED" in report

    async def test_a_repair_that_changes_nothing_is_not_accepted(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.team.repaired_sources = {"calc.py": CALC_BUGGY}
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert len(harness.team.calls_of("repair")) == 4
        assert harness.team.models_for("repair") == [QWEN, QWEN, GEMMA, GEMMA]

    async def test_acceptance_failure_goes_to_repair_not_to_model_failover(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert harness.team.models_for("implement") == [QWEN]
        assert harness.team.models_for("repair") == [QWEN]

    async def test_a_review_finding_triggers_repair_even_when_tests_pass(self, harness):
        import json

        harness.team.script(
            "review",
            Text(
                json.dumps(
                    {
                        "verdict": "changes_requested",
                        "summary": "Missing docs.",
                        "consistency_with_design": "Mostly.",
                        "findings": [
                            {
                                "severity": "major",
                                "file_path": "calc.py",
                                "description": "No docstring.",
                                "recommendation": "Add one.",
                            }
                        ],
                    }
                )
            ),
        )
        harness.team.repaired_sources = {"calc.py": '"""Docs."""\n' + CALC_SOURCE}
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert "No docstring" in harness.team.calls_of("repair")[0].prompt
        assert len(harness.team.calls_of("review")) == 2
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED

    async def test_test_engineer_can_fail_a_passing_run(self, harness):
        import json

        harness.team.script(
            "assess",
            Text(json.dumps({"verdict": "fail", "summary": "Bad.", "concerns": ["Too shallow"]})),
        )
        harness.team.repaired_sources = {"calc.py": "# Revised.\n" + CALC_SOURCE}
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert "Too shallow" in harness.team.calls_of("repair")[0].prompt

    async def test_test_engineer_cannot_pass_a_failing_run(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert harness.team.calls_of("assess") == []
        assert workflow.state.phase == ProductPhase.BLOCKED


class TestModifyingAfterFailedVerification:
    """A new proposal must earn fresh evidence, never inherit the old proposal's."""

    async def _failed(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert workflow.state.blocked.kind == BlockKind.VERIFICATION_FAILED
        harness.team.sources = {"calc.py": CALC_SOURCE}
        return workflow

    async def test_stale_failing_evidence_is_not_reused(self, harness):
        workflow = await self._failed(harness)

        await workflow.request_step_modification(HumanDecision("Fix the logic"))
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.store.exists("steps/step-01/test_runs/v1-c1.json")
        assert workflow.store.exists("steps/step-01/test_runs/v2-c1.json")
        evidence = workflow.store.load_model(
            workflow.state.steps["step-01"].evidence_file, VerificationEvidence
        )
        assert evidence.proposal_version == 2
        assert evidence.passed

    async def test_later_tasks_run_again_for_the_new_proposal(self, harness):
        workflow = await self._failed(harness)
        closures_before = len(harness.team.calls_of("closure"))

        await workflow.request_step_modification(HumanDecision("Fix the logic"))
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert len(harness.team.calls_of("closure")) == closures_before + 1
        assert len(harness.team.calls_of("assess")) == 1
        assert len(harness.team.calls_of("review")) == 1

    async def test_a_repair_in_a_new_proposal_is_a_new_task(self, harness):
        workflow = await self._failed(harness)
        harness.policy.workflow.max_repair_cycles = 2
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.team.repaired_sources = {"calc.py": CALC_SOURCE}

        await workflow.request_step_modification(HumanDecision("Fix the logic"))
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert len(harness.team.calls_of("repair")) == 1
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED


class TestCompletionEvidence:
    async def test_completion_requires_recorded_evidence(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        assert workflow.completion_problems("step-01") == []

    async def test_a_code_change_after_testing_invalidates_the_evidence(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        (workflow.store.workspace_dir / "calc.py").write_text("# changed\n")
        assert "the code changed after it was tested" in workflow.completion_problems("step-01")

    async def test_a_run_with_no_passing_tests_is_not_evidence(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        progress = workflow.state.steps["step-01"]
        evidence = workflow.store.load_model(progress.evidence_file, VerificationEvidence)
        evidence.runs[0].passed_count = 0
        workflow.store.save_model(progress.evidence_file, evidence)
        assert any("no passing tests" in item for item in workflow.completion_problems("step-01"))

    async def test_conditions_unmet_stop_a_step_even_with_passing_tests(self, harness):
        import json

        report = {
            "message": "Not quite.",
            "what_was_delivered": "Part.",
            "what_was_tested": "All.",
            "conditions": [{"condition": "Addition works", "met": False, "evidence": "Partly."}],
        }
        harness.team.script("closure", Text(json.dumps(report)))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.kind == BlockKind.CONDITIONS_UNMET
        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED

        workflow.accept_with_limitations(HumanDecision("/accept"))

        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.steps["step-01"].accepted_by_human is True
        kinds = [record.kind for record in workflow.store.read_approvals()]
        assert ApprovalKind.STEP_ACCEPTANCE in kinds

    async def test_accepting_is_impossible_when_tests_failed(self, harness):
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        with pytest.raises(GateViolation):
            workflow.accept_with_limitations(HumanDecision("/accept"))
        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED

    async def test_a_closure_report_must_judge_each_condition(self, harness):
        import json

        report = {
            "message": "Done.",
            "what_was_delivered": "x",
            "what_was_tested": "y",
            "conditions": [],
        }
        harness.team.script("closure", Text(json.dumps(report)))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert len(harness.team.calls_of("closure")) == 2


class TestBlockedState:
    async def test_exhausted_models_persist_a_blocked_state(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        blocked = workflow.state.blocked
        assert blocked.kind == BlockKind.MODEL_FAILURE
        assert blocked.resume_phase == ProductPhase.STEP_ARCHITECT
        assert blocked.failure_class.value == "model_connection"
        assert "Both models failed after 4 attempts" in blocked.message
        assert "check that Ollama is running" in blocked.message
        assert workflow.state.active_task is None
        assert workflow.state.last_failed_operation is not None
        assert harness.team.models_for("architect") == [QWEN, QWEN, GEMMA, GEMMA]

    async def test_blocked_state_survives_a_restart_and_can_be_retried(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()

        restarted = harness.restart(workflow)
        assert restarted.state.phase == ProductPhase.BLOCKED

        restarted.retry_blocked(HumanDecision("yes"))
        await restarted.advance()

        assert restarted.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert len(harness.team.calls_of("step_brief")) == 1

    async def test_discovery_failure_keeps_the_users_message(self, harness):
        harness.team.script("discovery", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await workflow.discovery_message("My precious idea")

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.pending_input == "My precious idea"

        workflow.retry_blocked(HumanDecision("yes"))
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_NAME
        assert "My precious idea" in harness.team.calls_of("discovery")[-1].prompt

    async def test_filesystem_failure_blocks_without_trying_the_other_model(self, harness):
        harness.team.script("architect", Fail(OSError(28, "No space left on device")))
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()

        assert workflow.state.blocked.kind == BlockKind.DETERMINISTIC_FAILURE
        assert workflow.state.blocked.failure_class.value == "filesystem"
        assert harness.team.models_for("architect") == [QWEN]

    async def test_failure_diagnostics_are_saved(self, harness):
        harness.team.script("architect", Text("garbage"), Text("garbage"))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        attempts = (workflow.store.task_dir("step-01-architect-v1") / "attempts.jsonl").read_text()
        assert "invalid_json" in attempts
        events = [event.type for event in workflow.store.read_events()]
        assert "ModelAttemptFailed" in events
        assert "ModelSwitched" in events
        records = workflow.store.read_text("steps/step-01/transcript.md")
        assert "failed over" in records
