"""Human intervention: a failure becomes a checkpoint a person can fix and continue from.

Only the models are scripted.  The reports, hashes, seals and test runs are real.
"""

from __future__ import annotations

import json
import os

import pytest

from lab.product.session import ProductSession, _ExitSession, _LeaveProduct
from lab.product.store import ProductStore
from lab.product.workflow import GateViolation, HumanDecision, VerifyStatus
from lab.schemas.product import (
    ApprovalKind,
    BlockKind,
    InterventionReport,
    ProductPhase,
    StepStatus,
    TestManifest,
)
from lab.tools.fsutil import sha256_file
from lab.tools.journal import workspace_fingerprint
from tests.fakes import (
    APPROVE,
    CALC_BUGGY,
    CALC_SOURCE,
    CALC_TEST,
    Fail,
    Harness,
    ScriptedTerminal,
    SimulatedPowerLoss,
    Text,
    complete_first_step,
    enable_autonomy,
    to_failing_verification,
    to_proposal_gate,
    to_step_start_gate,
)

VERIFY = HumanDecision("/verify")
NOT_JSON = "this is not json"


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def session_for(harness: Harness, *answers: str) -> tuple[ProductSession, ScriptedTerminal]:
    terminal = ScriptedTerminal(*answers)
    return (
        ProductSession(harness.manager, harness.router, harness.policy, harness.playbooks, terminal),
        terminal,
    )


def report_of(workflow) -> InterventionReport:
    intervention = workflow.state.intervention
    return workflow.store.load_model(
        f"interventions/{intervention.intervention_id}/report.json", InterventionReport
    )


def report_text(workflow) -> str:
    return workflow.store.read_text(workflow.state.intervention.report_path)


def event_types(workflow) -> list[str]:
    return [event.type for event in workflow.store.read_events()]


def fix_calc(workflow) -> None:
    (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE)


def sealed_test(workflow, name: str = "test_calc.py", step: str = "step-01"):
    return workflow.store.verification_dir / step / name


def edit_sealed(workflow, content: str, name: str = "test_calc.py") -> None:
    path = sealed_test(workflow, name)
    os.chmod(path, 0o644)
    path.write_text(content)


async def blocked_on_test_file(harness) -> object:
    harness.team.script("test_file", *[Text(NOT_JSON)] * 4)
    workflow = harness.new_workflow()
    await to_step_start_gate(workflow)
    workflow.start_step(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.BLOCKED
    return workflow


async def blocked_implementing(harness) -> object:
    harness.team.script("implement", *[Text("Here is the code.")] * 4)
    workflow = harness.new_workflow()
    await to_proposal_gate(workflow)
    workflow.approve_step_proposal(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.BLOCKED
    return workflow


class TestTheReport:
    async def test_retry_exhaustion_creates_it(self, harness):
        workflow = await blocked_on_test_file(harness)

        intervention = workflow.state.intervention
        assert intervention.intervention_id == "INT-001"
        assert workflow.store.exists("interventions/INT-001/report.md")
        assert workflow.store.exists("interventions/INT-001/report.json")
        assert "HumanInterventionRequired" in event_types(workflow)

    async def test_context_section(self, harness):
        workflow = await blocked_on_test_file(harness)
        report = report_of(workflow)

        assert report.product == "Pocket Calc"
        assert (report.step_number, report.step_name) == (1, "Core arithmetic")
        assert report.phase == "step_test_plan"
        assert report.responsible_agent == "Test Engineer"
        assert (report.attempts_used, report.attempts_allowed) == (4, 4)
        assert {report.preferred_model, report.fallback_model} == {
            "gemma4:latest",
            "qwen2.5:7b",
        }

    async def test_exhausted_repairs_name_the_agent_and_its_models(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        report = report_of(workflow)

        assert report.responsible_agent == "Developer"
        assert (report.preferred_model, report.fallback_model) == (
            "qwen2.5:7b",
            "gemma4:latest",
        )
        assert any("0 of 0 repair cycles" in line for line in report.confirmed_facts)

    async def test_a_runtime_error_is_a_confirmed_fact(self, harness):
        workflow = harness.new_workflow()
        harness.team.script(
            "test_file",
            Text(json.dumps({"path": "test_calc.py", "content": "def test_x(:\n"})),
        )
        await to_failing_verification(harness, workflow)

        facts = report_of(workflow).confirmed_facts

        assert any(line.startswith("Reported error: SyntaxError") for line in facts)
        assert any("could not run its tests" in line for line in facts)

    async def test_spent_repairs_are_reported_as_the_attempts_used(self, harness):
        workflow = harness.new_workflow()
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.team.repaired_sources = {"calc.py": CALC_BUGGY + "# touched\n"}
        harness.policy.workflow.max_repair_cycles = 1
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        report = report_of(workflow)

        assert (report.repair_cycles_used, report.repair_cycles_allowed) == (1, 1)
        assert "Attempts used: 1 of 1 repair cycles" in report_text(workflow)
        assert "Used: 1 of 1 repair cycles." in workflow.intervention_summary()

    async def test_unmet_conditions_name_the_business_analyst(self, harness):
        closure = harness.team._default("closure", "step-01")
        closure["conditions"][0]["met"] = False
        harness.team.script("closure", Text(json.dumps(closure)))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        report = report_of(workflow)
        assert report.responsible_agent == "Business Analyst"
        assert "/accept" in " ".join(report.next_commands)

    async def test_it_names_the_file_and_the_actual_errors(self, harness):
        workflow = await blocked_on_test_file(harness)
        report = report_of(workflow)

        assert "verification/step-01/test_calc.py" in report.files
        assert any("no valid JSON object" in line for line in report.errors)
        assert "Attempt 4" in report_text(workflow)

    async def test_it_says_what_the_tests_were_meant_to_cover(self, harness):
        workflow = await blocked_on_test_file(harness)
        report = report_of(workflow)

        assert any(line.startswith("AC-1: Works.") for line in report.acceptance_criteria)
        assert any(line.startswith("TC-1 (positive") for line in report.test_cases)
        assert any(line.startswith("FR-1") for line in report.requirements)

    async def test_failing_tests_name_tests_and_expected_against_actual(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        report = report_of(workflow)

        names = {item.name for item in report.failed_tests}
        assert names == {"test_add_positive", "test_add_negative_numbers"}
        first = next(i for i in report.failed_tests if i.name == "test_add_positive")
        assert (first.actual, first.expected) == ("-1", "5")
        assert "workspace/calc.py" in report.files
        assert report.suites[0].diagnostic_output

    async def test_confirmed_facts_are_separate_from_suspicions(self, harness):
        workflow = await blocked_on_test_file(harness)
        report = report_of(workflow)

        assert any("4 of 4 model attempts failed" in line for line in report.confirmed_facts)
        assert report.unconfirmed_causes
        assert not set(report.confirmed_facts) & set(report.unconfirmed_causes)
        assert not any(" may " in line for line in report.confirmed_facts)
        assert all(" may " in line for line in report.unconfirmed_causes)

    async def test_the_saved_report_labels_each_kind_of_statement(self, harness):
        workflow = await blocked_on_test_file(harness)
        text = report_text(workflow)

        confirmed = text.index("### Confirmed facts")
        diagnosis = text.index("### The agents' diagnosis")
        possible = text.index("### Possible causes, not confirmed")
        assert confirmed < diagnosis < possible
        assert "has not been verified" in text
        assert "nothing has shown them to be true" in text

    async def test_it_invents_no_diagnosis_when_the_agents_gave_none(self, harness):
        workflow = await blocked_on_test_file(harness)
        assert report_of(workflow).agent_diagnosis == []
        assert "The agents offered no diagnosis." in report_text(workflow)

    async def test_the_agents_opinion_is_quoted_and_attributed(self, harness):
        harness.policy.workflow.max_repair_cycles = 0
        review = {
            "verdict": "changes_requested",
            "summary": "It does not match the design.",
            "consistency_with_design": "Off.",
            "findings": [
                {"severity": "major", "description": "Wrong shape.", "recommendation": "Fix."}
            ],
        }
        harness.team.script("review", Text(json.dumps(review)))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        notes = report_of(workflow).agent_diagnosis
        assert "Software Architect: It does not match the design." in notes
        assert any("Wrong shape." in line for line in notes)

    async def test_recommendations_are_practical(self, harness):
        workflow = await blocked_on_test_file(harness)
        action = report_of(workflow).actions[0]

        assert "verification/step-01/test_calc.py" in action.files
        assert "Create the file yourself" in action.action
        assert "2 approved test case(s)" in action.action
        assert "/verify" in action.action
        assert action.why_it_matters

    async def test_failed_model_outputs_are_kept_as_files(self, harness):
        workflow = await blocked_on_test_file(harness)
        outputs = report_of(workflow).preserved_outputs

        assert len(outputs) == 4
        assert workflow.store.read_text(outputs[-1]) == NOT_JSON

    async def test_failing_to_keep_an_output_does_not_change_the_failure(
        self, harness, monkeypatch
    ):
        """An OSError here would be classed as a filesystem fault, which stops all retries."""
        harness.team.script("test_file", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)

        def disk_full(*args):
            raise OSError("No space left on device")

        monkeypatch.setattr(workflow.store, "save_failed_output", disk_full)
        workflow.start_step(APPROVE)
        await workflow.advance()

        assert len(harness.team.calls_of("test_file")) == 4
        assert workflow.state.blocked.kind == BlockKind.MODEL_FAILURE
        assert workflow.state.blocked.failure_class.value == "invalid_json"

    async def test_a_failed_output_never_reaches_the_next_model(self, harness):
        await blocked_on_test_file(harness)
        last = harness.team.calls_of("test_file")[-1]
        assert NOT_JSON not in last.prompt

    async def test_a_summary_is_shown_in_the_terminal(self, harness):
        terminal = ScriptedTerminal("/exit")
        harness.team.script("test_file", *[Text(NOT_JSON)] * 4)
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )

        with pytest.raises(_ExitSession):
            await session._on_blocked(workflow)

        shown = "\n".join(terminal.said)
        assert "BLOCKED / HUMAN INTERVENTION REQUIRED" in shown
        assert "verification/step-01/test_calc.py" in shown
        assert "Why it matters" in shown
        assert "products_log/pocket-calc/interventions/INT-001/report.md" in shown

    async def test_status_points_to_the_report(self, harness):
        workflow = await blocked_on_test_file(harness)
        status = workflow.status_report()

        assert "BLOCKED / HUMAN INTERVENTION REQUIRED" in status
        assert "Intervention report: products_log/pocket-calc/interventions/INT-001" in status
        assert "/verify" in workflow.state.next_expected_action

    async def test_the_product_list_points_to_the_report(self, harness):
        await blocked_on_test_file(harness)
        assert "Intervention report:" in harness.manager.discover()[0][0].render()

    async def test_a_report_failure_never_hides_the_block(self, harness, monkeypatch):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        terminal = ScriptedTerminal()
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_step_start_gate(workflow)
        monkeypatch.setattr(
            workflow, "_write_report", lambda *a, **k: (_ for _ in ()).throw(ValueError("broken"))
        )
        workflow.start_step(APPROVE)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.intervention is None
        assert any("could not write the intervention report" in w for w in terminal.warnings)


class TestThePreservedCheckpoint:
    async def test_nothing_is_reset_or_overwritten(self, harness):
        workflow = await blocked_implementing(harness)
        step = workflow.state.steps["step-01"]

        assert workflow.state.blocked.resume_phase == ProductPhase.IMPLEMENTING
        assert workflow.state.current_step_id == "step-01"
        assert step.approved_version == 1
        assert workflow.verification.is_sealed("step-01")
        assert workflow.store.exists("steps/step-01/proposals/v1/proposal.json")

    async def test_the_attempt_ledger_is_kept(self, harness):
        workflow = await blocked_implementing(harness)
        attempts = workflow._attempts_of(workflow.state.blocked.task_id)

        assert len(attempts) == 4
        assert {item.status for item in attempts} == {"failed"}

    async def test_the_lock_is_released_on_leaving(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        session, terminal = session_for(harness, "/exit")

        await session.run_product(session.open_product(workflow.store.slug))

        assert not workflow.store.lock_path.exists()

    async def test_the_state_survives_a_restart(self, harness):
        workflow = await blocked_implementing(harness)
        intervention = workflow.state.intervention

        restarted = harness.restart(workflow)

        assert restarted.state.phase == ProductPhase.BLOCKED
        assert restarted.state.intervention == intervention
        assert restarted.state.blocked == workflow.state.blocked

    async def test_a_block_from_an_older_version_gets_a_report_when_opened(self, harness):
        workflow = await blocked_implementing(harness)
        workflow.state.intervention = None
        workflow._save()

        restarted = harness.restart(workflow)

        assert restarted.ensure_intervention() is not None
        assert restarted.store.exists(restarted.state.intervention.report_path)

    async def test_interrupted_between_the_block_and_its_report(self, harness, monkeypatch):
        harness.team.script("implement", *[Text("Here is the code.")] * 4)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)

        def power_loss() -> None:
            raise SimulatedPowerLoss

        monkeypatch.setattr(workflow, "_open_intervention", power_loss)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()

        restarted = harness.restart(workflow)
        assert restarted.state.phase == ProductPhase.BLOCKED
        assert restarted.state.intervention is None
        assert restarted.ensure_intervention().intervention_id == "INT-001"

    async def test_each_block_gets_its_own_numbered_report(self, harness):
        workflow = await blocked_implementing(harness)
        workflow.retry_blocked(APPROVE)
        harness.team.script("implement", *[Text("Here is the code.")] * 4)
        await workflow.advance()

        assert workflow.state.intervention.intervention_id == "INT-002"
        assert workflow.store.exists("interventions/INT-001/report.md")


class TestDetectingManualChanges:
    async def test_edited_files_are_found_with_their_hashes(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        before = sha256_file(workflow.store.workspace_dir / "calc.py")
        fix_calc(workflow)
        after = sha256_file(workflow.store.workspace_dir / "calc.py")

        workflow.verify_manual_fix(VERIFY)

        detected = next(
            e for e in workflow.store.read_events() if e.type == "ManualChangesDetected"
        )
        assert detected.payload["changes"] == [
            {
                "path": "workspace/calc.py",
                "status": "modified",
                "old_sha256": before,
                "new_sha256": after,
            }
        ]

    async def test_added_and_removed_files_are_found(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        (workflow.store.workspace_dir / "helper.py").write_text("x = 1\n")
        (workflow.store.workspace_dir / "calc.py").unlink()

        changes = workflow._record_manual_changes(workflow.state.intervention)

        assert {(item.path, item.status) for item in changes} == {
            ("workspace/calc.py", "removed"),
            ("workspace/helper.py", "added"),
        }

    async def test_a_copy_of_the_change_is_kept(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)

        workflow.verify_manual_fix(VERIFY)

        kept = workflow.store.root / "interventions/INT-001/manual_changes/1/workspace/calc.py"
        assert kept.read_text() == CALC_SOURCE

    async def test_the_journal_knows_so_nothing_is_reported_as_drift(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)

        workflow.verify_manual_fix(VERIFY)

        report = workflow._workspace("step-01").reconcile()
        assert report.drifted == []
        assert "calc.py" in report.confirmed

    async def test_it_is_recorded_as_a_decision(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)

        workflow.verify_manual_fix(VERIFY)

        assert "Manual changes recorded for INT-001" in workflow.store.read_text("decisions.md")

    async def test_nothing_changed_is_not_a_change_set(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)

        workflow.verify_manual_fix(VERIFY)

        assert workflow.state.intervention.change_sets == 0
        assert "ManualChangesDetected" not in event_types(workflow)

    async def test_a_finder_file_in_the_tests_is_not_tampering(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        (workflow.store.verification_dir / "step-01" / ".DS_Store").write_bytes(b"\x00")
        fix_calc(workflow)

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.PASSED


class TestVerifying:
    async def test_it_reruns_the_authoritative_tests(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)

        workflow.verify_manual_fix(VERIFY)

        path = "steps/step-01/test_runs/v1-c2.json"
        evidence = json.loads(workflow.store.read_text(path))
        assert evidence["passed"] is True
        assert evidence["manifest_sha256"] == workflow.verification.manifest_digest("step-01")
        assert [run["scope"] for run in evidence["runs"]] == ["acceptance"]
        assert evidence["tested_workspace_fingerprint"] == workspace_fingerprint(
            workflow.store.workspace_dir
        )

    async def test_it_also_runs_earlier_steps(self, harness):
        harness.policy.workflow.max_repair_cycles = 0
        harness.team.scripts.clear()
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        original = harness.team._developer_writes
        harness.team._developer_writes = lambda kind, task_id: {
            "stats.py": "def mean(values):\n    return 0\n"
        }
        workflow.start_step(APPROVE)
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        harness.team._developer_writes = original
        assert workflow.state.phase == ProductPhase.BLOCKED
        (workflow.store.workspace_dir / "stats.py").write_text(
            "def mean(values):\n    return sum(values) / len(values)\n"
        )

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.PASSED
        evidence = json.loads(workflow.store.read_text("steps/step-02/test_runs/v1-c2.json"))
        assert [(run["scope"], run["step_id"]) for run in evidence["runs"]] == [
            ("acceptance", "step-02"),
            ("regression", "step-01"),
        ]

    async def test_a_broken_earlier_step_fails_verification(self, harness):
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        original = harness.team._developer_writes
        harness.team._developer_writes = lambda kind, task_id: {
            "stats.py": "def mean(values):\n    return 0\n"
        }
        workflow.start_step(APPROVE)
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        harness.team._developer_writes = original
        (workflow.store.workspace_dir / "stats.py").write_text(
            "def mean(values):\n    return sum(values) / len(values)\n"
        )
        (workflow.store.workspace_dir / "calc.py").write_text(CALC_BUGGY)

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.FAILED
        assert "regression suite step-01" in outcome.message

    async def test_a_passing_fix_continues_with_the_review_chain(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        calls = len(harness.team.calls)

        outcome = workflow.verify_manual_fix(VERIFY)
        assert outcome.status == VerifyStatus.PASSED
        assert workflow.state.phase == ProductPhase.VERIFYING_TESTS
        assert workflow.state.intervention is None
        assert workflow.state.blocked is None

        await workflow.advance()

        assert [call.kind for call in harness.team.calls[calls:]] == [
            "assess",
            "review",
            "closure",
        ]
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START

    async def test_the_next_step_then_works_normally(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        workflow.start_step(APPROVE)
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE

    async def test_completion_is_proven_for_the_manual_code(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert workflow.completion_problems("step-01") == []

    async def test_code_changed_after_verification_is_not_completed(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        workflow.verify_manual_fix(VERIFY)
        (workflow.store.workspace_dir / "calc.py").write_text(CALC_SOURCE + "# later\n")

        await workflow.advance()

        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_a_verified_fix_is_recorded_in_the_execution_record(self, harness):
        workflow = await blocked_implementing(harness)
        fix_calc(workflow)

        workflow.verify_manual_fix(VERIFY)

        tasks = workflow.state.steps["step-01"].implementation_tasks
        assert [(item.status, item.model) for item in tasks] == [("done", "human")]
        record = json.loads(workflow.store.read_text("steps/step-01/execution.json"))
        assert record["written_files"]["calc.py"] == sha256_file(
            workflow.store.workspace_dir / "calc.py"
        )

    async def test_blocked_implementation_resumes_at_verification(self, harness):
        workflow = await blocked_implementing(harness)
        fix_calc(workflow)
        calls = len(harness.team.calls)

        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert "implement" not in [call.kind for call in harness.team.calls[calls:]]
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED

    async def test_a_fix_to_the_environment_needs_no_file_change(self, harness, monkeypatch):
        from lab.product import workflow as module
        from lab.tools.verification import VerificationToolMissing

        real = module.run_profile

        def missing(*args, **kwargs):
            raise VerificationToolMissing("'node' is required but is not installed.")

        monkeypatch.setattr(module, "run_profile", missing)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert workflow.state.blocked.kind == BlockKind.DETERMINISTIC_FAILURE
        assert "Install the missing tool" in report_of(workflow).actions[0].action

        still = workflow.verify_manual_fix(VERIFY)
        assert still.status == VerifyStatus.FAILED
        assert workflow.state.phase == ProductPhase.BLOCKED

        monkeypatch.setattr(module, "run_profile", real)
        assert workflow.verify_manual_fix(VERIFY).status == VerifyStatus.PASSED

    async def test_verify_is_refused_when_nothing_is_blocked(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        with pytest.raises(GateViolation):
            workflow.verify_manual_fix(VERIFY)


class TestFailedVerification:
    async def _failed(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        (workflow.store.workspace_dir / "calc.py").write_text(
            "def add(a, b):\n    return a + b + 1\n"
        )
        return workflow, workflow.verify_manual_fix(VERIFY)

    async def test_it_returns_to_the_intervention_state(self, harness):
        workflow, outcome = await self._failed(harness)

        assert outcome.status == VerifyStatus.FAILED
        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.intervention is not None
        assert "ManualVerificationFailed" in event_types(workflow)

    async def test_the_report_is_updated_and_the_old_one_kept(self, harness):
        workflow, _ = await self._failed(harness)

        assert workflow.state.intervention.report_version == 2
        assert workflow.store.exists("interventions/INT-001/report.v1.md")
        assert workflow.store.exists("interventions/INT-001/report.v2.md")
        text = report_text(workflow)
        assert "report version 2" in text
        assert "After your fix" in text
        assert "workspace/calc.py (modified)" in text

    async def test_the_remaining_failures_are_identified(self, harness):
        workflow, _ = await self._failed(harness)
        report = report_of(workflow)

        assert "test_add_positive" in {item.name for item in report.failed_tests}
        first = next(i for i in report.failed_tests if i.name == "test_add_positive")
        assert (first.actual, first.expected) == ("6", "5")

    async def test_the_changes_stay_on_disk(self, harness):
        workflow, _ = await self._failed(harness)
        assert "a + b + 1" in (workflow.store.workspace_dir / "calc.py").read_text()

    async def test_no_model_repair_starts_by_itself(self, harness):
        workflow, _ = await self._failed(harness)
        calls = len(harness.team.calls)

        for _ in range(3):
            await workflow.advance()

        assert len(harness.team.calls) == calls
        assert harness.team.calls_of("repair") == []
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_another_correction_compares_with_the_last_attempt(self, harness):
        workflow, _ = await self._failed(harness)
        first_attempt = sha256_file(workflow.store.workspace_dir / "calc.py")
        fix_calc(workflow)

        changes = workflow._record_manual_changes(workflow.state.intervention)

        assert [item.status for item in changes] == ["modified"]
        assert changes[0].old_sha256 == first_attempt
        assert workflow.state.intervention.change_sets == 2

    async def test_verifying_again_does_not_report_old_changes_again(self, harness):
        workflow, _ = await self._failed(harness)

        workflow.verify_manual_fix(VERIFY)

        assert workflow.state.intervention.change_sets == 1
        assert event_types(workflow).count("ManualChangesDetected") == 1

    async def test_a_second_try_can_succeed(self, harness):
        workflow, _ = await self._failed(harness)
        fix_calc(workflow)

        assert workflow.verify_manual_fix(VERIFY).status == VerifyStatus.PASSED

    async def test_a_person_can_authorize_another_ai_attempt(self, harness):
        workflow, _ = await self._failed(harness)

        workflow.retry_blocked(HumanDecision("/retry"))
        await workflow.advance()

        assert len(harness.team.calls_of("repair")) == 1
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED

    async def test_a_new_ai_attempt_keeps_a_copy_of_the_person_s_work(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        (workflow.store.workspace_dir / "calc.py").write_text("# my notes\n")

        workflow.retry_blocked(HumanDecision("/retry"))

        kept = workflow.store.root / "interventions/INT-001/manual_changes/1/workspace/calc.py"
        assert kept.read_text() == "# my notes\n"


class TestNoBudgetIsTouched:
    async def _ledger(self, workflow, task_id: str) -> tuple[bytes, bool]:
        folder = workflow.store.task_dir(task_id)
        return (folder / "attempts.jsonl").read_bytes(), (folder / "round.txt").exists()

    async def test_verify_calls_no_model_when_tests_fail(self, harness):
        workflow = await blocked_implementing(harness)
        calls = len(harness.team.calls)
        task_id = workflow.state.blocked.task_id
        ledger = await self._ledger(workflow, task_id)

        for _ in range(3):
            assert workflow.verify_manual_fix(VERIFY).status == VerifyStatus.FAILED

        assert len(harness.team.calls) == calls
        assert await self._ledger(workflow, task_id) == ledger

    async def test_verify_itself_calls_no_model_when_tests_pass(self, harness):
        workflow = await blocked_implementing(harness)
        fix_calc(workflow)
        calls = len(harness.team.calls)

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.PASSED
        assert len(harness.team.calls) == calls

    async def test_the_failed_task_keeps_its_exhausted_budget(self, harness):
        workflow = await blocked_implementing(harness)
        task_id = workflow.state.blocked.task_id
        fix_calc(workflow)

        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert workflow.store.current_round(task_id) == 1
        assert len(workflow._attempts_of(task_id)) == 4

    async def test_retry_policy_is_not_altered(self, harness):
        assert harness.policy.retry.attempts_per_model == 2
        workflow = await blocked_implementing(harness)
        assert len(harness.team.calls_of("implement")) == 4
        assert report_of(workflow).attempts_allowed == 4


class TestAfterAManualFix:
    """A person's fix is theirs: nothing may repair over it without being asked."""

    def rejecting_review(self) -> Text:
        return Text(
            json.dumps(
                {
                    "verdict": "changes_requested",
                    "summary": "No.",
                    "consistency_with_design": "Off.",
                    "findings": [
                        {"severity": "major", "description": "Bad.", "recommendation": "Fix."}
                    ],
                }
            )
        )

    async def test_a_review_finding_comes_back_to_the_person(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        harness.policy.workflow.max_repair_cycles = 2
        workflow.state.steps["step-01"].repair_cycle_limit = 2
        harness.team.script("review", self.rejecting_review())
        fix_calc(workflow)
        workflow.verify_manual_fix(VERIFY)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert "Your fix passed the tests" in workflow.state.blocked.message
        assert harness.team.calls_of("repair") == []
        assert (workflow.store.workspace_dir / "calc.py").read_text() == CALC_SOURCE

    async def test_a_test_engineer_concern_comes_back_to_the_person(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        workflow.state.steps["step-01"].repair_cycle_limit = 2
        harness.team.script(
            "assess",
            Text(json.dumps({"verdict": "fail", "summary": "No.", "concerns": ["Too thin."]})),
        )
        fix_calc(workflow)
        workflow.verify_manual_fix(VERIFY)

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert harness.team.calls_of("repair") == []

    async def test_a_later_repair_never_reuses_earlier_evidence(self, harness):
        """Cycles once followed repairs one to one.  /verify breaks that assumption."""
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        harness.team.script("review", self.rejecting_review())
        harness.team.repaired_sources = {"calc.py": CALC_SOURCE + "# repaired\n"}
        fix_calc(workflow)
        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()
        assert workflow.state.steps["step-01"].verification_cycle == 2

        workflow.retry_blocked(HumanDecision("/retry"))
        await workflow.advance()

        step = workflow.state.steps["step-01"]
        assert step.verification_cycle == 3
        evidence = json.loads(workflow.store.read_text("steps/step-01/test_runs/v1-c3.json"))
        assert evidence["tested_workspace_fingerprint"] == workspace_fingerprint(
            workflow.store.workspace_dir
        )
        assert step.status == StepStatus.COMPLETED

    async def test_retry_is_refused_while_test_changes_await_a_decision(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, CALC_TEST + "\n# edited\n")
        workflow.verify_manual_fix(VERIFY)

        with pytest.raises(GateViolation, match="approve or reject"):
            workflow.retry_blocked(HumanDecision("/retry"))


class TestSuppliedResultsBeforeSealing:
    """A task that exhausted its budget before approval: a person can do its job."""

    async def test_a_test_file_written_by_hand_is_accepted(self, harness):
        workflow = await blocked_on_test_file(harness)
        sealed_test(workflow).parent.mkdir(parents=True, exist_ok=True)
        sealed_test(workflow).write_text(CALC_TEST)
        calls = len(harness.team.calls_of("test_file"))

        outcome = workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert outcome.status == VerifyStatus.PASSED
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert len(harness.team.calls_of("test_file")) == calls

    async def test_the_result_is_marked_as_written_by_a_person(self, harness):
        workflow = await blocked_on_test_file(harness)
        task_id = workflow.state.blocked.task_id
        sealed_test(workflow).parent.mkdir(parents=True, exist_ok=True)
        sealed_test(workflow).write_text(CALC_TEST)

        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        result = workflow.store.task_result(task_id)
        assert (result.model, result.attempts) == ("human", 0)
        assert "ManualResultAccepted" in event_types(workflow)

    async def test_it_is_sealed_with_the_proposal_and_enforced(self, harness):
        workflow = await blocked_on_test_file(harness)
        sealed_test(workflow).parent.mkdir(parents=True, exist_ok=True)
        sealed_test(workflow).write_text(CALC_TEST)
        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        manifest = workflow.verification.manifest("step-01")
        assert manifest.files["test_calc.py"] == sha256_file(sealed_test(workflow))
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED

    async def test_a_missing_file_stays_blocked(self, harness):
        workflow = await blocked_on_test_file(harness)

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.FAILED
        assert "cannot find verification/step-01/test_calc.py" in outcome.message
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_a_file_with_no_tests_stays_blocked(self, harness):
        workflow = await blocked_on_test_file(harness)
        sealed_test(workflow).parent.mkdir(parents=True, exist_ok=True)
        sealed_test(workflow).write_text("import calc\n")

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.FAILED
        assert "defines no test cases" in outcome.message
        assert workflow.state.intervention.report_version == 2

    async def test_a_design_written_by_hand_is_held_to_the_same_schema(self, harness):
        harness.team.script("architect", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        blocked = workflow.state.blocked
        assert "handoff_summary" in blocked.artifact_fields
        workflow.store.write_text(blocked.artifact, json.dumps({"summary": "Too little."}))

        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert "is not valid yet" in workflow.state.blocked.message
        assert "components" in workflow.state.blocked.message

    async def test_a_test_plan_written_by_hand_is_held_to_the_task_s_own_rules(self, harness):
        harness.team.script("test_plan", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        plan = harness.team._default("test_plan", "step-01")
        plan["verification_profile"] = "a_profile_that_does_not_exist"
        workflow.store.write_text(workflow.state.blocked.artifact, json.dumps(plan))

        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert "a_profile_that_does_not_exist" in workflow.state.blocked.message
        assert harness.team.calls_of("test_file") == []

    async def test_a_valid_design_written_by_hand_continues_the_workflow(self, harness):
        harness.team.script("architect", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        design = harness.team._default("architect", "step-01")
        workflow.store.write_text(workflow.state.blocked.artifact, json.dumps(design))
        calls = len(harness.team.calls_of("architect"))

        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert len(harness.team.calls_of("architect")) == calls

    async def test_a_developer_plan_written_by_hand_leaves_model_routing_alone(self, harness):
        harness.team.script("developer_plan", *[Text(NOT_JSON)] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        plan = harness.team._default("developer_plan", "step-01")
        workflow.store.write_text(workflow.state.blocked.artifact, json.dumps(plan))

        workflow.verify_manual_fix(VERIFY)
        await workflow.advance()

        assert workflow.state.steps["step-01"].developer_plan_model is None
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    async def test_a_model_problem_has_nothing_on_disk_to_check(self, harness):
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        workflow.state.blocked.artifact = None

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.NOTHING_TO_VERIFY
        assert "/retry" in outcome.message
        assert "Check that Ollama is running" in report_of(workflow).actions[0].action


class TestEditedTests:
    async def _edited(self, harness, content: str = CALC_TEST + "\n# clearer now\n"):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        edit_sealed(workflow, content)
        return workflow, workflow.verify_manual_fix(VERIFY)

    async def test_they_are_not_used_until_approved(self, harness):
        workflow, outcome = await self._edited(harness)

        assert outcome.status == VerifyStatus.NEEDS_TEST_APPROVAL
        assert not workflow.store.exists("steps/step-01/test_runs/v1-c2.json")
        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.pending_approval == ApprovalKind.TEST_REVISION

    async def test_verifying_again_does_not_get_past_the_decision(self, harness):
        workflow, _ = await self._edited(harness)

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.NEEDS_TEST_APPROVAL
        assert not workflow.store.exists("steps/step-01/test_runs/v1-c2.json")

    async def test_verifying_again_keeps_the_same_decision_open(self, harness):
        workflow, _ = await self._edited(harness)
        pending = workflow.state.intervention.pending_revision

        workflow.verify_manual_fix(VERIFY)

        assert workflow.state.intervention.pending_revision == pending
        assert event_types(workflow).count("TestRevisionProposed") == 1

    async def test_the_old_and_new_hashes_are_recorded(self, harness):
        workflow, _ = await self._edited(harness)
        change = workflow.state.intervention.pending_revision.changes[0]

        old = workflow.verification.manifest("step-01").files["test_calc.py"]
        assert change.path == "verification/step-01/test_calc.py"
        assert change.status == "modified"
        assert change.old_sha256 == old
        assert change.new_sha256 == sha256_file(sealed_test(workflow))
        assert change.old_sha256 != change.new_sha256

    async def test_the_person_is_shown_the_differences(self, harness):
        workflow, _ = await self._edited(harness)
        summary = workflow.revision_summary()

        assert "+# clearer now" in summary
        assert "AC-1: Works." in summary
        revision = workflow.state.intervention.pending_revision
        assert "+# clearer now" in workflow.store.read_text(revision.diff_path)

    async def test_approval_seals_the_new_suite_and_runs_it(self, harness):
        workflow, _ = await self._edited(harness)

        outcome = workflow.approve_test_revision(HumanDecision("yes"))

        assert outcome.status == VerifyStatus.PASSED
        manifest = workflow.verification.manifest("step-01")
        assert manifest.revision == 1
        assert manifest.previous_manifest_sha256
        assert manifest.files["test_calc.py"] == sha256_file(sealed_test(workflow))
        assert sealed_test(workflow).stat().st_mode & 0o222 == 0
        assert workflow.verification.verify_seal("step-01")

    async def test_approval_is_recorded_as_a_human_decision(self, harness):
        workflow, _ = await self._edited(harness)
        new_hash = workflow.state.intervention.pending_revision.changes[0].new_sha256

        workflow.approve_test_revision(HumanDecision("yes"))

        record = next(
            item
            for item in workflow.store.read_approvals()
            if item.kind == ApprovalKind.TEST_REVISION
        )
        assert record.decided_by == "human"
        assert record.decision == "approved"
        assert record.artifact_sha256 == workflow.verification.manifest_digest("step-01")
        assert new_hash[:12] in workflow.store.read_text("decisions.md")
        event = next(
            e for e in workflow.store.read_events() if e.type == "TestRevisionApproved"
        )
        assert event.payload["old_manifest_sha256"] != event.payload["new_manifest_sha256"]

    async def test_an_autonomous_policy_cannot_approve_them(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        enable_autonomy(workflow)
        fix_calc(workflow)
        edit_sealed(workflow, CALC_TEST + "\n# clearer now\n")
        workflow.verify_manual_fix(VERIFY)
        calls = len(harness.team.calls)

        for _ in range(3):
            await workflow.advance()

        assert workflow.state.intervention.pending_revision is not None
        assert workflow.verification.manifest("step-01").revision == 0
        assert len(harness.team.calls) == calls

    async def test_rejection_restores_the_sealed_tests(self, harness):
        workflow, _ = await self._edited(harness)

        workflow.reject_test_revision(HumanDecision("no"))

        assert sealed_test(workflow).read_text().strip() == CALC_TEST.strip()
        assert workflow.verification.verify_seal("step-01")
        assert workflow.state.intervention.pending_revision is None
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_rejection_keeps_the_person_s_version(self, harness):
        workflow, _ = await self._edited(harness)
        revision_id = workflow.state.intervention.pending_revision.revision_id

        workflow.reject_test_revision(HumanDecision("no"))

        kept = workflow.store.root / f"interventions/INT-001/rejected_tests/{revision_id}"
        assert "# clearer now" in (kept / "test_calc.py").read_text()

    async def test_after_rejection_verify_uses_the_original_suite(self, harness):
        workflow, _ = await self._edited(harness)
        workflow.reject_test_revision(HumanDecision("no"))

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.PASSED
        assert workflow.verification.manifest("step-01").revision == 0

    async def test_a_file_added_to_the_suite_needs_approval(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        (workflow.store.verification_dir / "step-01" / "conftest.py").write_text("x = 1\n")

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.NEEDS_TEST_APPROVAL
        change = workflow.state.intervention.pending_revision.changes[0]
        assert (change.status, change.old_sha256) == ("added", None)

    async def test_a_tampered_suite_can_be_reviewed_through_verify(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        edit_sealed(workflow, CALC_TEST + "\n# changed early\n")
        await workflow.advance()
        assert workflow.state.blocked.kind == BlockKind.INTEGRITY
        assert "verification/step-01/test_calc.py" in report_of(workflow).files

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.NEEDS_TEST_APPROVAL
        assert "+# changed early" in workflow.state.intervention.pending_revision.diff_excerpt

    async def test_an_edit_made_before_the_failure_can_still_be_compared(self, harness):
        """The copy has to exist from the moment of sealing.  Taking it at the failure
        would be too late, because by then the sealed bytes may already be gone."""
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        original = sealed_test(workflow).read_text()
        edit_sealed(workflow, original + "\n# changed early\n")
        await workflow.advance()

        workflow.verify_manual_fix(VERIFY)

        revision = workflow.state.intervention.pending_revision
        assert not any("not available" in line for line in revision.warnings)
        assert "-" + original.splitlines()[0] not in revision.diff_excerpt
        assert "+" + original.splitlines()[0] not in revision.diff_excerpt

        workflow.reject_test_revision(HumanDecision("no"))
        assert sealed_test(workflow).read_text() == original

    async def test_a_decision_is_needed_for_each_changed_suite(self, harness):
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        original = harness.team._developer_writes
        harness.team._developer_writes = lambda kind, task_id: {
            "stats.py": "def mean(values):\n    return 0\n"
        }
        workflow.start_step(APPROVE)
        await workflow.advance()
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        harness.team._developer_writes = original
        (workflow.store.workspace_dir / "stats.py").write_text(
            "def mean(values):\n    return sum(values) / len(values)\n"
        )
        for step in ("step-01", "step-02"):
            name = "test_calc.py" if step == "step-01" else "test_stats.py"
            path = sealed_test(workflow, name, step)
            os.chmod(path, 0o644)
            path.write_text(path.read_text() + "\n# reviewed\n")

        first = workflow.verify_manual_fix(VERIFY)
        assert first.status == VerifyStatus.NEEDS_TEST_APPROVAL
        assert workflow.state.intervention.pending_revision.step_id == "step-01"

        second = workflow.approve_test_revision(HumanDecision("yes"))
        assert second.status == VerifyStatus.NEEDS_TEST_APPROVAL
        assert workflow.state.intervention.pending_revision.step_id == "step-02"

        assert workflow.approve_test_revision(HumanDecision("yes")).status == VerifyStatus.PASSED

    async def test_a_restart_does_not_lose_the_pending_decision(self, harness):
        workflow, _ = await self._edited(harness)

        restarted = harness.restart(workflow)

        assert restarted.state.intervention.pending_revision is not None
        assert restarted.approve_test_revision(HumanDecision("yes")).status == VerifyStatus.PASSED

    async def test_an_older_product_without_a_copy_shows_hashes(self, harness):
        import shutil

        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        edit_sealed(workflow, CALC_TEST + "\n# edited\n")
        shutil.rmtree(workflow.store.step_dir("step-01") / "sealed_tests")

        workflow.verify_manual_fix(VERIFY)

        revision = workflow.state.intervention.pending_revision
        assert revision.changes[0].old_sha256
        assert any("not available" in line for line in revision.warnings)
        with pytest.raises(GateViolation, match="no copy of the sealed tests"):
            workflow.reject_test_revision(HumanDecision("no"))


class TestAcceptanceCriteriaStayAuthoritative:
    WEAK = "import calc\n\n\ndef test_add_positive():\n    assert True\n"

    async def test_removing_a_test_is_flagged(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, self.WEAK)

        workflow.verify_manual_fix(VERIFY)

        warnings = " ".join(workflow.state.intervention.pending_revision.warnings)
        assert "test_add_negative_numbers" in warnings
        assert "fewer assertions" in warnings

    async def test_a_plain_yes_is_not_enough_for_a_weaker_suite(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, self.WEAK)
        workflow.verify_manual_fix(VERIFY)

        for text in ("yes", "", "ok", "looks good"):
            with pytest.raises(GateViolation, match="plain yes is not enough"):
                workflow.approve_test_revision(HumanDecision(text))

        assert workflow.verification.manifest("step-01").revision == 0
        assert workflow.state.intervention.pending_revision is not None

    async def test_a_stated_reason_is_recorded(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, self.WEAK)
        workflow.verify_manual_fix(VERIFY)

        workflow.approve_test_revision(HumanDecision("The negative case belongs to step 2"))

        decisions = workflow.store.read_text("decisions.md")
        assert "The negative case belongs to step 2" in decisions
        assert "Warnings accepted" in decisions

    async def test_skipping_a_test_is_flagged(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(
            workflow,
            CALC_TEST.replace(
                "def test_add_positive",
                "import pytest\n\n\n@pytest.mark.skip\ndef test_add_positive",
            ),
        )

        workflow.verify_manual_fix(VERIFY)

        assert any(
            "skipped" in line for line in workflow.state.intervention.pending_revision.warnings
        )

    async def test_a_skipped_suite_cannot_pass_verification(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(
            workflow,
            "import pytest\n\n\n@pytest.mark.skip\ndef test_add_positive():\n    assert False\n",
        )
        workflow.verify_manual_fix(VERIFY)

        outcome = workflow.approve_test_revision(HumanDecision("Skipping on purpose"))

        assert outcome.status == VerifyStatus.FAILED
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_editing_the_approved_criteria_is_refused(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        path = workflow.store.root / "steps/step-01/proposals/v1/test_plan.json"
        plan = json.loads(path.read_text())
        plan["acceptance_criteria"][0]["then"] = "Anything at all"
        path.write_text(json.dumps(plan))
        fix_calc(workflow)

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.REFUSED
        assert "steps/step-01/proposals/v1/test_plan.json" in outcome.message
        assert "/change" in outcome.message
        assert workflow.state.phase == ProductPhase.BLOCKED
        assert not workflow.store.exists("steps/step-01/test_runs/v1-c2.json")

    async def test_editing_the_criteria_before_the_failure_is_also_refused(self, harness):
        workflow = harness.new_workflow()
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        path = workflow.store.root / "steps/step-01/proposals/v1/test_plan.json"
        path.write_text(path.read_text().replace("Correct result", "Any result"))
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.BLOCKED

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.REFUSED

    async def test_editing_the_approved_proposal_is_refused(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        path = workflow.store.root / "steps/step-01/proposals/v1/proposal.json"
        path.write_text(path.read_text().replace("A working feature.", "Something else."))

        assert workflow.verify_manual_fix(VERIFY).status == VerifyStatus.REFUSED

    async def test_editing_the_approved_roadmap_is_refused(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        path = workflow.store.root / "artifacts/roadmap.v1.json"
        path.write_text(path.read_text().replace("Addition works", "Nothing needed"))

        assert workflow.verify_manual_fix(VERIFY).status == VerifyStatus.REFUSED

    async def test_rewriting_the_manifest_to_match_is_refused(self, harness):
        """Without this, a matching edit of a test and its hash would look untouched."""
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, self.WEAK)
        path = workflow.store.verification_dir / "step-01" / "manifest.json"
        manifest = TestManifest.model_validate_json(path.read_text())
        manifest.files["test_calc.py"] = sha256_file(sealed_test(workflow))
        os.chmod(path, 0o644)
        path.write_text(manifest.model_dump_json(indent=2))
        assert workflow.verification.verify_seal("step-01")

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.REFUSED
        assert "manifest.json" in outcome.message

    async def test_deleting_the_manifest_does_not_unseal_the_step(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        (workflow.store.verification_dir / "step-01" / "manifest.json").unlink()

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.REFUSED
        assert "manifest.json" in outcome.message
        assert workflow.state.phase == ProductPhase.BLOCKED

    async def test_a_corrupted_manifest_is_refused_not_a_crash(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        workflow.state.steps["step-01"].sealed_manifest_sha256 = None
        path = workflow.store.verification_dir / "step-01" / "manifest.json"
        os.chmod(path, 0o644)
        path.write_text("not json")

        outcome = workflow.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.REFUSED
        assert "can no longer be read" in outcome.message

    async def test_a_real_requirement_change_goes_through_a_new_proposal(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        enable_autonomy(workflow)
        calls = len(harness.team.calls_of("implement"))

        await workflow.request_step_modification(HumanDecision("Change what add must do"))
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert workflow.state.intervention is None
        assert len(harness.team.calls_of("implement")) == calls
        assert workflow.state.steps["step-01"].approved_version is None

    async def test_the_suite_keeps_its_old_copy_after_a_revision(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        edit_sealed(workflow, CALC_TEST + "\n# clearer\n")
        workflow.verify_manual_fix(VERIFY)
        workflow.approve_test_revision(HumanDecision("yes"))

        copies = workflow.store.step_dir("step-01") / "sealed_tests"
        assert "# clearer" not in (copies / "r0" / "test_calc.py").read_text()
        assert "# clearer" in (copies / "r1" / "test_calc.py").read_text()


class TestRecoveryAcrossRestarts:
    async def test_fix_after_a_restart(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)

        restarted = harness.restart(workflow)
        fix_calc(restarted)
        outcome = restarted.verify_manual_fix(VERIFY)
        await restarted.advance()

        assert outcome.status == VerifyStatus.PASSED
        assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED

    async def test_changes_made_while_the_application_was_closed_are_found(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)

        restarted = harness.restart(workflow)
        changes = restarted._record_manual_changes(restarted.state.intervention)

        assert [item.path for item in changes] == ["workspace/calc.py"]

    async def test_power_loss_during_verification(self, harness, monkeypatch):
        from lab.product import workflow as module

        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        real = module.run_profile

        def power_loss(*args, **kwargs):
            raise SimulatedPowerLoss

        monkeypatch.setattr(module, "run_profile", power_loss)
        with pytest.raises(SimulatedPowerLoss):
            workflow.verify_manual_fix(VERIFY)
        monkeypatch.setattr(module, "run_profile", real)

        restarted = harness.restart(workflow)
        assert restarted.state.phase == ProductPhase.BLOCKED
        assert restarted.state.intervention is not None
        outcome = restarted.verify_manual_fix(VERIFY)

        assert outcome.status == VerifyStatus.PASSED

    async def test_power_loss_after_verification_passed(self, harness):
        harness.team.script("assess", Fail(SimulatedPowerLoss()))
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        workflow.verify_manual_fix(VERIFY)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()

        restarted = harness.restart(workflow)
        await restarted.advance()

        assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED
        # The evidence was reused: the tests ran once for that cycle.
        verified = [
            e for e in restarted.store.read_events()
            if e.type == "VerificationRecorded" and e.payload.get("manual")
        ]
        assert len(verified) == 1

    async def test_the_real_store_reloads_every_new_field(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        enable_autonomy(workflow, "until step 2")

        state = ProductStore(workflow.store.root).load_state()

        assert state.intervention.intervention_id == "INT-001"
        assert state.intervention_count == 1
        assert state.autonomy.authorization_id == workflow.state.autonomy.authorization_id


class TestTheConversation:
    async def test_verify_from_the_blocked_prompt(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        session, terminal = session_for(harness, "/verify", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED
        assert any("authoritative tests pass" in line for line in terminal.said)

    async def test_a_failed_verify_shows_the_updated_report(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        session, terminal = session_for(harness, "/verify", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        summaries = [line for line in terminal.said if "HUMAN INTERVENTION REQUIRED" in line]
        assert len(summaries) == 2
        assert any("Tests still fail" in line for line in terminal.warnings)
        assert harness.team.calls_of("repair") == []

    async def test_approving_revised_tests_in_the_conversation(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        fix_calc(workflow)
        edit_sealed(workflow, CALC_TEST + "\n# clearer\n")
        session, terminal = session_for(harness, "/verify", "yes", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        assert any("+# clearer" in line for line in terminal.said)
        assert any("new authoritative test suite" in line for line in terminal.prompts)
        assert workflow.verification.manifest("step-01").revision == 1
        assert workflow.state.steps["step-01"].status == StepStatus.COMPLETED

    async def test_rejecting_revised_tests_in_the_conversation(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, CALC_TEST + "\n# clearer\n")
        session, terminal = session_for(harness, "/verify", "/reject", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        assert any("sealed tests are back" in line for line in terminal.said)
        assert workflow.verification.verify_seal("step-01")

    async def test_a_weakening_edit_cannot_be_waved_through(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, TestAcceptanceCriteriaStayAuthoritative.WEAK)
        session, terminal = session_for(harness, "/verify", "yes", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        assert any("plain yes is not enough" in line for line in terminal.warnings)
        assert workflow.verification.manifest("step-01").revision == 0

    async def test_change_is_honoured_at_the_revision_prompt(self, harness):
        harness.team.scripts["route_modification"] = [
            Text(json.dumps({"restated_request": "New rule", "affected_agents": ["developer"]}))
        ]
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        edit_sealed(workflow, TestAcceptanceCriteriaStayAuthoritative.WEAK)
        session, terminal = session_for(harness, "/verify", "/change a new rule", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        # The edited tests are not carried forward, even for a Developer-only change.
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert workflow.state.intervention is None
        assert workflow.state.steps["step-01"].proposal_version == 2
        assert len(harness.team.calls_of("test_file")) == 2
        assert "assert True" not in sealed_test(workflow).read_text()

    async def test_report_command_repeats_the_summary(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        session, terminal = session_for(harness, "/report", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        assert len([t for t in terminal.said if "HUMAN INTERVENTION REQUIRED" in t]) == 2

    async def test_report_when_nothing_is_blocked(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        assert "Nothing is waiting" in workflow.intervention_summary()

    async def test_verify_at_a_gate_explains_itself(self, harness):
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        session, terminal = session_for(harness, "/verify", "/pause")

        with pytest.raises(_LeaveProduct):
            await session._product_loop(workflow, None)

        assert any("Nothing is waiting for verification" in line for line in terminal.said)
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    async def test_help_lists_the_new_commands(self, harness):
        from lab.product.session import HELP_TEXT

        for command in ("/autonomy on", "/autonomy until", "/autonomy off", "/verify", "/stop"):
            assert command in HELP_TEXT

    async def test_the_block_is_shown_again_after_reopening(self, harness):
        workflow = harness.new_workflow()
        await to_failing_verification(harness, workflow)
        session, terminal = session_for(harness, "/exit")

        await session.run_product(session.open_product(workflow.store.slug))

        assert any("HUMAN INTERVENTION REQUIRED" in line for line in terminal.said)
