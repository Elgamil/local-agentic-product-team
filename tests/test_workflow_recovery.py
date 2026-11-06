"""Crash recovery: power loss at each phase, then a restart from disk alone."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest

import lab.tools.journal as journal
from lab.product.session import ProductSession
from lab.product.store import ProductStore
from lab.product.workflow import ProductWorkflow
from lab.schemas.product import ProductPhase, StepStatus
from tests.fakes import (
    APPROVE,
    CALC_BUGGY,
    CALC_SOURCE,
    QWEN,
    Fail,
    Harness,
    ScriptedTerminal,
    SimulatedPowerLoss,
    to_name_gate,
    to_proposal_gate,
    to_roadmap_gate,
    to_step_start_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Every model-backed task in a normal step, including the verification phases.
MODEL_TASK_KINDS = [
    "discovery",
    "brief",
    "roadmap",
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


async def drive_first_step(workflow: ProductWorkflow) -> None:
    """Act as a human who approves each gate until step 1 is finished."""
    for _ in range(60):
        await workflow.advance()
        phase = workflow.state.phase
        if phase == ProductPhase.DISCOVERY:
            await workflow.discovery_message("I want a small calculator.")
        elif phase == ProductPhase.AWAITING_NAME:
            workflow.confirm_name(APPROVE)
        elif phase == ProductPhase.AWAITING_ROADMAP_APPROVAL:
            workflow.approve_roadmap(APPROVE)
        elif phase == ProductPhase.AWAITING_STEP_START:
            if workflow.state.steps["step-01"].status == StepStatus.COMPLETED:
                return
            workflow.start_step(APPROVE)
        elif phase == ProductPhase.AWAITING_STEP_APPROVAL:
            workflow.approve_step_proposal(APPROVE)
        else:
            raise AssertionError(f"Stuck in {phase}: {workflow.state.blocked}")
    raise AssertionError("Workflow did not finish")


async def baseline_counts(tmp_path: Path) -> Counter:
    harness = Harness(tmp_path / "baseline")
    await drive_first_step(harness.new_workflow())
    return Counter(harness.team.kinds())


@pytest.mark.parametrize("kind", MODEL_TASK_KINDS)
async def test_power_loss_during_a_model_call(tmp_path, kind):
    """The interrupted task resumes, and no finished task runs a second time."""
    expected = await baseline_counts(tmp_path)
    harness = Harness(tmp_path / "crash")
    harness.team.script(kind, Fail(SimulatedPowerLoss()))
    workflow = harness.new_workflow()

    with pytest.raises(SimulatedPowerLoss):
        await drive_first_step(workflow)

    interrupted = harness.manager.discover()[0][0]
    assert interrupted.interrupted_task is not None

    restarted = harness.restart(workflow)
    await drive_first_step(restarted)

    actual = Counter(harness.team.kinds())
    expected[kind] += 1  # Only the call the crash interrupted is repeated.
    assert actual == expected
    assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED
    assert restarted.completion_problems("step-01") == []


@pytest.mark.parametrize("kind", MODEL_TASK_KINDS)
async def test_power_loss_after_the_artifact_was_saved(tmp_path, kind, monkeypatch):
    """The artifact is on disk but the checkpoint is not: nothing runs twice."""
    expected = await baseline_counts(tmp_path)
    harness = Harness(tmp_path / "crash")
    original = ProductWorkflow._finish_task
    armed = {"kind": kind}

    kinds_by_result = {
        "brief": "product_brief",
        "architect": "architecture",
        "proposal": "step_proposal",
        "implement": "implement_file",
        "assess": "test_assessment",
        "review": "implementation_review",
        "closure": "step_closure",
    }
    target = kinds_by_result.get(kind, kind)

    def crash_once(self, result, *args, **kwargs):
        if armed.get("kind") and result.kind == target:
            armed.clear()
            raise SimulatedPowerLoss
        return original(self, result, *args, **kwargs)

    monkeypatch.setattr(ProductWorkflow, "_finish_task", crash_once)
    workflow = harness.new_workflow()

    with pytest.raises(SimulatedPowerLoss):
        await drive_first_step(workflow)

    restarted = harness.restart(workflow)
    await drive_first_step(restarted)

    assert Counter(harness.team.kinds()) == expected
    assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED


#: Phases a normal first step passes through, each one a checkpoint boundary.
BOUNDARY_PHASES = [
    ProductPhase.BRIEF_DRAFTING,
    ProductPhase.AWAITING_NAME,
    ProductPhase.ROADMAP_DRAFTING,
    ProductPhase.AWAITING_ROADMAP_APPROVAL,
    ProductPhase.AWAITING_STEP_START,
    ProductPhase.STEP_PREPARING,
    ProductPhase.STEP_ARCHITECT,
    ProductPhase.STEP_DEVELOPER_PLAN,
    ProductPhase.STEP_TEST_PLAN,
    ProductPhase.STEP_PROPOSAL,
    ProductPhase.AWAITING_STEP_APPROVAL,
    ProductPhase.IMPLEMENTING,
    ProductPhase.VERIFYING_TESTS,
    ProductPhase.REVIEWING,
    ProductPhase.STEP_CLOSING,
]


@pytest.mark.parametrize("target", BOUNDARY_PHASES, ids=lambda phase: phase.value)
@pytest.mark.parametrize("moment", ["before_checkpoint", "after_checkpoint"])
async def test_power_loss_at_each_phase_boundary(tmp_path, monkeypatch, target, moment):
    """Dying either side of a checkpoint repeats no model work and loses no artifact."""
    expected = await baseline_counts(tmp_path)
    harness = Harness(tmp_path / "crash")
    original = ProductStore.save_state
    armed = {"target": target}

    def crash_once(self, state):
        if armed and state.phase == target:
            armed.clear()
            if moment == "after_checkpoint":
                original(self, state)
            raise SimulatedPowerLoss
        return original(self, state)

    monkeypatch.setattr(ProductStore, "save_state", crash_once)
    workflow = harness.new_workflow()

    with pytest.raises(SimulatedPowerLoss):
        await drive_first_step(workflow)
    assert not armed, "the crash point was reached"

    restarted = harness.restart(workflow)
    await drive_first_step(restarted)

    assert Counter(harness.team.kinds()) == expected
    assert restarted.state.roadmap_version == 1, "a crash must not draft an extra roadmap"
    assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED
    assert restarted.completion_problems("step-01") == []


@pytest.mark.parametrize(
    ("reach", "phase"),
    [
        (to_name_gate, ProductPhase.AWAITING_NAME),
        (to_roadmap_gate, ProductPhase.AWAITING_ROADMAP_APPROVAL),
        (to_step_start_gate, ProductPhase.AWAITING_STEP_START),
        (to_proposal_gate, ProductPhase.AWAITING_STEP_APPROVAL),
    ],
)
async def test_a_restart_at_a_gate_asks_again_and_repeats_no_work(tmp_path, reach, phase):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()
    await reach(workflow)
    calls_before = len(harness.team.calls)
    question_before = workflow.gate().question

    restarted = harness.restart(workflow)
    await restarted.advance()

    assert restarted.state.phase == phase
    assert restarted.gate().question == question_before
    assert len(harness.team.calls) == calls_before


async def test_a_restart_never_skips_the_proposal_gate_after_a_crash(tmp_path):
    harness = Harness(tmp_path)
    harness.team.script("proposal", Fail(SimulatedPowerLoss()))
    workflow = harness.new_workflow()
    await to_step_start_gate(workflow)
    workflow.start_step(APPROVE)
    with pytest.raises(SimulatedPowerLoss):
        await workflow.advance()

    restarted = harness.restart(workflow)
    await restarted.advance()

    assert restarted.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
    assert harness.team.calls_of("implement") == []


async def test_a_real_process_kill_mid_task_can_be_resumed(tmp_path):
    """Not simulated: a separate process dies with os._exit during a model call."""
    root = tmp_path / "killed"
    script = textwrap.dedent(
        f"""
        import asyncio, os, sys
        sys.path.insert(0, {str(PROJECT_ROOT)!r})
        from pathlib import Path
        from tests.fakes import Harness, to_step_start_gate, APPROVE

        def die(model, prompt, tools):
            os._exit(137)

        async def main():
            harness = Harness(Path({str(root)!r}))
            harness.team.script("developer_plan", die)
            workflow = harness.new_workflow()
            await to_step_start_gate(workflow)
            workflow.start_step(APPROVE)
            await workflow.advance()

        asyncio.run(main())
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 137, result.stderr

    harness = Harness(root)
    slug = harness.manager.discover()[0][0].slug
    workflow = harness.workflow(harness.manager.open(slug))
    assert workflow.state.active_task.kind == "developer_plan"
    assert workflow.state.phase == ProductPhase.STEP_DEVELOPER_PLAN

    notes = workflow.resume()
    await workflow.advance()

    assert any("interrupted" in note for note in notes)
    assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
    assert harness.team.calls_of("architect") == [], "finished work was not repeated"
    assert len(harness.team.calls_of("developer_plan")) == 1


async def test_resume_rebuilds_the_same_task_from_saved_inputs(tmp_path):
    harness = Harness(tmp_path)
    harness.team.script("architect", Fail(SimulatedPowerLoss()))
    workflow = harness.new_workflow()
    await to_step_start_gate(workflow)
    workflow.start_step(APPROVE)
    with pytest.raises(SimulatedPowerLoss):
        await workflow.advance()
    first_prompt = harness.team.calls_of("architect")[0].prompt

    restarted = harness.restart(workflow)
    await restarted.advance()

    assert harness.team.calls_of("architect")[1].prompt == first_prompt


async def test_failed_attempts_before_a_crash_still_count(tmp_path):
    harness = Harness(tmp_path)
    harness.team.script(
        "architect",
        Fail(ConnectionError("down")),
        Fail(ConnectionError("down")),
        Fail(SimulatedPowerLoss()),
    )
    workflow = harness.new_workflow()
    await to_step_start_gate(workflow)
    workflow.start_step(APPROVE)
    with pytest.raises(SimulatedPowerLoss):
        await workflow.advance()

    restarted = harness.restart(workflow)
    await restarted.advance()

    models = harness.team.models_for("architect")
    assert models[:2] == [QWEN, QWEN]
    assert set(models[2:]) == {"gemma4:latest"}, "the alternate continues after the restart"
    assert "Handover" in harness.team.calls_of("architect")[-1].prompt


class TestInterruptedImplementation:
    async def _approved(self, tmp_path, sources) -> tuple[Harness, ProductWorkflow]:
        harness = Harness(tmp_path)
        harness.team.sources = sources
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        return harness, workflow

    async def test_a_write_that_landed_before_the_crash_is_not_repeated(
        self, tmp_path, monkeypatch
    ):
        harness, workflow = await self._approved(tmp_path, {"calc.py": CALC_SOURCE})
        real_write = journal.atomic_write_bytes

        def land_then_die(target, data, mode=None):
            real_write(target, data, mode=mode)
            raise SimulatedPowerLoss

        monkeypatch.setattr(journal, "atomic_write_bytes", land_then_die)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()
        monkeypatch.setattr(journal, "atomic_write_bytes", real_write)

        restarted = harness.restart(workflow)
        await restarted.advance()

        assert len(harness.team.calls_of("implement")) == 1
        assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED
        stages = [
            operation.stage
            for operation in restarted._workspace("step-01").operations()
        ]
        assert "recovered" in stages

    async def test_a_write_that_never_landed_is_redone(self, tmp_path, monkeypatch):
        harness, workflow = await self._approved(tmp_path, {"calc.py": CALC_SOURCE})
        real_write = journal.atomic_write_bytes

        def die_first(target, data, mode=None):
            raise SimulatedPowerLoss

        monkeypatch.setattr(journal, "atomic_write_bytes", die_first)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()
        monkeypatch.setattr(journal, "atomic_write_bytes", real_write)
        assert not (workflow.store.workspace_dir / "calc.py").exists()

        restarted = harness.restart(workflow)
        await restarted.advance()

        assert len(harness.team.calls_of("implement")) == 2
        assert (restarted.store.workspace_dir / "calc.py").read_text() == CALC_SOURCE

    async def test_a_partly_finished_plan_resumes_at_the_next_file(self, tmp_path):
        harness = Harness(tmp_path)
        import json

        plan = harness.team._default("developer_plan", "step-01")
        plan["file_changes"] = [
            {"path": "calc.py", "action": "create", "purpose": "Logic."},
            {"path": "helpers.py", "action": "create", "purpose": "Helpers."},
        ]
        from tests.fakes import Text

        harness.team.script("developer_plan", Text(json.dumps(plan)))

        def write_calc(model, prompt, tools):
            tools[0]("calc.py", CALC_SOURCE)
            return "Done."

        harness.team.script("implement", write_calc, Fail(SimulatedPowerLoss()))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()

        def write_helpers(model, prompt, tools):
            tools[0]("helpers.py", "VALUE = 1\n")
            return "Done."

        harness.team.script("implement", write_helpers)
        restarted = harness.restart(workflow)
        await restarted.advance()

        targets = [call.prompt for call in harness.team.calls_of("implement")]
        assert "'calc.py'" in targets[0]
        assert "'helpers.py'" in targets[1]
        assert "'helpers.py'" in targets[2]
        assert len(targets) == 3

    async def test_a_file_changed_by_hand_is_reported_and_left_alone(self, tmp_path):
        harness, workflow = await self._approved(tmp_path, {"calc.py": CALC_BUGGY})
        harness.team.script("assess", Fail(SimulatedPowerLoss()))
        harness.team.repaired_sources = {"calc.py": CALC_SOURCE}
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()
        hand_written = "# edited by hand\n" + CALC_SOURCE
        (workflow.store.workspace_dir / "calc.py").write_text(hand_written)

        report = workflow._workspace("step-01").reconcile()

        assert report.drifted == ["calc.py"]
        assert (workflow.store.workspace_dir / "calc.py").read_text() == hand_written

    async def test_changes_during_a_repair_interruption_are_reconciled(self, tmp_path):
        harness, workflow = await self._approved(tmp_path, {"calc.py": CALC_BUGGY})
        harness.team.script("repair", Fail(SimulatedPowerLoss()))
        harness.team.repaired_sources = {"calc.py": CALC_SOURCE}
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()
        assert workflow.state.phase == ProductPhase.REPAIRING

        restarted = harness.restart(workflow)
        await restarted.advance()

        assert restarted.state.steps["step-01"].status == StepStatus.COMPLETED


class TestLongPause:
    async def test_pause_survives_a_restart_and_resume_clears_it(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.pause(APPROVE)

        reopened = harness.workflow(harness.manager.open(workflow.store.slug))
        assert reopened.state.paused is True
        await reopened.advance()
        assert reopened.state.paused is True

        reopened.resume()
        assert reopened.state.paused is False
        assert reopened.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    async def test_a_paused_product_does_no_work(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        workflow.pause(APPROVE)
        calls = len(harness.team.calls)

        await workflow.advance()

        assert len(harness.team.calls) == calls


class TestStartupDiscovery:
    async def test_startup_offers_an_interrupted_task(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", Fail(SimulatedPowerLoss()))
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()

        terminal = ScriptedTerminal("4")
        harness.router = harness.new_router()
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        await session.run()

        said = "\n".join(terminal.said)
        assert "Welcome back" in said
        assert "Pocket Calc" in said
        assert "interrupted" in said

    async def test_session_resume_by_name_returns_to_the_gate(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        calls = len(harness.team.calls)

        terminal = ScriptedTerminal("2", "Pocket Calc", "/exit")
        harness.router = harness.new_router()
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        await session.run()

        assert any("Picking up Pocket Calc" in text for text in terminal.said)
        assert any("Do you approve this proposal?" in text for text in terminal.prompts)
        assert len(harness.team.calls) == calls

    async def test_session_resume_by_number_finishes_an_interrupted_task(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", Fail(SimulatedPowerLoss()))
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        with pytest.raises(SimulatedPowerLoss):
            await workflow.advance()

        terminal = ScriptedTerminal("2", "1", "/exit")
        harness.router = harness.new_router()
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        await session.run()

        assert any("was interrupted" in text for text in terminal.said)
        assert any("Do you approve this proposal?" in text for text in terminal.prompts)
