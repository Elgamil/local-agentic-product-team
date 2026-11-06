"""A whole product through the real workflow, with only the models scripted."""

from __future__ import annotations

import json

import pytest

from lab.product.session import GREETING, ProductSession
from lab.schemas.product import (
    ApprovalKind,
    HandoffRecord,
    ProductPhase,
    StepStatus,
    VerificationEvidence,
)
from lab.tools.fsutil import read_jsonl
from tests.fakes import (
    APPROVE,
    CALC_BUGGY,
    GEMMA,
    QWEN,
    Harness,
    ScriptedTerminal,
    complete_first_step,
    to_proposal_gate,
)


async def test_two_step_product_reaches_completion_with_real_test_evidence(tmp_path):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()

    await complete_first_step(workflow)

    first = workflow.state.steps["step-01"]
    assert first.status == StepStatus.COMPLETED
    evidence = workflow.store.load_model(first.evidence_file, VerificationEvidence)
    assert evidence.passed
    assert [run.scope for run in evidence.runs] == ["acceptance"]
    assert evidence.runs[0].passed_count == 2
    assert (workflow.store.workspace_dir / "calc.py").is_file()

    workflow.start_step(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
    workflow.approve_step_proposal(APPROVE)
    await workflow.advance()

    assert workflow.state.phase == ProductPhase.PRODUCT_COMPLETE
    second = workflow.state.steps["step-02"]
    evidence = workflow.store.load_model(second.evidence_file, VerificationEvidence)
    assert [run.scope for run in evidence.runs] == ["acceptance", "regression"]
    assert all(run.success for run in evidence.runs)


async def test_expected_files_exist_for_a_finished_step(tmp_path):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()
    await complete_first_step(workflow)
    root = workflow.store.root

    for name in ("product.json", "state.json", "brief.md", "roadmap.md", "decisions.md"):
        assert (root / name).is_file(), name
    step = root / "steps" / "step-01"
    for name in (
        "brief.md",
        "architect.md",
        "developer.md",
        "test_plan.md",
        "proposal.md",
        "implementation.md",
        "review.md",
        "test_results.md",
        "closure.md",
        "events.jsonl",
        "handoffs.jsonl",
        "transcript.md",
        "file_ops.jsonl",
    ):
        assert (step / name).is_file(), name
    assert (root / "verification" / "step-01" / "test_calc.py").is_file()
    assert (root / "verification" / "step-01" / "manifest.json").is_file()


async def test_default_routing_matches_the_policy(tmp_path):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()
    await complete_first_step(workflow)
    team = harness.team

    assert set(team.models_for("discovery")) == {GEMMA}
    assert set(team.models_for("proposal")) == {GEMMA}
    assert set(team.models_for("architect")) == {QWEN}
    assert set(team.models_for("review")) == {QWEN}
    assert set(team.models_for("developer_plan")) == {QWEN}
    assert set(team.models_for("implement")) == {QWEN}
    # Opposite of the Developer's model, both when planning and when assessing.
    assert set(team.models_for("test_plan")) == {GEMMA}
    assert set(team.models_for("assess")) == {GEMMA}


async def test_each_handoff_records_who_what_and_how(tmp_path):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()
    await complete_first_step(workflow)

    records, _ = read_jsonl(workflow.store.step_dir("step-01") / "handoffs.jsonl")
    handoffs = [HandoffRecord.model_validate(record) for record in records]
    design = next(item for item in handoffs if "architect" in item.task_id)
    assert (design.sender, design.receiver) == ("software_architect", "developer")
    assert design.model == QWEN
    assert design.validation_status == "validated"
    assert design.decisions == ["Use a single calc module"]
    assert design.rationale and design.handoff_summary
    assert design.relevant_context
    transcript = (workflow.store.step_dir("step-01") / "transcript.md").read_text()
    assert "Software Architect → Developer" in transcript


async def test_test_engineer_never_sees_the_developers_plan(tmp_path):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()
    await to_proposal_gate(workflow)

    prompt = harness.team.calls_of("test_plan")[0].prompt
    assert "## Developer plan" not in prompt
    assert "## Public contracts" in prompt


async def test_failing_tests_are_repaired_from_real_output(tmp_path):
    harness = Harness(tmp_path)
    harness.team.sources = {"calc.py": CALC_BUGGY}
    workflow = harness.new_workflow()
    await complete_first_step(workflow)

    progress = workflow.state.steps["step-01"]
    assert progress.repair_cycles_used == 1
    assert progress.verification_cycle == 2
    assert progress.status == StepStatus.COMPLETED
    repair_prompt = harness.team.calls_of("repair")[0].prompt
    assert "AssertionError" in repair_prompt or "assert" in repair_prompt
    assert "failed" in repair_prompt
    failed = workflow.store.load_model(
        "steps/step-01/test_runs/v1-c1.json", VerificationEvidence
    )
    assert not failed.passed
    assert failed.runs[0].failed_count == 2


async def test_interactive_session_from_greeting_to_first_step(tmp_path):
    harness = Harness(tmp_path)
    terminal = ScriptedTerminal(
        "1",
        "I want a small calculator",
        "yes",  # keep the name
        "yes",  # approve roadmap
        "yes",  # start step 1
        "yes",  # approve proposal
        "/pause",  # at "shall we proceed to step 2"
        "4",
    )
    session = ProductSession(
        harness.manager, harness.router, harness.policy, harness.playbooks, terminal
    )
    await session.run()

    assert GREETING in terminal.said
    assert any("I suggest the product name Pocket Calc" in text for text in terminal.said)
    assert any("products_log/pocket-calc" in text for text in terminal.said)
    assert any("Proposed roadmap for Pocket Calc" in text for text in terminal.said)
    assert any("Proposal for Step 1" in text for text in terminal.said)
    assert any("Step 1 is completed. Shall we proceed to Step 2" in p for p in terminal.prompts)

    summaries, _ = harness.manager.discover()
    assert len(summaries) == 1
    assert summaries[0].paused is True
    assert harness.lifecycle.loaded == set(), "models are released on exit"


async def test_session_output_stays_in_business_language(tmp_path):
    harness = Harness(tmp_path)
    terminal = ScriptedTerminal("1", "I want a small calculator", "yes", "yes", "yes", "/exit")
    session = ProductSession(
        harness.manager, harness.router, harness.policy, harness.playbooks, terminal
    )
    await session.run()

    for line in [*terminal.said, *terminal.prompts]:
        assert "{\"" not in line
        assert "ready_for_brief" not in line
        assert "schema" not in line.lower()


async def test_approval_log_binds_each_decision_to_an_artifact_hash(tmp_path):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()
    await complete_first_step(workflow)

    approvals = workflow.store.read_approvals()
    assert {record.decided_by for record in approvals} == {"human"}
    proposal = next(item for item in approvals if item.kind == ApprovalKind.STEP_PROPOSAL)
    assert proposal.artifact_sha256 == workflow.state.steps["step-01"].approved_proposal_sha256
    assert json.loads(
        (workflow.store.logs_dir / "approvals.jsonl").read_text().splitlines()[0]
    )


@pytest.mark.parametrize("answer", ["no", "/reject"])
async def test_declining_a_proposal_builds_nothing(tmp_path, answer):
    harness = Harness(tmp_path)
    workflow = harness.new_workflow()
    await to_proposal_gate(workflow)
    terminal = ScriptedTerminal(answer)
    session = ProductSession(
        harness.manager, harness.router, harness.policy, harness.playbooks, terminal
    )
    await session._on_gate(workflow, answer)

    assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
    assert harness.team.calls_of("implement") == []
    assert list(workflow.store.workspace_dir.iterdir()) == []
