"""What a model receives: bounded context, and clean reconstruction after a switch."""

from __future__ import annotations

import json

import pytest

from lab.product.context import ContextBuilder, ContextRequest
from lab.schemas.product import ModelAttempt, FailureClass
from tests.fakes import (
    APPROVE,
    GEMMA,
    QWEN,
    Fail,
    Harness,
    Text,
    complete_first_step,
    to_proposal_gate,
    to_step_start_gate,
)


@pytest.fixture
def harness(tmp_path):
    return Harness(tmp_path)


def failure(model: str = QWEN, number: int = 1) -> ModelAttempt:
    return ModelAttempt(
        attempt_id=f"a{number}",
        task_id="t",
        model=model,
        model_attempt=number,
        overall_attempt=number,
        status="failed",
        failure_class=FailureClass.SCHEMA_VALIDATION,
        error="components: Field required",
    )


class TestSwitchReconstruction:
    async def _failed_over_prompt(self, harness) -> str:
        harness.team.script("developer_plan", Text("MALFORMED_OUTPUT"), Text("MALFORMED_OUTPUT"))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        calls = harness.team.calls_of("developer_plan")
        assert [call.model for call in calls] == [QWEN, QWEN, GEMMA]
        return calls[2].prompt

    async def test_the_alternate_receives_everything_the_spec_lists(self, harness):
        prompt = await self._failed_over_prompt(harness)
        for expected in (
            "Original task",
            "Plan the implementation of Step 1",
            "Product brief",
            "Roadmap",
            "Current roadmap step",
            "Approved design",
            "Earlier decisions",
            "Last successful handoff",
            "Exact failure to avoid",
            "Previous attempts",
            "Already completed for this step",
            "What remains",
        ):
            assert expected in prompt, expected

    async def test_the_alternate_is_told_to_resume_not_restart(self, harness):
        prompt = await self._failed_over_prompt(harness)
        assert "Resume this one task" in prompt
        assert "Do NOT restart the product" in prompt

    async def test_it_never_contains_the_failed_output(self, harness):
        prompt = await self._failed_over_prompt(harness)
        assert "MALFORMED_OUTPUT" not in prompt

    async def test_it_lists_what_succeeded(self, harness):
        prompt = await self._failed_over_prompt(harness)
        assert "architecture (software_architect) finished" in prompt

    async def test_each_failed_attempt_is_summarised(self, harness):
        prompt = await self._failed_over_prompt(harness)
        assert "Attempt 1 on qwen2.5:7b: invalid_json" in prompt
        assert "Attempt 2 on qwen2.5:7b: invalid_json" in prompt

    async def test_failover_does_not_transfer_the_transcript(self, harness):
        workflow = harness.new_workflow()
        await workflow.discovery_message("A distinctive first message about zebras.")
        await workflow.advance()
        harness.team.script("roadmap", Text("x"), Text("x"))
        workflow.confirm_name(APPROVE)
        await workflow.advance()
        assert "zebras" not in harness.team.calls_of("roadmap")[-1].prompt

    async def test_the_handover_stays_within_the_alternates_budget(self, harness):
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        huge = {"Y": "y" * 60000}
        (workflow.store.workspace_dir / "large.py").write_text(json.dumps(huge))
        harness.team.script("architect", Text("x"), Text("x"))
        workflow.start_step(APPROVE)
        await workflow.advance()

        prompt = harness.team.calls_of("architect")[2].prompt
        schema_allowance = 6000
        assert len(prompt) < workflow.context.budget_chars(GEMMA) + schema_allowance

    async def test_a_second_attempt_on_the_alternate_keeps_the_handover(self, harness):
        harness.team.script(
            "developer_plan", Text("bad"), Text("bad"), Text("bad")
        )
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        last = harness.team.calls_of("developer_plan")[3].prompt
        assert "Handover" in last
        assert "CORRECTION" in last


class TestAttemptAudit:
    async def test_the_exact_prompt_of_each_attempt_is_saved(self, harness):
        harness.team.script("architect", Text("bad"))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        folder = workflow.store.task_dir("step-01-architect-v1")
        calls = harness.team.calls_of("architect")
        assert (folder / "input-r1-a1.md").read_text() == calls[0].prompt
        assert (folder / "input-r1-a2.md").read_text() == calls[1].prompt

    async def test_a_cut_off_answer_gets_specific_advice(self, harness):
        truncated = '{"summary": "A plain module", "design": '
        harness.team.script("architect", Text(truncated))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        assert "keep every field much shorter" in harness.team.calls_of("architect")[1].prompt


class TestBudget:
    def test_budget_follows_each_models_context_window(self, harness):
        store = harness.manager.create("Sample")
        builder = ContextBuilder(store, harness.policy, harness.playbooks, {})
        assert builder.budget_chars(QWEN) > builder.budget_chars(GEMMA)

    def test_sections_are_capped(self, harness):
        store = harness.manager.create("Sample")
        store.write_text("brief.md", "b" * 100000)
        builder = ContextBuilder(store, harness.policy, harness.playbooks, {})
        prompt = builder.task_prompt(
            "Do it.", ContextRequest(sections=("brief",)), GEMMA
        )
        assert len(prompt) < harness.policy.context.max_section_chars + 200
        assert "shortened to fit" in prompt

    def test_total_size_never_exceeds_the_budget(self, harness):
        store = harness.manager.create("Sample")
        store.write_text("brief.md", "b" * 100000)
        store.write_text("decisions.md", "d" * 100000)
        builder = ContextBuilder(store, harness.policy, harness.playbooks, {})
        request = ContextRequest(
            sections=("brief", "decisions", "one", "two"),
            extra={"one": "o" * 100000, "two": "t" * 100000},
        )
        prompt = builder.task_prompt("Do it.", request, GEMMA)
        assert len(prompt) <= builder.budget_chars(GEMMA) + 200

    def test_the_task_itself_is_never_cut(self, harness):
        store = harness.manager.create("Sample")
        store.write_text("brief.md", "b" * 100000)
        builder = ContextBuilder(store, harness.policy, harness.playbooks, {})
        prompt = builder.task_prompt(
            "THE ACTUAL TASK", ContextRequest(sections=("brief",)), GEMMA
        )
        assert prompt.endswith("THE ACTUAL TASK")


class TestRetrieval:
    def _builder(self, harness):
        store = harness.manager.create("Sample")
        (store.workspace_dir / "cart.py").write_text("def checkout_cart(): pass\n")
        (store.workspace_dir / "weather.py").write_text("def forecast(): pass\n")
        (store.workspace_dir / "logo.png").write_bytes(b"\x89PNG")
        return ContextBuilder(store, harness.policy, harness.playbooks, {})

    def test_files_are_ranked_by_relevance(self, harness):
        builder = self._builder(harness)
        chosen = builder._retrieve(ContextRequest(sections=(), query="checkout cart total"))
        assert chosen == ["cart.py"]

    def test_named_paths_always_come_first(self, harness):
        builder = self._builder(harness)
        chosen = builder._retrieve(
            ContextRequest(sections=(), query="checkout cart", paths=("weather.py",))
        )
        assert chosen[0] == "weather.py"

    def test_binary_files_are_not_retrieved(self, harness):
        builder = self._builder(harness)
        assert "logo.png" not in builder._retrieve(ContextRequest(sections=(), query="png logo"))

    def test_retrieval_is_bounded(self, harness):
        harness.policy.context.retrieval_top_k = 1
        store = harness.manager.create("Many")
        for index in range(5):
            (store.workspace_dir / f"m{index}.py").write_text("shared_word\n")
        builder = ContextBuilder(store, harness.policy, harness.playbooks, {})
        assert len(builder._retrieve(ContextRequest(sections=(), query="shared_word"))) == 1


class TestRelevantContextPerRole:
    async def test_a_later_step_sees_the_existing_architecture(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()

        prompt = harness.team.calls_of("architect")[-1].prompt
        assert "Existing architecture from completed steps" in prompt
        assert "A plain calc module" in prompt
        assert "calc.py" in prompt

    async def test_the_architect_sees_earlier_decisions(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        assert "Use a single calc module" in harness.team.calls_of("architect")[-1].prompt

    async def test_repair_receives_actual_test_output(self, harness):
        from tests.fakes import CALC_BUGGY

        harness.team.sources = {"calc.py": CALC_BUGGY}
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        prompt = harness.team.calls_of("repair")[0].prompt
        assert "test_add_positive" in prompt
        assert "Command:" in prompt

    async def test_the_reviewer_sees_the_actual_files(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        prompt = harness.team.calls_of("review")[0].prompt
        assert "def add(a, b):" in prompt
        assert "2 passed" in prompt

    async def test_closure_is_told_the_systems_verdict(self, harness):
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        prompt = harness.team.calls_of("closure")[0].prompt
        assert "verification verdict: PASSED" in prompt

    async def test_failover_of_the_business_analyst_uses_qwen(self, harness):
        harness.team.script("discovery", Fail(ConnectionError("x")), Fail(ConnectionError("x")))
        workflow = harness.new_workflow()
        await workflow.discovery_message("A calculator.")
        assert harness.team.models_for("discovery") == [GEMMA, GEMMA, QWEN]
