"""Workspace boundaries, tool binding, independent test protection, and the journal."""

from __future__ import annotations

import os

import pytest

from lab.schemas.product import (
    BlockKind,
    DeveloperPlan,
    ExecutableTestFile,
    PlannedFileChange,
    PlannedTestFile,
    ProductPhase,
)
from lab.tools.developer_tools import ProjectWorkspace, SecurityError, build_developer_tools
from lab.tools.journal import JournaledWorkspace, workspace_fingerprint
from pydantic import ValidationError
from tests.fakes import (
    APPROVE,
    CALC_SOURCE,
    Harness,
    complete_first_step,
    to_name_gate,
    to_proposal_gate,
)


def tools_for(workspace):
    write, read, listing = build_developer_tools(workspace)
    return write, read, listing


class TestToolBinding:
    def test_tools_keep_the_names_the_prompts_use(self, tmp_path):
        names = [tool.__name__ for tool in build_developer_tools(ProjectWorkspace(tmp_path))]
        assert names == ["write_project_file", "read_project_file", "list_project_files"]

    def test_tools_are_bound_to_their_own_product(self, tmp_path):
        first = ProjectWorkspace(tmp_path / "one")
        second = ProjectWorkspace(tmp_path / "two")
        write_one, _, _ = tools_for(first)
        write_two, _, list_two = tools_for(second)

        write_one("shared.txt", "from one")
        write_two("other.txt", "from two")

        assert list_two() == ["other.txt"]
        assert (tmp_path / "one" / "shared.txt").read_text() == "from one"

    def test_binding_does_not_depend_on_the_legacy_singleton(self, tmp_path):
        from lab.tools.developer_tools import set_active_workspace

        workspace = ProjectWorkspace(tmp_path / "product")
        write, _, _ = tools_for(workspace)
        set_active_workspace(tmp_path / "elsewhere")

        write("a.txt", "x")

        assert (tmp_path / "product" / "a.txt").exists()
        assert not (tmp_path / "elsewhere" / "a.txt").exists()

    def test_importing_the_tools_creates_no_directories(self, tmp_path):
        import subprocess
        import sys

        code = "import lab.tools.developer_tools as d; print(d._ACTIVE_WORKSPACE)"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, cwd=tmp_path
        )
        assert result.stdout.strip() == "None"

    def test_refused_paths_return_an_error_the_model_can_read(self, tmp_path):
        write, read, _ = tools_for(ProjectWorkspace(tmp_path / "w"))
        assert write("../outside.txt", "x").startswith("ERROR:")
        assert read("/etc/passwd").startswith("ERROR:")
        assert read("missing.txt").startswith("ERROR:")
        assert not (tmp_path / "outside.txt").exists()


class TestBoundaries:
    @pytest.mark.parametrize(
        "path",
        [
            "../escape.txt",
            "a/../../escape.txt",
            "/etc/passwd",
            "~/secrets",
            ".git/config",
            ".venv/bin/python",
            "tests/acceptance/test_hack.py",
            "nested/acceptance_tests/x.py",
            "   ",
        ],
    )
    def test_dangerous_paths_are_refused(self, tmp_path, path):
        workspace = ProjectWorkspace(tmp_path / "w")
        with pytest.raises(SecurityError):
            if path == "~/secrets":
                workspace._validate_safe_path("../~/secrets")
            else:
                workspace.write_file(path, "x")

    def test_a_symlink_cannot_lead_out_of_the_workspace(self, tmp_path):
        workspace = ProjectWorkspace(tmp_path / "w")
        (tmp_path / "secret.txt").write_text("private")
        os.symlink(tmp_path / "secret.txt", workspace.root / "link.txt")
        with pytest.raises(SecurityError):
            workspace.read_file("link.txt")

    def test_protected_words_in_the_parent_path_do_not_break_a_workspace(self, tmp_path):
        """A product may live anywhere, including under a folder called tests/acceptance."""
        workspace = ProjectWorkspace(tmp_path / ".git-backups" / "tests" / "acceptance" / "w")
        workspace.write_file("app.py", "x = 1")
        assert (workspace.root / "app.py").exists()

    def test_ordinary_dotfiles_are_allowed(self, tmp_path):
        workspace = ProjectWorkspace(tmp_path / "w")
        workspace.write_file(".gitignore", "*.pyc\n")
        workspace.write_file(".github/workflows/ci.yml", "x: 1\n")
        assert (workspace.root / ".gitignore").exists()

    def test_listing_skips_generated_directories(self, tmp_path):
        workspace = ProjectWorkspace(tmp_path / "w")
        (workspace.root / "__pycache__").mkdir()
        (workspace.root / "__pycache__" / "x.pyc").write_text("x")
        workspace.write_file("real.py", "x = 1")
        assert workspace.list_files() == ["real.py"]

    @pytest.mark.parametrize("path", ["../x.py", "/abs/x.py", "a/../../x.py", "..\\x.py"])
    def test_model_supplied_paths_are_rejected_at_the_schema(self, path):
        with pytest.raises(ValidationError):
            PlannedFileChange(path=path, action="create", purpose="x")
        with pytest.raises(ValidationError):
            PlannedTestFile(path=path, purpose="x")
        with pytest.raises(ValidationError):
            ExecutableTestFile(path=path, content="x")

    def test_a_plan_cannot_list_the_same_file_twice(self):
        change = {"path": "a.py", "action": "create", "purpose": "x"}
        with pytest.raises(ValidationError):
            DeveloperPlan(
                summary="s", approach="a", file_changes=[change, change], handoff_summary="h"
            )


class TestIndependentVerification:
    async def test_the_developer_workspace_cannot_reach_verification(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)

        workspace = workflow._workspace("step-01")
        write, read, _ = build_developer_tools(workspace)

        assert write("../verification/step-01/test_calc.py", "weakened").startswith("ERROR:")
        assert read("../verification/step-01/test_calc.py").startswith("ERROR:")
        sealed = (workflow.store.verification_dir / "step-01" / "test_calc.py").read_text()
        assert "weakened" not in sealed

    async def test_a_developer_attempt_to_change_tests_is_recorded_and_harmless(self, tmp_path):
        harness = Harness(tmp_path)

        def tries_to_cheat(model, prompt, tools):
            result = tools[0]("../verification/step-01/test_calc.py", "def test_x(): pass")
            assert result.startswith("ERROR:")
            tools[0]("calc.py", CALC_SOURCE)
            return "Done."

        harness.team.script("implement", tries_to_cheat)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert "def test_add_positive" in (
            workflow.store.verification_dir / "step-01" / "test_calc.py"
        ).read_text()

    async def test_the_developer_is_given_only_three_workspace_tools(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await complete_first_step(workflow)
        assert harness.team.calls_of("implement")[0].tools == [
            "write_project_file",
            "read_project_file",
            "list_project_files",
        ]

    async def test_tests_are_sealed_at_approval_not_before(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        assert not workflow.verification.is_sealed("step-01")

        workflow.approve_step_proposal(APPROVE)

        assert workflow.verification.is_sealed("step-01")
        mode = (workflow.store.verification_dir / "step-01" / "test_calc.py").stat().st_mode
        assert mode & 0o222 == 0, "sealed tests are read-only"

    async def test_tampering_with_a_sealed_test_stops_verification(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        sealed = workflow.store.verification_dir / "step-01" / "test_calc.py"
        os.chmod(sealed, 0o644)
        sealed.write_text("def test_nothing():\n    assert True\n")

        await workflow.advance()

        assert workflow.state.phase == ProductPhase.BLOCKED
        assert workflow.state.blocked.kind == BlockKind.INTEGRITY
        assert "test_calc.py" in workflow.state.blocked.message

    async def test_adding_a_file_to_a_sealed_suite_is_detected(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        (workflow.store.verification_dir / "step-01" / "conftest.py").write_text("x = 1\n")

        await workflow.advance()

        assert workflow.state.blocked.kind == BlockKind.INTEGRITY

    async def test_retrying_a_tampered_suite_is_refused(self, tmp_path):
        from lab.product.workflow import GateViolation, HumanDecision

        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        (workflow.store.verification_dir / "step-01" / "conftest.py").write_text("x = 1\n")
        await workflow.advance()

        with pytest.raises(GateViolation):
            workflow.retry_blocked(HumanDecision("yes"))

    async def test_a_new_proposal_regenerates_a_tampered_suite(self, tmp_path):
        from lab.product.workflow import HumanDecision

        harness = Harness(tmp_path)
        harness.team.script("route_modification")
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        extra = workflow.store.verification_dir / "step-01" / "conftest.py"
        extra.write_text("x = 1\n")
        await workflow.advance()

        await workflow.request_step_modification(HumanDecision("Rebuild the tests"))
        await workflow.advance()

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert not extra.exists()
        assert len(harness.team.calls_of("test_plan")) == 2

    async def test_integrity_failure_regenerates_tests_even_if_routing_says_otherwise(
        self, tmp_path
    ):
        """The BA's routing is advisory: it cannot keep a suite that failed its seal."""
        import json

        from lab.product.workflow import HumanDecision
        from tests.fakes import Text

        harness = Harness(tmp_path)
        harness.team.script(
            "route_modification",
            Text(json.dumps({"restated_request": "Tweak", "affected_agents": ["developer"]})),
        )
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        extra = workflow.store.verification_dir / "step-01" / "conftest.py"
        extra.write_text("x = 1\n")
        await workflow.advance()

        await workflow.request_step_modification(HumanDecision("Tweak the code"))
        await workflow.advance()

        assert not extra.exists()
        assert len(harness.team.calls_of("test_plan")) == 2

    async def test_a_tampered_suite_cannot_be_resealed_by_approval(self, tmp_path):
        """A suite carried into a new proposal must prove itself before it is sealed."""
        import json

        from lab.product.workflow import GateViolation, HumanDecision
        from lab.schemas.product import StepStatus
        from tests.fakes import CALC_BUGGY, Text

        harness = Harness(tmp_path)
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        harness.team.script(
            "route_modification",
            Text(json.dumps({"restated_request": "Fix", "affected_agents": ["developer"]})),
        )
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        workflow.approve_step_proposal(APPROVE)
        await workflow.advance()
        assert workflow.state.blocked.kind == BlockKind.VERIFICATION_FAILED

        await workflow.request_step_modification(HumanDecision("Fix the code"))
        await workflow.advance()
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        (workflow.store.verification_dir / "step-01" / "conftest.py").write_text("x = 1\n")

        with pytest.raises(GateViolation, match="changed after approval"):
            workflow.approve_step_proposal(APPROVE)
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert workflow.state.steps["step-01"].status != StepStatus.COMPLETED

    async def test_a_test_path_outside_the_profile_is_refused(self, tmp_path):
        import json

        from tests.fakes import Text

        harness = Harness(tmp_path)
        plan = harness.team._default("test_plan", "step-01")
        plan["test_files"] = [{"path": "not_a_test.txt", "purpose": "x"}]
        harness.team.script("test_plan", Text(json.dumps(plan)))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        calls = harness.team.calls_of("test_plan")
        assert len(calls) == 2
        assert "does not match" in calls[1].prompt

    async def test_an_unknown_verification_profile_is_refused(self, tmp_path):
        import json

        from tests.fakes import Text

        harness = Harness(tmp_path)
        plan = harness.team._default("test_plan", "step-01")
        plan["verification_profile"] = "rm -rf /"
        harness.team.script("test_plan", Text(json.dumps(plan)))
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        assert "cannot be used here" in harness.team.calls_of("test_plan")[1].prompt

    async def test_a_model_cannot_ask_for_a_command(self, tmp_path):
        """The test plan schema carries a profile id.  There is no field for a command."""
        from lab.schemas.product import StepTestPlan

        fields = set(StepTestPlan.model_fields)
        assert not fields & {"command", "commands", "shell", "args", "cwd"}


class TestJournal:
    @pytest.fixture
    def workspace(self, tmp_path):
        return JournaledWorkspace(tmp_path / "w", tmp_path / "ops.jsonl")

    def test_a_write_records_intent_and_commit_with_hashes(self, workspace):
        workspace.current_task = "task-1"
        workspace.write_file("a.txt", "one")
        intent, commit = workspace.operations()
        assert (intent.stage, commit.stage) == ("intent", "commit")
        assert intent.op_id == commit.op_id
        assert intent.sha256_before is None
        assert intent.sha256_after == commit.sha256_after
        assert commit.task_id == "task-1"

    def test_repeating_the_same_write_is_a_recorded_no_op(self, workspace):
        workspace.write_file("a.txt", "one")
        message = workspace.write_file("a.txt", "one")
        assert "nothing to do" in message
        assert workspace.operations()[-1].unchanged is True

    def test_committed_for_task_is_scoped(self, workspace):
        workspace.current_task = "one"
        workspace.write_file("a.txt", "1")
        workspace.current_task = "two"
        workspace.write_file("b.txt", "2")
        assert set(workspace.committed_for_task("one")) == {"a.txt"}

    def test_refused_writes_never_reach_the_journal(self, workspace):
        with pytest.raises(SecurityError):
            workspace.write_file("../x.txt", "no")
        assert workspace.operations() == []
        assert workspace.operations_log[-1].success is False

    def test_paths_are_normalized_in_the_journal(self, workspace):
        workspace.write_file("./sub//a.txt", "x")
        assert set(workspace.latest_committed()) == {"sub/a.txt"}

    def test_reconcile_confirms_a_clean_write(self, workspace):
        workspace.write_file("a.txt", "one")
        assert workspace.reconcile().confirmed == ["a.txt"]

    def test_reconcile_finds_an_interrupted_write_that_landed(self, workspace, tmp_path):
        from lab.schemas.product import JournaledFileOperation
        from lab.tools.fsutil import append_line, sha256_text

        (workspace.root / "a.txt").write_text("new")
        append_line(
            workspace.journal_path,
            JournaledFileOperation(
                op_id="1", stage="intent", path="a.txt", sha256_after=sha256_text("new")
            ).model_dump_json(),
        )
        report = workspace.reconcile()
        assert report.recovered == ["a.txt"]
        assert workspace.operations()[-1].stage == "recovered"
        assert workspace.reconcile().confirmed == ["a.txt"]

    def test_reconcile_finds_a_write_that_never_landed(self, workspace):
        from lab.schemas.product import JournaledFileOperation
        from lab.tools.fsutil import append_line, sha256_text

        (workspace.root / "a.txt").write_text("old")
        append_line(
            workspace.journal_path,
            JournaledFileOperation(
                op_id="1",
                stage="intent",
                path="a.txt",
                sha256_before=sha256_text("old"),
                sha256_after=sha256_text("new"),
            ).model_dump_json(),
        )
        assert workspace.reconcile().not_applied == ["a.txt"]

    def test_reconcile_flags_changes_made_outside_the_platform(self, workspace):
        workspace.write_file("a.txt", "one")
        (workspace.root / "a.txt").write_text("edited by hand")
        report = workspace.reconcile()
        assert report.drifted == ["a.txt"]
        assert (workspace.root / "a.txt").read_text() == "edited by hand"


class TestFingerprint:
    def test_it_changes_with_source_and_ignores_runtime_noise(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1")
        before = workspace_fingerprint(tmp_path)

        (tmp_path / "app.db").write_text("runtime data")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "app.pyc").write_text("x")
        assert workspace_fingerprint(tmp_path) == before

        (tmp_path / "app.py").write_text("x = 2")
        assert workspace_fingerprint(tmp_path) != before


class TestProductIndependence:
    async def test_no_minishop_reference_in_the_generic_platform(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "lab"
        for name in ("workflow.py", "session.py", "context.py", "store.py", "objectives.py"):
            assert "minishop" not in (root / "product" / name).read_text().lower(), name
        assert "minishop" not in (root / "runtime" / "router.py").read_text().lower()

    async def test_a_first_product_does_not_touch_the_repository(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_name_gate(workflow)
        assert harness.root in workflow.store.root.parents
