"""Reading real test output.  A report is only as good as this.

The runners here are real where possible, so a change in how a test tool prints its
results fails these tests rather than quietly producing an empty report.
"""

from __future__ import annotations

import shutil

import pytest

from lab.config import load_verification_profiles
from lab.product.intervention import (
    ReportInputs,
    build_report,
    compare_suites,
    defined_tests,
    diagnose_run,
    report_markdown,
    revision_summary,
    terminal_summary,
    unified_diff,
)
from lab.schemas.product import (
    BlockedInfo,
    BlockKind,
    FailureClass,
    ModelAttempt,
    ProductPhase,
    VerificationEvidence,
    VerificationRun,
)
from lab.tools.verification import run_profile

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")

# The exact shape of the live failure in the Tic-Tac-Toe run, with its paths shortened.
LIVE_NODE_OUTPUT = """TAP version 13
# file:///p/verification/step-01/test_board_structure/test_board_structure.test.mjs:1
# import { test, describe, expect } from 'node:test';
#                          ^^^^^^
# SyntaxError: The requested module 'node:test' does not provide an export named 'expect'
#     at \\#_instantiate (node:internal/modules/esm/module_job:254:21)
#     at async ModuleJob.run (node:internal/modules/esm/module_job:369:5)
# Node.js v24.9.0
# Subtest: /p/verification/step-01/test_board_structure/test_board_structure.test.mjs
not ok 1 - /p/verification/step-01/test_board_structure/test_board_structure.test.mjs
  ---
  duration_ms: 43.089834
  type: 'test'
  location: '/p/verification/step-01/test_board_structure/test_board_structure.test.mjs:1:1'
  failureType: 'testCodeFailure'
  exitCode: 1
  signal: ~
  error: 'test failed'
  code: 'ERR_TEST_FAILURE'
  ...
1..1
# tests 1
# suites 0
# pass 0
# fail 1
# cancelled 0
# skipped 0
# todo 0
# duration_ms 47.315292
"""
LIVE_TEST_FILE = "import { test, describe, expect } from 'node:test';\n"


def run_of(output: str, *, exit_code=1, passed=0, failed=1, timed_out=False) -> VerificationRun:
    return VerificationRun(
        command="[]",
        exit_code=exit_code,
        passed_count=passed,
        failed_count=failed,
        duration_ms=1.0,
        stdout=output,
        stderr="",
        success=exit_code == 0 and failed == 0 and passed > 0,
        step_id="step-01",
        scope="acceptance",
        profile_id="p",
        cycle=1,
        cwd="/w",
        timed_out=timed_out,
    )


def real_run(profile_id: str, tests, workspace):
    profile = load_verification_profiles()[profile_id]
    raw = run_profile(profile, tests, workspace)
    return (
        VerificationRun(
            command=raw.command,
            exit_code=raw.exit_code,
            passed_count=raw.passed_count,
            failed_count=raw.failed_count,
            duration_ms=raw.duration_ms,
            stdout=raw.stdout,
            stderr=raw.stderr,
            success=raw.success,
            step_id="step-01",
            scope="acceptance",
            profile_id=profile_id,
            cycle=1,
            cwd=raw.cwd,
            timed_out=raw.timed_out,
        ),
        profile,
    )


def diagnose_real(profile_id: str, files: dict[str, str], tmp_path, product: dict[str, str] = {}):
    tests = tmp_path / "tests"
    workspace = tmp_path / "workspace"
    tests.mkdir()
    workspace.mkdir()
    for name, text in files.items():
        target = tests / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    for name, text in product.items():
        (workspace / name).write_text(text)
    run, profile = real_run(profile_id, tests, workspace)
    return diagnose_run(
        run, profile.result_parser, files, profile.test_file_patterns, "verification/step-01"
    )


class TestTheLiveFailure:
    def diagnose(self):
        return diagnose_run(
            run_of(LIVE_NODE_OUTPUT),
            "tap",
            {"test_board_structure/test_board_structure.test.mjs": LIVE_TEST_FILE},
            ["*.test.mjs"],
            "verification/step-01",
        )

    def test_the_suite_is_recognised_as_unable_to_run(self):
        assert self.diagnose().executable is False

    def test_the_exact_error_is_found(self):
        assert self.diagnose().errors == [
            "SyntaxError: The requested module 'node:test' does not provide an export "
            "named 'expect'"
        ]

    def test_the_file_is_named(self):
        assert self.diagnose().files == [
            "verification/step-01/test_board_structure/test_board_structure.test.mjs"
        ]

    def test_having_no_test_cases_is_a_finding(self):
        assert "defines no test cases" in self.diagnose().static_findings[0]

    def test_the_file_is_not_presented_as_a_failing_test(self):
        failure = self.diagnose().failed_tests[0]
        assert failure.name == "test_board_structure.test.mjs (the file did not run)"
        assert failure.message == ""


class TestPytestOutput:
    def test_assertion_failures(self, tmp_path):
        found = diagnose_real(
            "python_pytest",
            {"test_calc.py": "import calc\n\ndef test_add():\n    assert calc.add(2, 3) == 5\n"},
            tmp_path,
            {"calc.py": "def add(a, b):\n    return a - b\n"},
        )

        assert found.executable is True
        failure = found.failed_tests[0]
        assert (failure.name, failure.file) == ("test_add", "test_calc.py")
        assert (failure.actual, failure.expected) == ("-1", "5")
        assert found.files == ["verification/step-01/test_calc.py"]

    def test_a_syntax_error_means_it_could_not_run(self, tmp_path):
        found = diagnose_real("python_pytest", {"test_bad.py": "def test_x(:\n"}, tmp_path)

        assert found.executable is False
        assert any("SyntaxError" in line for line in found.errors)
        assert found.files == ["verification/step-01/test_bad.py"]

    def test_a_failed_import_means_it_could_not_run(self, tmp_path):
        found = diagnose_real(
            "python_pytest", {"test_x.py": "import missing_thing\n\ndef test_a():\n    pass\n"},
            tmp_path,
        )

        assert found.executable is False
        assert any("missing_thing" in line for line in found.errors)

    def test_a_file_with_no_tests(self, tmp_path):
        found = diagnose_real("python_pytest", {"test_x.py": "x = 1\n"}, tmp_path)

        assert found.executable is False
        assert (found.passed, found.failed) == (0, 0)
        assert "defines no test cases" in found.static_findings[0]

    def test_only_the_failing_file_is_named(self, tmp_path):
        found = diagnose_real(
            "python_pytest",
            {
                "test_good.py": "def test_ok():\n    assert True\n",
                "test_bad.py": "def test_no():\n    assert False\n",
            },
            tmp_path,
        )

        assert found.executable is True
        assert found.files == ["verification/step-01/test_bad.py"]

    def test_a_support_file_is_never_blamed(self, tmp_path):
        found = diagnose_real(
            "python_pytest",
            {"pytest.ini": "[pytest]\n", "test_bad.py": "def test_no():\n    assert False\n"},
            tmp_path,
        )

        assert found.files == ["verification/step-01/test_bad.py"]


class TestUnittestOutput:
    SOURCE = (
        "import unittest\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_one(self):\n"
        "        self.assertEqual(1, 2)\n"
    )

    def test_assertion_failures(self, tmp_path):
        found = diagnose_real("python_unittest", {"test_a.py": self.SOURCE}, tmp_path)

        assert found.executable is True
        failure = found.failed_tests[0]
        assert (failure.name, failure.file) == ("test_one", "test_a.py")
        assert "1 != 2" in failure.message

    def test_a_failed_import_means_it_could_not_run(self, tmp_path):
        found = diagnose_real(
            "python_unittest",
            {"test_a.py": "import missing_thing\n"},
            tmp_path,
        )

        assert found.executable is False
        assert found.files == ["verification/step-01/test_a.py"]
        assert any("missing_thing" in line for line in found.errors)


@needs_node
class TestNodeOutput:
    def test_a_bad_import_could_not_run(self, tmp_path):
        found = diagnose_real("node_test", {"a.test.mjs": LIVE_TEST_FILE}, tmp_path)

        assert found.executable is False
        assert any("does not provide an export" in line for line in found.errors)

    def test_assertion_failures_with_expected_and_actual(self, tmp_path):
        source = (
            "import { test } from 'node:test';\n"
            "import assert from 'node:assert';\n"
            "test('adds', () => { assert.strictEqual(1 + 1, 3); });\n"
            "test('fine', () => { assert.ok(true); });\n"
        )
        found = diagnose_real("node_test", {"a.test.mjs": source}, tmp_path)

        assert found.executable is True
        failure = found.failed_tests[0]
        assert failure.name == "adds"
        assert failure.file == "a.test.mjs"
        assert (failure.actual, failure.expected) == ("2", "3")


class TestRunsThatNeverFinished:
    def test_a_timeout_is_not_a_test_failure(self):
        found = diagnose_run(
            run_of("", exit_code=-1, failed=0, timed_out=True), "pytest", {}, ["test_*.py"], "v"
        )
        assert found.executable is False
        assert found.timed_out is True

    def test_output_is_bounded(self):
        found = diagnose_run(run_of("x" * 90_000), "pytest", {}, ["test_*.py"], "v")
        assert len(found.diagnostic_output) <= 4000


class TestReadingSource:
    @pytest.mark.parametrize(
        ("source", "parser", "names"),
        [
            ("def test_a():\n    pass\n\nasync def test_b():\n    pass\n", "pytest", ["test_a", "test_b"]),
            ("class T:\n    def test_c(self):\n        pass\n", "unittest", ["test_c"]),
            ("test('adds', () => {});\nit(\"keeps\", () => {});\n", "tap", ["adds", "keeps"]),
            ("describe('x', () => { test('in', () => {}); });\n", "tap", ["in"]),
            ("x = 1\n", "pytest", []),
            (LIVE_TEST_FILE, "tap", []),
        ],
    )
    def test_defined_tests(self, source, parser, names):
        assert defined_tests(source, parser) == names


class TestComparingSuites:
    PATTERNS = ["test_*.py"]
    OLD = "def test_a():\n    assert 1\n\n\ndef test_b():\n    assert 2\n"

    def compare(self, new: dict[str, str], old: dict | None = None) -> list[str]:
        return compare_suites(old or {"test_x.py": self.OLD}, new, "pytest", self.PATTERNS)

    def test_a_clean_correction_raises_nothing(self):
        better = self.OLD.replace("assert 1", "assert 1 == 1")
        assert self.compare({"test_x.py": better}) == []

    def test_more_tests_raise_nothing(self):
        assert self.compare({"test_x.py": self.OLD + "\n\ndef test_c():\n    assert 3\n"}) == []

    def test_a_removed_test(self):
        found = self.compare({"test_x.py": "def test_a():\n    assert 1\n"})
        assert any("test_b" in line for line in found)

    def test_a_removed_file(self):
        found = self.compare({}, {"test_x.py": self.OLD, "test_y.py": self.OLD})
        assert any("test_x.py was removed" in line for line in found)
        assert any("test_y.py was removed" in line for line in found)

    def test_a_renamed_test_is_still_reported(self):
        found = self.compare({"test_x.py": self.OLD.replace("test_b", "test_z")})
        assert any("test_b" in line for line in found)

    def test_fewer_assertions(self):
        found = self.compare({"test_x.py": self.OLD.replace("    assert 2\n", "    pass\n")})
        assert any("fewer assertions" in line for line in found)

    def test_a_new_skip(self):
        found = self.compare(
            {"test_x.py": "import pytest\n\n\n@pytest.mark.skip\n" + self.OLD}
        )
        assert any("skipped" in line for line in found)

    def test_a_new_helper_file(self):
        found = self.compare({"test_x.py": self.OLD, "conftest.py": "x = 1\n"})
        assert any("conftest.py" in line for line in found)

    def test_a_new_test_file_is_fine(self):
        assert self.compare({"test_x.py": self.OLD, "test_new.py": "def test_n():\n    assert 1\n"}) == []

    def test_no_tests_left(self):
        found = self.compare({"test_x.py": "x = 1\n"})
        assert any("no test cases at all" in line for line in found)

    def test_a_lost_original_is_a_warning(self):
        found = self.compare({"test_x.py": self.OLD}, {"test_x.py": None})
        assert any("not available" in line for line in found)

    def test_replacing_a_suite_with_no_tests_is_fine(self):
        found = compare_suites(
            {"a.test.mjs": LIVE_TEST_FILE},
            {"a.test.mjs": "test('a', () => { assert.ok(1); });\n"},
            "tap",
            ["*.test.mjs"],
        )
        assert found == []

    def test_node_skips_are_noticed(self):
        found = compare_suites(
            {"a.test.mjs": "test('a', () => { assert.ok(1); });\n"},
            {"a.test.mjs": "test.skip('b', () => { assert.ok(1); });\n"},
            "tap",
            ["*.test.mjs"],
        )
        assert any("skipped" in line for line in found)


class TestDiffs:
    def test_a_file_without_a_final_newline_stays_readable(self):
        diff = unified_diff("a.py", "one", "two\nthree\n")
        assert "-one\n+two\n+three" in diff

    def test_the_summary_shows_what_a_person_must_decide(self):
        text = revision_summary(
            "-old\n+new", "x.diff", ["Fewer assertions."], ["AC-1: Works."], ["a.py (modified)"]
        )
        for part in ("a.py (modified)", "+new", "x.diff", "AC-1: Works.", "may weaken"):
            assert part in text


def inputs(**overrides) -> ReportInputs:
    values = {
        "intervention_id": "INT-001",
        "version": 1,
        "product": "Board",
        "slug": "board",
        "execution_mode": "Manual",
        "blocked": BlockedInfo(
            kind=BlockKind.MODEL_FAILURE,
            resume_phase=ProductPhase.STEP_ARCHITECT,
            message="The Architect failed.",
            failure_class=FailureClass.MODEL_CONNECTION,
            task_id="t",
            role="software_architect",
        ),
    }
    return ReportInputs(**{**values, **overrides})


def attempts(failure: FailureClass, count: int = 4) -> list[ModelAttempt]:
    return [
        ModelAttempt(
            attempt_id=str(index),
            task_id="t",
            model="m",
            model_attempt=1,
            overall_attempt=index,
            status="failed",
            failure_class=failure,
            error=f"failure {index}",
        )
        for index in range(1, count + 1)
    ]


class TestBuildingReports:
    def test_a_connection_failure_tells_the_person_to_check_ollama(self):
        report = build_report(inputs(attempts=attempts(FailureClass.MODEL_CONNECTION)))
        assert "python -m lab doctor" in report.actions[0].action

    def test_a_timeout_suggests_freeing_memory_as_advice_not_fact(self):
        blocked = inputs().blocked.model_copy(
            update={"failure_class": FailureClass.INFERENCE_TIMEOUT}
        )
        report = build_report(
            inputs(blocked=blocked, attempts=attempts(FailureClass.INFERENCE_TIMEOUT))
        )
        assert any("Memory pressure" in line for line in report.unconfirmed_causes)
        assert not any("Memory" in line for line in report.confirmed_facts)

    def test_commands_depend_on_the_kind_of_block(self):
        integrity = inputs().blocked.model_copy(
            update={"kind": BlockKind.INTEGRITY, "failure_class": None}
        )
        commands = " ".join(build_report(inputs(blocked=integrity)).next_commands)
        assert "/verify" in commands
        assert "/retry" not in commands

        unmet = inputs().blocked.model_copy(
            update={"kind": BlockKind.CONDITIONS_UNMET, "failure_class": None}
        )
        commands = " ".join(build_report(inputs(blocked=unmet)).next_commands)
        assert "/accept" in commands

    def test_a_block_with_nothing_else_known_still_gives_an_action(self):
        blocked = BlockedInfo(
            kind=BlockKind.VERIFICATION_FAILED,
            resume_phase=ProductPhase.REPAIRING,
            message="Something failed.",
        )
        report = build_report(inputs(blocked=blocked))
        assert report.actions[0].problem == "Something failed."

    def test_only_the_failing_suite_is_reported(self):
        passing = run_of("1 passed", exit_code=0, passed=1, failed=0)
        failing = run_of("FAILED test_a.py::test_x - assert 1 == 2\n1 failed in 0.1s")
        failing.step_id = "step-02"
        failing.scope = "regression"
        for run in (passing, failing):
            run.profile_id = "python_pytest"
        evidence = VerificationEvidence(
            step_id="step-01",
            cycle=1,
            proposal_version=1,
            tested_workspace_fingerprint="x",
            manifest_sha256="y",
            runs=[passing, failing],
        )

        report = build_report(inputs(evidence=evidence, profiles=load_verification_profiles()))

        assert [item.step_id for item in report.suites] == ["step-02"]
        assert "broke earlier behaviour" in report.actions[0].why_it_matters

    def test_the_markdown_has_all_four_sections_in_order(self):
        text = report_markdown(build_report(inputs(attempts=attempts(FailureClass.INVALID_JSON))))
        positions = [
            text.index(heading)
            for heading in (
                "## A. Context",
                "## B. Exact problem",
                "## C. Root-cause assessment",
                "## D. Recommended manual intervention",
            )
        ]
        assert positions == sorted(positions)

    def test_the_terminal_summary_gives_the_location(self):
        report = build_report(inputs())
        text = terminal_summary(report, "products_log/board/interventions/INT-001/report.md")
        assert text.startswith("BLOCKED / HUMAN INTERVENTION REQUIRED")
        assert "products_log/board/interventions/INT-001/report.md" in text

    def test_a_block_before_any_step_has_no_step(self):
        text = report_markdown(build_report(inputs()))
        assert "Roadmap step: None yet" in text
