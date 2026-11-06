"""The profile-based test runner, exercised through real subprocesses."""

from __future__ import annotations

import os
import shutil
import time

import pytest

from lab.config import load_verification_profiles
from lab.tools.verification import (
    InvalidTestFile,
    VerificationStore,
    VerificationTamperedError,
    VerificationToolMissing,
    _parse_pytest,
    _parse_tap,
    _parse_unittest,
    build_command,
    run_profile,
)

PROFILES = load_verification_profiles()


@pytest.fixture
def dirs(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    return workspace, tmp_path / "verification"


def stage(store, name, content, profile_id="python_pytest"):
    store.stage("step-01", name, content, PROFILES[profile_id])


class TestRealRunners:
    def test_pytest_pass(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(store, "test_ok.py", "import calc\n\ndef test_a():\n    assert calc.add(1, 1) == 2\n")
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert run.success
        assert (run.passed_count, run.failed_count, run.exit_code) == (1, 0, 0)
        assert run.cwd == str(workspace.resolve())

    def test_pytest_failure_is_reported_with_real_output(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(store, "test_bad.py", "def test_a():\n    assert 1 == 2\n")
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert not run.success
        assert run.failed_count == 1
        assert "assert 1 == 2" in run.stdout

    def test_no_tests_collected_is_not_a_success(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(store, "test_empty.py", "# nothing here\n")
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert not run.success

    def test_a_collection_error_is_not_a_success(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(store, "test_broken.py", "import module_that_does_not_exist\n")
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert not run.success
        assert run.failed_count >= 1

    def test_pytest_ignores_the_repository_configuration(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(store, "test_ok.py", "def test_a():\n    assert True\n")
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert "rootdir: " in run.stdout
        assert "pytest.ini" in run.stdout

    def test_unittest_profile(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(
            store,
            "test_unit.py",
            "import unittest, calc\n\n"
            "class T(unittest.TestCase):\n"
            "    def test_a(self):\n        self.assertEqual(calc.add(1, 1), 2)\n"
            "    def test_b(self):\n        self.assertEqual(calc.add(1, 1), 3)\n",
            "python_unittest",
        )
        run = run_profile(PROFILES["python_unittest"], store.step_dir("step-01"), workspace)
        assert (run.passed_count, run.failed_count) == (1, 1)
        assert not run.success

    @pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
    def test_node_profile_with_the_builtin_runner(self, dirs):
        workspace, root = dirs
        (workspace / "game.mjs").write_text("export const add = (a, b) => a + b;\n")
        store = VerificationStore(root)
        stage(
            store,
            "game.test.mjs",
            "import test from 'node:test';\n"
            "import assert from 'node:assert';\n"
            "import path from 'node:path';\n"
            "const { add } = await import("
            "path.join(process.env.PRODUCT_WORKSPACE, 'game.mjs'));\n"
            "test('adds', () => assert.strictEqual(add(1, 2), 3));\n"
            "test('fails', () => assert.strictEqual(add(1, 2), 4));\n",
            "node_test",
        )
        run = run_profile(PROFILES["node_test"], store.step_dir("step-01"), workspace)
        assert (run.passed_count, run.failed_count) == (1, 1)
        assert not run.success


class TestProcessControl:
    def test_a_hung_suite_is_killed_at_the_timeout(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(store, "test_hang.py", "import time\n\ndef test_a():\n    time.sleep(60)\n")
        profile = PROFILES["python_pytest"].model_copy(update={"timeout_seconds": 1.5})

        started = time.monotonic()
        run = run_profile(profile, store.step_dir("step-01"), workspace)

        assert time.monotonic() - started < 20
        assert run.timed_out
        assert not run.success
        assert "timed out" in run.stderr

    def test_a_process_left_running_by_a_test_is_reaped(self, dirs, tmp_path):
        workspace, root = dirs
        marker = tmp_path / "survived.txt"
        store = VerificationStore(root)
        stage(
            store,
            "test_spawn.py",
            "import subprocess, sys\n\n"
            "def test_a():\n"
            "    code = \"import time; time.sleep(2); "
            f"open({str(marker)!r}, 'w').write('x')\"\n"
            "    subprocess.Popen([sys.executable, '-c', code])\n",
        )
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert run.success
        time.sleep(2.5)
        assert not marker.exists()

    def test_the_environment_is_scrubbed(self, dirs, monkeypatch):
        workspace, root = dirs
        monkeypatch.setenv("MY_SECRET_TOKEN", "hunter2")
        store = VerificationStore(root)
        stage(
            store,
            "test_env.py",
            "import os\n\n"
            "def test_a():\n"
            "    assert 'MY_SECRET_TOKEN' not in os.environ\n"
            "    assert os.environ['PRODUCT_WORKSPACE']\n",
        )
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert run.success, run.stdout

    def test_output_is_bounded(self, dirs):
        workspace, root = dirs
        store = VerificationStore(root)
        stage(
            store,
            "test_noisy.py",
            "def test_a():\n    print('x' * 500000)\n    assert False\n",
        )
        run = run_profile(PROFILES["python_pytest"], store.step_dir("step-01"), workspace)
        assert run.truncated
        assert len(run.stdout) < 100_000

    def test_a_missing_tool_is_a_dependency_problem(self, dirs):
        workspace, root = dirs
        profile = PROFILES["python_pytest"].model_copy(
            update={"executable": "definitely-not-installed-tool"}
        )
        with pytest.raises(VerificationToolMissing):
            build_command(profile, root, workspace)

    def test_commands_run_without_a_shell(self, dirs):
        workspace, root = dirs
        argv = build_command(PROFILES["python_pytest"], root / "a;rm -rf x", workspace)
        assert isinstance(argv, list)
        assert str(root / "a;rm -rf x") in argv, "metacharacters stay inside one argument"


class TestSealing:
    @pytest.fixture
    def sealed(self, dirs):
        _, root = dirs
        store = VerificationStore(root)
        stage(store, "test_ok.py", "def test_a():\n    assert True\n")
        store.seal("step-01", "python_pytest", 1)
        return store

    def test_a_clean_seal_verifies(self, sealed):
        assert sealed.verify_seal("step-01").profile_id == "python_pytest"

    def test_changing_a_file_is_detected(self, sealed):
        path = sealed.step_dir("step-01") / "test_ok.py"
        os.chmod(path, 0o644)
        path.write_text("def test_a():\n    pass\n")
        with pytest.raises(VerificationTamperedError, match="test_ok.py"):
            sealed.verify_seal("step-01")

    def test_deleting_a_file_is_detected(self, sealed):
        (sealed.step_dir("step-01") / "test_ok.py").unlink()
        with pytest.raises(VerificationTamperedError):
            sealed.verify_seal("step-01")

    def test_a_missing_manifest_is_not_a_valid_seal(self, dirs):
        _, root = dirs
        with pytest.raises(VerificationTamperedError):
            VerificationStore(root).verify_seal("step-09")

    def test_a_sealed_suite_cannot_be_staged_into(self, sealed):
        with pytest.raises(VerificationTamperedError):
            stage(sealed, "test_more.py", "def test_b():\n    pass\n")

    def test_archive_preserves_tests_for_the_audit_trail(self, sealed, tmp_path):
        archive = tmp_path / "archive"
        sealed.archive_and_clear("step-01", archive)
        assert (archive / "test_ok.py").exists()
        assert (archive / "manifest.json").exists()
        assert not sealed.step_dir("step-01").exists()

    def test_test_file_names_must_match_the_profile(self, dirs):
        _, root = dirs
        with pytest.raises(InvalidTestFile):
            stage(VerificationStore(root), "evil.sh", "rm -rf /")

    def test_staging_refuses_traversal(self, dirs):
        _, root = dirs
        with pytest.raises(InvalidTestFile):
            stage(VerificationStore(root), "../../test_escape.py", "x")


class TestParsers:
    def test_pytest_summary_only(self):
        output = "test_passed_and_failed_names PASSED\n=== 3 passed, 1 failed, 2 errors in 0.5s ==="
        assert _parse_pytest(output) == (3, 3)

    def test_pytest_ignores_counts_outside_the_summary(self):
        assert _parse_pytest("99 passed somewhere in a log line\n=== 1 failed in 0.1s ===") == (0, 1)

    def test_unittest_ok(self):
        assert _parse_unittest("Ran 3 tests in 0.1s\n\nOK\n") == (3, 0)

    def test_unittest_failures(self):
        assert _parse_unittest("Ran 4 tests in 0.1s\n\nFAILED (failures=1, errors=1)\n") == (2, 2)

    def test_tap(self):
        assert _parse_tap("# tests 3\n# pass 2\n# fail 1\n# cancelled 0\n") == (2, 1)
