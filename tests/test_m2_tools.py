"""Unit tests for developer tools and test runner safety controls."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from lab.tools.developer_tools import ProjectWorkspace, SecurityError
from lab.tools.test_runner import run_controlled_tests


class TestProjectWorkspaceSafety:
    @pytest.fixture
    def workspace(self, tmp_path):
        return ProjectWorkspace(tmp_path / "sandbox")

    def test_safe_write_and_read(self, workspace):
        msg = workspace.write_file("sub/test.txt", "Hello World")
        assert "Successfully wrote" in msg
        content = workspace.read_file("sub/test.txt")
        assert content == "Hello World"

    def test_list_files(self, workspace):
        workspace.write_file("a.py", "# a")
        workspace.write_file("nested/b.py", "# b")
        files = workspace.list_files()
        assert "a.py" in files
        assert "nested/b.py" in files

    def test_reject_path_traversal(self, workspace):
        with pytest.raises(SecurityError, match="Path traversal"):
            workspace.write_file("../escape.txt", "Danger")

    def test_reject_absolute_path_outside_root(self, workspace):
        with pytest.raises(SecurityError):
            workspace.write_file("/etc/passwd", "Danger")

    def test_reject_protected_test_directory(self, workspace):
        with pytest.raises(SecurityError, match="forbidden"):
            workspace.write_file("tests/acceptance/test_hack.py", "Danger")

    def test_reject_git_directory(self, workspace):
        with pytest.raises(SecurityError, match="forbidden"):
            workspace.write_file(".git/config", "Danger")

    def test_reject_empty_path(self, workspace):
        with pytest.raises(SecurityError, match="empty"):
            workspace.write_file("   ", "Danger")

    def test_reject_oversized_file(self, workspace):
        huge = "x" * (1_048_576 + 10)
        with pytest.raises(SecurityError, match="exceeds maximum permitted size"):
            workspace.write_file("big.txt", huge)


class TestTestRunner:
    def test_run_controlled_tests_on_dummy_dir(self, tmp_path):
        test_file = tmp_path / "test_sample.py"
        test_file.write_text("def test_ok(): assert 1 == 1\n")
        result = run_controlled_tests(test_target=str(test_file), cwd=tmp_path)
        assert result.exit_code == 0
        assert result.success is True
        assert result.passed_count == 1
