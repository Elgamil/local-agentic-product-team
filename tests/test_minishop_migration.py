"""Preserving the original MiniShop as a historical product."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from lab.config import load_verification_profiles
from lab.playbooks import PlaybookLibrary
from lab.product.manager import ProductManager
from lab.product.migration import SLUG, migrate_minishop
from lab.product.session import ProductSession
from lab.schemas.product import ProductPhase
from lab.tools import fsutil
from lab.tools.verification import run_profile
from tests.fakes import Harness, ScriptedTerminal

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path):
    """A copy of the real sources, so these tests never touch the originals."""
    copy = tmp_path / "repo"
    shutil.copytree(PROJECT_ROOT / "minishop", copy / "minishop",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(PROJECT_ROOT / "tests" / "acceptance", copy / "tests" / "acceptance",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (copy / "runs").mkdir()
    return copy


@pytest.fixture
def manager(tmp_path):
    return ProductManager(tmp_path / "products_log", PlaybookLibrary())


class TestMigration:
    def test_code_and_tests_land_in_separate_places(self, manager, repo):
        report = migrate_minishop(manager, repo)
        assert report.status == "migrated"
        root = manager.root / SLUG
        assert (root / "workspace" / "minishop" / "app.py").is_file()
        assert (root / "workspace" / "minishop" / "templates" / "catalog.html").is_file()
        assert (root / "verification" / "legacy" / "test_catalog_acceptance.py").is_file()
        assert not (root / "workspace" / "tests").exists()

    def test_originals_are_kept(self, manager, repo):
        before = sorted(path.name for path in (repo / "minishop").iterdir())
        migrate_minishop(manager, repo)
        assert sorted(path.name for path in (repo / "minishop").iterdir()) == before

    def test_each_copied_file_is_hash_verified(self, manager, repo):
        migrate_minishop(manager, repo)
        root = manager.root / SLUG
        manifest = json.loads((root / "logs" / "migration_manifest.json").read_text())
        assert manifest["originals_kept"] is True
        assert manifest["files"]
        for relative, digest in manifest["files"].items():
            assert fsutil.sha256_file(root / relative) == digest

    def test_a_copy_that_does_not_verify_stops_the_migration(self, manager, repo, monkeypatch):
        import lab.product.migration as migration

        real = migration.sha256_file
        calls = {"n": 0}

        def corrupt(path):
            calls["n"] += 1
            return "wrong" if calls["n"] == 2 else real(path)

        monkeypatch.setattr(migration, "sha256_file", corrupt)
        with pytest.raises(OSError, match="Verification failed"):
            migrate_minishop(manager, repo)
        assert (repo / "minishop" / "app.py").is_file()

    def test_generated_files_are_not_copied(self, manager, repo):
        cache = repo / "minishop" / "__pycache__"
        cache.mkdir()
        (cache / "app.cpython-311.pyc").write_text("x")
        (repo / "minishop" / ".DS_Store").write_text("x")
        migrate_minishop(manager, repo)
        workspace = manager.root / SLUG / "workspace" / "minishop"
        assert not (workspace / "__pycache__").exists()
        assert not (workspace / ".DS_Store").exists()

    def test_it_is_safe_to_run_twice(self, manager, repo):
        migrate_minishop(manager, repo)
        marker = manager.root / SLUG / "workspace" / "note.txt"
        marker.write_text("kept")
        report = migrate_minishop(manager, repo)
        assert report.status == "already_migrated"
        assert marker.read_text() == "kept"

    def test_an_unrelated_folder_with_that_name_is_left_alone(self, manager, repo):
        (manager.root / SLUG).mkdir(parents=True)
        (manager.root / SLUG / "mine.txt").write_text("mine")
        report = migrate_minishop(manager, repo)
        assert report.status == "conflict"
        assert (manager.root / SLUG / "mine.txt").read_text() == "mine"

    def test_a_missing_source_is_reported(self, manager, tmp_path):
        assert migrate_minishop(manager, tmp_path).status == "source_missing"

    def test_existing_reports_are_preserved(self, manager, repo):
        (repo / "runs" / "report.md").write_text("old report")
        migrate_minishop(manager, repo)
        preserved = manager.root / SLUG / "logs" / "legacy_runs" / "report.md"
        assert preserved.read_text() == "old report"

    def test_the_migrated_copy_passes_its_own_acceptance_tests(self, manager, repo):
        """A real run, against the copy, proving the preserved product still works."""
        migrate_minishop(manager, repo)
        root = manager.root / SLUG
        profile = load_verification_profiles()["python_pytest"]
        run = run_profile(profile, root / "verification" / "legacy", root / "workspace")
        assert run.success, run.stdout[-2000:]
        assert run.passed_count == 7


class TestProductList:
    def test_it_appears_as_a_historical_product(self, manager, repo):
        migrate_minishop(manager, repo)
        summary = manager.discover()[0][0]
        assert summary.display_name == "Mini Shop Lite"
        assert summary.phase == ProductPhase.HISTORICAL
        assert summary.category_label == "Website / E-commerce"
        assert summary.finished is True

    async def test_it_is_not_treated_as_unfinished_work(self, tmp_path, repo):
        harness = Harness(tmp_path)
        migrate_minishop(harness.manager, repo)
        terminal = ScriptedTerminal("4")
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        await session.run()
        assert not any("Welcome back" in text for text in terminal.said)

    async def test_opening_it_runs_no_model_and_changes_nothing(self, tmp_path, repo):
        harness = Harness(tmp_path)
        migrate_minishop(harness.manager, repo)
        before = (harness.root / SLUG / "state.json").read_bytes()
        terminal = ScriptedTerminal("2", "Mini Shop Lite", "4")
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        await session.run()
        assert harness.team.calls == []
        assert any("preserved for reference" in text for text in terminal.said)
        assert (harness.root / SLUG / "state.json").read_bytes() == before

    async def test_a_new_product_never_defaults_to_it(self, tmp_path, repo):
        harness = Harness(tmp_path)
        migrate_minishop(harness.manager, repo)
        store = harness.new_workflow().store
        assert store.slug != SLUG
        assert list(store.workspace_dir.iterdir()) == []
