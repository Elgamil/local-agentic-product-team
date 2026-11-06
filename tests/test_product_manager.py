"""Creating, naming, listing and selecting products."""

from __future__ import annotations

import pytest

from lab.playbooks import PlaybookLibrary
from lab.product.manager import ProductManager, slugify
from lab.product.store import ProductStore
from lab.schemas.product import ProductMetadata, ProductPhase, Roadmap
from pydantic import ValidationError
from tests.fakes import APPROVE, Harness, complete_first_step, to_name_gate, to_proposal_gate


@pytest.fixture
def manager(tmp_path):
    return ProductManager(tmp_path / "products_log", PlaybookLibrary())


class TestSlugs:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Dairy Corner", "dairy-corner"),
            ("  Tic-Tac-Toe!  ", "tic-tac-toe"),
            ("Café Délice", "cafe-delice"),
            ("../../etc/passwd", "etc-passwd"),
            ("a/b\\c", "a-b-c"),
            ("محل ألبان", "product"),
            ("???", "product"),
        ],
    )
    def test_slugify_produces_safe_names(self, name, expected):
        assert slugify(name) == expected

    def test_slug_length_is_bounded(self):
        assert len(slugify("x" * 500)) == 60

    def test_metadata_refuses_an_unsafe_slug(self):
        with pytest.raises(ValidationError):
            ProductMetadata(product_id="1", display_name="x", slug="../evil")


class TestCreation:
    def test_creates_the_expected_layout(self, manager):
        store = manager.create("Dairy Corner")
        assert store.slug == "dairy-corner"
        for name in ("workspace", "verification", "artifacts", "steps", "logs"):
            assert (store.root / name).is_dir()
        assert store.load_metadata().display_name == "Dairy Corner"
        assert store.load_state().phase == ProductPhase.DISCOVERY

    def test_a_collision_never_overwrites_the_existing_product(self, manager):
        first = manager.create("Dairy Corner")
        (first.workspace_dir / "precious.txt").write_text("keep me")

        second = manager.create("Dairy Corner")
        third = manager.create("Dairy Corner")

        assert (second.slug, third.slug) == ("dairy-corner-2", "dairy-corner-3")
        assert (first.workspace_dir / "precious.txt").read_text() == "keep me"
        assert first.load_metadata().product_id != second.load_metadata().product_id

    def test_a_folder_that_is_not_a_product_still_counts_as_taken(self, manager):
        (manager.root / "dairy-corner").mkdir(parents=True)
        assert manager.create("Dairy Corner").slug == "dairy-corner-2"

    def test_unnamed_product_gets_a_durable_draft_folder(self, manager):
        store = manager.create()
        assert store.slug.startswith("draft-")
        assert store.load_metadata().name_confirmed is False


class TestRename:
    def test_rename_moves_the_folder_and_keeps_its_contents(self, manager):
        store = manager.create()
        draft = store.root
        (store.workspace_dir / "a.txt").write_text("hello")

        assert manager.rename(store, "Dairy Corner") == "dairy-corner"

        assert not draft.exists()
        assert (manager.root / "dairy-corner" / "workspace" / "a.txt").read_text() == "hello"
        assert store.load_metadata().slug == "dairy-corner"

    def test_rename_onto_a_taken_name_never_overwrites(self, manager):
        existing = manager.create("Dairy Corner")
        (existing.workspace_dir / "precious.txt").write_text("keep me")
        store = manager.create()

        assert manager.rename(store, "Dairy Corner") == "dairy-corner-2"
        assert (existing.workspace_dir / "precious.txt").read_text() == "keep me"

    def test_crash_between_rename_and_metadata_update_self_heals(self, manager):
        store = manager.create()
        target = manager.root / "dairy-corner"
        store.root.rename(target)

        reopened = ProductStore(target)
        assert reopened.load_metadata().slug == "dairy-corner"


class TestNamingFlow:
    async def test_llm_name_collision_is_handled_and_reported(self, tmp_path):
        harness = Harness(tmp_path)
        harness.manager.create("Pocket Calc")
        workflow = harness.new_workflow()
        await to_name_gate(workflow)

        assert workflow.store.slug == "pocket-calc-2"
        assert "products_log/pocket-calc-2" in workflow.gate().message

    async def test_the_user_can_choose_a_different_name(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_name_gate(workflow)

        slug = workflow.change_name(type(APPROVE)("Milk Market"))

        assert slug == "milk-market"
        assert workflow.meta.display_name == "Milk Market"
        assert workflow.state.phase == ProductPhase.ROADMAP_DRAFTING
        assert "# Milk Market" in workflow.store.read_text("brief.md")
        assert not (harness.root / "pocket-calc").exists()

    async def test_renaming_leaves_no_stray_draft_folders(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await complete_first_step(workflow)

        assert [path.name for path in harness.root.iterdir()] == ["pocket-calc"]
        assert (workflow.store.root / "verification" / "step-01" / "manifest.json").is_file()

    async def test_each_product_has_independent_state_and_workspace(self, tmp_path):
        harness = Harness(tmp_path)
        first = harness.new_workflow()
        await complete_first_step(first)

        harness.team.product_name = "Second Product"
        second = harness.new_workflow()
        await to_name_gate(second)

        assert first.store.root != second.store.root
        assert second.state.steps == {}
        assert list(second.store.workspace_dir.iterdir()) == []
        assert first.store.read_events()[0].event_id != second.store.read_events()[0].event_id


class TestDiscovery:
    def test_empty_root_is_fine(self, manager):
        assert manager.discover() == ([], [])

    def test_ignores_folders_that_are_not_products(self, manager):
        manager.create("Real Product")
        (manager.root / "not-a-product").mkdir()
        (manager.root / ".hidden").mkdir()
        (manager.root / "loose-file.txt").write_text("x")

        summaries, invalid = manager.discover()

        assert [item.slug for item in summaries] == ["real-product"]
        assert invalid == []

    def test_damage_to_one_product_does_not_hide_the_others(self, manager):
        manager.create("Healthy")
        broken = manager.create("Broken")
        broken.metadata_path.write_text("{ not json")

        summaries, invalid = manager.discover()

        assert [item.slug for item in summaries] == ["healthy"]
        assert [item.slug for item in invalid] == ["broken"]

    async def test_summary_shows_status_progress_and_next_action(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)

        summary = harness.manager.discover()[0][0]

        assert summary.display_name == "Pocket Calc"
        assert summary.category_label == "Command-Line Tool"
        assert summary.progress == "Step 1 of 2"
        assert summary.next_action == "Approve implementation proposal for Step 1."
        text = summary.render()
        assert "Pocket Calc — Command-Line Tool" in text
        assert "Status: Step 1: proposal waiting for approval" in text
        assert "Last activity:" in text

    async def test_summary_after_a_completed_step(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await complete_first_step(workflow)

        summary = harness.manager.discover()[0][0]

        assert summary.status == "Step 1 completed"
        assert summary.progress == "Step 2 of 2"
        assert summary.next_action == "Start Step 2: Statistics."
        assert workflow.store.load_model("artifacts/roadmap.v1.json", Roadmap).version == 1


class TestSelection:
    @pytest.fixture
    def summaries(self, manager):
        manager.create("Dairy Corner")
        manager.create("Tic Tac Toe")
        manager.create("Dairy Express")
        return manager.discover()[0]

    def test_select_by_number(self, manager, summaries):
        assert manager.select("2", summaries) is summaries[1]

    def test_select_by_exact_name_ignoring_case(self, manager, summaries):
        assert manager.select("dairy corner", summaries).slug == "dairy-corner"

    def test_select_by_unique_partial_name(self, manager, summaries):
        assert manager.select("tic", summaries).slug == "tic-tac-toe"

    @pytest.mark.parametrize("selection", ["dairy", "99", "0", "", "unknown"])
    def test_ambiguous_or_unknown_selection_picks_nothing(self, manager, summaries, selection):
        assert manager.select(selection, summaries) is None
