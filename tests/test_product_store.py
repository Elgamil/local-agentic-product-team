"""Durability: atomic checkpoints, append-only history, session locking."""

from __future__ import annotations

import json
import os

import pytest

from lab.playbooks import PlaybookLibrary
from lab.product.manager import ProductManager
from lab.product.store import ProductBusyError, ProductLock, StateCorruptedError
from lab.schemas.product import ModelAttempt, ProductPhase
from lab.tools import fsutil


@pytest.fixture
def store(tmp_path):
    return ProductManager(tmp_path / "products_log", PlaybookLibrary()).create("Sample")


class TestAtomicWrites:
    def test_crash_during_write_keeps_the_previous_file(self, tmp_path, monkeypatch):
        target = tmp_path / "state.json"
        fsutil.atomic_write_text(target, "version one")

        def power_cut(*args, **kwargs):
            raise OSError("power lost")

        monkeypatch.setattr(fsutil.os, "replace", power_cut)
        with pytest.raises(OSError):
            fsutil.atomic_write_text(target, "version two")

        assert target.read_text() == "version one"
        assert [path.name for path in tmp_path.iterdir()] == ["state.json"], "no temp left behind"

    def test_preserves_an_existing_file_mode(self, tmp_path):
        target = tmp_path / "run.sh"
        target.write_text("old")
        os.chmod(target, 0o755)
        fsutil.atomic_write_text(target, "new")
        assert target.stat().st_mode & 0o777 == 0o755


class TestCheckpoints:
    def test_state_round_trips_through_a_fresh_read(self, store):
        state = store.load_state()
        state.phase = ProductPhase.AWAITING_NAME
        state.discovery_round = 3
        store.save_state(state)

        loaded = store.load_state()

        assert loaded.phase == ProductPhase.AWAITING_NAME
        assert loaded.discovery_round == 3

    def test_revision_increases_with_each_checkpoint(self, store):
        state = store.load_state()
        before = state.revision
        store.save_state(state)
        store.save_state(state)
        assert store.load_state().revision == before + 2

    def test_corrupt_checkpoint_falls_back_to_the_previous_one(self, store):
        state = store.load_state()
        state.discovery_round = 1
        store.save_state(state)
        state.discovery_round = 2
        store.save_state(state)
        store.state_path.write_text('{"phase": "trunc')

        assert store.load_state().discovery_round == 1
        assert any(event.type == "CheckpointRecovered" for event in store.read_events())

    def test_an_invalid_phase_is_treated_as_corruption(self, store):
        store.save_state(store.load_state())
        raw = json.loads(store.state_path.read_text())
        raw["phase"] = "skip_all_approvals"
        store.state_path.write_text(json.dumps(raw))

        assert store.load_state().phase == ProductPhase.DISCOVERY

    def test_unreadable_checkpoints_raise_rather_than_guess(self, store):
        store.state_path.write_text("garbage")
        store.previous_state_path.write_text("garbage")
        with pytest.raises(StateCorruptedError):
            store.load_state()


class TestEventHistory:
    def test_events_are_append_only_and_identified(self, store):
        before = (store.logs_dir / "events.jsonl").read_bytes()

        first = store.append_event("One", payload={"n": 1})
        second = store.append_event("Two", step_id="step-01")

        after = (store.logs_dir / "events.jsonl").read_bytes()
        assert after.startswith(before), "earlier history is never rewritten"
        assert first.event_id != second.event_id
        assert first.timestamp <= second.timestamp

    def test_step_events_are_mirrored_to_the_step(self, store):
        store.append_event("Global")
        store.append_event("Local", step_id="step-01")
        assert [event.type for event in store.read_events("step-01")] == ["Local"]

    def test_a_torn_final_line_loses_only_that_record(self, store):
        store.append_event("Good")
        with open(store.logs_dir / "events.jsonl", "ab") as handle:
            handle.write(b'{"event_id": "torn')

        store.append_event("AfterCrash")

        types = [event.type for event in store.read_events()]
        assert "Good" in types
        assert "AfterCrash" in types

    def test_the_store_offers_no_way_to_rewrite_history(self, store):
        for name in ("delete_event", "rewrite_events", "clear_events", "truncate_events"):
            assert not hasattr(store, name)

    def test_decisions_file_only_grows(self, store):
        store.append_decision("First decision.")
        before = (store.root / "decisions.md").read_text()
        store.append_decision("Second decision.")
        after = (store.root / "decisions.md").read_text()
        assert after.startswith(before)
        assert "Second decision." in after


class TestAttemptLog:
    def test_an_attempt_with_no_outcome_reads_as_interrupted(self, store):
        from lab.product.store import DurableAttemptLog

        log = DurableAttemptLog(store)
        log.record(
            ModelAttempt(
                attempt_id="a", task_id="t", model="m", model_attempt=1, overall_attempt=1
            )
        )
        assert log.load("t")[0].status == "interrupted"


class TestLock:
    def test_second_live_session_is_refused(self, store):
        ProductLock(store).acquire()
        import subprocess
        import sys

        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "from lab.product.store import ProductStore, ProductLock, ProductBusyError\n"
            f"lock = ProductLock(ProductStore(Path({str(store.root)!r})))\n"
            "try:\n    lock.acquire()\nexcept ProductBusyError:\n    sys.exit(7)\n"
        )
        result = subprocess.run([sys.executable, "-c", code])
        assert result.returncode == 7

    def test_a_stale_lock_from_a_dead_process_is_replaced(self, store):
        process = __import__("subprocess").Popen(["true"])
        process.wait()
        store.lock_path.write_text(
            json.dumps({"pid": process.pid, "host": __import__("socket").gethostname()})
        )
        lock = ProductLock(store)
        lock.acquire()
        lock.release()
        assert not store.lock_path.exists()

    def test_lock_busy_error_is_raised_for_a_foreign_live_holder(self, store):
        store.lock_path.write_text(
            json.dumps({"pid": os.getppid(), "host": __import__("socket").gethostname()})
        )
        with pytest.raises(ProductBusyError):
            ProductLock(store).acquire()
