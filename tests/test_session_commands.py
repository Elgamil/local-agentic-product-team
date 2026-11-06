"""Reading human input, and the terminal commands."""

from __future__ import annotations

import json

import pytest

from lab.cli import main
from lab.product.decisions import Intent, extract_name, is_affirmation, parse_input
from lab.product.session import ProductSession
from lab.schemas.product import ProductPhase
from tests.fakes import (
    CALC_BUGGY,
    Fail,
    Harness,
    ScriptedTerminal,
    Text,
    to_name_gate,
    to_proposal_gate,
    to_roadmap_gate,
    to_step_start_gate,
)


class TestApprovalsAreDeterministic:
    @pytest.mark.parametrize(
        "text", ["yes", "Yes!", "y", "approve", "Approved.", "looks good", "LGTM", "go ahead",
                 "/approve"],
    )
    def test_plain_approvals(self, text):
        assert parse_input(text).intent == Intent.APPROVE

    @pytest.mark.parametrize(
        "text",
        [
            "yes but use a different colour",
            "approve the first part only",
            "looks good except the header",
            "I do not approve",
            "don't approve this",
            "not yet",
            "yes, after you change the name",
            "maybe",
            "ok so what about the cart",
        ],
    )
    def test_anything_conditional_is_not_an_approval(self, text):
        assert parse_input(text).intent != Intent.APPROVE

    @pytest.mark.parametrize("text", ["ok?", "yes?", "approve?", "looks good?", "proceed?"])
    def test_a_yes_with_a_question_mark_is_a_question_not_an_approval(self, text):
        assert parse_input(text).intent == Intent.QUESTION

    @pytest.mark.parametrize("text", ["no?", "reject?"])
    def test_a_no_with_a_question_mark_is_a_question_not_a_rejection(self, text):
        assert parse_input(text).intent == Intent.QUESTION

    @pytest.mark.parametrize("text", ["no", "No.", "reject", "/reject too big", "nope"])
    def test_rejections(self, text):
        assert parse_input(text).intent == Intent.REJECT

    def test_reject_command_keeps_the_reason(self):
        assert parse_input("/reject too expensive").text == "too expensive"

    @pytest.mark.parametrize(
        "text", ["why this design?", "What does this cost", "how will you test it?", "/ask why"]
    )
    def test_questions(self, text):
        assert parse_input(text).intent == Intent.QUESTION

    @pytest.mark.parametrize("text", ["make it blue", "add a search box", "/change use SQLite"])
    def test_changes(self, text):
        assert parse_input(text).intent == Intent.MODIFY

    def test_remove_step(self):
        parsed = parse_input("remove step 3")
        assert (parsed.intent, parsed.number) == (Intent.REMOVE_STEP, 3)

    def test_slash_remove(self):
        assert parse_input("/remove 2").number == 2

    def test_move_step(self):
        parsed = parse_input("move step 4 to 2")
        assert (parsed.intent, parsed.number, parsed.position) == (Intent.MOVE_STEP, 4, 2)

    @pytest.mark.parametrize(
        ("text", "intent"),
        [("/pause", Intent.PAUSE), ("/status", Intent.STATUS), ("/team", Intent.TEAM),
         ("/help", Intent.HELP), ("/exit", Intent.EXIT), ("quit", Intent.EXIT),
         ("/retry", Intent.RETRY), ("/accept", Intent.ACCEPT), ("/done", Intent.DONE),
         ("", Intent.EMPTY), ("/nonsense", Intent.HELP)],
    )
    def test_commands(self, text, intent):
        assert parse_input(text).intent == intent

    @pytest.mark.parametrize(
        ("text", "name"),
        [
            ("Milk Market", "Milk Market"),
            ("call it Milk Market", "Milk Market"),
            ("Let's call it 'Milk Market'.", "Milk Market"),
            ("how about Fresh Dairy", "Fresh Dairy"),
        ],
    )
    def test_name_extraction(self, text, name):
        assert extract_name(text) == name

    def test_keep_is_an_affirmation(self):
        assert is_affirmation("keep the name")


def session_for(harness: Harness, *answers: str) -> tuple[ProductSession, ScriptedTerminal]:
    terminal = ScriptedTerminal(*answers)
    session = ProductSession(
        harness.manager, harness.router, harness.policy, harness.playbooks, terminal
    )
    return session, terminal


class TestConversationAtGates:
    async def test_typing_an_idea_at_the_menu_starts_a_product(self, tmp_path):
        harness = Harness(tmp_path)
        session, terminal = session_for(harness, "I want a calculator", "/exit")
        await session.run()
        assert harness.team.calls_of("discovery")
        assert any("Pocket Calc" in text for text in terminal.said)

    async def test_listing_products_from_the_menu(self, tmp_path):
        harness = Harness(tmp_path)
        harness.manager.create("Dairy Corner")
        session, terminal = session_for(harness, "3", "4")
        await session.run()
        assert any("Dairy Corner" in text for text in terminal.said)

    async def test_an_empty_product_list_is_handled(self, tmp_path):
        harness = Harness(tmp_path)
        session, terminal = session_for(harness, "2", "4")
        await session.run()
        assert "There are no products yet." in terminal.said

    async def test_an_unknown_selection_is_handled(self, tmp_path):
        harness = Harness(tmp_path)
        harness.manager.create("Dairy Corner")
        session, terminal = session_for(harness, "2", "nothing like that", "4")
        await session.run()
        assert any("could not match" in text for text in terminal.said)

    async def test_eof_leaves_gracefully(self, tmp_path):
        harness = Harness(tmp_path)
        session, terminal = session_for(harness)
        await session.run()
        assert any("Goodbye" in text for text in terminal.said)

    async def test_renaming_through_the_conversation(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_name_gate(workflow)
        session, terminal = session_for(harness)
        await session._on_gate(workflow, "call it Milk Market")
        assert workflow.meta.display_name == "Milk Market"
        assert any("products_log/milk-market" in text for text in terminal.said)

    @pytest.mark.parametrize(
        "answer",
        ["why that name?", "I would rather have something that sounds much friendlier please"],
    )
    async def test_a_sentence_at_the_name_gate_does_not_become_the_name(self, tmp_path, answer):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_name_gate(workflow)
        session, terminal = session_for(harness)

        await session._on_gate(workflow, answer)

        assert workflow.meta.display_name == "Pocket Calc"
        assert workflow.state.phase == ProductPhase.AWAITING_NAME
        assert any("tell me the name" in text for text in terminal.said)

    async def test_a_question_between_steps_does_not_change_the_roadmap(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        session, _ = session_for(harness)

        await session._on_gate(workflow, "what happens next?")

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert workflow.state.roadmap_change_request is None

    async def test_ok_with_a_question_mark_does_not_approve_a_proposal(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        session, _ = session_for(harness)

        await session._on_gate(workflow, "ok?")

        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert workflow.state.steps["step-01"].approved_version is None
        assert harness.team.calls_of("implement") == []

    async def test_a_conditional_answer_to_a_proposal_never_approves(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        session, _ = session_for(harness)

        await session._on_gate(workflow, "yes but use a different structure")

        assert workflow.state.phase != ProductPhase.IMPLEMENTING
        assert workflow.state.steps["step-01"].approved_version is None
        assert harness.team.calls_of("implement") == []

    async def test_a_question_at_the_proposal_gate_stays_at_the_gate(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        session, terminal = session_for(harness)
        await session._on_gate(workflow, "Why this design?")
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert any("Software Architect" in text for text in terminal.said)

    async def test_remove_step_at_the_roadmap_gate(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        session, _ = session_for(harness)
        await session._on_gate(workflow, "remove step 2")
        assert len(workflow.roadmap.steps) == 1
        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL

    async def test_an_invalid_edit_explains_itself(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        session, terminal = session_for(harness)
        await session._on_gate(workflow, "remove step 9")
        assert "There is no step 9." in terminal.warnings
        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL

    async def test_declining_the_next_step_pauses(self, tmp_path):
        from lab.product.session import _LeaveProduct

        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        session, _ = session_for(harness)
        with pytest.raises(_LeaveProduct):
            await session._on_gate(workflow, "no")
        assert workflow.state.paused is True
        assert workflow.state.current_step_id is None

    async def test_status_does_not_change_anything(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        revision = workflow.state.revision
        session, terminal = session_for(harness)
        await session._on_gate(workflow, "/status")
        assert any("Next action: Approve implementation proposal" in t for t in terminal.said)
        assert workflow.state.revision == revision

    async def test_team_shows_specialist_decisions(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        session, terminal = session_for(harness)
        await session._on_gate(workflow, "/team")
        output = "\n".join(terminal.said)
        assert "Software Architect to Developer" in output
        assert "Decision: Use a single calc module" in output

    async def test_the_user_is_told_about_a_model_switch(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", Text("bad"), Text("bad"))
        terminal = ScriptedTerminal()
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_proposal_gate(workflow)
        assert any("switching to the backup model" in text for text in terminal.warnings)

    async def test_a_connection_problem_is_described_as_one(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("discovery", Fail(ConnectionError("down")))
        terminal = ScriptedTerminal()
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await workflow.discovery_message("An idea.")
        assert any("could not reach the local model" in text for text in terminal.warnings)
        assert not any("unusable answer" in text for text in terminal.warnings)

    async def test_progress_lines_name_each_consultation(self, tmp_path):
        harness = Harness(tmp_path)
        terminal = ScriptedTerminal()
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_proposal_gate(workflow)
        text = "\n".join(terminal.statuses)
        assert "Preparing the roadmap" in text
        assert "Consulting the Software Architect" in text
        assert "Test Engineer" in text

    async def test_the_blocked_conversation_retries_on_yes(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        session, terminal = session_for(harness, "yes", "/exit")
        from lab.product.session import _ExitSession

        await session._on_gate(workflow, "yes")
        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        assert any("I need your help to continue" in text for text in terminal.warnings)
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    async def test_a_failed_step_offers_repair_or_change(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.sources = {"calc.py": CALC_BUGGY}
        harness.policy.workflow.max_repair_cycles = 0
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        session, terminal = session_for(harness, "/pause")
        from lab.product.session import _LeaveProduct

        await session._on_gate(workflow, "yes")
        with pytest.raises(_LeaveProduct):
            await session._product_loop(workflow, None)

        assert any("did not pass verification" in text for text in terminal.warnings)
        assert any("repair attempts" in text for text in terminal.prompts)

    async def test_a_busy_product_is_refused_politely(self, tmp_path):
        import os

        harness = Harness(tmp_path)
        store = harness.manager.create("Busy Thing")
        store.lock_path.write_text(
            json.dumps({"pid": os.getppid(), "host": __import__("socket").gethostname()})
        )
        session, terminal = session_for(harness)
        await session.open_product("busy-thing")
        assert any("open in another session" in text for text in terminal.warnings)

    async def test_the_lock_is_released_when_leaving(self, tmp_path):
        harness = Harness(tmp_path)
        session, _ = session_for(harness, "/exit")
        store = harness.manager.create("Sample")
        with pytest.raises(BaseException):
            await session.open_product(store.slug)
        assert not store.lock_path.exists()


class TestInterrupt:
    def test_ctrl_c_during_a_prompt_stops_at_once_and_cleans_up(self):
        from lab.cli import run_interruptibly

        cleaned = []

        async def blocked_at_prompt():
            raise KeyboardInterrupt

        async def cleanup():
            cleaned.append(True)

        run_interruptibly(blocked_at_prompt(), cleanup)
        assert cleaned == [True]

    def test_a_real_sigint_stops_a_blocked_prompt_with_one_press(self, tmp_path):
        """Not simulated: a real SIGINT to a process waiting in input()."""
        import signal
        import subprocess
        import sys
        import time
        from pathlib import Path

        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})\n"
            "from lab.cli import run_interruptibly\n"
            "async def prompt():\n"
            "    print('READY', flush=True)\n"
            "    input()\n"
            "async def cleanup():\n"
            "    print('CLEANED', flush=True)\n"
            "run_interruptibly(prompt(), cleanup)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert process.stdout.readline().strip() == "READY"
        time.sleep(0.2)
        process.send_signal(signal.SIGINT)
        stdout, _ = process.communicate(timeout=20)

        assert process.returncode == 0
        assert "Stopped." in stdout
        assert "CLEANED" in stdout


class TestCommandLine:
    def test_list_with_no_products(self, tmp_path, capsys):
        assert main(["--products-dir", str(tmp_path / "p"), "list"]) == 0
        assert "no products yet" in capsys.readouterr().out

    def test_list_and_status(self, tmp_path, capsys):
        root = tmp_path / "p"
        from lab.playbooks import PlaybookLibrary
        from lab.product.manager import ProductManager

        ProductManager(root, PlaybookLibrary()).create("Dairy Corner")

        assert main(["--products-dir", str(root), "list"]) == 0
        assert "Dairy Corner" in capsys.readouterr().out
        assert main(["--products-dir", str(root), "status", "dairy"]) == 0
        assert "Status:" in capsys.readouterr().out

    def test_status_of_an_unknown_product(self, tmp_path, capsys):
        assert main(["--products-dir", str(tmp_path / "p"), "status", "nothing"]) == 1

    def test_internal_logs_stay_out_of_the_terminal(self, tmp_path, capsys):
        import logging

        main(["--products-dir", str(tmp_path / "p"), "list"])
        logging.getLogger("lab.anything").warning("internal detail")
        captured = capsys.readouterr()
        assert "internal detail" not in captured.out
        assert "internal detail" not in captured.err
        assert "internal detail" in (tmp_path / "p" / ".session.log").read_text()
