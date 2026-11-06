"""Speaker labels, color handling and empty input."""

from __future__ import annotations

import io
import json
import os
import pty
import select
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from lab.cli import main
from lab.product import console
from lab.product.console import ConsoleTerminal, strip_control, supports_color
from lab.product.session import EMPTY_INPUT_HINT, ProductSession, _ExitSession
from lab.product.workflow import GateViolation, HumanDecision
from lab.schemas.product import ProductPhase
from tests.fakes import (
    APPROVE,
    Fail,
    Harness,
    ScriptedTerminal,
    to_name_gate,
    to_proposal_gate,
    to_roadmap_gate,
    to_step_start_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ESC = "\x1b"
COLOR_ENV = {"TERM": "xterm-256color"}
ROLES = {
    "business_analyst": "Business Analyst:",
    "software_architect": "Software Architect:",
    "developer": "Developer:",
    "test_engineer": "Test Engineer:",
}


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


def color_terminal(*lines: str) -> tuple[ConsoleTerminal, FakeTty]:
    stream = FakeTty()
    answers = iter(lines)
    return ConsoleTerminal(stream, lambda: next(answers), COLOR_ENV), stream


def plain_terminal(*lines: str) -> tuple[ConsoleTerminal, io.StringIO]:
    stream = io.StringIO()
    answers = iter(lines)
    return ConsoleTerminal(stream, lambda: next(answers), COLOR_ENV), stream


class TestSpeakerLabels:
    @pytest.mark.parametrize(("role", "label"), ROLES.items())
    def test_each_agent_is_labelled_with_its_real_role(self, role, label):
        terminal, stream = plain_terminal()
        terminal.agent(role, "Hello.")
        assert stream.getvalue() == f"\n{label}\nHello.\n"

    def test_say_is_the_business_analyst(self):
        terminal, stream = plain_terminal()
        terminal.say("Let me ask a few questions.")
        assert "Business Analyst:\nLet me ask a few questions." in stream.getvalue()

    def test_the_user_prompt_is_labelled(self):
        terminal, stream = plain_terminal("A clean, modern interface.")
        assert terminal.ask("Any preferences?") == "A clean, modern interface."
        assert stream.getvalue().endswith("\nYou:\n")
        assert "Any preferences?" in stream.getvalue()

    def test_the_conversation_reads_like_the_requested_layout(self):
        terminal, stream = plain_terminal("I want a clean interface.")
        terminal.say("Let me ask a few questions.")
        terminal.ask("")
        terminal.agent("software_architect", "Here is the design.")
        assert stream.getvalue() == (
            "\nBusiness Analyst:\nLet me ask a few questions.\n"
            "\nYou:\n"
            "\nSoftware Architect:\nHere is the design.\n"
        )

    def test_consecutive_messages_share_one_label(self):
        terminal, stream = plain_terminal()
        terminal.say("One.")
        terminal.say("Two.")
        assert stream.getvalue().count("Business Analyst:") == 1

    def test_a_new_speaker_gets_a_new_label(self):
        terminal, stream = plain_terminal()
        terminal.say("One.")
        terminal.agent("developer", "Two.")
        terminal.say("Three.")
        assert stream.getvalue().count("Business Analyst:") == 2

    def test_the_label_returns_after_a_status_line(self):
        terminal, stream = plain_terminal()
        terminal.say("One.")
        terminal.status("Working…")
        terminal.say("Two.")
        assert stream.getvalue().count("Business Analyst:") == 2

    def test_a_blank_prompt_shows_only_the_user_label(self):
        terminal, stream = plain_terminal("x")
        terminal.ask("")
        assert stream.getvalue() == "\nYou:\n"

    async def test_all_four_agents_appear_in_a_real_workflow(self, tmp_path):
        harness = Harness(tmp_path)
        terminal, stream = plain_terminal()
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_proposal_gate(workflow)

        output = stream.getvalue()
        for label in ROLES.values():
            assert label in output
        assert "Software Architect:\nBuild calc.py with a plain function." in output

    async def test_the_session_labels_the_greeting_and_the_user(self, tmp_path):
        harness = Harness(tmp_path)
        terminal, stream = plain_terminal("4")
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        await session.run()
        output = stream.getvalue()
        assert "Business Analyst:\nHi! What product can I help you with today?" in output
        assert "You:" in output


class TestColors:
    def test_agents_use_the_agent_color(self):
        terminal, stream = color_terminal()
        terminal.say("Hello.")
        output = stream.getvalue()
        assert f"{console.AGENT_LABEL}Business Analyst:{console.RESET}" in output
        assert f"{console.AGENT_TEXT}Hello.{console.RESET}" in output

    def test_typed_input_is_shown_in_the_user_color(self):
        terminal, stream = color_terminal("typed")
        terminal.ask("Question?")
        output = stream.getvalue()
        assert output.endswith(
            f"{console.USER_LABEL}You:{console.RESET}\n{console.USER_TEXT}{console.RESET}"
        )

    def test_the_input_color_is_active_while_the_user_types(self):
        stream = FakeTty()
        seen: list[str] = []

        def read_line() -> str:
            seen.append(stream.getvalue())
            return "typed"

        ConsoleTerminal(stream, read_line, COLOR_ENV).ask("Question?")
        assert seen[0].endswith(console.USER_TEXT)

    def test_agent_and_user_colors_differ(self):
        assert console.AGENT_TEXT != console.USER_TEXT
        assert console.AGENT_LABEL != console.USER_LABEL

    def test_warnings_and_errors_have_their_own_colors(self):
        terminal, stream = color_terminal()
        terminal.warn("Careful.")
        terminal.error("Broken.")
        output = stream.getvalue()
        assert f"{console.WARNING_TEXT}Warning: Careful.{console.RESET}" in output
        assert f"{console.ERROR_TEXT}Error: Broken.{console.RESET}" in output

    def test_status_lines_are_subdued(self):
        terminal, stream = color_terminal()
        terminal.status("Working…")
        assert f"{console.STATUS_TEXT}  … Working…{console.RESET}" in stream.getvalue()

    def test_the_five_styles_are_all_different(self):
        styles = {
            console.AGENT_TEXT, console.USER_TEXT, console.WARNING_TEXT,
            console.ERROR_TEXT, console.STATUS_TEXT,
        }
        assert len(styles) == 5

    def test_each_line_of_a_block_is_colored_and_reset(self):
        terminal, stream = color_terminal()
        terminal.say("One.\n\nTwo.")
        output = stream.getvalue()
        assert f"{console.AGENT_TEXT}One.{console.RESET}\n\n" in output
        assert f"{console.AGENT_TEXT}Two.{console.RESET}" in output

    @pytest.mark.parametrize("failure", [EOFError, KeyboardInterrupt])
    def test_color_is_reset_if_input_is_interrupted(self, failure):
        stream = FakeTty()

        def read_line() -> str:
            raise failure

        with pytest.raises(failure):
            ConsoleTerminal(stream, read_line, COLOR_ENV).ask("Question?")
        assert stream.getvalue().endswith(console.RESET)

    def test_labels_survive_with_color_on(self):
        terminal, stream = color_terminal()
        terminal.agent("test_engineer", "Done.")
        assert "Test Engineer:" in stream.getvalue()


class TestNoColorFallback:
    def _exercise(self, terminal: ConsoleTerminal) -> None:
        terminal.say("Agent.")
        terminal.agent("developer", "Agent.")
        terminal.status("Status.")
        terminal.warn("Warning.")
        terminal.error("Error.")
        terminal.ask("Question?")

    @pytest.mark.parametrize(
        "environ",
        [
            {"TERM": "xterm-256color", "NO_COLOR": "1"},
            {"TERM": "xterm-256color", "NO_COLOR": "anything"},
            {"TERM": "dumb"},
            {"TERM": ""},
            {},
        ],
        ids=["no-color", "no-color-any-value", "dumb-terminal", "blank-term", "no-term"],
    )
    def test_no_escape_codes_reach_a_terminal_without_color(self, environ):
        stream = FakeTty()
        self._exercise(ConsoleTerminal(stream, lambda: "x", environ))
        assert ESC not in stream.getvalue()
        assert "You:" in stream.getvalue()
        assert "Warning: Warning." in stream.getvalue()
        assert "Error: Error." in stream.getvalue()

    def test_no_escape_codes_reach_redirected_output(self):
        stream = io.StringIO()
        self._exercise(ConsoleTerminal(stream, lambda: "x", COLOR_ENV))
        assert ESC not in stream.getvalue()

    def test_an_empty_no_color_variable_does_not_disable_color(self):
        assert supports_color(FakeTty(), {"TERM": "xterm", "NO_COLOR": ""}) is True

    def test_a_closed_stream_is_treated_as_no_color(self):
        stream = io.StringIO()
        stream.close()
        assert supports_color(stream, COLOR_ENV) is False

    def test_a_stream_without_isatty_is_treated_as_no_color(self):
        assert supports_color(object(), COLOR_ENV) is False  # type: ignore[arg-type]

    def test_the_real_environment_is_used_by_default(self, monkeypatch):
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setenv("NO_COLOR", "1")
        assert supports_color(FakeTty()) is False
        monkeypatch.delenv("NO_COLOR")
        assert supports_color(FakeTty()) is True


class TestModelOutputCannotControlTheTerminal:
    @pytest.mark.parametrize(
        "text",
        [
            "\x1b[2J\x1b[Hscreen cleared",
            "\x1b[32mYou:\x1b[0m fake user",
            "\x1b]0;window title\x07title set",
            "\x07bell",
            "\x9b31mcontrol sequence",
        ],
    )
    def test_escape_sequences_are_removed_from_output(self, text):
        terminal, stream = plain_terminal()
        terminal.say(text)
        assert ESC not in stream.getvalue()
        assert "\x07" not in stream.getvalue()
        assert "\x9b" not in stream.getvalue()

    def test_normal_text_is_untouched(self):
        text = "Line one\n\tLine two: café — “quotes” … 3 > 2"
        assert strip_control(text) == text

    def test_windows_line_endings_are_normalized(self):
        assert strip_control("one\r\ntwo") == "one\ntwo"


def run_in_terminal(arguments: list[str], environ: dict[str, str], *lines: bytes) -> str:
    """Run the real CLI on a real pseudo-terminal, typing a line at each prompt."""
    master, replica = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-m", "lab", *arguments],
        stdin=replica,
        stdout=replica,
        stderr=replica,
        cwd=PROJECT_ROOT,
        env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), **environ},
    )
    os.close(replica)

    output = b""
    pending = list(lines)
    prompts_answered = 0
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master], [], [], 0.2)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                output += chunk
            if pending and output.count(b"You:") > prompts_answered:
                prompts_answered += 1
                os.write(master, pending.pop(0))
            if process.poll() is not None and not ready:
                break
    finally:
        os.close(master)
        process.kill()
        process.wait()
    return output.decode("utf-8", "replace")


class TestRealTerminal:
    """The real CLI, with nothing mocked, stopped at the menu so no model is involved."""

    def test_colors_appear_on_a_real_terminal(self, tmp_path):
        output = run_in_terminal(
            ["--products-dir", str(tmp_path)], {"TERM": "xterm-256color"}, b"4\n"
        )
        assert f"{console.AGENT_LABEL}Business Analyst:{console.RESET}" in output
        assert f"{console.USER_LABEL}You:{console.RESET}" in output
        assert "Goodbye" in output

    def test_what_you_type_is_echoed_in_the_user_color(self, tmp_path):
        output = run_in_terminal(
            ["--products-dir", str(tmp_path)], {"TERM": "xterm-256color"}, b"4\n"
        )
        assert f"{console.USER_TEXT}4" in output
        assert output.rstrip().endswith(console.RESET), "the shell gets its colors back"

    def test_no_color_on_a_real_terminal(self, tmp_path):
        output = run_in_terminal(
            ["--products-dir", str(tmp_path)],
            {"TERM": "xterm-256color", "NO_COLOR": "1"},
            b"4\n",
        )
        assert ESC not in output
        assert "Business Analyst:" in output
        assert "You:" in output

    def test_dumb_terminal(self, tmp_path):
        output = run_in_terminal(["--products-dir", str(tmp_path)], {"TERM": "dumb"}, b"4\n")
        assert ESC not in output
        assert "You:" in output

    def test_redirected_output_has_no_color(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "lab", "--products-dir", str(tmp_path)],
            input="4\n",
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={"PATH": os.environ["PATH"], "TERM": "xterm-256color"},
        )
        assert result.returncode == 0
        assert ESC not in result.stdout
        assert ESC not in result.stderr
        assert "Business Analyst:" in result.stdout

    def test_pressing_enter_at_a_real_prompt_only_asks_again(self, tmp_path):
        output = run_in_terminal(
            ["--products-dir", str(tmp_path)], {"TERM": "xterm-256color"}, b"\n", b"\n", b"4\n"
        )
        assert output.count("I did not catch anything there.") == 2
        assert output.count("Hi! What product") == 1
        assert list(tmp_path.glob("draft-*")) == []

    def test_end_of_input_leaves_cleanly(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "-m", "lab", "--products-dir", str(tmp_path)],
            input="",
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={"PATH": os.environ["PATH"]},
        )
        assert result.returncode == 0
        assert "Goodbye" in result.stdout


def session_for(harness: Harness, *answers: str) -> tuple[ProductSession, ScriptedTerminal]:
    terminal = ScriptedTerminal(*answers)
    return (
        ProductSession(harness.manager, harness.router, harness.policy, harness.playbooks, terminal),
        terminal,
    )


class TestEmptyInputDuringDiscovery:
    async def _in_discovery(self, harness: Harness):
        harness.team.discovery_ready = False
        workflow = harness.new_workflow()
        await workflow.discovery_message("I want a tic-tac-toe game.")
        return workflow

    @pytest.mark.parametrize(
        "blank",
        ["", "   ", "\t", " \t  ", "\x1b[A", "\x1b[A\x1b[B", "\x00"],
        ids=["empty", "spaces", "tab", "mixed", "arrow-key", "arrow-keys", "control"],
    )
    async def test_a_blank_line_never_reaches_a_model(self, tmp_path, blank):
        harness = Harness(tmp_path)
        workflow = await self._in_discovery(harness)
        calls = len(harness.team.calls)
        revision = workflow.state.revision
        conversation = workflow.store.read_conversation()
        session, terminal = session_for(harness, blank, "/exit")

        with pytest.raises(_ExitSession):
            await session._on_discovery(workflow)

        assert len(harness.team.calls) == calls
        assert workflow.state.revision == revision
        assert workflow.state.discovery_round == 1
        assert workflow.state.pending_input is None
        assert workflow.store.read_conversation() == conversation
        assert EMPTY_INPUT_HINT in terminal.said

    async def test_no_attempt_is_recorded(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await self._in_discovery(harness)
        session, _ = session_for(harness, "", "", "", "/exit")

        with pytest.raises(_ExitSession):
            await session._on_discovery(workflow)

        assert not workflow.store.task_dir("discovery-2").exists()
        assert [path.name for path in (workflow.store.logs_dir / "tasks").iterdir()] == [
            "discovery-1"
        ]

    async def test_the_question_is_not_repeated_for_each_blank_line(self, tmp_path):
        """The reported symptom: repeated "Your thoughts?" prompts."""
        harness = Harness(tmp_path)
        workflow = await self._in_discovery(harness)
        session, terminal = session_for(harness, "", "", "", "/exit")

        with pytest.raises(_ExitSession):
            await session._on_discovery(workflow)

        assert terminal.prompts.count("Your thoughts?") == 1
        assert terminal.said.count(EMPTY_INPUT_HINT) == 3

    async def test_the_conversation_continues_with_the_next_real_answer(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await self._in_discovery(harness)
        session, _ = session_for(harness, "", "   ", "Two players.")

        await session._on_discovery(workflow)

        assert workflow.state.discovery_round == 2
        assert len(harness.team.calls_of("discovery")) == 2
        assert "Two players." in harness.team.calls_of("discovery")[-1].prompt
        assert workflow.store.read_conversation()[-2]["text"] == "Two players."

    async def test_a_blank_line_does_not_use_up_discovery_rounds(self, tmp_path):
        harness = Harness(tmp_path)
        harness.policy.workflow.discovery_max_rounds = 2
        workflow = await self._in_discovery(harness)
        session, _ = session_for(harness, *[""] * 10, "/exit")

        with pytest.raises(_ExitSession):
            await session._on_discovery(workflow)

        assert workflow.state.phase == ProductPhase.DISCOVERY

    async def test_whitespace_around_a_real_answer_is_removed(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await self._in_discovery(harness)
        session, _ = session_for(harness, "   Two players.   ")
        await session._on_discovery(workflow)
        assert workflow.store.read_conversation()[-2]["text"] == "Two players."

    async def test_slash_commands_during_discovery_are_not_sent_to_a_model(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await self._in_discovery(harness)
        calls = len(harness.team.calls)

        for command in ("/approve", "/change", "/ask", "/retry", "/accept"):
            session, terminal = session_for(harness, command)
            await session._on_discovery(workflow)
            assert any("That command is for later" in text for text in terminal.said)

        assert len(harness.team.calls) == calls

    async def test_a_slash_command_with_text_sends_only_the_text(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await self._in_discovery(harness)
        session, _ = session_for(harness, "/change Make it a two player game")
        await session._on_discovery(workflow)
        assert workflow.store.read_conversation()[-2]["text"] == "Make it a two player game"


class TestEmptyInputEverywhereElse:
    @pytest.mark.parametrize(
        "reach",
        [to_name_gate, to_roadmap_gate, to_step_start_gate, to_proposal_gate],
        ids=["name", "roadmap", "step-start", "proposal"],
    )
    async def test_a_blank_line_at_a_gate_changes_nothing(self, tmp_path, reach):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await reach(workflow)
        calls = len(harness.team.calls)
        revision = workflow.state.revision
        phase = workflow.state.phase
        name = workflow.meta.display_name
        session, terminal = session_for(harness, "", "   ", "/exit")

        with pytest.raises(_ExitSession):
            await session._product_loop(workflow, None)

        assert len(harness.team.calls) == calls
        assert workflow.state.revision == revision
        assert workflow.state.phase == phase
        assert workflow.meta.display_name == name
        assert terminal.said.count(EMPTY_INPUT_HINT) == 2

    async def test_blank_at_the_menu(self, tmp_path):
        harness = Harness(tmp_path)
        session, terminal = session_for(harness, "", "  ", "4")
        await session.run()
        assert terminal.said.count(EMPTY_INPUT_HINT) == 2
        assert harness.team.calls == []
        assert not harness.root.exists() or list(harness.root.iterdir()) == []

    async def test_a_blank_idea_creates_no_product(self, tmp_path):
        harness = Harness(tmp_path)
        session, terminal = session_for(harness, "1", "", "   ", "A real idea", "/exit")
        await session.run()
        assert terminal.said.count(EMPTY_INPUT_HINT) == 2
        assert len(harness.team.calls_of("discovery")) == 1
        assert "A real idea" in harness.team.calls_of("discovery")[0].prompt
        assert len(harness.manager.discover()[0]) == 1

    async def test_start_new_refuses_blank_text(self, tmp_path):
        harness = Harness(tmp_path)
        session, terminal = session_for(harness)
        await session.start_new("   ")
        assert harness.manager.discover()[0] == []
        assert harness.team.calls == []
        assert EMPTY_INPUT_HINT in terminal.said

    def test_a_blank_idea_on_the_command_line_creates_no_product(self, tmp_path, capsys):
        root = tmp_path / "p"
        assert main(["--products-dir", str(root), "new", "  "]) == 0
        assert "I did not catch anything there." in capsys.readouterr().out
        assert [path.name for path in root.iterdir()] == [".session.log"]

    async def test_blank_at_the_product_selection_prompt(self, tmp_path):
        harness = Harness(tmp_path)
        harness.manager.create("Dairy Corner")
        session, terminal = session_for(harness, "2", "", "unknown", "4")
        await session.run()
        assert EMPTY_INPUT_HINT in terminal.said

    async def test_blank_when_blocked_does_not_retry(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.script("architect", *[Fail(ConnectionError("down"))] * 4)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        workflow.start_step(APPROVE)
        await workflow.advance()
        calls = len(harness.team.calls)
        session, terminal = session_for(harness, "", "/exit")

        with pytest.raises(_ExitSession):
            await session._on_blocked(workflow)

        assert len(harness.team.calls) == calls
        assert workflow.state.phase == ProductPhase.BLOCKED
        assert EMPTY_INPUT_HINT in terminal.said


class TestCommandsWithNoText:
    @pytest.mark.parametrize("command", ["/change", "/ask"])
    async def test_at_the_proposal_gate(self, tmp_path, command):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        calls = len(harness.team.calls)
        session, terminal = session_for(harness)

        await session._on_gate(workflow, command)

        assert len(harness.team.calls) == calls
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert workflow.state.steps["step-01"].proposal_version == 1
        assert terminal.warnings

    @pytest.mark.parametrize("command", ["/change", "/ask"])
    async def test_at_the_roadmap_gate(self, tmp_path, command):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        calls = len(harness.team.calls)
        session, terminal = session_for(harness)

        await session._on_gate(workflow, command)

        assert len(harness.team.calls) == calls
        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL
        assert terminal.warnings

    async def test_at_the_step_start_gate(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_step_start_gate(workflow)
        calls = len(harness.team.calls)
        session, terminal = session_for(harness)

        await session._on_gate(workflow, "/change")

        assert len(harness.team.calls) == calls
        assert workflow.state.phase == ProductPhase.AWAITING_STEP_START
        assert terminal.warnings


class TestWorkflowRefusesBlankText:
    """The session filters blanks first.  These guard the workflow in its own right."""

    async def test_discovery(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        with pytest.raises(GateViolation):
            await workflow.discovery_message("   ")
        assert workflow.state.discovery_round == 0
        assert harness.team.calls == []

    async def test_questions(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        with pytest.raises(GateViolation):
            await workflow.ask_team(HumanDecision(" "))

    async def test_step_changes(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_proposal_gate(workflow)
        with pytest.raises(GateViolation):
            await workflow.request_step_modification(HumanDecision(""))

    async def test_roadmap_changes(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_roadmap_gate(workflow)
        with pytest.raises(GateViolation):
            workflow.request_roadmap_change(HumanDecision(""))


async def paused_workspace(harness: Harness) -> "ProductWorkflow":  # noqa: F821
    """Rebuild the situation the reported bug left behind, at the Step 1 proposal."""
    harness.team.product_name = "BoardBattle"
    workflow = harness.new_workflow()
    await to_name_gate(workflow)
    workflow.change_name(HumanDecision("ok for the name"))
    await workflow.advance()
    workflow.approve_roadmap(APPROVE)
    workflow.start_step(APPROVE)
    await workflow.advance()
    assert workflow.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
    return workflow


def dead_process_id() -> int:
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


class TestResumingTheExistingWorkspace:
    async def test_a_stale_lock_does_not_stop_a_resume(self, tmp_path):
        """The real workspace has a lock left by a process that no longer exists."""
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)
        workflow.store.lock_path.write_text(
            json.dumps({"pid": dead_process_id(), "host": socket.gethostname()})
        )
        harness.router = harness.new_router()
        session, terminal = session_for(harness, "2", "ok for the name", "/exit")

        await session.run()

        assert any("Picking up ok for the name" in text for text in terminal.said)
        assert any("Do you approve this proposal?" in text for text in terminal.prompts)
        assert not workflow.store.lock_path.exists()

    async def test_the_proposal_is_shown_again_without_any_model_work(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)
        calls = len(harness.team.calls)
        harness.router = harness.new_router()
        session, terminal = session_for(harness, "2", "1", "/exit")

        await session.run()

        assert len(harness.team.calls) == calls
        assert any("Proposal for Step 1" in text for text in terminal.said)

    async def test_resuming_keeps_the_name_folder_and_gate(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)
        harness.router = harness.new_router()
        session, _ = session_for(harness, "2", "1", "/exit")

        await session.run()

        reopened = harness.workflow(harness.manager.open("ok-for-the-name"))
        assert reopened.meta.display_name == "ok for the name"
        assert reopened.store.slug == "ok-for-the-name"
        assert reopened.state.phase == ProductPhase.AWAITING_STEP_APPROVAL
        assert reopened.state.steps["step-01"].proposal_version == 1
        assert reopened.state.steps["step-01"].approved_version is None

    async def test_audit_history_only_grows_across_a_resume(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)
        logs = workflow.store.logs_dir
        before = {
            name: (logs / name).read_bytes()
            for name in ("events.jsonl", "approvals.jsonl", "handoffs.jsonl", "conversation.jsonl")
        }
        harness.router = harness.new_router()
        session, _ = session_for(harness, "2", "1", "/exit")

        await session.run()

        for name, content in before.items():
            assert (logs / name).read_bytes().startswith(content), name

    async def test_the_proposal_can_be_approved_after_resuming(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)
        harness.router = harness.new_router()
        session, _ = session_for(harness, "2", "1", "yes", "/exit")

        await session.run()

        reopened = harness.workflow(harness.manager.open("ok-for-the-name"))
        assert reopened.state.phase == ProductPhase.AWAITING_STEP_START
        assert harness.team.calls_of("implement")
        assert reopened.completion_problems("step-01") == []

    async def test_a_formally_paused_workspace_resumes_the_same_way(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)
        workflow.pause(APPROVE)
        harness.router = harness.new_router()
        session, terminal = session_for(harness, "2", "1", "/exit")

        await session.run()

        assert any("Do you approve this proposal?" in text for text in terminal.prompts)

    async def test_a_name_like_phrase_at_the_proposal_gate_is_not_a_rename(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)
        session, _ = session_for(harness)

        await session._on_gate(workflow, "yes")

        assert workflow.meta.display_name == "ok for the name"

    def test_the_cli_lists_and_finds_the_product_by_its_phrase_name(self, tmp_path, capsys):
        import asyncio

        harness = Harness(tmp_path)
        asyncio.run(paused_workspace(harness))
        assert main(["--products-dir", str(harness.root), "status", "ok for the name"]) == 0
        output = capsys.readouterr().out
        assert "ok for the name — Command-Line Tool" in output
        assert "Next action: Approve implementation proposal for Step 1." in output


class TestLeavingFromADirectCommand:
    """`resume` and `new` skip the menu, and must leave the same way it does."""

    def _resume(self, root: Path, typed: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "lab", "--products-dir", str(root), "resume", "ok for the name"],
            input=typed,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={"PATH": os.environ["PATH"]},
        )

    @pytest.mark.parametrize("typed", ["/exit\n", "quit\n", ""], ids=["exit", "quit", "eof"])
    async def test_resume_leaves_without_a_traceback(self, tmp_path, typed):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)

        result = self._resume(harness.root, typed)

        assert result.returncode == 0
        assert "Traceback" not in result.stderr
        assert "Goodbye" in result.stdout
        assert not workflow.store.lock_path.exists()

    async def test_resume_can_pause_from_the_command_line(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = await paused_workspace(harness)

        result = self._resume(harness.root, "/pause\n")

        assert result.returncode == 0
        assert "is paused" in result.stdout
        reopened = harness.workflow(harness.manager.open(workflow.store.slug))
        assert reopened.state.paused is True
        assert reopened.state.phase == ProductPhase.AWAITING_STEP_APPROVAL

    async def test_new_leaves_cleanly_from_the_conversation(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.discovery_ready = False
        session, terminal = session_for(harness, "/exit")

        await session.run_product(session.start_new("A small game"))

        assert any("Goodbye" in text for text in terminal.said)
        summaries, _ = harness.manager.discover()
        assert len(summaries) == 1

    async def test_models_are_released_when_a_direct_command_ends(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.discovery_ready = False
        session, _ = session_for(harness, "/exit")
        await session.run_product(session.start_new("A small game"))
        assert harness.lifecycle.loaded == set()


REAL_WORKSPACE = PROJECT_ROOT / "products_log" / "ok-for-the-name"


@pytest.mark.skipif(
    not (REAL_WORKSPACE / "product.json").is_file(),
    reason="Runs only on the machine that has this workspace",
)
class TestTheRealWorkspaceCopy:
    """Works on a copy, so the real folder is only ever read.

    The product moves on as it is used, so these make no claim about which checkpoint
    it has reached: only that whatever is on disk still loads and resumes unchanged.
    """

    @pytest.fixture
    def copy(self, tmp_path):
        harness = Harness(tmp_path)
        harness.root.mkdir()
        target = harness.root / "ok-for-the-name"
        shutil.copytree(REAL_WORKSPACE, target)
        (target / ".session.lock").unlink(missing_ok=True)
        return harness

    def test_the_saved_checkpoint_still_loads(self, copy):
        saved = json.loads((REAL_WORKSPACE / "state.json").read_text())
        workflow = copy.workflow(copy.manager.open("ok-for-the-name"))
        assert workflow.state.phase.value == saved["phase"]
        assert workflow.state.current_step_id == saved["current_step_id"]
        assert workflow.state.revision == saved["revision"]

    def test_resuming_keeps_the_checkpoint(self, copy):
        saved = json.loads((REAL_WORKSPACE / "state.json").read_text())
        workflow = copy.workflow(copy.manager.open("ok-for-the-name"))
        workflow.resume()
        assert workflow.state.phase.value == saved["phase"]
        assert workflow.state.current_step_id == saved["current_step_id"]

    def test_it_appears_in_the_product_list(self, copy):
        summaries, invalid = copy.manager.discover()
        assert [item.slug for item in summaries] == ["ok-for-the-name"]
        assert invalid == []

    def test_an_interrupted_task_is_reported_and_kept(self, copy):
        saved = json.loads((REAL_WORKSPACE / "state.json").read_text())
        workflow = copy.workflow(copy.manager.open("ok-for-the-name"))
        notes = workflow.resume()
        if saved["active_task"] is None:
            assert workflow.state.active_task is None
        else:
            assert workflow.state.active_task.task_id == saved["active_task"]["task_id"]
            assert any("interrupted" in note for note in notes)
