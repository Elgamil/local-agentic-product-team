"""The interactive terminal session: the Business Analyst as the interface.

The session turns what the person types into :class:`HumanDecision` calls on the
workflow.  It never decides anything itself, and ordinary output stays in business
language.  Internal logs go to the files under ``products_log/``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Protocol

from lab.config import ModelPolicy
from lab.playbooks import PlaybookLibrary
from lab.product.console import ConsoleTerminal, strip_control
from lab.product.decisions import Intent, NameAnswer, ParsedInput, parse_input, read_name_reply
from lab.product.manager import ProductManager, ProductSummary, slugify
from lab.product.store import ProductBusyError, ProductLock, ProductStore, StateCorruptedError
from lab.product.workflow import (
    GateViolation,
    HumanDecision,
    ProductWorkflow,
    VerifyOutcome,
    VerifyStatus,
)
from lab.runtime.router import ModelRouter
from lab.schemas.product import BlockKind, ProductPhase, TestRevision

logger = logging.getLogger(__name__)

GREETING = "Hi! What product can I help you with today?"
EMPTY_INPUT_HINT = "I did not catch anything there. Please type a response, or /help for options."
GOODBYE = "Goodbye! Everything is saved, so we can pick up exactly where we left off."
MENU = (
    "  1. Start a new product\n"
    "  2. Continue an existing product\n"
    "  3. List existing products\n"
    "  4. Exit"
)
HELP_TEXT = """You can talk to me normally. These shortcuts are always available:

  /status            Where this product stands
  /team              What the specialists decided recently
  /products          List your products
  /approve           Approve what is on the table
  /reject [reason]   Reject the proposal
  /change <request>  Ask for a change
  /ask <question>    Ask the team a question
  /retry             Authorize another AI attempt after a blocked task
  /verify            I fixed it by hand: check it and continue
  /report            Show what is blocked and how to fix it
  /accept            Accept a step with known limitations
  /done              Finish the discussion and move on

  /autonomy on             Run the rest of the roadmap without asking at each step
  /autonomy until <point>  Run until a step or milestone, for example 'step 3', then stop
  /autonomy off            Return to Manual Mode, where you approve every step
  /autonomy status         Show the current execution mode

  /pause             Pause this product (your progress is saved)
  /stop              Pause and end any autonomous run
  /exit              Leave (your progress is saved)"""
RUNNING_HELP = (
    "The team is working. You can type /status, /team, /report, /pause, /stop, /exit, "
    "/autonomy off or /autonomy until <point>. The current model call and file write "
    "always finish first."
)
#: How often typed commands are noticed while the team works.
CONTROL_POLL_SECONDS = 0.25


class Terminal(Protocol):
    heartbeat_seconds: float

    def ask(self, prompt: str) -> str: ...

    def say(self, text: str) -> None: ...

    def agent(self, role: str, text: str) -> None: ...

    def status(self, text: str) -> None: ...

    def warn(self, text: str) -> None: ...

    def error(self, text: str) -> None: ...

    def progress(self, text: str, context: str = "") -> None: ...

    def progress_done(self, text: str, context: str = "") -> None: ...


class _LeaveProduct(Exception):
    """Return to the main menu."""


class _ExitSession(Exception):
    """Leave the program."""


class ProductSession:
    def __init__(
        self,
        manager: ProductManager,
        router: ModelRouter,
        policy: ModelPolicy,
        playbooks: PlaybookLibrary,
        terminal: Terminal | None = None,
    ) -> None:
        self.manager = manager
        self.router = router
        self.policy = policy
        self.playbooks = playbooks
        self.terminal: Terminal = terminal or ConsoleTerminal()
        # Conversation state for the name gate.  It lives only in this session: losing
        # it costs one repeated question, and it changes nothing in the product.
        self._offered_name: str | None = None
        self._name_requested = False
        self._question_answered = False
        self._shown_report: str | None = None
        self._shown_revision: str | None = None
        self._announced_authorization: str | None = None
        self._exit_requested = False

    # ------------------------------------------------------------------
    # Reading input
    # ------------------------------------------------------------------

    def _read(self, prompt: str) -> str:
        """Ask until the person types something.

        A blank line goes no further than this loop: nothing is saved, the workflow
        does not move, and no model is called.
        """
        text = strip_control(self.terminal.ask(prompt)).strip()
        while not text:
            self.terminal.say(EMPTY_INPUT_HINT)
            text = strip_control(self.terminal.ask("")).strip()
        return text

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """The main menu loop."""
        try:
            self._show_unfinished()
            self.terminal.say(GREETING)
            while True:
                self.terminal.say(MENU)
                choice = self._read("You can pick a number, or just tell me your idea.")
                await self._dispatch_menu(choice)
        except (_ExitSession, EOFError, KeyboardInterrupt):
            self.terminal.say(GOODBYE)
        finally:
            await self.release_models()

    async def run_product(self, conversation: Awaitable[None]) -> None:
        """Run one product directly, for the `new` and `resume` commands.

        Leaving works here exactly as it does from the menu.
        """
        try:
            await conversation
        except (_ExitSession, EOFError, KeyboardInterrupt):
            self.terminal.say(GOODBYE)
        finally:
            await self.release_models()

    async def release_models(self) -> None:
        try:
            await asyncio.wait_for(self.router.release_all(), timeout=30)
        except Exception as exc:  # Leaving must not fail because memory cleanup did.
            logger.warning("Could not release models on exit: %s", exc)

    async def _dispatch_menu(self, choice: str) -> None:
        text = choice.strip()
        lowered = text.lower()
        if lowered in ("1", "new", "new product", "start", "start a new product"):
            await self.start_new(self._read("Tell me about your product idea."))
        elif lowered in ("2", "continue", "resume", "continue an existing product"):
            await self._choose_and_resume()
        elif lowered in ("3", "list", "list products", "list existing products"):
            self.show_products()
        elif lowered in ("4", "exit", "quit", "bye", "/exit", "/quit"):
            raise _ExitSession
        else:
            await self.start_new(text)

    # ------------------------------------------------------------------
    # Product list
    # ------------------------------------------------------------------

    def _show_unfinished(self) -> None:
        summaries, invalid = self.manager.discover()
        unfinished = [item for item in summaries if not item.finished]
        if unfinished:
            lines = [f"Welcome back. You have {len(unfinished)} product(s) in progress:"]
            for item in unfinished:
                lines.append(f"  - {item.display_name}: {item.status}")
                if item.interrupted_task:
                    lines.append(
                        "    A task was interrupted last time. I can resume it from its "
                        "saved inputs."
                    )
                else:
                    lines.append(f"    Next: {item.next_action}")
            self.terminal.say("\n".join(lines))
        for item in invalid:
            self.terminal.error(
                f"I could not read the product folder '{item.slug}': {item.reason}"
            )

    def show_products(self) -> list[ProductSummary]:
        summaries, invalid = self.manager.discover()
        if not summaries:
            self.terminal.say("There are no products yet.")
        else:
            blocks = [
                f"{index}. {item.render()}" for index, item in enumerate(summaries, start=1)
            ]
            self.terminal.say("\n\n".join(blocks))
        for item in invalid:
            self.terminal.error(f"Needs attention: {item.slug} ({item.reason})")
        return summaries

    async def _choose_and_resume(self) -> None:
        summaries = self.show_products()
        if not summaries:
            return
        answer = self._read("Which one? Give me its number or name.")
        chosen = self.manager.select(answer, summaries)
        if chosen is None:
            self.terminal.say("I could not match that to one product.")
            return
        await self.open_product(chosen.slug)

    # ------------------------------------------------------------------
    # Opening a product
    # ------------------------------------------------------------------

    def _workflow(self, store: ProductStore) -> ProductWorkflow:
        return ProductWorkflow(
            store, self.manager, self.router, self.policy, self.playbooks, reporter=self.terminal
        )

    async def start_new(self, idea: str) -> None:
        if not idea.strip():
            self.terminal.say(EMPTY_INPUT_HINT)
            return
        store = self.manager.create()
        await self._drive(store, first_message=idea)

    async def open_product(self, slug: str) -> None:
        try:
            store = self.manager.open(slug)
            await self._drive(store)
        except FileNotFoundError as exc:
            self.terminal.error(str(exc))
        except StateCorruptedError as exc:
            self.terminal.error(f"I cannot open this product safely: {exc}")

    async def _drive(self, store: ProductStore, first_message: str | None = None) -> None:
        lock = ProductLock(store)
        try:
            lock.acquire()
        except ProductBusyError as exc:
            self.terminal.warn(str(exc))
            return
        workflow: ProductWorkflow | None = None
        try:
            workflow = self._workflow(store)
            if first_message is None:
                self._announce_resume(workflow)
            await self._product_loop(workflow, first_message)
        except _LeaveProduct:
            pass
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Ctrl-C is a person stopping the run, so it must not resume by itself.
            if workflow is not None and workflow.autonomy_active:
                workflow.pause(HumanDecision("Ctrl-C"))
            raise
        finally:
            lock.release()

    def _announce_resume(self, workflow: ProductWorkflow) -> None:
        if workflow.state.phase == ProductPhase.HISTORICAL:
            return
        name = workflow.meta.display_name
        held = workflow.state.paused and workflow.autonomy_active
        notes = workflow.resume(hold=held)
        self.terminal.say(f"Picking up {name}.\n{workflow.status_report()}")
        for note in notes:
            self.terminal.say(note)
        if held:
            self._confirm_autonomous_resume(workflow)

    def _confirm_autonomous_resume(self, workflow: ProductWorkflow) -> None:
        """A paused autonomous run needs a word from the person before it moves again."""
        while workflow.state.paused:
            text = self._read(
                f"You paused this product, and its execution mode is still "
                f"{workflow.execution_mode_label()}. Say yes to continue, or use /autonomy "
                "off to return to Manual Mode first."
            )
            parsed = parse_input(text)
            if parsed.intent == Intent.APPROVE:
                workflow.release_pause(HumanDecision(text))
            elif parsed.intent == Intent.AUTONOMY:
                self._change_autonomy(workflow, parsed.text)
                if not workflow.autonomy_active:
                    workflow.release_pause(HumanDecision(text))
            else:
                self._common(workflow, parsed)

    # ------------------------------------------------------------------
    # The conversation
    # ------------------------------------------------------------------

    async def _product_loop(self, workflow: ProductWorkflow, first_message: str | None) -> None:
        shown_gate: str | None = None
        if first_message is not None:
            await workflow.discovery_message(first_message)

        while True:
            await self._advance(workflow)
            if self._exit_requested:
                self._exit_requested = False
                raise _ExitSession
            if workflow.state.paused:
                self.terminal.say(
                    f"{workflow.status_report()}\n\n"
                    f"{workflow.meta.display_name} is paused at a safe point. Nothing is lost."
                )
                raise _LeaveProduct
            phase = workflow.state.phase

            if phase == ProductPhase.HISTORICAL:
                self.terminal.say(
                    f"{workflow.status_report()}\n\n"
                    "This product is preserved for reference, so there is nothing to continue."
                )
                return
            if phase == ProductPhase.PRODUCT_COMPLETE:
                await self._on_complete(workflow)
                continue
            if phase == ProductPhase.BLOCKED:
                await self._on_blocked(workflow)
                continue
            if phase == ProductPhase.DISCOVERY:
                await self._on_discovery(workflow)
                continue

            gate = workflow.gate()
            if gate is None:
                # Nothing to run and nothing to ask, so waiting here would spin forever.
                self.terminal.say(workflow.status_report())
                return
            if gate.key != shown_gate:
                if gate.message:
                    self.terminal.say(gate.message)
                shown_gate = gate.key
            question = gate.question
            if self._question_answered:
                question = ""
            else:
                # The gate's own question replaces any earlier offer.  Otherwise a "yes"
                # meant for this question would apply a rename offered before it.
                self._offered_name = None
                self._name_requested = False
            self._question_answered = False
            await self._on_gate(workflow, self._read(question))

    # ------------------------------------------------------------------
    # Control during an autonomous run
    # ------------------------------------------------------------------

    async def _advance(self, workflow: ProductWorkflow) -> None:
        """Run the workflow, listening for commands if a person is not being asked.

        Manual Mode reads no input here, so anything typed ahead still reaches the next
        gate, exactly as it always has.
        """
        if not workflow.autonomy_active:
            await workflow.advance()
            return
        self._announce_autonomy(workflow)
        workflow.on_boundary = lambda: self._drain_controls(workflow)
        watcher = asyncio.create_task(self._watch_controls(workflow))
        try:
            await workflow.advance()
        finally:
            workflow.on_boundary = None
            watcher.cancel()

    def _announce_autonomy(self, workflow: ProductWorkflow) -> None:
        authorization = workflow.state.autonomy.authorization_id
        if authorization != self._announced_authorization:
            self._announced_authorization = authorization
            self.terminal.status(
                f"Execution mode: {workflow.execution_mode_label()}. {RUNNING_HELP}"
            )

    async def _watch_controls(self, workflow: ProductWorkflow) -> None:
        while True:
            await asyncio.sleep(CONTROL_POLL_SECONDS)
            self._drain_controls(workflow)

    def _drain_controls(self, workflow: ProductWorkflow) -> None:
        poll = getattr(self.terminal, "poll_line", None)
        if poll is None:
            return
        while (line := poll()) is not None:
            self._on_control(workflow, strip_control(line).strip())

    def _on_control(self, workflow: ProductWorkflow, text: str) -> None:
        """One line typed during a run.  It records a request and starts nothing.

        The running model call or file write is never interrupted.  A mode change is
        saved now and read at the next gate; a pause is honoured at the next safe point.
        """
        parsed = parse_input(text)
        if parsed.intent == Intent.EMPTY:
            return
        if parsed.intent == Intent.AUTONOMY:
            self._change_autonomy(workflow, parsed.text)
        elif parsed.intent == Intent.PAUSE:
            workflow.pause(HumanDecision(text))
            self.terminal.status("Pausing at the next safe point.")
        elif parsed.intent == Intent.STOP:
            workflow.stop(HumanDecision(text))
            self.terminal.status(
                "Stopping at the next safe point, and returning to Manual Mode."
            )
        elif parsed.intent == Intent.EXIT:
            workflow.pause(HumanDecision(text))
            self._exit_requested = True
            self.terminal.status("Leaving at the next safe point.")
        elif parsed.intent in (Intent.STATUS, Intent.TEAM, Intent.REPORT):
            self._common(workflow, parsed)
        else:
            self.terminal.say(RUNNING_HELP)

    def _change_autonomy(self, workflow: ProductWorkflow, text: str) -> None:
        try:
            self.terminal.say(workflow.change_autonomy(HumanDecision(text)))
        except GateViolation as exc:
            self.terminal.warn(str(exc))

    def _common(self, workflow: ProductWorkflow, parsed: ParsedInput) -> bool:
        """Handle input that means the same everywhere.  True when it was consumed."""
        if parsed.intent == Intent.STATUS:
            self.terminal.say(workflow.status_report())
        elif parsed.intent == Intent.TEAM:
            self.terminal.say(workflow.team_activity())
        elif parsed.intent == Intent.AUTONOMY:
            self._change_autonomy(workflow, parsed.text)
        elif parsed.intent == Intent.REPORT:
            self.terminal.say(workflow.intervention_summary())
        elif parsed.intent == Intent.VERIFY and workflow.state.phase != ProductPhase.BLOCKED:
            self.terminal.say("Nothing is waiting for verification right now.")
        elif parsed.intent == Intent.STOP:
            workflow.stop(HumanDecision(parsed.text))
            self.terminal.say(
                f"{workflow.meta.display_name} is stopped and back in Manual Mode. Nothing is lost."
            )
            raise _LeaveProduct
        elif parsed.intent == Intent.HELP:
            self.terminal.say(HELP_TEXT)
        elif parsed.intent == Intent.PRODUCTS:
            self.show_products()
        elif parsed.intent == Intent.EMPTY:
            pass
        elif parsed.intent == Intent.PAUSE:
            workflow.pause(HumanDecision(parsed.text))
            self.terminal.say(
                f"{workflow.meta.display_name} is paused. Nothing is lost, even if it stays "
                "paused for months."
            )
            raise _LeaveProduct
        elif parsed.intent == Intent.EXIT:
            raise _ExitSession
        else:
            return False
        return True

    async def _on_discovery(self, workflow: ProductWorkflow) -> None:
        text = self._read("Your thoughts?")
        parsed = parse_input(text)
        if parsed.intent == Intent.DONE:
            workflow.finish_discovery(HumanDecision(text))
            return
        if text.startswith("/"):
            if self._common(workflow, parsed):
                return
            if parsed.intent in (Intent.MODIFY, Intent.QUESTION) and parsed.text:
                await workflow.discovery_message(parsed.text)
            else:
                # A command is an instruction to the system, not a message for a model.
                self.terminal.say(
                    "That command is for later. For now, tell me about the product, or "
                    "type /done to move on."
                )
            return
        if parsed.intent in (Intent.EXIT, Intent.PAUSE) and self._common(workflow, parsed):
            return
        await workflow.discovery_message(text)

    async def _on_gate(self, workflow: ProductWorkflow, text: str) -> None:
        parsed = parse_input(text)
        if self._common(workflow, parsed):
            return
        workflow.store.append_conversation("user", text.strip())
        try:
            phase = workflow.state.phase
            if phase == ProductPhase.AWAITING_NAME:
                self._at_name_gate(workflow, text)
            elif phase == ProductPhase.AWAITING_ROADMAP_APPROVAL:
                await self._at_roadmap_gate(workflow, parsed)
            elif phase == ProductPhase.AWAITING_STEP_START:
                self._at_step_start_gate(workflow, parsed)
            elif phase == ProductPhase.AWAITING_STEP_APPROVAL:
                await self._at_proposal_gate(workflow, parsed)
        except GateViolation as exc:
            self.terminal.warn(str(exc))

    def _ask_about_name(self, text: str) -> None:
        """Say *text* in place of repeating the gate's own question."""
        self.terminal.say(text)
        self._question_answered = True

    def _rename(self, workflow: ProductWorkflow, name: str) -> None:
        slug = workflow.change_name(HumanDecision(name))
        message = f"Done. The product is now {name}, in products_log/{slug}."
        if slug != slugify(name):
            message += " That folder name was taken, so I picked the nearest free one."
        self.terminal.say(message)

    def _at_name_gate(self, workflow: ProductWorkflow, text: str) -> None:
        current = workflow.meta.display_name
        reply = read_name_reply(text, current)
        offered, self._offered_name = self._offered_name, None
        requested, self._name_requested = self._name_requested, False

        if offered is not None and reply.answer == NameAnswer.ACCEPT:
            self._rename(workflow, offered)
        elif offered is not None and reply.answer == NameAnswer.DECLINE:
            self.terminal.say(f"All right, I will not rename it. The name is still {current}.")
        elif reply.answer == NameAnswer.ACCEPT:
            workflow.confirm_name(HumanDecision(text))
            self.terminal.say(f"{current} it is.")
        elif reply.answer == NameAnswer.DECLINE:
            self._name_requested = True
            self._ask_about_name("Sure. What name would you prefer?")
        elif reply.answer == NameAnswer.NEW_NAME:
            self._rename(workflow, reply.name)
        elif reply.answer == NameAnswer.POSSIBLE_NAME and requested:
            self._rename(workflow, reply.name)
        elif reply.answer == NameAnswer.POSSIBLE_NAME:
            self._offered_name = reply.name
            self._ask_about_name(
                f"Should I rename the product to {reply.name}? "
                f"Say yes to rename it, or no to keep {current}."
            )
        else:
            self._name_requested = requested
            self._ask_about_name(
                f"I want to get this right. To keep {current}, say yes. To change it, "
                "tell me the name like this: call it Sunny Market."
            )

    async def _at_roadmap_gate(self, workflow: ProductWorkflow, parsed: ParsedInput) -> None:
        decision = HumanDecision(parsed.text)
        if parsed.intent == Intent.APPROVE:
            workflow.approve_roadmap(decision)
            self.terminal.say("The roadmap is approved and is now our baseline.")
        elif parsed.intent == Intent.REMOVE_STEP and parsed.number is not None:
            workflow.remove_roadmap_step(parsed.number, decision)
        elif parsed.intent == Intent.MOVE_STEP and parsed.number and parsed.position:
            workflow.move_roadmap_step(parsed.number, parsed.position, decision)
        elif parsed.intent == Intent.QUESTION:
            self.terminal.say(await workflow.ask_team(decision))
        elif parsed.intent == Intent.REJECT:
            self.terminal.say("No problem. Tell me what you would like to change.")
        else:
            workflow.request_roadmap_change(decision)

    def _at_step_start_gate(self, workflow: ProductWorkflow, parsed: ParsedInput) -> None:
        decision = HumanDecision(parsed.text)
        if parsed.intent == Intent.APPROVE:
            workflow.start_step(decision)
        elif parsed.intent == Intent.REJECT:
            workflow.pause(decision)
            self.terminal.say("All right, I will hold here. Your progress is saved.")
            raise _LeaveProduct
        elif parsed.intent == Intent.QUESTION:
            self.terminal.say(
                "We are between steps, so there is no proposal to discuss yet. Say yes to "
                "continue, or tell me how the roadmap should change."
            )
        else:
            workflow.request_roadmap_change(decision)

    async def _at_proposal_gate(self, workflow: ProductWorkflow, parsed: ParsedInput) -> None:
        decision = HumanDecision(parsed.text)
        if parsed.intent == Intent.APPROVE:
            workflow.approve_step_proposal(decision)
            self.terminal.say("Approved. The Developer is starting now.")
        elif parsed.intent == Intent.REJECT:
            workflow.reject_step_proposal(decision)
            self.terminal.say("Rejected. The proposal stays in the history.")
        elif parsed.intent == Intent.QUESTION:
            self.terminal.say(await workflow.ask_team(decision))
        else:
            self.terminal.status("Passing your change back to the team…")
            await workflow.request_step_modification(decision)

    async def _on_blocked(self, workflow: ProductWorkflow) -> None:
        blocked = workflow.state.blocked
        assert blocked is not None
        self.terminal.warn(f"I need your help to continue.\n{blocked.message}")
        intervention = workflow.ensure_intervention()
        if intervention is not None:
            report = f"{intervention.intervention_id}:{intervention.report_version}"
            if report != self._shown_report:
                self._shown_report = report
                self.terminal.say(workflow.intervention_summary())
            if intervention.pending_revision is not None:
                await self._on_test_revision(workflow, intervention.pending_revision)
                return
        options = {
            BlockKind.MODEL_FAILURE: (
                "Should I try again? Say yes to retry, /verify after fixing it yourself, or "
                "/pause to come back later."
            ),
            BlockKind.DETERMINISTIC_FAILURE: (
                "This needs fixing on your computer. Tell me when it is done with /retry or "
                "/verify, or /pause."
            ),
            BlockKind.VERIFICATION_FAILED: (
                "Say yes for more repair attempts, /verify after fixing it yourself, or "
                "describe what should change."
            ),
            BlockKind.CONDITIONS_UNMET: (
                "Say yes for more repair attempts, /verify after fixing it yourself, /accept "
                "to take the step as it is, or describe what should change."
            ),
            BlockKind.INTEGRITY: (
                "Use /verify to review a change you made, describe what the new proposal "
                "should do, or /pause."
            ),
        }
        text = self._read(options[blocked.kind])
        parsed = parse_input(text)
        if self._common(workflow, parsed):
            return
        try:
            if parsed.intent == Intent.VERIFY:
                self.terminal.status("Checking your changes…")
                self._show_outcome(workflow.verify_manual_fix(HumanDecision(text)))
            elif parsed.intent in (Intent.APPROVE, Intent.RETRY):
                workflow.retry_blocked(HumanDecision(text))
            elif parsed.intent == Intent.ACCEPT:
                workflow.accept_with_limitations(HumanDecision(text))
            elif parsed.intent == Intent.MODIFY and blocked.kind in (
                BlockKind.VERIFICATION_FAILED,
                BlockKind.CONDITIONS_UNMET,
                BlockKind.INTEGRITY,
            ):
                await workflow.request_step_modification(HumanDecision(parsed.text))
            else:
                self.terminal.say("I did not catch a decision there.")
        except GateViolation as exc:
            self.terminal.warn(str(exc))

    def _show_outcome(self, outcome: VerifyOutcome) -> None:
        if outcome.status == VerifyStatus.PASSED:
            self.terminal.say(outcome.message)
        elif outcome.status == VerifyStatus.NOTHING_TO_VERIFY:
            self.terminal.say(outcome.message)
        elif outcome.status != VerifyStatus.NEEDS_TEST_APPROVAL:
            self.terminal.warn(outcome.message)

    async def _on_test_revision(
        self, workflow: ProductWorkflow, revision: TestRevision
    ) -> None:
        """Nothing runs against an edited suite until a person has seen and approved it."""
        if revision.revision_id != self._shown_revision:
            self._shown_revision = revision.revision_id
            self.terminal.say(workflow.revision_summary())
        question = (
            "Do you approve this as the new authoritative test suite? It may correct how a "
            "criterion is tested, not what it requires. To change a requirement, use /change."
        )
        if revision.warnings:
            question += " Because this may weaken the tests, approve with /approve <reason>."
        text = self._read(f"{question} /reject restores the sealed tests.")
        parsed = parse_input(text)
        if self._common(workflow, parsed):
            return
        try:
            if parsed.intent == Intent.APPROVE:
                self._show_outcome(workflow.approve_test_revision(HumanDecision(parsed.text)))
            elif parsed.intent == Intent.REJECT:
                self.terminal.say(workflow.reject_test_revision(HumanDecision(parsed.text)))
            elif parsed.intent == Intent.MODIFY and text.startswith("/"):
                await workflow.request_step_modification(HumanDecision(parsed.text))
            else:
                self.terminal.say("I did not catch a decision on your revised tests.")
        except GateViolation as exc:
            self.terminal.warn(str(exc))

    async def _on_complete(self, workflow: ProductWorkflow) -> None:
        self.terminal.say(workflow.completion_message())
        text = self._read(
            "Would you like to add anything to the roadmap? Describe it, or say no to finish."
        )
        parsed = parse_input(text)
        if self._common(workflow, parsed):
            return
        if parsed.intent in (Intent.REJECT, Intent.APPROVE):
            raise _LeaveProduct
        try:
            workflow.request_roadmap_change(HumanDecision(parsed.text))
        except GateViolation as exc:
            self.terminal.warn(str(exc))
