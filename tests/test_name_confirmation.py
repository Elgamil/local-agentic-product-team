"""Confirming a product name.  Regression tests for the `ok-for-the-name` bug."""

from __future__ import annotations

import pytest

from lab.product.decisions import NameAnswer, read_name_reply
from lab.product.session import ProductSession, _ExitSession
from lab.schemas.product import ProductPhase
from tests.fakes import Harness, ScriptedTerminal, to_name_gate

PROPOSED = "BoardBattle"


def answer(text: str, proposed: str = PROPOSED) -> NameAnswer:
    return read_name_reply(text, proposed).answer


class TestAcceptance:
    @pytest.mark.parametrize(
        "text",
        [
            # The examples from the bug report.
            "yes", "y", "ok", "okay", "looks good", "keep it", "use this name",
            "ok for the name",
            # Case and punctuation.
            "YES", "Ok.", "Ok for the name.", "OK FOR THE NAME!", "  ok for the name  ",
            # The same idea in other words.
            "okay for the name", "ok with the name", "fine with this name", "yes to the name",
            "the name is fine", "the name is good", "good name", "great name", "that works",
            "works for me", "fine by me", "it's fine", "that's good", "sounds great",
            "perfect", "i like it", "i like the name", "i love it", "use that name", "use it",
            "keep the name", "keep this name", "i prefer this name", "go with it",
            "happy with the name", "ok thanks", "yes please",
            # Naming the suggestion is agreeing with it.
            "BoardBattle", "boardbattle", "use BoardBattle", "BoardBattle is fine",
            "keep BoardBattle", "go with BoardBattle", "call it BoardBattle",
        ],
    )
    def test_affirmative_replies_keep_the_suggestion(self, text):
        assert answer(text) == NameAnswer.ACCEPT

    def test_spacing_differences_in_the_suggestion_do_not_matter(self):
        assert answer("board battle", "BoardBattle") == NameAnswer.ACCEPT
        assert answer("boardbattle", "Board Battle") == NameAnswer.ACCEPT

    def test_a_name_starting_with_no_is_not_declined_when_repeated(self):
        assert answer("No Limits", "No Limits") == NameAnswer.ACCEPT
        assert answer("no", "No Limits") == NameAnswer.DECLINE


class TestDeclining:
    @pytest.mark.parametrize(
        "text",
        [
            "no", "n", "nope", "not ok", "no thanks", "not good", "don't use this name",
            "i don't like it", "not really", "a different name", "something else",
            "another one", "i want a different name", "change it", "rename it", "new name",
            "i prefer a different one",
        ],
    )
    def test_a_refusal_is_never_an_acceptance_or_a_name(self, text):
        assert answer(text) == NameAnswer.DECLINE


class TestExplicitNames:
    @pytest.mark.parametrize(
        "text",
        [
            "call it Milk Market", "Call it Milk Market.", "name it Milk Market",
            "rename it to Milk Market", "rename to Milk Market", "change it to Milk Market",
            "change the name to Milk Market", "the name should be Milk Market",
            "use the name Milk Market", "use Milk Market", "I prefer Milk Market",
            "I'd prefer Milk Market", "how about Milk Market?", "let's go with Milk Market",
            "no, call it Milk Market", "yes but call it Milk Market",
            "call it Milk Market please", "call it Milk Market instead",
            '"Milk Market"', "'Milk Market'", "“Milk Market”", "I want 'Milk Market' instead",
            "/change Milk Market",
        ],
    )
    def test_a_stated_name_is_used(self, text):
        reply = read_name_reply(text, PROPOSED)
        assert reply.answer == NameAnswer.NEW_NAME
        assert reply.name == "Milk Market"

    def test_an_apostrophe_is_not_a_quoted_name(self):
        assert answer("that's fine, don't worry") != NameAnswer.NEW_NAME


class TestAmbiguity:
    @pytest.mark.parametrize(
        "text",
        [
            "maybe", "hmm", "not sure", "i don't know", "idk", "whatever", "i guess",
            "this one", "the name", "???", "why that name?", "is it ok?", "ok?",
            "yes but change it", "ok but maybe shorter", "ok for now", "cool beans",
            "how about something shorter", "I love dairy", "I think we should talk",
            "i would rather have something that sounds much friendlier please",
        ],
    )
    def test_unclear_replies_are_not_a_name_or_a_decision(self, text):
        assert answer(text) == NameAnswer.UNCLEAR

    @pytest.mark.parametrize("text", ["Milk Market", "Green Apple", "Quest", "The Milk Market"])
    def test_bare_text_is_only_ever_a_possible_name(self, text):
        reply = read_name_reply(text, PROPOSED)
        assert reply.answer == NameAnswer.POSSIBLE_NAME
        assert reply.name == text

    def test_a_long_title_is_treated_as_a_sentence(self):
        assert answer("one two three four five six seven") == NameAnswer.UNCLEAR


class TestNameGateConversation:
    @pytest.fixture
    async def gate(self, tmp_path):
        harness = Harness(tmp_path)
        harness.team.product_name = PROPOSED
        workflow = harness.new_workflow()
        await to_name_gate(workflow)
        return harness, workflow

    async def _reply(self, gate, *answers: str) -> ScriptedTerminal:
        """Answer the real gate through the real loop, then leave."""
        harness, workflow = gate
        terminal = ScriptedTerminal(*answers, "/exit")
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        with pytest.raises((_ExitSession, EOFError)):
            await session._product_loop(workflow, None)
        return terminal

    async def test_the_reported_bug(self, gate):
        """The exact failure: this phrase created a workspace called ok-for-the-name."""
        harness, workflow = gate
        await self._reply(gate, "ok for the name", "/exit")

        assert workflow.meta.display_name == PROPOSED
        assert workflow.store.slug == "boardbattle"
        assert workflow.meta.name_confirmed is True
        assert workflow.state.phase == ProductPhase.AWAITING_ROADMAP_APPROVAL
        assert not (harness.root / "ok-for-the-name").exists()
        assert [path.name for path in harness.root.iterdir()] == ["boardbattle"]

    @pytest.mark.parametrize(
        "phrase",
        ["yes", "y", "ok", "okay", "looks good", "keep it", "use this name", "ok for the name"],
    )
    async def test_each_reported_phrase_keeps_the_folder(self, gate, phrase):
        harness, workflow = gate
        await self._reply(gate, phrase, "/exit")
        assert workflow.store.slug == "boardbattle"
        assert workflow.meta.name_confirmed is True

    async def test_ambiguity_asks_again_and_changes_nothing(self, gate):
        harness, workflow = gate
        revision = workflow.state.revision
        terminal = await self._reply(gate, "maybe")

        assert workflow.state.phase == ProductPhase.AWAITING_NAME
        assert workflow.state.revision == revision
        assert workflow.meta.display_name == PROPOSED
        assert workflow.meta.name_confirmed is False
        assert workflow.store.slug == "boardbattle"
        assert any("To keep BoardBattle, say yes" in text for text in terminal.said)
        assert any("call it" in text for text in terminal.said)

    async def test_a_clarification_is_not_followed_by_the_original_question(self, gate):
        terminal = await self._reply(gate, "maybe")
        asked = [prompt for prompt in terminal.prompts if "Would you prefer" in prompt]
        assert len(asked) == 1

    async def test_a_possible_name_is_confirmed_before_anything_changes(self, gate):
        harness, workflow = gate
        terminal = await self._reply(gate, "Milk Market")

        assert any("Should I rename the product to Milk Market?" in t for t in terminal.said)
        assert workflow.meta.display_name == PROPOSED
        assert not (harness.root / "milk-market").exists()

    async def test_confirming_a_possible_name_renames(self, gate):
        harness, workflow = gate
        await self._reply(gate, "Milk Market", "yes", "/exit")
        assert workflow.meta.display_name == "Milk Market"
        assert workflow.store.slug == "milk-market"

    async def test_refusing_a_possible_name_keeps_the_suggestion(self, gate):
        harness, workflow = gate
        terminal = await self._reply(gate, "Milk Market", "no")
        assert workflow.meta.display_name == PROPOSED
        assert workflow.state.phase == ProductPhase.AWAITING_NAME
        assert any("I will not rename it" in text for text in terminal.said)

    async def test_an_offer_can_be_replaced_by_a_stated_name(self, gate):
        harness, workflow = gate
        await self._reply(gate, "Milk Market", "call it Fresh Dairy", "/exit")
        assert workflow.meta.display_name == "Fresh Dairy"

    async def test_a_status_check_cancels_a_pending_offer(self, gate):
        """A yes to the re-asked question must keep the name, not apply the old offer."""
        harness, workflow = gate
        await self._reply(gate, "Milk Market", "/status", "yes", "/exit")
        assert workflow.meta.display_name == PROPOSED
        assert workflow.store.slug == "boardbattle"
        assert workflow.meta.name_confirmed is True

    async def test_a_name_given_after_being_asked_for_one_needs_no_confirmation(self, gate):
        harness, workflow = gate
        terminal = await self._reply(gate, "no", "Milk Market", "/exit")
        assert any("What name would you prefer?" in text for text in terminal.said)
        assert workflow.meta.display_name == "Milk Market"

    async def test_changing_your_mind_after_being_asked_for_a_name(self, gate):
        harness, workflow = gate
        await self._reply(gate, "no", "ok for the name", "/exit")
        assert workflow.meta.display_name == PROPOSED

    async def test_being_unclear_after_being_asked_keeps_asking_for_a_name(self, gate):
        harness, workflow = gate
        await self._reply(gate, "no", "hmm", "Milk Market", "/exit")
        assert workflow.meta.display_name == "Milk Market"

    async def test_a_stated_name_renames_at_once(self, gate):
        harness, workflow = gate
        terminal = await self._reply(gate, "call it Milk Market", "/exit")
        assert workflow.store.slug == "milk-market"
        assert any("products_log/milk-market" in text for text in terminal.said)

    async def test_a_stated_name_that_is_taken_gets_a_free_folder(self, gate):
        harness, workflow = gate
        existing = harness.manager.create("Milk Market")
        terminal = await self._reply(gate, "call it Milk Market", "/exit")
        assert workflow.store.slug == "milk-market-2"
        assert existing.root.is_dir()
        assert any("That folder name was taken" in text for text in terminal.said)

    async def test_an_offer_from_one_product_never_reaches_another(self, tmp_path):
        """Leaving mid-offer, then saying yes in another product, must not rename it."""
        harness = Harness(tmp_path)
        harness.team.product_name = "First Idea"
        first = harness.new_workflow()
        await to_name_gate(first)
        harness.team.product_name = "Second Idea"
        second = harness.new_workflow()
        await to_name_gate(second)
        terminal = ScriptedTerminal("Milk Market", "/pause", "yes", "/exit")
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )

        await session.open_product("first-idea")
        with pytest.raises(_ExitSession):
            await session.open_product("second-idea")

        reopened = harness.workflow(harness.manager.open("second-idea"))
        assert reopened.meta.display_name == "Second Idea"
        assert reopened.meta.name_confirmed is True
        assert not (harness.root / "milk-market").exists()
        asked = [prompt for prompt in terminal.prompts if "Would you prefer" in prompt]
        assert len(asked) == 2, "the second product asks its own question"

    async def test_a_new_session_does_not_inherit_an_old_offer(self, gate):
        harness, workflow = gate
        await self._reply(gate, "Milk Market")

        fresh = harness.restart(workflow)
        terminal = ScriptedTerminal("yes", "/exit")
        session = ProductSession(
            harness.manager, harness.router, harness.policy, harness.playbooks, terminal
        )
        with pytest.raises(_ExitSession):
            await session._product_loop(fresh, None)
        assert fresh.meta.display_name == PROPOSED
