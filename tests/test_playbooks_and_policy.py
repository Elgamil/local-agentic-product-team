"""Category playbooks and the configurable model policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lab.config import ModelPolicy, load_model_policy, load_verification_profiles
from lab.playbooks import REQUIRED_SECTIONS, PlaybookLibrary, parse_playbook
from tests.fakes import GEMMA, QWEN, Harness, to_name_gate

_REQUIRED_IDS = {"game", "website", "ecommerce", "api_service", "cli_tool", "data_automation"}


class TestPlaybooks:
    def test_all_initial_playbooks_load(self):
        assert _REQUIRED_IDS <= set(PlaybookLibrary().ids())

    @pytest.mark.parametrize("category", sorted(_REQUIRED_IDS))
    def test_each_playbook_has_every_required_section(self, category):
        playbook = PlaybookLibrary().get(category)
        for section in REQUIRED_SECTIONS:
            assert playbook.section(section).strip(), section

    def test_the_game_playbook_covers_the_specified_topics(self):
        text = PlaybookLibrary().get("game").body.lower()
        for topic in ("invalid moves", "draw", "restart", "state transitions", "user input"):
            assert topic in text

    def test_the_ecommerce_playbook_treats_payment_as_optional(self):
        text = PlaybookLibrary().get("ecommerce").body.lower()
        assert "payment integration, only when the user asks" in text

    @pytest.mark.parametrize(
        ("idea", "expected"),
        [
            ("a mini shop for dairy products", "ecommerce"),
            ("tic-tac-toe game for two players", "game"),
            ("portfolio website", "website"),
            ("rest api for bookings", "api_service"),
            ("command line tool for renaming", "cli_tool"),
            ("etl pipeline for csv files", "data_automation"),
        ],
    )
    def test_keyword_suggestion(self, idea, expected):
        assert PlaybookLibrary().suggest(idea) == expected

    def test_unrelated_text_suggests_nothing(self):
        assert PlaybookLibrary().suggest("hello there") is None

    def test_a_new_category_needs_only_a_new_file(self, tmp_path):
        body = "\n\n".join(f"## {title}\n\nText." for title in (
            "Considerations",
            "Typical Development Phases",
            "Quality Requirements",
            "Common Edge Cases",
            "Suggested Acceptance Checks",
        ))
        (tmp_path / "mobile.md").write_text(
            f"---\nid: mobile\ntitle: Mobile App\nkeywords: [ios, android]\n---\n\n{body}\n"
        )
        library = PlaybookLibrary(tmp_path)
        assert library.ids() == ["mobile"]
        assert library.suggest("an ios app") == "mobile"

    def test_an_incomplete_playbook_is_rejected(self, tmp_path):
        path = tmp_path / "bad.md"
        path.write_text("---\ntitle: Bad\n---\n\n## Considerations\n\nOnly this.\n")
        with pytest.raises(ValueError, match="missing sections"):
            parse_playbook(path)

    async def test_the_matching_playbook_reaches_the_business_analyst(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await workflow.discovery_message("I want to build a tic-tac-toe game.")

        prompt = harness.team.calls_of("discovery")[0].prompt

        assert "Internal Game guidance" in prompt
        assert "Invalid moves" in prompt

    async def test_playbook_text_is_not_shown_to_the_user(self, tmp_path):
        from tests.fakes import ScriptedTerminal

        harness = Harness(tmp_path)
        terminal = ScriptedTerminal()
        workflow = harness.workflow(harness.manager.create(), reporter=terminal)
        await to_name_gate(workflow)

        assert "Internal" not in terminal.transcript
        assert "guidance" not in terminal.transcript

    async def test_the_final_category_comes_from_the_agent_within_known_ids(self, tmp_path):
        harness = Harness(tmp_path)
        workflow = harness.new_workflow()
        await to_name_gate(workflow)
        assert workflow.meta.category == "cli_tool"
        assert workflow.meta.category_label == "Command-Line Tool"

    async def test_an_invented_category_is_rejected_and_retried(self, tmp_path):
        harness = Harness(tmp_path)
        bad = (
            '{"reply": "Hi", "understanding": "x", "category": "blockchain_metaverse", '
            '"ready_for_brief": false}'
        )
        from tests.fakes import Text

        harness.team.script("discovery", Text(bad))
        workflow = harness.new_workflow()
        await workflow.discovery_message("Something unusual.")

        calls = harness.team.calls_of("discovery")
        assert len(calls) == 2
        assert "Unknown category" in calls[1].prompt


class TestModelPolicy:
    def test_the_specified_preferences(self):
        policy = load_model_policy()
        assert policy.chain_for("business_analyst") == [GEMMA, QWEN]
        assert policy.chain_for("software_architect") == [QWEN, GEMMA]
        assert policy.chain_for("developer") == [QWEN, GEMMA]

    @pytest.mark.parametrize(("developer", "expected"), [(QWEN, GEMMA), (GEMMA, QWEN)])
    def test_test_engineer_prefers_the_opposite_of_the_developer(self, developer, expected):
        chain = load_model_policy().chain_for("test_engineer", developer)
        assert chain == [expected, developer]

    def test_routing_is_configurable(self, tmp_path):
        raw = load_model_policy().model_dump()
        raw["roles"]["business_analyst"] = {"primary": "qwen", "fallback": "gemma"}
        assert ModelPolicy.model_validate(raw).chain_for("business_analyst") == [QWEN, GEMMA]

    def test_more_than_two_attempts_per_model_is_refused(self):
        raw = load_model_policy().model_dump()
        raw["retry"]["attempts_per_model"] = 3
        with pytest.raises(ValidationError):
            ModelPolicy.model_validate(raw)

    def test_a_role_naming_an_unknown_model_is_refused(self):
        raw = load_model_policy().model_dump()
        raw["roles"]["developer"]["primary"] = "gpt"
        with pytest.raises(ValidationError):
            ModelPolicy.model_validate(raw)

    def test_a_typo_in_the_policy_is_refused(self):
        raw = load_model_policy().model_dump()
        raw["memory"]["unload_between_modelz"] = True
        with pytest.raises(ValidationError):
            ModelPolicy.model_validate(raw)

    def test_only_local_ollama_models_are_configured(self):
        ids = [spec.id for spec in load_model_policy().models.values()]
        assert sorted(ids) == sorted([GEMMA, QWEN])

    def test_output_ceiling_must_fit_inside_the_context_window(self):
        raw = load_model_policy().model_dump()
        raw["models"]["gemma"]["max_output_tokens"] = raw["models"]["gemma"]["num_ctx"]
        with pytest.raises(ValidationError):
            ModelPolicy.model_validate(raw)


class TestVerificationProfiles:
    def test_profiles_are_not_tied_to_pytest(self):
        profiles = load_verification_profiles()
        assert {"python_pytest", "python_unittest", "node_test"} <= set(profiles)

    def test_no_profile_runs_developer_defined_scripts(self):
        for profile in load_verification_profiles().values():
            argv = " ".join([profile.executable, *profile.args])
            assert "npm" not in argv
            assert "make" not in argv

    def test_a_product_can_add_its_own_profile(self, tmp_path):
        extra = tmp_path / "profiles.yaml"
        extra.write_text(
            "profiles:\n"
            "  custom:\n"
            "    description: A custom suite.\n"
            "    executable: python\n"
            "    args: ['-m', 'pytest', '{tests_dir}']\n"
            "    result_parser: pytest\n"
            "    test_file_patterns: ['check_*.py']\n"
        )
        assert "custom" in load_verification_profiles(extra)

    def test_a_shell_string_is_not_a_valid_executable(self, tmp_path):
        extra = tmp_path / "profiles.yaml"
        extra.write_text(
            "profiles:\n"
            "  evil:\n"
            "    description: Nope.\n"
            "    executable: 'sh -c rm'\n"
            "    args: []\n"
            "    result_parser: pytest\n"
            "    test_file_patterns: ['*']\n"
        )
        with pytest.raises(ValidationError):
            load_verification_profiles(extra)
