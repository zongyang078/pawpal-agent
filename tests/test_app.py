"""Tests for the Streamlit UI, driven through Streamlit's own AppTest.

These exercise app.py as a user meets it -- widgets, session state, reruns --
which is where the last round of bugs actually surfaced. The unit suite and
the eval harness were both green at the time.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from llm import FailingClient, FakeClient, LLMResponse

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A fresh app with its state file inside tmp_path.

    chdir matters: the app resolves data.json and logs/ relative to the cwd,
    so without it a test run writes into the repository. The API keys are
    cleared so these assertions describe rule-based mode whether or not the
    developer running them has credentials exported.
    """
    monkeypatch.chdir(tmp_path)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL",
                "ANTHROPIC_MODEL", "PAWPAL_MODEL"):
        monkeypatch.delenv(key, raising=False)

    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def assistant_text(at: AppTest) -> str:
    return at.chat_message[-1].markdown[0].value


class TestFirstLoad:
    def test_renders_without_error(self, app):
        assert app.title[0].value == "🐾 PawPal+ Agent"
        assert [s.value for s in app.sidebar.subheader] == [
            "Status", "Your Pets", "Agent Log"
        ]

    def test_reports_rule_based_mode_without_a_key(self, app):
        assert "Rule-based mode" in app.sidebar.info[0].value

    def test_welcomes_with_no_pets(self, app):
        assert "Welcome to PawPal+" in assistant_text(app)
        assert any("No pets yet" in c.value for c in app.sidebar.caption)


class TestConversation:
    def test_adding_a_pet_updates_the_sidebar_the_same_turn(self, app):
        """Streamlit draws the sidebar before the chat handler runs, so this
        only holds because the handler reruns."""
        app.chat_input[0].set_value("Add Mochi, a dog").run()

        assert "Added Mochi" in assistant_text(app)
        assert any("Mochi" in m.value for m in app.sidebar.markdown)

    def test_care_question_reaches_the_knowledge_base(self, app):
        app.chat_input[0].set_value("How often should I bathe my dog?").run()

        assert "Dog grooming basics" in assistant_text(app)
        assert [e.label for e in app.expander] == ["Agent reasoning"]

    def test_emergency_overrides_and_warns(self, app):
        app.chat_input[0].set_value("My dog had a seizure!").run()

        assert "emergency" in assistant_text(app).lower()
        assert any("seizure" in w.value for w in app.warning)
        assert app.expander == [], "no tool should have run"

    def test_medical_keyword_does_not_suppress_the_answer(self, app):
        """The worst of the guardrail defects, seen from the UI."""
        app.chat_input[0].set_value("My dog has a lump. How often should I feed him?").run()

        message = assistant_text(app)
        assert "veterinarian" in message.lower()
        assert len(message) > 200, "the care answer must survive the disclaimer"

    def test_plump_is_not_read_as_a_lump(self, app):
        app.chat_input[0].set_value("Is a plump hamster unhealthy?").run()

        assert "emergency" not in assistant_text(app).lower()
        assert "Hamster care basics" in assistant_text(app)

    def test_retrieval_does_not_cross_species(self, app):
        app.chat_input[0].set_value("How often should I feed my dog?").run()

        message = assistant_text(app)
        assert "Dog feeding guidelines" in message
        assert "Cat feeding guidelines" not in message


class TestResetSession:
    """The reset button exists because session state pins the agent instance."""

    def test_rebuilds_the_agent_and_clears_history(self, app):
        app.chat_input[0].set_value("Add Mochi, a dog").run()
        app.chat_input[0].set_value("What's on today's schedule?").run()
        assert len(app.chat_message) > 1

        # Hold the old agent, rather than comparing id() across the reset:
        # ids are only unique among live objects, and CPython reuses the
        # address of the freed agent often enough to fail intermittently.
        old_agent = app.session_state.agent

        next(b for b in app.sidebar.button if b.label == "Reset session").click().run()

        assert app.session_state.agent is not old_agent
        assert app.session_state.agent.logger.get_summary()["total_interactions"] == 0
        assert len(app.chat_message) == 1, "only the welcome message should remain"

    def test_keeps_the_pets(self, app):
        app.chat_input[0].set_value("Add Mochi, a dog").run()

        next(b for b in app.sidebar.button if b.label == "Reset session").click().run()

        assert [p.name for p in app.session_state.owner.pets] == ["Mochi"]
        assert any("Mochi" in m.value for m in app.sidebar.markdown)

    def test_chat_still_works_afterwards(self, app):
        next(b for b in app.sidebar.button if b.label == "Reset session").click().run()

        app.chat_input[0].set_value("How often should I bathe my dog?").run()

        assert not app.exception
        assert "Dog grooming basics" in assistant_text(app)


class TestDegradedProvider:
    """A configured provider is not a working one, and the UI must say so.

    This is the demo-day failure: the sidebar showed a green "LLM mode:
    anthropic" badge while every request 401'd and the answers came from
    keyword matching.
    """

    @pytest.fixture
    def broken_app(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")

        at = AppTest.from_file(str(APP_PATH), default_timeout=30)
        at.run()
        # Swap in a client that always fails, standing in for a bad key.
        at.session_state.agent.llm_client = FailingClient("401 invalid api key")
        return at

    def test_badge_is_green_before_anything_is_tried(self, broken_app):
        assert broken_app.sidebar.success, "a configured key looks fine until used"

    def test_badge_turns_red_after_a_failed_call(self, broken_app):
        broken_app.chat_input[0].set_value("How often should I bathe my dog?").run()

        assert not broken_app.sidebar.success, "the green badge must not survive"
        errors = [e.value for e in broken_app.sidebar.error]
        assert any("call failed" in e for e in errors)
        assert any("401" in c.value for c in broken_app.sidebar.caption)

    def test_the_answer_still_arrives(self, broken_app):
        """Degrading beats erroring out -- that part of the old behaviour was right."""
        broken_app.chat_input[0].set_value("How often should I bathe my dog?").run()

        assert "Dog grooming basics" in assistant_text(broken_app)

    def test_recovery_clears_the_warning(self, broken_app):
        broken_app.chat_input[0].set_value("How often should I bathe my dog?").run()
        assert broken_app.sidebar.error

        broken_app.session_state.agent.llm_client = FakeClient([LLMResponse(text="ok")])
        broken_app.chat_input[0].set_value("hi").run()

        assert not broken_app.sidebar.error
        assert broken_app.sidebar.success


class TestProviderSelector:
    """The one thing two providers actually buy a demo: switching live."""

    def _app(self, tmp_path, monkeypatch, **env):
        monkeypatch.chdir(tmp_path)
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL",
                    "ANTHROPIC_MODEL", "PAWPAL_MODEL"):
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)

        at = AppTest.from_file(str(APP_PATH), default_timeout=30)
        at.run()
        assert not at.exception, at.exception
        return at

    def test_hidden_with_no_keys(self, tmp_path, monkeypatch):
        assert self._app(tmp_path, monkeypatch).selectbox == []

    def test_hidden_with_one_key(self, tmp_path, monkeypatch):
        """Nothing to choose between, so do not ask."""
        app = self._app(tmp_path, monkeypatch, ANTHROPIC_API_KEY="sk-ant-x")
        assert app.selectbox == []
        assert "anthropic" in app.sidebar.success[0].value

    def test_offered_with_both_keys(self, tmp_path, monkeypatch):
        app = self._app(tmp_path, monkeypatch,
                        ANTHROPIC_API_KEY="sk-ant-x", OPENAI_API_KEY="sk-x")

        selector = app.sidebar.selectbox[0]
        assert list(selector.options) == ["anthropic", "openai"]
        assert selector.value == "anthropic", "the default provider is preselected"

    def test_switching_rebuilds_the_agent_on_the_chosen_provider(self, tmp_path, monkeypatch):
        app = self._app(tmp_path, monkeypatch,
                        ANTHROPIC_API_KEY="sk-ant-x", OPENAI_API_KEY="sk-x")

        app.sidebar.selectbox[0].select("openai").run()

        assert app.session_state.agent.api_provider == "openai"
        assert app.session_state.agent.llm_client.api_key == "sk-x"
        assert "openai" in app.sidebar.success[0].value

    def test_switching_keeps_the_interaction_log(self, tmp_path, monkeypatch):
        """Rebuilding the agent must not silently reset the log counter."""
        app = self._app(tmp_path, monkeypatch,
                        ANTHROPIC_API_KEY="sk-ant-x", OPENAI_API_KEY="sk-x")
        logger = app.session_state.agent.logger

        app.sidebar.selectbox[0].select("openai").run()

        assert app.session_state.agent.logger is logger
