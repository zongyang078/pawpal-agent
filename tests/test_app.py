"""Tests for the Streamlit UI, driven through Streamlit's own AppTest.

These exercise app.py as a user meets it -- widgets, session state, reruns --
which is where the last round of bugs actually surfaced. The unit suite and
the eval harness were both green at the time.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
        before = id(app.session_state.agent)
        assert len(app.chat_message) > 1

        next(b for b in app.sidebar.button if b.label == "Reset session").click().run()

        assert id(app.session_state.agent) != before
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
