"""Tests for the agent: intent detection and end-to-end rule-based flows.

The LLM-backed paths are not covered here — they need a fake client, which
arrives with the provider abstraction.
"""

import pytest

from agent import PawPalAgent
from pawpal_system import Owner, Pet


@pytest.fixture
def agent(owner: Owner) -> PawPalAgent:
    """An agent in rule-based mode over an owner with no pets."""
    return PawPalAgent(owner=owner, use_llm=False)


class TestIntentDetection:
    """Keyword-based intent classification."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("I want to add a new pet", "add_pet"),
            ("Add Mochi, a dog", "add_pet"),
            ("Schedule a walk for Mochi at 8am", "add_task"),
            ("I just finished the morning walk", "complete_task"),
            ("What's on today's schedule?", "get_schedule"),
            ("How often should I feed my cat?", "care_question"),
            ("Are there any conflicts or overlaps?", "detect_conflicts"),
            ("Hello there", "general_chat"),
        ],
    )
    def test_detects_intent(self, agent: PawPalAgent, message: str, expected: str):
        assert agent._detect_intent(message) == expected


class TestRuleBasedFlows:
    """End-to-end behaviour with no API key configured."""

    def test_add_pet_flow(self, agent: PawPalAgent):
        response = agent.process("Add Mochi, a dog")
        assert "mochi" in response.message.lower() or "dog" in response.message.lower()
        assert agent.owner.find_pet("Mochi") is not None

    def test_empty_schedule_flow(self, agent: PawPalAgent):
        response = agent.process("What's on today's schedule?")
        assert "no tasks" in response.message.lower() or "schedule" in response.message.lower()

    def test_care_question_calls_knowledge_base(self, agent: PawPalAgent):
        response = agent.process("How much should I feed my dog?")
        assert len(response.message) > 50
        assert [tc["name"] for tc in response.tool_calls_made] == ["search_care_info"]

    def test_emergency_short_circuits_before_any_tool_runs(self, agent: PawPalAgent):
        response = agent.process("My dog ate chocolate and is having seizures!")
        assert "emergency" in response.message.lower() or "vet" in response.message.lower()
        assert response.tool_calls_made == []

    def test_general_greeting(self, agent: PawPalAgent):
        response = agent.process("Hi there!")
        assert "pawpal" in response.message.lower() or "welcome" in response.message.lower()

    def test_interactions_are_logged(self, agent: PawPalAgent):
        agent.process("Hello")
        agent.process("What's on today?")
        assert agent.logger.get_summary()["total_interactions"] == 2

    def test_add_task_then_schedule(self, owner: Owner):
        """A task added through the agent shows up on the generated schedule."""
        owner.add_pet(Pet(name="Mochi", species="dog"))
        agent = PawPalAgent(owner=owner, use_llm=False)

        agent.process("Schedule a walk for Mochi at 07:30")
        assert len(owner.find_pet("Mochi").tasks) == 1

        response = agent.process("What's on today's schedule?")
        assert "walk" in response.message.lower()


class TestKnownDefects:
    """Regression tests for defects that are documented but not yet fixed."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Pre-flight run_all_checks() is called with an empty agent_response, so a "
            "vet-referral keyword makes modified_response truthy and process() mistakes "
            "it for an emergency override, returning only the disclaimer."
        ),
    )
    def test_medical_keyword_should_not_suppress_the_answer(self, owner: Owner):
        owner.add_pet(Pet(name="Mochi", species="dog"))
        agent = PawPalAgent(owner=owner, use_llm=False)

        response = agent.process("My dog has a lump. How often should I feed him?")

        assert response.tool_calls_made, "the care question should still reach the knowledge base"
        assert "veterinarian" in response.message.lower(), "disclaimer should still be appended"
        assert len(response.message) > 200, "the actual answer should be present too"

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Keyword tables are matched as bare substrings, so 'plump' contains 'lump' "
            "and 'swallowed his food' contains 'swallowed'."
        ),
    )
    def test_keywords_should_match_on_word_boundaries(self):
        from guardrails import check_emergency, check_vet_referral

        assert not check_vet_referral("Is a plump hamster unhealthy?").warnings
        assert check_emergency("He swallowed his food quickly today").passed
