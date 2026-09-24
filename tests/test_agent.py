"""Tests for the agent: intent detection and end-to-end rule-based flows.

The LLM-backed paths are not covered here — they need a fake client, which
arrives with the provider abstraction.
"""

import pytest

from agent import MAX_REACT_ITERATIONS, PawPalAgent
from guardrails import check_emergency, check_vet_referral
from llm import (
    AssistantTurn,
    FailingClient,
    FakeClient,
    LLMResponse,
    ToolCall,
    ToolResultsTurn,
    UserTurn,
)
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


class TestGuardrailRegressions:
    """Regressions for guardrail defects found by earlier test restructuring."""

    def test_medical_keyword_does_not_suppress_the_answer(self, owner: Owner):
        owner.add_pet(Pet(name="Mochi", species="dog"))
        agent = PawPalAgent(owner=owner, use_llm=False)

        response = agent.process("My dog has a lump. How often should I feed him?")

        assert response.tool_calls_made, "the care question should still reach the knowledge base"
        assert "veterinarian" in response.message.lower(), "disclaimer should still be appended"
        assert len(response.message) > 200, "the actual answer should be present too"
        assert response.confidence < 1.0, "only an emergency override should claim certainty"

    def test_emergency_still_overrides(self, owner: Owner):
        """Narrowing the pre-flight check must not weaken the emergency path."""
        owner.add_pet(Pet(name="Mochi", species="dog"))
        agent = PawPalAgent(owner=owner, use_llm=False)

        response = agent.process("My dog had a seizure! How often should I feed him?")

        assert response.tool_calls_made == [], "no tool should run on an emergency"
        assert "emergency" in response.message.lower()
        assert response.confidence == 1.0

    def test_keywords_match_on_word_boundaries(self):
        """"plump" must not read as "lump", nor "swallowed his food" as a choking."""
        assert not check_vet_referral("Is a plump hamster unhealthy?").warnings
        assert check_emergency("He swallowed his food quickly today").passed

    def test_word_boundaries_do_not_break_real_matches(self):
        """The narrowing must not cost us true positives."""
        assert check_vet_referral("My cat has a lump on her side").warnings
        assert not check_emergency("My dog swallowed a sock!").passed


class TestReActLoop:
    """The LLM reasoning loop, driven by a scripted client."""

    def _agent(self, owner: Owner, responses: list[LLMResponse]) -> PawPalAgent:
        return PawPalAgent(owner=owner, llm_client=FakeClient(responses))

    def test_text_only_reply_ends_the_turn(self, owner: Owner):
        agent = self._agent(owner, [LLMResponse(text="Mochi is all set.")])

        response = agent.process("how's Mochi doing?")

        assert response.message == "Mochi is all set."
        assert response.tool_calls_made == []
        assert len(agent.llm_client.calls) == 1

    def test_tool_call_is_executed_and_fed_back(self, owner: Owner):
        agent = self._agent(owner, [
            LLMResponse(tool_calls=[
                ToolCall(id="c1", name="add_pet",
                         arguments={"name": "Mochi", "species": "dog"})
            ]),
            LLMResponse(text="Added Mochi."),
        ])

        response = agent.process("add my dog Mochi")

        assert owner.find_pet("Mochi") is not None
        assert response.message == "Added Mochi."
        assert [tc["name"] for tc in response.tool_calls_made] == ["add_pet"]

        # The second request must carry the assistant turn and the tool result.
        second = agent.llm_client.calls[1]["transcript"]
        assert isinstance(second[0], UserTurn)
        assert isinstance(second[1], AssistantTurn)
        assert isinstance(second[2], ToolResultsTurn)
        assert second[2].results[0].id == "c1"
        assert "Added Mochi the dog" in second[2].results[0].content

    def test_parallel_tool_calls_in_one_reply(self, owner: Owner):
        agent = self._agent(owner, [
            LLMResponse(tool_calls=[
                ToolCall(id="c1", name="add_pet",
                         arguments={"name": "Mochi", "species": "dog"}),
                ToolCall(id="c2", name="add_pet",
                         arguments={"name": "Luna", "species": "cat"}),
            ]),
            LLMResponse(text="Both added."),
        ])

        response = agent.process("add Mochi the dog and Luna the cat")

        assert len(response.tool_calls_made) == 2
        assert {p.name for p in owner.pets} == {"Mochi", "Luna"}
        assert len(agent.llm_client.calls[1]["transcript"][2].results) == 2

    def test_multi_step_chain(self, owner: Owner):
        """Two rounds of tools before the model answers."""
        agent = self._agent(owner, [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="add_pet",
                                             arguments={"name": "Mochi", "species": "dog"})]),
            LLMResponse(tool_calls=[ToolCall(id="c2", name="get_schedule", arguments={})]),
            LLMResponse(text="Mochi is added and has nothing scheduled."),
        ])

        response = agent.process("add Mochi then show me today")

        assert [tc["name"] for tc in response.tool_calls_made] == ["add_pet", "get_schedule"]
        assert response.message.startswith("Mochi is added")

    def test_loop_stops_at_the_iteration_cap(self, owner: Owner):
        """A model that only ever calls tools must not loop forever."""
        agent = self._agent(owner, [
            LLMResponse(tool_calls=[ToolCall(id=f"c{i}", name="get_schedule", arguments={})])
            for i in range(MAX_REACT_ITERATIONS + 2)
        ])

        response = agent.process("what's on today?")

        assert len(agent.llm_client.calls) == MAX_REACT_ITERATIONS
        assert len(response.tool_calls_made) == MAX_REACT_ITERATIONS
        assert "completed the requested actions" in response.message

    def test_tool_failure_is_reported_back_not_raised(self, owner: Owner):
        """A bad tool call becomes an observation the model can react to."""
        agent = self._agent(owner, [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="add_task",
                                             arguments={"pet_name": "Ghost",
                                                        "description": "Walk",
                                                        "time": "09:00",
                                                        "duration_minutes": 30})]),
            LLMResponse(text="I could not find a pet called Ghost."),
        ])

        response = agent.process("schedule a walk for Ghost")

        assert "not found" in response.tool_calls_made[0]["result"]
        assert response.message.startswith("I could not find")

    def test_unknown_tool_name_does_not_crash_the_loop(self, owner: Owner):
        agent = self._agent(owner, [
            LLMResponse(tool_calls=[ToolCall(id="c1", name="teleport_pet", arguments={})]),
            LLMResponse(text="I cannot do that."),
        ])

        response = agent.process("teleport Mochi")

        assert "Unknown tool" in response.tool_calls_made[0]["result"]

    def test_empty_reply_gets_a_fallback_message(self, owner: Owner):
        agent = self._agent(owner, [LLMResponse(text=None)])
        assert "not sure how to help" in agent.process("???").message

    def test_system_prompt_carries_current_state(self, owner: Owner):
        owner.add_pet(Pet(name="Mochi", species="dog"))
        agent = self._agent(owner, [LLMResponse(text="ok")])

        agent.process("hi")

        system = agent.llm_client.calls[0]["system"]
        assert "Mochi (dog)" in system
        assert "Jordan" in system

    def test_every_tool_is_advertised(self, owner: Owner):
        agent = self._agent(owner, [LLMResponse(text="ok")])
        agent.process("hi")

        names = {t["name"] for t in agent.llm_client.calls[0]["tools"]}
        assert "search_care_info" in names
        assert len(names) == 8

    def test_guardrails_still_wrap_the_llm_path(self, owner: Owner):
        """An emergency must short-circuit before the provider is called."""
        agent = self._agent(owner, [LLMResponse(text="Let me check the schedule.")])

        response = agent.process("My dog had a seizure!")

        assert agent.llm_client.calls == [], "no provider call on an emergency"
        assert "emergency" in response.message.lower()

    def test_toxic_suggestion_from_the_model_is_flagged(self, owner: Owner):
        owner.add_pet(Pet(name="Mochi", species="dog"))
        agent = self._agent(owner, [LLMResponse(text="Chocolate makes a great treat!")])

        response = agent.process("what treat should I give him?")

        assert "Safety warning" in response.message
        assert any("toxic" in w for w in response.guardrail_warnings)


class TestProviderDegradation:
    """What happens when the provider is unavailable."""

    def test_failure_falls_back_to_rule_based(self, owner: Owner, capsys):
        agent = PawPalAgent(owner=owner, llm_client=FailingClient())

        response = agent.process("How much should I feed my dog?")

        assert [tc["name"] for tc in response.tool_calls_made] == ["search_care_info"]
        assert len(response.message) > 50
        assert "falling back to rule-based" in capsys.readouterr().out

    def test_no_client_means_rule_based_mode(self, owner: Owner):
        agent = PawPalAgent(owner=owner, api_key=None, api_provider="openai")
        assert agent.use_llm is False
        assert agent.api_provider == "rule-based"
        assert agent.model is None

    def test_injected_client_reports_its_provider(self, owner: Owner):
        agent = PawPalAgent(owner=owner, llm_client=FakeClient([]))
        assert agent.use_llm is True
        assert agent.api_provider == "fake"
        assert agent.model == "fake-model"
