"""Tests for the safety guardrail layer."""

from guardrails import (
    check_emergency,
    check_toxic_food_mention,
    check_vet_referral,
    compute_confidence,
    run_all_checks,
)


class TestToxicFood:
    """Tests for species-aware toxic food detection."""

    def test_unwarned_mention_is_flagged(self):
        result = check_toxic_food_mention("You should give your dog chocolate as a treat!", "dog")
        assert not result.passed
        assert result.warnings

    def test_mention_with_existing_warning_passes(self):
        """The response already warns against the item, so don't double-flag it."""
        result = check_toxic_food_mention("Never give your dog chocolate, it is toxic.", "dog")
        assert result.passed

    def test_species_specific_list(self):
        """Raw eggs are on the cat list but not the dog list."""
        assert not check_toxic_food_mention("Try feeding raw eggs.", "cat").passed
        assert check_toxic_food_mention("Try feeding raw eggs.", "dog").passed

    def test_safe_is_not_treated_as_a_warning(self):
        """"safe" used approvingly must not read as a warning; "not safe" must."""
        assert not check_toxic_food_mention("Chocolate is a safe treat for dogs.", "dog").passed
        assert check_toxic_food_mention("Chocolate is not safe for dogs.", "dog").passed

    def test_every_mention_needs_its_own_warning(self):
        """Warning once up front must not license an unwarned mention later."""
        response = (
            "Chocolate is toxic to dogs and should never be given. "
            + "Dogs need daily walks, fresh water, and plenty of play time. " * 3
            + "For a special reward, a small piece of chocolate makes a nice treat."
        )
        assert not check_toxic_food_mention(response, "dog").passed


class TestEmergency:
    """Tests for emergency keyword detection."""

    def test_emergency_detected(self):
        result = check_emergency("My dog ate chocolate and is having seizures!")
        assert not result.passed
        assert result.modified_response is not None
        assert "emergency" in result.modified_response.lower()

    def test_ordinary_question_is_not_an_emergency(self):
        assert check_emergency("When should I feed my dog?").passed


class TestVetReferral:
    """Tests for the medical-topic disclaimer trigger."""

    def test_medical_topic_detected(self):
        assert check_vet_referral("My cat has blood in stool, what should I do?").warnings

    def test_non_medical_topic_is_clean(self):
        assert not check_vet_referral("What time should I walk my dog?").warnings


class TestConfidence:
    """Tests for the heuristic confidence score."""

    def test_rule_based_fallback_scores_low(self):
        """Nothing matched, so the generic reply really is weak."""
        assert compute_confidence([]) == 0.3

    def test_an_llm_answering_in_text_is_not_low_confidence(self):
        """A model replying to a greeting without tools has done the right
        thing. Scoring that 0.3 put a "Low confidence" warning under good
        answers, which is worse than saying nothing."""
        assert compute_confidence([], llm_answered=True) > 0.5

    def test_successful_tools_score_above_baseline(self):
        score = compute_confidence(
            ["Added Mochi the dog.", "Today's schedule (2026-04-13):"]
        )
        assert score > 0.5

    def test_error_results_score_below_baseline(self):
        assert compute_confidence(["Error: something went wrong"]) < 0.5

    def test_tool_results_outweigh_how_the_answer_was_produced(self):
        """Once tools ran, their outcome decides; llm_answered is irrelevant."""
        errors = ["Error: something went wrong"]
        assert compute_confidence(errors) == compute_confidence(errors, llm_answered=True)

    def test_score_is_clamped_to_unit_interval(self):
        many_good = ["Added a task. " * 20] * 10
        many_bad = ["Error: not found"] * 10
        assert 0.0 <= compute_confidence(many_good) <= 1.0
        assert 0.0 <= compute_confidence(many_bad) <= 1.0


class TestRunAllChecks:
    """Tests for the combined guardrail pipeline."""

    def test_emergency_overrides_everything(self):
        result = run_all_checks(
            user_message="My dog is not breathing!",
            agent_response="Let me check the schedule...",
            tool_results=[],
        )
        assert not result.passed
        assert "emergency" in result.modified_response.lower()

    def test_clean_interaction_passes_untouched(self):
        result = run_all_checks(
            user_message="What's on the schedule today?",
            agent_response="Here are today's tasks.",
            tool_results=["Today's schedule..."],
        )
        assert result.passed
        assert result.modified_response is None

    def test_medical_topic_appends_disclaimer(self):
        result = run_all_checks(
            user_message="My cat has blood in stool",
            agent_response="Here is some general guidance.",
            tool_results=["Some knowledge base content"],
        )
        assert result.modified_response is not None
        assert result.modified_response.startswith("Here is some general guidance.")
        assert "veterinarian" in result.modified_response.lower()

    def test_disclaimer_is_not_duplicated(self):
        """Knowledge base articles close with their own referral already."""
        result = run_all_checks(
            user_message="My cat has blood in stool",
            agent_response=(
                "Watch for changes in litter box habits. Note: this information is "
                "for general guidance only. For specific medical concerns, always "
                "consult a veterinarian."
            ),
            tool_results=["Some knowledge base content"],
        )
        assert result.modified_response is None
