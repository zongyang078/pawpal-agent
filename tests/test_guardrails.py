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

    def test_no_tools_scores_low(self):
        assert compute_confidence([], "hello") == 0.3

    def test_successful_tools_score_above_baseline(self):
        score = compute_confidence(
            ["Added Mochi the dog.", "Today's schedule (2026-04-13):"],
            "add a pet and show schedule",
        )
        assert score > 0.5

    def test_error_results_score_below_baseline(self):
        assert compute_confidence(["Error: something went wrong"], "add a pet") < 0.5

    def test_score_is_clamped_to_unit_interval(self):
        many_good = ["Added a task. " * 20] * 10
        many_bad = ["Error: not found"] * 10
        assert 0.0 <= compute_confidence(many_good, "q") <= 1.0
        assert 0.0 <= compute_confidence(many_bad, "q") <= 1.0


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
