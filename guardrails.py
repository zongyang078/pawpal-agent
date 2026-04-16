"""
Guardrails for PawPal+ Agent.

Validates Agent responses and tool calls to ensure safety and reliability.
Includes toxic food detection, confidence scoring, and medical disclaimer injection.
"""

from dataclasses import dataclass


# --- Toxic substances by species ---

TOXIC_FOODS: dict[str, list[str]] = {
    "dog": [
        "chocolate", "grapes", "raisins", "onions", "garlic", "xylitol",
        "avocado", "macadamia", "alcohol", "caffeine", "cooked bones",
    ],
    "cat": [
        "onions", "garlic", "chocolate", "caffeine", "alcohol", "raw eggs",
        "grapes", "raisins", "dog food", "lilies", "xylitol",
    ],
    "bird": [
        "avocado", "chocolate", "caffeine", "fruit pits", "alcohol", "salt",
    ],
    "hamster": [
        "chocolate", "caffeine", "citrus", "almonds", "garlic", "onions",
    ],
}

# Keywords that suggest a medical emergency
EMERGENCY_KEYWORDS = [
    "seizure", "seizures", "not breathing", "unconscious", "bleeding heavily",
    "poisoned", "ate poison", "ate chocolate", "swallowed", "choking",
    "collapsed", "paralyzed", "not moving", "hit by car", "broken bone",
]

# Keywords that suggest the user needs vet advice, not AI advice
VET_REFERRAL_KEYWORDS = [
    "blood in stool", "blood in urine", "lump", "tumor", "cancer",
    "limping for days", "not eating for days", "vomiting blood",
    "diarrhea for days", "eye infection", "ear infection", "skin rash",
    "breathing problems", "heart", "diabetes", "kidney",
]


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    warnings: list[str]
    modified_response: str | None = None  # If set, replaces the original response


def check_toxic_food_mention(response: str, species: str | None = None) -> GuardrailResult:
    """Check if the response recommends something toxic to the pet's species.

    Args:
        response: The Agent's planned response text.
        species: The pet's species for species-specific checks. If None, checks all.

    Returns:
        GuardrailResult with warnings if toxic items are mentioned approvingly.
    """
    response_lower = response.lower()
    warnings = []

    species_to_check = [species] if species else list(TOXIC_FOODS.keys())

    for sp in species_to_check:
        for item in TOXIC_FOODS.get(sp, []):
            if item in response_lower:
                # Check ALL occurrences of the toxic item in the response.
                # If ANY occurrence is near a warning word, consider it warned.
                danger_words = ["toxic", "avoid", "never", "dangerous", "harmful",
                                "do not", "don't", "not", "safe"]
                already_warned = False
                start = 0
                while True:
                    idx = response_lower.find(item, start)
                    if idx == -1:
                        break
                    # Check a 100-char window around this occurrence
                    window = response_lower[max(0, idx - 80):idx + len(item) + 80]
                    if any(dw in window for dw in danger_words):
                        already_warned = True
                        break
                    start = idx + 1

                if not already_warned:
                    warnings.append(
                        f"WARNING: '{item}' is toxic to {sp}s. "
                        f"The response should explicitly warn against it."
                    )

    return GuardrailResult(passed=len(warnings) == 0, warnings=warnings)


def check_emergency(user_message: str) -> GuardrailResult:
    """Check if the user's message indicates a pet emergency.

    Returns a guardrail result that may override the normal response
    with emergency instructions.
    """
    message_lower = user_message.lower()
    warnings = []

    for keyword in EMERGENCY_KEYWORDS:
        if keyword in message_lower:
            warnings.append(f"Emergency keyword detected: '{keyword}'")

    if warnings:
        emergency_response = (
            "This sounds like it could be a pet emergency. "
            "Please contact your veterinarian or an emergency animal hospital immediately. "
            "If you need to find an emergency vet, search for '24-hour emergency vet near me'. "
            "Time is critical in pet emergencies — please seek professional help right away."
        )
        return GuardrailResult(
            passed=False,
            warnings=warnings,
            modified_response=emergency_response,
        )

    return GuardrailResult(passed=True, warnings=[])


def check_vet_referral(user_message: str) -> GuardrailResult:
    """Check if the user's question requires professional veterinary advice.

    Unlike emergencies, these don't override the response but add a disclaimer.
    """
    message_lower = user_message.lower()
    warnings = []

    for keyword in VET_REFERRAL_KEYWORDS:
        if keyword in message_lower:
            warnings.append(f"Medical topic detected: '{keyword}'")

    if warnings:
        return GuardrailResult(
            passed=True,  # Allow response but flag it
            warnings=warnings,
            modified_response=None,  # Disclaimer will be appended by the agent
        )

    return GuardrailResult(passed=True, warnings=[])


def compute_confidence(tool_results: list[str], user_query: str) -> float:
    """Estimate confidence in the Agent's response based on tool results.

    Heuristic scoring:
    - Higher if tool results contain substantive content
    - Lower if tools returned errors or 'not found' messages
    - Lower if the query seems complex but few tools were called

    Returns a float between 0.0 and 1.0.
    """
    if not tool_results:
        return 0.3  # No tools called = low confidence

    score = 0.5  # Base score

    for result in tool_results:
        result_lower = result.lower()

        # Positive signals
        if len(result) > 100:
            score += 0.1
        if "added" in result_lower or "completed" in result_lower:
            score += 0.15
        if "schedule" in result_lower or "tasks for" in result_lower:
            score += 0.1

        # Negative signals
        if "error" in result_lower:
            score -= 0.2
        if "not found" in result_lower:
            score -= 0.15
        if "no relevant information" in result_lower:
            score -= 0.1

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))


def run_all_checks(
    user_message: str,
    agent_response: str,
    tool_results: list[str],
    pet_species: str | None = None,
) -> GuardrailResult:
    """Run all guardrail checks and return a combined result.

    Check order (short-circuits on emergency):
    1. Emergency check (may override response entirely)
    2. Vet referral check (may add disclaimer)
    3. Toxic food check (may add warning)
    4. Confidence scoring

    Returns the combined GuardrailResult with all warnings aggregated.
    """
    all_warnings = []

    # 1. Emergency check — highest priority
    emergency = check_emergency(user_message)
    if not emergency.passed:
        return emergency

    # 2. Vet referral check
    vet_check = check_vet_referral(user_message)
    all_warnings.extend(vet_check.warnings)

    # 3. Toxic food check on the response
    toxic_check = check_toxic_food_mention(agent_response, pet_species)
    all_warnings.extend(toxic_check.warnings)

    # 4. Confidence scoring
    confidence = compute_confidence(tool_results, user_message)

    # Build final result
    modified = None
    if vet_check.warnings:
        disclaimer = (
            "\n\nNote: This question touches on a medical topic. "
            "The information above is for general guidance only. "
            "Please consult your veterinarian for professional advice."
        )
        modified = agent_response + disclaimer

    if not toxic_check.passed:
        toxic_warning = (
            "\n\nSafety warning: Some items mentioned may be harmful to pets. "
            "Please double-check any food or substance recommendations."
        )
        base = modified if modified else agent_response
        modified = base + toxic_warning

    if confidence < 0.4:
        all_warnings.append(f"Low confidence ({confidence:.2f})")

    return GuardrailResult(
        passed=toxic_check.passed and len(all_warnings) == 0,
        warnings=all_warnings,
        modified_response=modified,
    )
