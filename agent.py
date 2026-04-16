"""
PawPal+ AI Agent — core reasoning engine.

Implements a ReAct (Reason-Act-Observe) loop that:
1. Detects user intent from natural language
2. Plans which tools to call
3. Executes tools and observes results
4. Self-checks the response via guardrails
5. Returns a natural language response

Supports both OpenAI and Anthropic APIs (configurable).
Falls back to a rule-based dispatcher when no API key is available.
"""

import json
import os
import re
from dataclasses import dataclass, field

from guardrails import compute_confidence, run_all_checks
from knowledge_base import KnowledgeBase
from logger import AgentLogger
from pawpal_system import Owner, Scheduler
from tools import TOOL_DEFINITIONS, execute_tool


# --- Intent categories ---

INTENTS = {
    "add_pet": [
        "add a pet", "new pet", "register pet", "i got a new", "i have a new",
        r"add \w+.* a (dog|cat|bird|hamster)", "add a (dog|cat|bird|hamster)",
        r"add \w+.* dog", r"add \w+.* cat", r"add \w+.* bird", r"add \w+.* hamster",
    ],
    "add_task": [
        "schedule", "add task", "set up", "remind me to", "add a walk",
        "feeding time", "groom", "medication", "vet appointment",
    ],
    "complete_task": [
        "done", "finished", "completed", "mark as done", "just did",
    ],
    "get_schedule": [
        "schedule", "today", "what's on", "daily plan", "what do i need to do",
        "what needs to be done",
    ],
    "get_pet_tasks": [
        "tasks for", "show me .* tasks", "what does .* need",
    ],
    "detect_conflicts": [
        "conflict", "overlap", "double booked", "same time",
    ],
    "suggest_time": [
        "when should i", "find a time", "suggest time", "available slot",
        "when is .* free", "next free", "free slot", "minute slot",
    ],
    "care_question": [
        "how often", "how much", "should i feed", "is it safe", "can .* eat",
        "what food", "toxic", "poison", "exercise", "groom", "train",
        "vaccine", "vet", "health", "sick", "symptom",
    ],
    "general_chat": [],  # fallback
}


@dataclass
class AgentResponse:
    """Structured response from the Agent."""

    message: str
    tool_calls_made: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    guardrail_warnings: list[str] = field(default_factory=list)


class PawPalAgent:
    """AI Agent for PawPal+ pet care management.

    The agent operates in two modes:
    - LLM mode: Uses an API (OpenAI/Anthropic) for reasoning and tool selection
    - Rule-based mode: Uses keyword matching for intent detection (no API needed)
    """

    def __init__(
        self,
        owner: Owner,
        api_key: str | None = None,
        api_provider: str = "openai",  # 'openai' or 'anthropic'
        model: str = "gpt-4o-mini",
        use_llm: bool = True,
    ):
        self.owner = owner
        self.scheduler = Scheduler(owner=owner)
        self.knowledge_base = KnowledgeBase()
        self.logger = AgentLogger()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        self.api_provider = api_provider
        self.model = model
        self.use_llm = use_llm and self.api_key is not None

        # Load additional knowledge documents if available

    def process(self, user_message: str) -> AgentResponse:
        """Process a user message through the full ReAct loop.

        Steps:
        1. Detect intent
        2. Check guardrails (emergency/vet referral)
        3. Plan and execute tools
        4. Self-check response
        5. Log everything
        6. Return response
        """
        # Step 1: Detect intent
        intent = self._detect_intent(user_message)

        # Start logging
        log = self.logger.start_interaction(user_message, intent)

        try:
            # Step 2: Pre-check guardrails (emergency detection)
            pre_check = run_all_checks(
                user_message=user_message,
                agent_response="",
                tool_results=[],
                pet_species=self._get_current_species(),
            )
            if pre_check.modified_response:
                # Emergency detected — return immediately
                self.logger.log_guardrail(
                    log,
                    passed=False,
                    warnings=pre_check.warnings,
                    response_modified=True,
                )
                self.logger.log_response(log, pre_check.modified_response)
                return AgentResponse(
                    message=pre_check.modified_response,
                    confidence=1.0,
                    guardrail_warnings=pre_check.warnings,
                )

            # Step 3: Plan and execute tools
            if self.use_llm:
                response_text, tool_calls = self._llm_reason_and_act(user_message, intent)
            else:
                response_text, tool_calls = self._rule_based_act(user_message, intent)

            tool_results = [tc.get("result", "") for tc in tool_calls]

            # Step 4: Self-check via guardrails
            post_check = run_all_checks(
                user_message=user_message,
                agent_response=response_text,
                tool_results=tool_results,
                pet_species=self._get_current_species(),
            )

            if post_check.modified_response:
                response_text = post_check.modified_response

            confidence = compute_confidence(tool_results, user_message)

            # Step 5: Log everything
            for tc in tool_calls:
                self.logger.log_tool_call(log, tc["name"], tc.get("args", {}), tc.get("result", ""))
            self.logger.log_guardrail(
                log,
                passed=post_check.passed,
                warnings=post_check.warnings,
                confidence=confidence,
                response_modified=post_check.modified_response is not None,
            )
            self.logger.log_response(log, response_text)

            return AgentResponse(
                message=response_text,
                tool_calls_made=tool_calls,
                confidence=confidence,
                guardrail_warnings=post_check.warnings,
            )

        except Exception as e:
            self.logger.log_error(log, str(e))
            return AgentResponse(
                message=f"Sorry, I encountered an error: {e}. Please try again.",
                confidence=0.0,
            )

    def _detect_intent(self, message: str) -> str:
        """Detect user intent using keyword matching.

        Returns the best-matching intent category.
        """
        message_lower = message.lower()
        best_intent = "general_chat"
        best_score = 0

        for intent, patterns in INTENTS.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    score += 1
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent

    def _get_current_species(self) -> str | None:
        """Get the species of the first pet (for guardrail checks)."""
        if self.owner.pets:
            return self.owner.pets[0].species
        return None

    def _rule_based_act(
        self, user_message: str, intent: str
    ) -> tuple[str, list[dict]]:
        """Execute tools based on detected intent without LLM.

        This is the fallback mode when no API key is configured.
        It uses pattern matching to extract parameters from the user message.
        """
        tool_calls = []
        message_lower = user_message.lower()

        if intent == "add_pet":
            name, species = self._extract_pet_info(user_message)
            if name and species:
                result = execute_tool(
                    "add_pet",
                    {"name": name, "species": species},
                    self.owner,
                    self.scheduler,
                )
                tool_calls.append({"name": "add_pet", "args": {"name": name, "species": species}, "result": result})
                return result, tool_calls
            return "What's your new pet's name and species? (e.g., 'Add Mochi, a dog')", tool_calls

        elif intent == "add_task":
            pet_name, description, time = self._extract_task_info(user_message)
            if pet_name and description:
                args = {
                    "pet_name": pet_name,
                    "description": description,
                    "time": time or "09:00",
                    "duration_minutes": self._guess_duration(description),
                    "priority": self._guess_priority(description),
                    "frequency": self._guess_frequency(user_message),
                }
                result = execute_tool("add_task", args, self.owner, self.scheduler)
                tool_calls.append({"name": "add_task", "args": args, "result": result})
                return result, tool_calls
            return "Which pet should I add the task for, and what's the task?", tool_calls

        elif intent == "complete_task":
            pet_name, task_desc = self._extract_completion_info(user_message)
            if pet_name and task_desc:
                args = {"pet_name": pet_name, "task_description": task_desc}
                result = execute_tool("complete_task", args, self.owner, self.scheduler)
                tool_calls.append({"name": "complete_task", "args": args, "result": result})
                return result, tool_calls
            return "Which task did you complete? (e.g., 'finished Mochi's morning walk')", tool_calls

        elif intent == "get_schedule":
            result = execute_tool("get_schedule", {}, self.owner, self.scheduler)
            tool_calls.append({"name": "get_schedule", "args": {}, "result": result})
            return result, tool_calls

        elif intent == "get_pet_tasks":
            pet_name = self._extract_pet_name(user_message)
            if pet_name:
                args = {"pet_name": pet_name, "pending_only": "pending" in message_lower}
                result = execute_tool("get_pet_tasks", args, self.owner, self.scheduler)
                tool_calls.append({"name": "get_pet_tasks", "args": args, "result": result})
                return result, tool_calls
            return "Which pet's tasks would you like to see?", tool_calls

        elif intent == "detect_conflicts":
            result = execute_tool("detect_conflicts", {}, self.owner, self.scheduler)
            tool_calls.append({"name": "detect_conflicts", "args": {}, "result": result})
            return result, tool_calls

        elif intent == "suggest_time":
            duration = self._extract_duration(user_message)
            args = {"duration_minutes": duration}
            result = execute_tool("suggest_time_slot", args, self.owner, self.scheduler)
            tool_calls.append({"name": "suggest_time_slot", "args": args, "result": result})
            return result, tool_calls

        elif intent == "care_question":
            result = execute_tool(
                "search_care_info",
                {"query": user_message},
                self.owner,
                self.scheduler,
                self.knowledge_base,
            )
            tool_calls.append({"name": "search_care_info", "args": {"query": user_message}, "result": result})
            return result, tool_calls

        else:
            # General chat — provide a helpful default
            return self._general_response(), tool_calls

    def _llm_reason_and_act(
        self, user_message: str, intent: str
    ) -> tuple[str, list[dict]]:
        """Use an LLM API to reason about the message and select tools.

        Implements a simplified ReAct loop:
        1. Send user message + tool definitions to LLM
        2. Parse tool calls from response
        3. Execute tools and collect results
        4. Send results back to LLM for final response
        """
        try:
            if self.api_provider == "openai":
                return self._openai_react(user_message)
            elif self.api_provider == "anthropic":
                return self._anthropic_react(user_message)
            else:
                # Fallback to rule-based
                return self._rule_based_act(user_message, intent)
        except Exception as e:
            # If LLM fails, fall back to rule-based
            print(f"LLM call failed ({e}), falling back to rule-based mode.")
            return self._rule_based_act(user_message, intent)

    def _openai_react(self, user_message: str) -> tuple[str, list[dict]]:
        """Execute ReAct loop using OpenAI API with function calling."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)

        # Build system prompt
        system_prompt = self._build_system_prompt()

        # Build OpenAI-compatible tool definitions
        oai_tools = [
            {
                "type": "function",
                "function": {
                    "name": td["name"],
                    "description": td["description"],
                    "parameters": td["parameters"],
                },
            }
            for td in TOOL_DEFINITIONS
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_calls_made = []

        # ReAct loop (max 3 iterations to prevent infinite loops)
        for _ in range(3):
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=oai_tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message)
                for tc in message.tool_calls:
                    func_name = tc.function.name
                    func_args = json.loads(tc.function.arguments)

                    result = execute_tool(
                        func_name,
                        func_args,
                        self.owner,
                        self.scheduler,
                        self.knowledge_base,
                    )

                    tool_calls_made.append({
                        "name": func_name,
                        "args": func_args,
                        "result": result,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                # No more tool calls — return the final response
                return message.content or "I'm not sure how to help with that.", tool_calls_made

        # Max iterations reached
        return "I've completed the requested actions. Is there anything else you need?", tool_calls_made

    def _anthropic_react(self, user_message: str) -> tuple[str, list[dict]]:
        """Execute ReAct loop using Anthropic API with tool use."""
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        system_prompt = self._build_system_prompt()

        # Build Anthropic-compatible tool definitions
        anth_tools = [
            {
                "name": td["name"],
                "description": td["description"],
                "input_schema": td["parameters"],
            }
            for td in TOOL_DEFINITIONS
        ]

        messages = [{"role": "user", "content": user_message}]
        tool_calls_made = []

        for _ in range(3):
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=anth_tools,
            )

            # Check if there are tool use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

            if tool_use_blocks:
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for tb in tool_use_blocks:
                    result = execute_tool(
                        tb.name,
                        tb.input,
                        self.owner,
                        self.scheduler,
                        self.knowledge_base,
                    )
                    tool_calls_made.append({
                        "name": tb.name,
                        "args": tb.input,
                        "result": result,
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tb.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
            else:
                text_blocks = [b.text for b in response.content if b.type == "text"]
                return " ".join(text_blocks) or "Done.", tool_calls_made

        return "I've completed the requested actions.", tool_calls_made

    def _build_system_prompt(self) -> str:
        """Build the system prompt with current state context."""
        pets_info = ""
        if self.owner.pets:
            pet_lines = []
            for pet in self.owner.pets:
                pending = len(pet.get_pending_tasks())
                pet_lines.append(f"  - {pet.name} ({pet.species}): {pending} pending task(s)")
            pets_info = "\n".join(pet_lines)
        else:
            pets_info = "  No pets registered yet."

        return f"""You are PawPal+ Assistant, an AI agent for pet care management.

Current state:
  Owner: {self.owner.name}
  Pets:
{pets_info}

Your capabilities:
- Manage pets and their care schedules (add pets, add/complete tasks)
- Detect scheduling conflicts and suggest available time slots
- Answer pet care questions using a built-in knowledge base
- Provide responsible, safety-conscious advice

Guidelines:
- Always confirm actions you take (e.g., "I've added the task...")
- Proactively check for conflicts after scheduling changes
- For health questions, include a disclaimer to consult a vet
- Never recommend foods toxic to the pet's species
- If you're unsure, say so and suggest consulting a professional
- Be concise and friendly
"""

    # --- Helper methods for rule-based parameter extraction ---

    def _extract_pet_info(self, message: str) -> tuple[str | None, str | None]:
        """Extract pet name and species from user message."""
        species_keywords = ["dog", "cat", "bird", "hamster"]
        message_lower = message.lower()

        species = None
        for s in species_keywords:
            if s in message_lower:
                species = s
                break

        # Try to extract a capitalized name
        words = message.split()
        name = None
        for word in words:
            cleaned = word.strip(",.!?")
            if cleaned and cleaned[0].isupper() and cleaned.lower() not in species_keywords:
                if cleaned.lower() not in ["add", "new", "my", "pet", "i", "a", "the", "got", "have", "named", "called"]:
                    name = cleaned
                    break

        return name, species or "other"

    def _extract_pet_name(self, message: str) -> str | None:
        """Extract a pet name by matching against known pets."""
        message_lower = message.lower()
        for pet in self.owner.pets:
            if pet.name.lower() in message_lower:
                return pet.name
        return None

    def _extract_task_info(self, message: str) -> tuple[str | None, str | None, str | None]:
        """Extract pet name, task description, and time from user message."""
        pet_name = self._extract_pet_name(message)

        # Extract time (HH:MM pattern)
        time_match = re.search(r"\b(\d{1,2}):(\d{2})\b", message)
        time = None
        if time_match:
            h, m = int(time_match.group(1)), int(time_match.group(2))
            time = f"{h:02d}:{m:02d}"

        # Extract task description (heuristic: look for common task words)
        task_keywords = {
            "walk": "Morning walk",
            "feed": "Feeding",
            "groom": "Grooming session",
            "bath": "Bath time",
            "vet": "Vet appointment",
            "medication": "Medication",
            "medicine": "Medication",
            "play": "Play session",
            "train": "Training session",
            "brush": "Brushing",
            "nail": "Nail trimming",
        }

        description = None
        message_lower = message.lower()
        for keyword, default_desc in task_keywords.items():
            if keyword in message_lower:
                description = default_desc
                break

        if description is None and pet_name:
            # Use the part of the message after the pet name as description
            parts = message.lower().split(pet_name.lower())
            if len(parts) > 1:
                desc_part = parts[1].strip(" ,.!?")
                if len(desc_part) > 3:
                    description = desc_part.capitalize()

        return pet_name, description, time

    def _extract_completion_info(self, message: str) -> tuple[str | None, str | None]:
        """Extract pet name and task description from a completion message."""
        pet_name = self._extract_pet_name(message)

        if pet_name:
            pet = self.owner.find_pet(pet_name)
            if pet:
                message_lower = message.lower()
                for task in pet.get_pending_tasks():
                    # Check if any task description words appear in the message
                    task_words = task.description.lower().split()
                    if any(word in message_lower for word in task_words):
                        return pet_name, task.description

        return pet_name, None

    def _extract_duration(self, message: str) -> int:
        """Extract a duration in minutes from the message."""
        match = re.search(r"(\d+)\s*min", message.lower())
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s*hour", message.lower())
        if match:
            return int(match.group(1)) * 60
        return 30  # default

    def _guess_duration(self, description: str) -> int:
        """Guess task duration based on description."""
        desc = description.lower()
        if "walk" in desc:
            return 30
        if "feed" in desc:
            return 10
        if "vet" in desc:
            return 60
        if "groom" in desc or "bath" in desc:
            return 45
        if "play" in desc:
            return 20
        if "medication" in desc or "medicine" in desc:
            return 5
        return 15

    def _guess_priority(self, description: str) -> str:
        """Guess task priority based on description."""
        desc = description.lower()
        if any(w in desc for w in ["vet", "medication", "medicine", "emergency"]):
            return "high"
        if any(w in desc for w in ["feed", "walk"]):
            return "high"
        if any(w in desc for w in ["groom", "bath", "train"]):
            return "medium"
        return "medium"

    def _guess_frequency(self, message: str) -> str:
        """Guess task frequency based on message content."""
        msg = message.lower()
        if any(w in msg for w in ["every day", "daily", "each day"]):
            return "daily"
        if any(w in msg for w in ["every week", "weekly", "each week"]):
            return "weekly"
        return "once"

    def _general_response(self) -> str:
        """Generate a helpful default response."""
        pet_count = len(self.owner.pets)
        if pet_count == 0:
            return (
                "Welcome to PawPal+! I can help you manage pet care schedules. "
                "Start by telling me about your pet — for example, 'Add Luna, a cat'."
            )
        schedule = self.scheduler.generate_schedule()
        task_count = len(schedule)
        return (
            f"Hi {self.owner.name}! You have {pet_count} pet(s) and "
            f"{task_count} task(s) on today's schedule. "
            "How can I help? You can ask me to add tasks, check the schedule, "
            "or ask pet care questions."
        )
