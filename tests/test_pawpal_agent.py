"""
PawPal+ Agent test suite.

Covers: original PawPal+ logic, tool execution, guardrails,
knowledge base retrieval, agent intent detection, and end-to-end flows.
"""

from datetime import date, timedelta

from agent import PawPalAgent
from guardrails import (
    check_emergency,
    check_toxic_food_mention,
    check_vet_referral,
    compute_confidence,
    run_all_checks,
)
from knowledge_base import KnowledgeBase
from logger import AgentLogger
from pawpal_system import Owner, Pet, Scheduler, Task
from tools import execute_tool


# ============================================================
# Section 1: Original PawPal+ logic tests
# ============================================================


class TestTask:
    """Tests for the Task dataclass."""

    def test_mark_complete_changes_status(self):
        task = Task(description="Walk", time="09:00", duration_minutes=30)
        assert task.completed is False
        assert task.mark_complete() is True
        assert task.completed is True

    def test_mark_complete_already_done(self):
        task = Task(description="Walk", time="09:00", duration_minutes=30, completed=True)
        assert task.mark_complete() is False

    def test_create_next_occurrence_daily(self):
        today = date.today()
        task = Task(
            description="Feeding", time="08:00", duration_minutes=10,
            frequency="daily", due_date=today, pet_name="Mochi",
        )
        next_task = task.create_next_occurrence()
        assert next_task is not None
        assert next_task.due_date == today + timedelta(days=1)
        assert next_task.completed is False
        assert next_task.pet_name == "Mochi"

    def test_create_next_occurrence_weekly(self):
        today = date.today()
        task = Task(
            description="Flea meds", time="10:00", duration_minutes=5,
            frequency="weekly", due_date=today,
        )
        next_task = task.create_next_occurrence()
        assert next_task is not None
        assert next_task.due_date == today + timedelta(days=7)

    def test_create_next_occurrence_once_returns_none(self):
        task = Task(description="Vet visit", time="11:00", duration_minutes=60, frequency="once")
        assert task.create_next_occurrence() is None

    def test_task_serialization(self):
        task = Task(description="Walk", time="09:00", duration_minutes=30, pet_name="Mochi")
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.description == "Walk"
        assert restored.pet_name == "Mochi"
        assert restored.due_date == task.due_date


class TestPet:
    """Tests for the Pet dataclass."""

    def test_add_task_sets_pet_name(self):
        pet = Pet(name="Mochi", species="dog")
        task = Task(description="Walk", time="09:00", duration_minutes=30)
        pet.add_task(task)
        assert task.pet_name == "Mochi"
        assert len(pet.tasks) == 1

    def test_remove_task(self):
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        assert pet.remove_task("Walk") is True
        assert len(pet.tasks) == 0

    def test_remove_nonexistent_task(self):
        pet = Pet(name="Mochi", species="dog")
        assert pet.remove_task("Nonexistent") is False

    def test_get_pending_tasks(self):
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=10, completed=True))
        pending = pet.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0].description == "Walk"


class TestScheduler:
    """Tests for the Scheduler class."""

    def setup_method(self):
        self.owner = Owner(name="Jordan")
        self.mochi = Pet(name="Mochi", species="dog")
        self.luna = Pet(name="Luna", species="cat")
        self.owner.add_pet(self.mochi)
        self.owner.add_pet(self.luna)
        self.scheduler = Scheduler(owner=self.owner)

    def test_sort_by_time(self):
        self.mochi.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        self.mochi.add_task(Task(description="Feed", time="07:00", duration_minutes=10))
        sorted_tasks = self.scheduler.sort_by_time()
        assert sorted_tasks[0].time == "07:00"
        assert sorted_tasks[1].time == "09:00"

    def test_sort_by_priority(self):
        self.mochi.add_task(Task(description="Play", time="14:00", duration_minutes=20, priority="low"))
        self.mochi.add_task(Task(description="Feed", time="08:00", duration_minutes=10, priority="high"))
        sorted_tasks = self.scheduler.sort_by_priority()
        assert sorted_tasks[0].priority == "high"

    def test_detect_conflicts(self):
        self.mochi.add_task(Task(description="Walk", time="08:00", duration_minutes=30))
        self.luna.add_task(Task(description="Feed", time="08:00", duration_minutes=10))
        conflicts = self.scheduler.detect_conflicts()
        assert len(conflicts) == 1
        assert "08:00" in conflicts[0]

    def test_no_conflicts_unique_times(self):
        self.mochi.add_task(Task(description="Walk", time="08:00", duration_minutes=30))
        self.luna.add_task(Task(description="Feed", time="09:00", duration_minutes=10))
        assert len(self.scheduler.detect_conflicts()) == 0

    def test_find_next_available_slot(self):
        self.mochi.add_task(Task(description="Walk", time="07:00", duration_minutes=60))
        slot = self.scheduler.find_next_available_slot(30)
        assert slot is not None
        assert slot >= "08:00"

    def test_mark_task_complete_recurring(self):
        self.mochi.add_task(Task(
            description="Feed", time="08:00", duration_minutes=10,
            frequency="daily", due_date=date.today(),
        ))
        task = self.mochi.tasks[0]
        next_task = self.scheduler.mark_task_complete(task)
        assert task.completed is True
        assert next_task is not None
        assert next_task.due_date == date.today() + timedelta(days=1)
        assert len(self.mochi.tasks) == 2


# ============================================================
# Section 2: Tool execution tests
# ============================================================


class TestTools:
    """Tests for tool execution layer."""

    def setup_method(self):
        self.owner = Owner(name="Jordan")
        self.owner.add_pet(Pet(name="Mochi", species="dog"))
        self.scheduler = Scheduler(owner=self.owner)

    def test_add_pet_tool(self):
        result = execute_tool("add_pet", {"name": "Luna", "species": "cat"}, self.owner, self.scheduler)
        assert "Added Luna" in result
        assert self.owner.find_pet("Luna") is not None

    def test_add_duplicate_pet(self):
        result = execute_tool("add_pet", {"name": "Mochi", "species": "dog"}, self.owner, self.scheduler)
        assert "already exists" in result

    def test_add_task_tool(self):
        result = execute_tool(
            "add_task",
            {"pet_name": "Mochi", "description": "Walk", "time": "09:00", "duration_minutes": 30},
            self.owner, self.scheduler,
        )
        assert "Added task" in result

    def test_add_task_unknown_pet(self):
        result = execute_tool(
            "add_task",
            {"pet_name": "Unknown", "description": "Walk", "time": "09:00", "duration_minutes": 30},
            self.owner, self.scheduler,
        )
        assert "not found" in result

    def test_get_schedule_empty(self):
        result = execute_tool("get_schedule", {}, self.owner, self.scheduler)
        assert "No tasks" in result

    def test_complete_task_tool(self):
        pet = self.owner.find_pet("Mochi")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        result = execute_tool(
            "complete_task",
            {"pet_name": "Mochi", "task_description": "Walk"},
            self.owner, self.scheduler,
        )
        assert "Completed" in result

    def test_detect_conflicts_tool(self):
        result = execute_tool("detect_conflicts", {}, self.owner, self.scheduler)
        assert "No scheduling conflicts" in result

    def test_suggest_time_slot_tool(self):
        result = execute_tool("suggest_time_slot", {"duration_minutes": 30}, self.owner, self.scheduler)
        assert "Suggested time slot" in result

    def test_unknown_tool(self):
        result = execute_tool("nonexistent_tool", {}, self.owner, self.scheduler)
        assert "Unknown tool" in result


# ============================================================
# Section 3: Guardrail tests
# ============================================================


class TestGuardrails:
    """Tests for safety guardrails."""

    def test_toxic_food_detected(self):
        result = check_toxic_food_mention("You should give your dog chocolate as a treat!", "dog")
        assert not result.passed
        assert len(result.warnings) > 0

    def test_toxic_food_already_warned(self):
        result = check_toxic_food_mention("Never give your dog chocolate, it is toxic.", "dog")
        assert result.passed

    def test_emergency_detected(self):
        result = check_emergency("My dog ate chocolate and is having seizures!")
        assert not result.passed
        assert result.modified_response is not None
        assert "emergency" in result.modified_response.lower()

    def test_no_emergency(self):
        result = check_emergency("When should I feed my dog?")
        assert result.passed

    def test_vet_referral_detected(self):
        result = check_vet_referral("My cat has blood in stool, what should I do?")
        assert len(result.warnings) > 0

    def test_confidence_no_tools(self):
        score = compute_confidence([], "hello")
        assert score == 0.3

    def test_confidence_successful_tools(self):
        score = compute_confidence(
            ["Added Mochi the dog.", "Today's schedule (2026-04-13):"],
            "add a pet and show schedule",
        )
        assert score > 0.5

    def test_confidence_error_results(self):
        score = compute_confidence(["Error: something went wrong"], "add a pet")
        assert score < 0.5

    def test_run_all_checks_emergency_overrides(self):
        result = run_all_checks(
            user_message="My dog is not breathing!",
            agent_response="Let me check the schedule...",
            tool_results=[],
        )
        assert not result.passed
        assert "emergency" in result.modified_response.lower()

    def test_run_all_checks_clean(self):
        result = run_all_checks(
            user_message="What's on the schedule today?",
            agent_response="Here are today's tasks.",
            tool_results=["Today's schedule..."],
        )
        assert result.passed


# ============================================================
# Section 4: Knowledge base tests
# ============================================================


class TestKnowledgeBase:
    """Tests for the pet care knowledge retrieval."""

    def setup_method(self):
        self.kb = KnowledgeBase()

    def test_search_dog_feeding(self):
        result = self.kb.search("how much should I feed my dog")
        assert "feeding" in result.lower() or "cup" in result.lower()

    def test_search_cat_health(self):
        result = self.kb.search("my cat is vomiting")
        assert "vet" in result.lower() or "health" in result.lower()

    def test_search_no_results(self):
        result = self.kb.search("quantum physics equations")
        assert "no relevant" in result.lower() or "consult" in result.lower()

    def test_search_returns_multiple_results(self):
        result = self.kb.search("dog health symptoms exercise")
        assert len(result) > 100  # Should return substantive content

    def test_empty_query(self):
        result = self.kb.search("")
        assert "specific" in result.lower() or "more" in result.lower()


# ============================================================
# Section 5: Agent intent detection tests
# ============================================================


class TestAgentIntent:
    """Tests for the Agent's intent detection."""

    def setup_method(self):
        self.owner = Owner(name="Test")
        self.agent = PawPalAgent(owner=self.owner, use_llm=False)

    def test_detect_add_pet(self):
        assert self.agent._detect_intent("I want to add a new pet") == "add_pet"

    def test_detect_add_task(self):
        assert self.agent._detect_intent("Schedule a walk for Mochi at 8am") == "add_task"

    def test_detect_complete_task(self):
        assert self.agent._detect_intent("I just finished the morning walk") == "complete_task"

    def test_detect_schedule(self):
        assert self.agent._detect_intent("What's on today's schedule?") == "get_schedule"

    def test_detect_care_question(self):
        assert self.agent._detect_intent("How often should I feed my cat?") == "care_question"

    def test_detect_conflicts(self):
        assert self.agent._detect_intent("Are there any conflicts or overlaps?") == "detect_conflicts"

    def test_detect_general(self):
        assert self.agent._detect_intent("Hello there") == "general_chat"

    def test_detect_add_pet_with_name(self):
        assert self.agent._detect_intent("Add Mochi, a dog") == "add_pet"


# ============================================================
# Section 6: Agent end-to-end tests (rule-based mode)
# ============================================================


class TestAgentEndToEnd:
    """End-to-end tests for the Agent in rule-based mode."""

    def setup_method(self):
        self.owner = Owner(name="Jordan")
        self.agent = PawPalAgent(owner=self.owner, use_llm=False)

    def test_add_pet_flow(self):
        response = self.agent.process("Add Mochi, a dog")
        assert "mochi" in response.message.lower() or "dog" in response.message.lower()

    def test_schedule_empty(self):
        response = self.agent.process("What's on today's schedule?")
        assert "no tasks" in response.message.lower() or "schedule" in response.message.lower()

    def test_care_question_flow(self):
        response = self.agent.process("How much should I feed my dog?")
        assert len(response.message) > 50  # Should return substantive info
        assert len(response.tool_calls_made) > 0

    def test_emergency_flow(self):
        response = self.agent.process("My dog ate chocolate and is having seizures!")
        assert "emergency" in response.message.lower() or "vet" in response.message.lower()

    def test_general_greeting(self):
        response = self.agent.process("Hi there!")
        assert "pawpal" in response.message.lower() or "welcome" in response.message.lower()

    def test_logging_records_interactions(self):
        self.agent.process("Hello")
        self.agent.process("What's on today?")
        summary = self.agent.logger.get_summary()
        assert summary["total_interactions"] == 2


# ============================================================
# Section 7: Logger tests
# ============================================================


class TestLogger:
    """Tests for the logging system."""

    def test_start_interaction(self):
        logger = AgentLogger(log_dir="/tmp/test_logs")
        log = logger.start_interaction("Hello", "general_chat")
        assert log.user_message == "Hello"
        assert log.intent == "general_chat"

    def test_log_tool_call(self):
        logger = AgentLogger(log_dir="/tmp/test_logs")
        log = logger.start_interaction("test", "test")
        logger.log_tool_call(log, "get_schedule", {}, "No tasks")
        assert len(log.tool_calls) == 1
        assert log.tool_calls[0].tool_name == "get_schedule"

    def test_summary(self):
        logger = AgentLogger(log_dir="/tmp/test_logs")
        log1 = logger.start_interaction("msg1", "add_pet")
        logger.log_tool_call(log1, "add_pet", {"name": "Mochi"}, "Added")
        log2 = logger.start_interaction("msg2", "get_schedule")
        logger.log_tool_call(log2, "get_schedule", {}, "No tasks")

        summary = logger.get_summary()
        assert summary["total_interactions"] == 2
        assert summary["total_tool_calls"] == 2
        assert summary["tool_usage"]["add_pet"] == 1

    def test_serialization(self):
        logger = AgentLogger(log_dir="/tmp/test_logs")
        log = logger.start_interaction("Hello", "general_chat")
        logger.log_response(log, "Hi there!")
        data = log.to_dict()
        assert data["user_message"] == "Hello"
        assert data["agent_response"] == "Hi there!"
