"""Tests for the tool execution layer."""

import pytest

from pawpal_system import Owner, Pet, Scheduler, Task
from tools import TOOL_DEFINITIONS, execute_tool


class TestToolDefinitions:
    """Sanity checks on the tool schemas exposed to the LLM."""

    def test_every_definition_has_a_json_schema(self):
        """Each tool advertises a name, a description, and an object schema."""
        for td in TOOL_DEFINITIONS:
            assert td["name"]
            assert td["description"]
            assert td["parameters"]["type"] == "object"

    def test_tool_names_are_unique(self):
        """Duplicate tool names would make dispatch ambiguous."""
        names = [td["name"] for td in TOOL_DEFINITIONS]
        assert len(names) == len(set(names))


class TestExecuteTool:
    """Tests for dispatching and running individual tools."""

    @pytest.fixture(autouse=True)
    def _setup(self, owner: Owner):
        """One dog, and a scheduler over it. Uses the tmp_path-bound owner
        fixture so the tool layer's autosave never touches the repo."""
        self.owner = owner
        self.owner.add_pet(Pet(name="Mochi", species="dog"))
        self.scheduler = Scheduler(owner=self.owner)

    def _run(self, name: str, args: dict) -> str:
        return execute_tool(name, args, self.owner, self.scheduler)

    def test_add_pet(self):
        result = self._run("add_pet", {"name": "Luna", "species": "cat"})
        assert "Added Luna" in result
        assert self.owner.find_pet("Luna") is not None

    def test_add_duplicate_pet(self):
        result = self._run("add_pet", {"name": "Mochi", "species": "dog"})
        assert "already exists" in result

    def test_add_task(self):
        result = self._run(
            "add_task",
            {"pet_name": "Mochi", "description": "Walk", "time": "09:00", "duration_minutes": 30},
        )
        assert "Added task" in result

    def test_add_task_unknown_pet(self):
        result = self._run(
            "add_task",
            {"pet_name": "Unknown", "description": "Walk", "time": "09:00", "duration_minutes": 30},
        )
        assert "not found" in result

    def test_get_schedule_empty(self):
        assert "No tasks" in self._run("get_schedule", {})

    def test_complete_task(self):
        self.owner.find_pet("Mochi").add_task(
            Task(description="Walk", time="09:00", duration_minutes=30)
        )
        result = self._run("complete_task", {"pet_name": "Mochi", "task_description": "Walk"})
        assert "Completed" in result

    def test_get_pet_tasks(self):
        self.owner.find_pet("Mochi").add_task(
            Task(description="Walk", time="09:00", duration_minutes=30)
        )
        result = self._run("get_pet_tasks", {"pet_name": "Mochi"})
        assert "Walk" in result

    def test_detect_conflicts(self):
        assert "No scheduling conflicts" in self._run("detect_conflicts", {})

    def test_suggest_time_slot(self):
        assert "Suggested time slot" in self._run("suggest_time_slot", {"duration_minutes": 30})

    def test_search_care_info_without_knowledge_base(self):
        """search_care_info degrades gracefully when no KB is wired in."""
        assert "not available" in self._run("search_care_info", {"query": "feeding"})

    def test_unknown_tool(self):
        assert "Unknown tool" in self._run("nonexistent_tool", {})

    def test_bad_arguments_are_reported_not_raised(self):
        """A schema violation surfaces as an error string, not an exception."""
        result = self._run("add_pet", {"wrong_arg": "Luna"})
        assert "Error executing add_pet" in result
