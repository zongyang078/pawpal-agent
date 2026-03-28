"""
Automated test suite for PawPal+ system.
Covers: task completion, task addition, sorting, recurring logic, conflict detection,
        JSON persistence, and next-available-slot finding.
"""

import os
import json
from datetime import date, timedelta
from pawpal_system import Task, Pet, Owner, Scheduler


# --- Task tests ---

class TestTask:
    """Tests for the Task class."""

    def test_mark_complete_changes_status(self):
        """Verify that calling mark_complete() changes the task's completed status."""
        task = Task(description="Walk", time="09:00", duration_minutes=30)
        assert task.completed is False
        result = task.mark_complete()
        assert task.completed is True
        assert result is True

    def test_mark_complete_already_done(self):
        """Calling mark_complete() on an already completed task returns False."""
        task = Task(description="Walk", time="09:00", duration_minutes=30, completed=True)
        result = task.mark_complete()
        assert result is False

    def test_create_next_occurrence_daily(self):
        """A daily task should produce a next occurrence one day later."""
        today = date.today()
        task = Task(
            description="Feeding",
            time="08:00",
            duration_minutes=10,
            frequency="daily",
            due_date=today,
            pet_name="Mochi",
        )
        next_task = task.create_next_occurrence()
        assert next_task is not None
        assert next_task.due_date == today + timedelta(days=1)
        assert next_task.completed is False
        assert next_task.pet_name == "Mochi"

    def test_create_next_occurrence_weekly(self):
        """A weekly task should produce a next occurrence seven days later."""
        today = date.today()
        task = Task(
            description="Flea meds",
            time="10:00",
            duration_minutes=5,
            frequency="weekly",
            due_date=today,
        )
        next_task = task.create_next_occurrence()
        assert next_task is not None
        assert next_task.due_date == today + timedelta(days=7)

    def test_create_next_occurrence_once_returns_none(self):
        """A one-time task should not produce a next occurrence."""
        task = Task(description="Vet visit", time="11:00", duration_minutes=60, frequency="once")
        assert task.create_next_occurrence() is None

    def test_task_to_dict_and_from_dict(self):
        """Task should round-trip through dict serialization."""
        today = date.today()
        task = Task(
            description="Walk",
            time="07:30",
            duration_minutes=30,
            priority="high",
            frequency="daily",
            completed=False,
            due_date=today,
            pet_name="Mochi",
        )
        data = task.to_dict()
        restored = Task.from_dict(data)
        assert restored.description == task.description
        assert restored.time == task.time
        assert restored.duration_minutes == task.duration_minutes
        assert restored.priority == task.priority
        assert restored.frequency == task.frequency
        assert restored.completed == task.completed
        assert restored.due_date == task.due_date
        assert restored.pet_name == task.pet_name


# --- Pet tests ---

class TestPet:
    """Tests for the Pet class."""

    def test_add_task_increases_count(self):
        """Adding a task to a Pet increases that pet's task count."""
        pet = Pet(name="Mochi", species="dog")
        assert len(pet.tasks) == 0
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        assert len(pet.tasks) == 1

    def test_add_task_sets_pet_name(self):
        """Adding a task should auto-set the task's pet_name."""
        pet = Pet(name="Luna", species="cat")
        task = Task(description="Play", time="14:00", duration_minutes=20)
        pet.add_task(task)
        assert task.pet_name == "Luna"

    def test_remove_task(self):
        """Removing a task by description should work correctly."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        assert pet.remove_task("Walk") is True
        assert len(pet.tasks) == 0

    def test_remove_task_not_found(self):
        """Removing a nonexistent task returns False."""
        pet = Pet(name="Mochi", species="dog")
        assert pet.remove_task("Nonexistent") is False

    def test_get_pending_tasks(self):
        """get_pending_tasks returns only incomplete tasks."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30, completed=False))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=10, completed=True))
        pending = pet.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0].description == "Walk"

    def test_pet_to_dict_and_from_dict(self):
        """Pet should round-trip through dict serialization."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=10))
        data = pet.to_dict()
        restored = Pet.from_dict(data)
        assert restored.name == "Mochi"
        assert restored.species == "dog"
        assert len(restored.tasks) == 2
        assert restored.tasks[0].pet_name == "Mochi"


# --- Owner tests ---

class TestOwner:
    """Tests for the Owner class."""

    def test_add_pet(self):
        """Adding a pet increases the owner's pet count."""
        owner = Owner(name="Jordan")
        owner.add_pet(Pet(name="Mochi", species="dog"))
        assert len(owner.pets) == 1

    def test_get_all_tasks_across_pets(self):
        """get_all_tasks should collect tasks from all pets."""
        owner = Owner(name="Jordan")
        dog = Pet(name="Mochi", species="dog")
        cat = Pet(name="Luna", species="cat")
        dog.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        cat.add_task(Task(description="Play", time="14:00", duration_minutes=20))
        owner.add_pet(dog)
        owner.add_pet(cat)
        assert len(owner.get_all_tasks()) == 2

    def test_save_and_load_json(self, tmp_path):
        """Owner should round-trip through JSON file save/load."""
        filepath = str(tmp_path / "test_data.json")
        owner = Owner(name="Jordan")
        dog = Pet(name="Mochi", species="dog")
        dog.add_task(Task(
            description="Walk",
            time="07:30",
            duration_minutes=30,
            priority="high",
            frequency="daily",
        ))
        cat = Pet(name="Luna", species="cat")
        cat.add_task(Task(description="Play", time="14:00", duration_minutes=20))
        owner.add_pet(dog)
        owner.add_pet(cat)

        # Save
        owner.save_to_json(filepath)
        assert os.path.exists(filepath)

        # Load and verify
        loaded = Owner.load_from_json(filepath)
        assert loaded.name == "Jordan"
        assert len(loaded.pets) == 2
        assert loaded.pets[0].name == "Mochi"
        assert len(loaded.pets[0].tasks) == 1
        assert loaded.pets[0].tasks[0].description == "Walk"
        assert loaded.pets[0].tasks[0].priority == "high"
        assert loaded.pets[0].tasks[0].frequency == "daily"
        assert loaded.pets[1].name == "Luna"

    def test_load_json_preserves_task_dates(self, tmp_path):
        """Dates should survive JSON serialization."""
        filepath = str(tmp_path / "test_dates.json")
        owner = Owner(name="Jordan")
        pet = Pet(name="Mochi", species="dog")
        today = date.today()
        pet.add_task(Task(description="Walk", time="07:30", duration_minutes=30, due_date=today))
        owner.add_pet(pet)

        owner.save_to_json(filepath)
        loaded = Owner.load_from_json(filepath)
        assert loaded.pets[0].tasks[0].due_date == today


# --- Scheduler tests ---

class TestScheduler:
    """Tests for the Scheduler class."""

    def _make_scheduler(self) -> Scheduler:
        """Helper to create a scheduler with sample data."""
        owner = Owner(name="Jordan")
        dog = Pet(name="Mochi", species="dog")
        cat = Pet(name="Luna", species="cat")

        dog.add_task(Task(description="Walk", time="07:30", duration_minutes=30, priority="high"))
        dog.add_task(Task(description="Feed Mochi", time="08:00", duration_minutes=10, priority="high"))
        cat.add_task(Task(description="Feed Luna", time="08:00", duration_minutes=10, priority="medium"))
        cat.add_task(Task(description="Play", time="14:00", duration_minutes=20, priority="low"))

        owner.add_pet(dog)
        owner.add_pet(cat)
        return Scheduler(owner=owner)

    def test_sort_by_time_chronological(self):
        """Tasks should be returned in chronological order after sorting."""
        scheduler = self._make_scheduler()
        sorted_tasks = scheduler.sort_by_time()
        times = [t.time for t in sorted_tasks]
        assert times == sorted(times)

    def test_sort_by_priority_descending(self):
        """High priority tasks should come before low priority tasks."""
        scheduler = self._make_scheduler()
        sorted_tasks = scheduler.sort_by_priority()
        assert sorted_tasks[0].priority == "high"
        assert sorted_tasks[-1].priority == "low"

    def test_filter_by_pet(self):
        """Filtering by pet name returns only that pet's tasks."""
        scheduler = self._make_scheduler()
        mochi_tasks = scheduler.filter_by_pet("Mochi")
        assert all(t.pet_name == "Mochi" for t in mochi_tasks)
        assert len(mochi_tasks) == 2

    def test_filter_by_status(self):
        """Filtering by status returns tasks matching the given completion state."""
        scheduler = self._make_scheduler()
        pending = scheduler.filter_by_status(completed=False)
        assert all(not t.completed for t in pending)

    def test_detect_conflicts_flags_same_time(self):
        """Scheduler should detect when two tasks share the same time slot."""
        scheduler = self._make_scheduler()
        conflicts = scheduler.detect_conflicts()
        assert len(conflicts) >= 1
        assert "08:00" in conflicts[0]

    def test_detect_conflicts_no_false_positives(self):
        """Scheduler should not flag unique time slots as conflicts."""
        owner = Owner(name="Jordan")
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="07:00", duration_minutes=30))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=10))
        owner.add_pet(pet)
        scheduler = Scheduler(owner=owner)
        assert len(scheduler.detect_conflicts()) == 0

    def test_mark_complete_recurring_creates_next(self):
        """Completing a daily task should auto-create the next occurrence."""
        owner = Owner(name="Jordan")
        pet = Pet(name="Mochi", species="dog")
        task = Task(
            description="Walk",
            time="07:30",
            duration_minutes=30,
            frequency="daily",
            due_date=date.today(),
        )
        pet.add_task(task)
        owner.add_pet(pet)
        scheduler = Scheduler(owner=owner)

        original_count = len(pet.tasks)
        next_task = scheduler.mark_task_complete(task)

        assert task.completed is True
        assert next_task is not None
        assert next_task.due_date == date.today() + timedelta(days=1)
        assert len(pet.tasks) == original_count + 1

    def test_generate_schedule_pending_only(self):
        """generate_schedule should only include pending tasks."""
        scheduler = self._make_scheduler()
        all_tasks = scheduler.get_all_tasks()
        all_tasks[0].mark_complete()

        schedule = scheduler.generate_schedule()
        assert all(not t.completed for t in schedule)

    def test_empty_pet_no_crash(self):
        """A pet with no tasks should not cause errors in the scheduler."""
        owner = Owner(name="Jordan")
        owner.add_pet(Pet(name="Ghost", species="hamster"))
        scheduler = Scheduler(owner=owner)
        assert scheduler.generate_schedule() == []
        assert scheduler.detect_conflicts() == []
        assert scheduler.sort_by_time() == []

    # --- Challenge 1: find_next_available_slot tests ---

    def test_find_slot_empty_schedule(self):
        """With no tasks, the first available slot should be the start time."""
        owner = Owner(name="Jordan")
        owner.add_pet(Pet(name="Mochi", species="dog"))
        scheduler = Scheduler(owner=owner)
        slot = scheduler.find_next_available_slot(30)
        assert slot == "07:00"

    def test_find_slot_skips_occupied(self):
        """Slot finder should skip over occupied time ranges."""
        owner = Owner(name="Jordan")
        pet = Pet(name="Mochi", species="dog")
        # Block 07:00-07:30
        pet.add_task(Task(description="Walk", time="07:00", duration_minutes=30))
        owner.add_pet(pet)
        scheduler = Scheduler(owner=owner)
        slot = scheduler.find_next_available_slot(30)
        assert slot == "07:30"

    def test_find_slot_respects_duration(self):
        """A long task should only fit in a slot wide enough for it."""
        owner = Owner(name="Jordan")
        pet = Pet(name="Mochi", species="dog")
        # Block 07:00-07:30 and 08:00-08:30
        pet.add_task(Task(description="Walk", time="07:00", duration_minutes=30))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=30))
        owner.add_pet(pet)
        scheduler = Scheduler(owner=owner)
        # 60-min task can't fit in 07:30-08:00 gap (only 30 min)
        slot = scheduler.find_next_available_slot(60)
        assert slot == "08:30"

    def test_find_slot_no_room_returns_none(self):
        """If the day is fully booked, return None."""
        owner = Owner(name="Jordan")
        pet = Pet(name="Mochi", species="dog")
        # Fill every 30-min slot from 07:00 to 21:00 (28 slots)
        for hour in range(7, 21):
            for minute in [0, 30]:
                pet.add_task(Task(
                    description=f"Task {hour}:{minute:02d}",
                    time=f"{hour:02d}:{minute:02d}",
                    duration_minutes=30,
                ))
        owner.add_pet(pet)
        scheduler = Scheduler(owner=owner)
        assert scheduler.find_next_available_slot(30) is None

    def test_find_slot_ignores_completed_tasks(self):
        """Completed tasks should not block time slots."""
        owner = Owner(name="Jordan")
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="07:00", duration_minutes=30, completed=True))
        owner.add_pet(pet)
        scheduler = Scheduler(owner=owner)
        slot = scheduler.find_next_available_slot(30)
        assert slot == "07:00"  # 07:00 is free because the task is completed
