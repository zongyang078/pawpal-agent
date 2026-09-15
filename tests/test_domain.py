"""Tests for the domain layer: Task, Pet, Owner, Scheduler.

Covers task completion and recurrence, task list management, JSON
persistence round-trips, sorting/filtering, conflict detection, and the
next-available-slot finder.
"""

import os
from datetime import date, timedelta

from pawpal_system import Owner, Pet, Scheduler, Task


class TestTask:
    """Tests for the Task dataclass."""

    def test_mark_complete_changes_status(self):
        """Calling mark_complete() flips the status and reports the change."""
        task = Task(description="Walk", time="09:00", duration_minutes=30)
        assert task.completed is False
        assert task.mark_complete() is True
        assert task.completed is True

    def test_mark_complete_already_done(self):
        """Completing an already-completed task is a no-op returning False."""
        task = Task(description="Walk", time="09:00", duration_minutes=30, completed=True)
        assert task.mark_complete() is False

    def test_create_next_occurrence_daily(self):
        """A daily task produces a next occurrence one day later."""
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
        """A weekly task produces a next occurrence seven days later."""
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
        """A one-time task does not recur."""
        task = Task(description="Vet visit", time="11:00", duration_minutes=60, frequency="once")
        assert task.create_next_occurrence() is None

    def test_round_trips_through_dict(self):
        """Every Task field survives to_dict() -> from_dict()."""
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
        restored = Task.from_dict(task.to_dict())
        assert restored.description == task.description
        assert restored.time == task.time
        assert restored.duration_minutes == task.duration_minutes
        assert restored.priority == task.priority
        assert restored.frequency == task.frequency
        assert restored.completed == task.completed
        assert restored.due_date == task.due_date
        assert restored.pet_name == task.pet_name


class TestPet:
    """Tests for the Pet dataclass."""

    def test_add_task_increases_count(self):
        """Adding a task grows the pet's task list."""
        pet = Pet(name="Mochi", species="dog")
        assert len(pet.tasks) == 0
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        assert len(pet.tasks) == 1

    def test_add_task_sets_pet_name(self):
        """Adding a task back-fills the task's pet_name."""
        pet = Pet(name="Luna", species="cat")
        task = Task(description="Play", time="14:00", duration_minutes=20)
        pet.add_task(task)
        assert task.pet_name == "Luna"

    def test_remove_task(self):
        """Removing by description drops the matching task."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        assert pet.remove_task("Walk") is True
        assert len(pet.tasks) == 0

    def test_remove_task_not_found(self):
        """Removing a task that does not exist reports False."""
        pet = Pet(name="Mochi", species="dog")
        assert pet.remove_task("Nonexistent") is False

    def test_get_pending_tasks(self):
        """get_pending_tasks() excludes completed tasks."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30, completed=False))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=10, completed=True))
        pending = pet.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0].description == "Walk"

    def test_round_trips_through_dict(self):
        """A pet and its nested tasks survive dict serialization."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="09:00", duration_minutes=30))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=10))
        restored = Pet.from_dict(pet.to_dict())
        assert restored.name == "Mochi"
        assert restored.species == "dog"
        assert len(restored.tasks) == 2
        assert restored.tasks[0].pet_name == "Mochi"


class TestOwner:
    """Tests for the Owner dataclass and its JSON persistence."""

    def test_add_pet(self, owner: Owner):
        """Adding a pet grows the owner's collection."""
        owner.add_pet(Pet(name="Mochi", species="dog"))
        assert len(owner.pets) == 1

    def test_get_all_tasks_across_pets(self, owner_with_pets: Owner):
        """get_all_tasks() aggregates across every pet."""
        owner_with_pets.find_pet("Mochi").add_task(
            Task(description="Walk", time="09:00", duration_minutes=30)
        )
        owner_with_pets.find_pet("Luna").add_task(
            Task(description="Play", time="14:00", duration_minutes=20)
        )
        assert len(owner_with_pets.get_all_tasks()) == 2

    def test_save_and_load_json(self, owner_with_pets: Owner, tmp_path):
        """The full owner/pet/task graph survives a JSON round-trip."""
        filepath = str(tmp_path / "test_data.json")
        owner_with_pets.find_pet("Mochi").add_task(
            Task(
                description="Walk",
                time="07:30",
                duration_minutes=30,
                priority="high",
                frequency="daily",
            )
        )
        owner_with_pets.find_pet("Luna").add_task(
            Task(description="Play", time="14:00", duration_minutes=20)
        )

        owner_with_pets.save_to_json(filepath)
        assert os.path.exists(filepath)

        loaded = Owner.load_from_json(filepath)
        assert loaded.name == "Jordan"
        assert len(loaded.pets) == 2
        assert loaded.pets[0].name == "Mochi"
        assert len(loaded.pets[0].tasks) == 1
        assert loaded.pets[0].tasks[0].description == "Walk"
        assert loaded.pets[0].tasks[0].priority == "high"
        assert loaded.pets[0].tasks[0].frequency == "daily"
        assert loaded.pets[1].name == "Luna"

    def test_load_json_preserves_task_dates(self, owner: Owner, tmp_path):
        """due_date survives JSON serialization as a date, not a string."""
        filepath = str(tmp_path / "test_dates.json")
        pet = Pet(name="Mochi", species="dog")
        today = date.today()
        pet.add_task(Task(description="Walk", time="07:30", duration_minutes=30, due_date=today))
        owner.add_pet(pet)

        owner.save_to_json(filepath)
        loaded = Owner.load_from_json(filepath)
        assert loaded.pets[0].tasks[0].due_date == today

    def test_load_json_missing_file_returns_none(self, tmp_path):
        """Loading a nonexistent file returns None rather than raising."""
        assert Owner.load_from_json(str(tmp_path / "does_not_exist.json")) is None

    def test_save_path_travels_with_the_owner(self, owner: Owner, tmp_path):
        """A later no-argument save reuses the path, not the "data.json" default.

        The tool layer calls owner.save_to_json() with no arguments, so the
        chosen path has to live on the object or writes escape to the cwd.
        """
        filepath = str(tmp_path / "custom.json")
        owner.save_to_json(filepath)
        assert owner.data_path == filepath

        owner.add_pet(Pet(name="Mochi", species="dog"))
        owner.save_to_json()
        assert Owner.load_from_json(filepath).pets[0].name == "Mochi"

    def test_loaded_owner_remembers_its_path(self, owner: Owner, tmp_path):
        """load_from_json() binds the owner to the file it came from."""
        filepath = str(tmp_path / "custom.json")
        owner.save_to_json(filepath)
        assert Owner.load_from_json(filepath).data_path == filepath


class TestScheduler:
    """Tests for sorting, filtering, and conflict detection."""

    def test_sort_by_time_chronological(self, busy_scheduler: Scheduler):
        """sort_by_time() returns tasks in ascending time order."""
        times = [t.time for t in busy_scheduler.sort_by_time()]
        assert times == sorted(times)

    def test_sort_by_priority_descending(self, busy_scheduler: Scheduler):
        """sort_by_priority() puts high priority first and low priority last."""
        sorted_tasks = busy_scheduler.sort_by_priority()
        assert sorted_tasks[0].priority == "high"
        assert sorted_tasks[-1].priority == "low"

    def test_filter_by_pet(self, busy_scheduler: Scheduler):
        """filter_by_pet() returns only that pet's tasks."""
        mochi_tasks = busy_scheduler.filter_by_pet("Mochi")
        assert len(mochi_tasks) == 2
        assert all(t.pet_name == "Mochi" for t in mochi_tasks)

    def test_filter_by_status(self, busy_scheduler: Scheduler):
        """filter_by_status() respects the requested completion state."""
        assert all(not t.completed for t in busy_scheduler.filter_by_status(completed=False))

    def test_detect_conflicts_flags_same_time(self, busy_scheduler: Scheduler):
        """Two tasks sharing an HH:MM slot are reported as a conflict."""
        conflicts = busy_scheduler.detect_conflicts()
        assert len(conflicts) >= 1
        assert "08:00" in conflicts[0]

    def test_detect_conflicts_no_false_positives(self, owner: Owner):
        """Distinct time slots are not reported as conflicts."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="07:00", duration_minutes=30))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=10))
        owner.add_pet(pet)
        assert Scheduler(owner=owner).detect_conflicts() == []

    def test_mark_complete_recurring_creates_next(self, owner: Owner):
        """Completing a daily task appends the next occurrence to the same pet."""
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

    def test_generate_schedule_pending_only(self, busy_scheduler: Scheduler):
        """generate_schedule() omits completed tasks."""
        busy_scheduler.get_all_tasks()[0].mark_complete()
        assert all(not t.completed for t in busy_scheduler.generate_schedule())

    def test_empty_pet_no_crash(self, owner: Owner):
        """A pet with no tasks does not break any scheduler operation."""
        owner.add_pet(Pet(name="Ghost", species="hamster"))
        scheduler = Scheduler(owner=owner)
        assert scheduler.generate_schedule() == []
        assert scheduler.detect_conflicts() == []
        assert scheduler.sort_by_time() == []


class TestFindNextAvailableSlot:
    """Tests for the duration-aware slot finder."""

    def test_empty_schedule_returns_start_of_day(self, scheduler: Scheduler):
        """With nothing booked, the first slot is the 07:00 window start."""
        assert scheduler.find_next_available_slot(30) == "07:00"

    def test_skips_occupied_slot(self, owner: Owner):
        """A booked 07:00-07:30 pushes the suggestion to 07:30."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="07:00", duration_minutes=30))
        owner.add_pet(pet)
        assert Scheduler(owner=owner).find_next_available_slot(30) == "07:30"

    def test_respects_requested_duration(self, owner: Owner):
        """A 60-minute request skips the 30-minute gap between two bookings."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(Task(description="Walk", time="07:00", duration_minutes=30))
        pet.add_task(Task(description="Feed", time="08:00", duration_minutes=30))
        owner.add_pet(pet)
        assert Scheduler(owner=owner).find_next_available_slot(60) == "08:30"

    def test_fully_booked_day_returns_none(self, owner: Owner):
        """When every 30-minute slot from 07:00 to 21:00 is taken, return None."""
        pet = Pet(name="Mochi", species="dog")
        for hour in range(7, 21):
            for minute in (0, 30):
                pet.add_task(
                    Task(
                        description=f"Task {hour}:{minute:02d}",
                        time=f"{hour:02d}:{minute:02d}",
                        duration_minutes=30,
                    )
                )
        owner.add_pet(pet)
        assert Scheduler(owner=owner).find_next_available_slot(30) is None

    def test_completed_tasks_do_not_block(self, owner: Owner):
        """A completed task frees up its time slot."""
        pet = Pet(name="Mochi", species="dog")
        pet.add_task(
            Task(description="Walk", time="07:00", duration_minutes=30, completed=True)
        )
        owner.add_pet(pet)
        assert Scheduler(owner=owner).find_next_available_slot(30) == "07:00"
