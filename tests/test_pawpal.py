"""
Automated test suite for PawPal+ system.
Covers: task completion, task addition, sorting, recurring logic, and conflict detection.
"""

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
        # Feed Mochi and Feed Luna are both at 08:00
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
        # Complete one task
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
