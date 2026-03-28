"""
PawPal+ Pet Care Management System
Logic layer containing all backend classes: Task, Pet, Owner, Scheduler.
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional


# Priority ranking for sorting (higher number = higher priority)
PRIORITY_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass
class Task:
    """Represents a single pet care activity."""

    description: str
    time: str  # HH:MM format
    duration_minutes: int
    priority: str = "medium"  # low, medium, high
    frequency: str = "once"  # once, daily, weekly
    completed: bool = False
    due_date: date = field(default_factory=date.today)
    pet_name: str = ""  # Track which pet this task belongs to

    def mark_complete(self) -> bool:
        """Mark this task as completed. Returns True if status changed."""
        if self.completed:
            return False
        self.completed = True
        return True

    def create_next_occurrence(self) -> Optional["Task"]:
        """For recurring tasks, create the next occurrence. Returns None if not recurring."""
        if self.frequency == "once":
            return None

        days_ahead = 1 if self.frequency == "daily" else 7
        next_date = self.due_date + timedelta(days=days_ahead)

        return Task(
            description=self.description,
            time=self.time,
            duration_minutes=self.duration_minutes,
            priority=self.priority,
            frequency=self.frequency,
            completed=False,
            due_date=next_date,
            pet_name=self.pet_name,
        )

    def to_dict(self) -> dict:
        """Convert task to a JSON-serializable dictionary."""
        return {
            "description": self.description,
            "time": self.time,
            "duration_minutes": self.duration_minutes,
            "priority": self.priority,
            "frequency": self.frequency,
            "completed": self.completed,
            "due_date": self.due_date.isoformat(),
            "pet_name": self.pet_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Create a Task from a dictionary (loaded from JSON)."""
        return cls(
            description=data["description"],
            time=data["time"],
            duration_minutes=data["duration_minutes"],
            priority=data.get("priority", "medium"),
            frequency=data.get("frequency", "once"),
            completed=data.get("completed", False),
            due_date=date.fromisoformat(data["due_date"]),
            pet_name=data.get("pet_name", ""),
        )

    def __str__(self) -> str:
        status = "Done" if self.completed else "Pending"
        return (
            f"[{status}] {self.time} - {self.description} "
            f"({self.duration_minutes}min, {self.priority} priority, "
            f"{self.frequency}, due {self.due_date})"
        )


@dataclass
class Pet:
    """Stores pet details and manages its task list."""

    name: str
    species: str
    tasks: list[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Add a task to this pet's task list."""
        task.pet_name = self.name
        self.tasks.append(task)

    def remove_task(self, description: str) -> bool:
        """Remove a task by description. Returns True if found and removed."""
        for i, task in enumerate(self.tasks):
            if task.description == description:
                self.tasks.pop(i)
                return True
        return False

    def get_pending_tasks(self) -> list[Task]:
        """Return all tasks that are not yet completed."""
        return [t for t in self.tasks if not t.completed]

    def to_dict(self) -> dict:
        """Convert pet to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "species": self.species,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pet":
        """Create a Pet from a dictionary (loaded from JSON)."""
        pet = cls(name=data["name"], species=data["species"])
        for task_data in data.get("tasks", []):
            task = Task.from_dict(task_data)
            task.pet_name = pet.name
            pet.tasks.append(task)
        return pet

    def __str__(self) -> str:
        return f"{self.name} ({self.species}) - {len(self.tasks)} task(s)"


@dataclass
class Owner:
    """Manages multiple pets and provides access to all their tasks."""

    name: str
    pets: list[Pet] = field(default_factory=list)

    def add_pet(self, pet: Pet) -> None:
        """Add a pet to the owner's pet list."""
        self.pets.append(pet)

    def remove_pet(self, name: str) -> bool:
        """Remove a pet by name. Returns True if found and removed."""
        for i, pet in enumerate(self.pets):
            if pet.name == name:
                self.pets.pop(i)
                return True
        return False

    def get_all_tasks(self) -> list[Task]:
        """Collect and return all tasks across all pets."""
        all_tasks = []
        for pet in self.pets:
            all_tasks.extend(pet.tasks)
        return all_tasks

    def find_pet(self, name: str) -> Optional[Pet]:
        """Find a pet by name. Returns None if not found."""
        for pet in self.pets:
            if pet.name == name:
                return pet
        return None

    def to_dict(self) -> dict:
        """Convert owner to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "pets": [p.to_dict() for p in self.pets],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Owner":
        """Create an Owner from a dictionary (loaded from JSON)."""
        owner = cls(name=data["name"])
        for pet_data in data.get("pets", []):
            owner.pets.append(Pet.from_dict(pet_data))
        return owner

    def save_to_json(self, filepath: str = "data.json") -> None:
        """Save the owner, pets, and tasks to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str = "data.json") -> "Owner":
        """Load an Owner (with pets and tasks) from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __str__(self) -> str:
        pet_names = ", ".join(p.name for p in self.pets) if self.pets else "No pets"
        return f"{self.name} - Pets: {pet_names}"


class Scheduler:
    """The scheduling brain — retrieves, organizes, and manages tasks across pets."""

    def __init__(self, owner: Owner):
        """Initialize scheduler with an owner."""
        self.owner = owner

    def get_all_tasks(self) -> list[Task]:
        """Get all tasks from the owner's pets."""
        return self.owner.get_all_tasks()

    def sort_by_time(self, tasks: Optional[list[Task]] = None) -> list[Task]:
        """Sort tasks by their scheduled time (HH:MM)."""
        target = tasks if tasks is not None else self.get_all_tasks()
        return sorted(target, key=lambda t: t.time)

    def sort_by_priority(self, tasks: Optional[list[Task]] = None) -> list[Task]:
        """Sort tasks by priority: high > medium > low (descending)."""
        target = tasks if tasks is not None else self.get_all_tasks()
        return sorted(
            target,
            key=lambda t: PRIORITY_ORDER.get(t.priority, 0),
            reverse=True,
        )

    def filter_by_pet(self, pet_name: str) -> list[Task]:
        """Return tasks belonging to a specific pet."""
        return [t for t in self.get_all_tasks() if t.pet_name == pet_name]

    def filter_by_status(self, completed: bool = False) -> list[Task]:
        """Filter tasks by completion status."""
        return [t for t in self.get_all_tasks() if t.completed == completed]

    def detect_conflicts(self) -> list[str]:
        """Detect tasks scheduled at the same time. Returns warning messages."""
        warnings = []
        tasks = self.get_all_tasks()
        seen: dict[str, list[Task]] = {}

        for task in tasks:
            if task.completed:
                continue
            if task.time not in seen:
                seen[task.time] = []
            seen[task.time].append(task)

        for time_slot, conflicting in seen.items():
            if len(conflicting) > 1:
                names = ", ".join(
                    f"'{t.description}' ({t.pet_name})" for t in conflicting
                )
                warnings.append(
                    f"Conflict at {time_slot}: {names} are scheduled at the same time."
                )

        return warnings

    def find_next_available_slot(
        self, duration_minutes: int, start_time: str = "07:00", end_time: str = "21:00"
    ) -> Optional[str]:
        """Find the next available time slot that fits a task of the given duration.

        Scans from start_time to end_time in 30-minute increments and returns
        the first slot where no existing pending task overlaps. Returns None if
        no slot is available.

        This algorithm converts HH:MM strings to minutes-since-midnight for
        easy arithmetic, then checks each candidate slot against all occupied
        time ranges (task.time to task.time + task.duration_minutes).
        """
        def to_minutes(hhmm: str) -> int:
            h, m = hhmm.split(":")
            return int(h) * 60 + int(m)

        def to_hhmm(minutes: int) -> str:
            return f"{minutes // 60:02d}:{minutes % 60:02d}"

        # Build list of occupied ranges from pending tasks
        occupied: list[tuple[int, int]] = []
        for task in self.get_all_tasks():
            if task.completed:
                continue
            start = to_minutes(task.time)
            occupied.append((start, start + task.duration_minutes))

        # Scan in 30-minute increments
        candidate = to_minutes(start_time)
        latest = to_minutes(end_time)

        while candidate + duration_minutes <= latest:
            candidate_end = candidate + duration_minutes
            conflict = False
            for occ_start, occ_end in occupied:
                # Check for overlap: two ranges overlap if one starts before the other ends
                if candidate < occ_end and candidate_end > occ_start:
                    conflict = True
                    break
            if not conflict:
                return to_hhmm(candidate)
            candidate += 30

        return None

    def mark_task_complete(self, task: Task) -> Optional[Task]:
        """Mark a task complete. If recurring, create and return the next occurrence."""
        task.mark_complete()
        next_task = task.create_next_occurrence()

        if next_task is not None:
            # Find the pet this task belongs to and add the next occurrence
            pet = self.owner.find_pet(task.pet_name)
            if pet:
                pet.add_task(next_task)

        return next_task

    def generate_schedule(self) -> list[Task]:
        """Generate today's schedule: pending tasks sorted by priority then time."""
        today = date.today()
        pending = [
            t
            for t in self.get_all_tasks()
            if not t.completed and t.due_date <= today
        ]
        # Sort by priority (high first), then by time within same priority
        return sorted(
            pending,
            key=lambda t: (-PRIORITY_ORDER.get(t.priority, 0), t.time),
        )
