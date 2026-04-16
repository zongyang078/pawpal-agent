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

        delta = timedelta(days=1) if self.frequency == "daily" else timedelta(days=7)
        return Task(
            description=self.description,
            time=self.time,
            duration_minutes=self.duration_minutes,
            priority=self.priority,
            frequency=self.frequency,
            completed=False,
            due_date=self.due_date + delta,
            pet_name=self.pet_name,
        )

    def to_dict(self) -> dict:
        """Serialize task to a dictionary for JSON storage."""
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
        """Deserialize a task from a dictionary."""
        data["due_date"] = date.fromisoformat(data["due_date"])
        return cls(**data)

    def __str__(self) -> str:
        status = "done" if self.completed else "pending"
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
            self.priority, ""
        )
        owner_label = f" [{self.pet_name}]" if self.pet_name else ""
        return (
            f"{priority_emoji} {self.time} - {self.description}"
            f" ({self.duration_minutes}min, {self.frequency}, {status}){owner_label}"
        )


@dataclass
class Pet:
    """Stores pet details and manages its task list."""

    name: str
    species: str
    tasks: list[Task] = field(default_factory=list)

    def add_task(self, task: Task) -> None:
        """Add a task and tag it with this pet's name."""
        task.pet_name = self.name
        self.tasks.append(task)

    def remove_task(self, description: str) -> bool:
        """Remove the first task matching the description. Returns True if found."""
        for i, task in enumerate(self.tasks):
            if task.description == description:
                self.tasks.pop(i)
                return True
        return False

    def get_pending_tasks(self) -> list[Task]:
        """Return all incomplete tasks for this pet."""
        return [t for t in self.tasks if not t.completed]

    def to_dict(self) -> dict:
        """Serialize pet to a dictionary."""
        return {
            "name": self.name,
            "species": self.species,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pet":
        """Deserialize a pet from a dictionary."""
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        return cls(name=data["name"], species=data["species"], tasks=tasks)

    def __str__(self) -> str:
        return f"{self.name} ({self.species}) - {len(self.tasks)} task(s)"


@dataclass
class Owner:
    """Manages multiple pets and aggregates tasks across all of them."""

    name: str
    pets: list[Pet] = field(default_factory=list)

    def add_pet(self, pet: Pet) -> None:
        """Add a pet to this owner's collection."""
        self.pets.append(pet)

    def find_pet(self, name: str) -> Optional[Pet]:
        """Find a pet by name (case-insensitive). Returns None if not found."""
        for pet in self.pets:
            if pet.name.lower() == name.lower():
                return pet
        return None

    def get_all_tasks(self) -> list[Task]:
        """Aggregate all tasks from all pets."""
        return [task for pet in self.pets for task in pet.tasks]

    def to_dict(self) -> dict:
        """Serialize owner to a dictionary."""
        return {
            "name": self.name,
            "pets": [p.to_dict() for p in self.pets],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Owner":
        """Deserialize an owner from a dictionary."""
        pets = [Pet.from_dict(p) for p in data.get("pets", [])]
        owner = cls(name=data["name"])
        owner.pets = pets
        return owner

    def save_to_json(self, filepath: str = "data.json") -> None:
        """Save the entire owner/pet/task graph to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str = "data.json") -> Optional["Owner"]:
        """Load an owner from a JSON file. Returns None if file not found."""
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except FileNotFoundError:
            return None

    def __str__(self) -> str:
        return f"Owner: {self.name} ({len(self.pets)} pet(s))"


@dataclass
class Scheduler:
    """Orchestration layer for sorting, filtering, conflict detection, and scheduling."""

    owner: Owner

    def get_all_tasks(self) -> list[Task]:
        """Shortcut to get all tasks from the owner."""
        return self.owner.get_all_tasks()

    def sort_by_time(self) -> list[Task]:
        """Return all tasks sorted by time (ascending)."""
        return sorted(self.get_all_tasks(), key=lambda t: t.time)

    def sort_by_priority(self) -> list[Task]:
        """Return all tasks sorted by priority (high first), then by time."""
        return sorted(
            self.get_all_tasks(),
            key=lambda t: (-PRIORITY_ORDER.get(t.priority, 0), t.time),
        )

    def filter_by_pet(self, pet_name: str) -> list[Task]:
        """Return tasks belonging to a specific pet."""
        return [t for t in self.get_all_tasks() if t.pet_name.lower() == pet_name.lower()]

    def filter_by_status(self, completed: bool = False) -> list[Task]:
        """Return tasks filtered by completion status."""
        return [t for t in self.get_all_tasks() if t.completed == completed]

    def detect_conflicts(self) -> list[str]:
        """Detect tasks scheduled at the same time. Returns list of warning strings."""
        time_map: dict[str, list[Task]] = {}
        for task in self.get_all_tasks():
            if not task.completed:
                time_map.setdefault(task.time, []).append(task)

        warnings = []
        for time_slot, tasks in time_map.items():
            if len(tasks) > 1:
                names = ", ".join(f"{t.description} [{t.pet_name}]" for t in tasks)
                warnings.append(f"Conflict at {time_slot}: {names}")
        return warnings

    def mark_task_complete(self, task: Task) -> Optional[Task]:
        """Complete a task and auto-create its next occurrence if recurring."""
        task.mark_complete()
        next_task = task.create_next_occurrence()

        if next_task is not None:
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
        return sorted(
            pending,
            key=lambda t: (-PRIORITY_ORDER.get(t.priority, 0), t.time),
        )

    def find_next_available_slot(self, duration_minutes: int) -> Optional[str]:
        """Find the next available time slot that fits the given duration.

        Scans from 07:00 to 21:00 in 30-minute increments, using minute-based
        overlap detection to find a gap large enough for the requested duration.
        """
        occupied = []
        for task in self.get_all_tasks():
            if not task.completed:
                h, m = map(int, task.time.split(":"))
                start = h * 60 + m
                occupied.append((start, start + task.duration_minutes))

        # Scan from 07:00 (420 min) to 21:00 (1260 min) in 30-min steps
        for candidate in range(420, 1260, 30):
            candidate_end = candidate + duration_minutes
            if candidate_end > 1260:
                break
            conflict = False
            for occ_start, occ_end in occupied:
                if candidate < occ_end and candidate_end > occ_start:
                    conflict = True
                    break
            if not conflict:
                return f"{candidate // 60:02d}:{candidate % 60:02d}"

        return None
