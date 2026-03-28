"""
PawPal+ CLI Demo Script
Demonstrates the backend logic by creating sample data and exercising all features.
"""

from datetime import date
from pawpal_system import Task, Pet, Owner, Scheduler


def main():
    # --- Setup: Create owner and pets ---
    owner = Owner(name="Jordan")

    mochi = Pet(name="Mochi", species="dog")
    luna = Pet(name="Luna", species="cat")

    owner.add_pet(mochi)
    owner.add_pet(luna)

    # --- Add tasks with different times, priorities, and frequencies ---
    mochi.add_task(Task(
        description="Morning walk",
        time="07:30",
        duration_minutes=30,
        priority="high",
        frequency="daily",
    ))
    mochi.add_task(Task(
        description="Breakfast feeding",
        time="08:00",
        duration_minutes=10,
        priority="high",
        frequency="daily",
    ))
    mochi.add_task(Task(
        description="Flea medication",
        time="09:00",
        duration_minutes=5,
        priority="medium",
        frequency="weekly",
    ))

    luna.add_task(Task(
        description="Breakfast feeding",
        time="08:00",
        duration_minutes=10,
        priority="high",
        frequency="daily",
    ))
    luna.add_task(Task(
        description="Play session",
        time="14:00",
        duration_minutes=20,
        priority="medium",
        frequency="daily",
    ))
    luna.add_task(Task(
        description="Vet appointment",
        time="10:30",
        duration_minutes=60,
        priority="high",
        frequency="once",
    ))

    # --- Initialize Scheduler ---
    scheduler = Scheduler(owner=owner)

    # --- Feature 1: Generate today's schedule (sorted by priority then time) ---
    print("=" * 60)
    print(f"  PawPal+ Daily Schedule for {owner.name}")
    print(f"  Date: {date.today()}")
    print("=" * 60)

    schedule = scheduler.generate_schedule()
    for i, task in enumerate(schedule, 1):
        print(f"  {i}. {task}")
    print()

    # --- Feature 2: Sort by time ---
    print("-" * 60)
    print("  All tasks sorted by time:")
    print("-" * 60)
    for task in scheduler.sort_by_time():
        print(f"  {task}")
    print()

    # --- Feature 3: Filter by pet ---
    print("-" * 60)
    print("  Tasks for Mochi only:")
    print("-" * 60)
    for task in scheduler.filter_by_pet("Mochi"):
        print(f"  {task}")
    print()

    # --- Feature 4: Conflict detection ---
    print("-" * 60)
    print("  Conflict warnings:")
    print("-" * 60)
    conflicts = scheduler.detect_conflicts()
    if conflicts:
        for warning in conflicts:
            print(f"  ⚠️  {warning}")
    else:
        print("  No conflicts detected.")
    print()

    # --- Feature 5: Mark task complete + recurring task generation ---
    print("-" * 60)
    print("  Completing Mochi's morning walk...")
    print("-" * 60)
    walk_task = mochi.tasks[0]
    next_task = scheduler.mark_task_complete(walk_task)
    print(f"  Completed: {walk_task}")
    if next_task:
        print(f"  Next occurrence created: {next_task}")
    print()

    # --- Feature 6: Filter by status ---
    print("-" * 60)
    print("  Remaining pending tasks:")
    print("-" * 60)
    pending = scheduler.filter_by_status(completed=False)
    for task in pending:
        print(f"  {task}")
    print()

    print("=" * 60)
    print("  Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
