"""
Tool definitions for PawPal+ Agent.

Each tool wraps an existing PawPal+ function with a standardized interface
so the Agent can discover and invoke them through function calling.
"""

from datetime import date
from pawpal_system import Task, Pet, Owner, Scheduler


# --- Tool definitions (schema + implementation) ---

TOOL_DEFINITIONS = [
    {
        "name": "add_pet",
        "description": "Add a new pet to the owner's collection. Use this when the user wants to register a new pet.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The pet's name"},
                "species": {
                    "type": "string",
                    "description": "The pet's species (dog, cat, bird, hamster, other)",
                },
            },
            "required": ["name", "species"],
        },
    },
    {
        "name": "add_task",
        "description": "Add a care task for a specific pet. Use this when the user wants to schedule feeding, walking, medication, grooming, vet visits, or any other pet care activity.",
        "parameters": {
            "type": "object",
            "properties": {
                "pet_name": {"type": "string", "description": "Name of the pet this task is for"},
                "description": {"type": "string", "description": "What the task involves"},
                "time": {"type": "string", "description": "Scheduled time in HH:MM format"},
                "duration_minutes": {"type": "integer", "description": "How long the task takes in minutes"},
                "priority": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "description": "Task priority level",
                },
                "frequency": {
                    "type": "string",
                    "enum": ["once", "daily", "weekly"],
                    "description": "How often the task repeats",
                },
            },
            "required": ["pet_name", "description", "time", "duration_minutes"],
        },
    },
    {
        "name": "complete_task",
        "description": "Mark a task as completed. If the task is recurring (daily/weekly), a new occurrence is automatically created. Use this when the user says they finished a task.",
        "parameters": {
            "type": "object",
            "properties": {
                "pet_name": {"type": "string", "description": "Name of the pet"},
                "task_description": {"type": "string", "description": "Description of the task to complete"},
            },
            "required": ["pet_name", "task_description"],
        },
    },
    {
        "name": "get_schedule",
        "description": "Get today's schedule for all pets, sorted by priority then time. Use this when the user asks what needs to be done today or wants to see the daily plan.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_pet_tasks",
        "description": "Get all tasks for a specific pet. Use this when the user asks about a particular pet's schedule or tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "pet_name": {"type": "string", "description": "Name of the pet"},
                "pending_only": {
                    "type": "boolean",
                    "description": "If true, only return incomplete tasks",
                },
            },
            "required": ["pet_name"],
        },
    },
    {
        "name": "detect_conflicts",
        "description": "Check for scheduling conflicts where multiple tasks overlap in time. Use this proactively when adding tasks or when the user asks about conflicts.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "suggest_time_slot",
        "description": "Find the next available time slot for a task of a given duration. Use this when the user wants to schedule something but hasn't specified a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration_minutes": {
                    "type": "integer",
                    "description": "How long the task needs in minutes",
                },
            },
            "required": ["duration_minutes"],
        },
    },
    {
        "name": "search_care_info",
        "description": "Search the pet care knowledge base for information about feeding, health, grooming, training, or general pet care. Use this when the user asks a care-related question.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The care question or topic to search for",
                },
            },
            "required": ["query"],
        },
    },
]


def execute_tool(
    tool_name: str,
    arguments: dict,
    owner: Owner,
    scheduler: Scheduler,
    knowledge_base=None,
) -> str:
    """Execute a tool by name with the given arguments.

    Returns a string describing the result for the Agent to interpret.
    """
    try:
        if tool_name == "add_pet":
            return _add_pet(owner, **arguments)
        elif tool_name == "add_task":
            return _add_task(owner, scheduler, **arguments)
        elif tool_name == "complete_task":
            return _complete_task(owner, scheduler, **arguments)
        elif tool_name == "get_schedule":
            return _get_schedule(scheduler)
        elif tool_name == "get_pet_tasks":
            return _get_pet_tasks(owner, **arguments)
        elif tool_name == "detect_conflicts":
            return _detect_conflicts(scheduler)
        elif tool_name == "suggest_time_slot":
            return _suggest_time_slot(scheduler, **arguments)
        elif tool_name == "search_care_info":
            if knowledge_base is None:
                return "Knowledge base not available."
            return knowledge_base.search(arguments["query"])
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing {tool_name}: {e}"


# --- Tool implementations ---


def _add_pet(owner: Owner, name: str, species: str) -> str:
    """Add a new pet to the owner."""
    if owner.find_pet(name) is not None:
        return f"A pet named '{name}' already exists."
    owner.add_pet(Pet(name=name, species=species))
    try:
        owner.save_to_json()
    except Exception:
        pass  # Persistence is best-effort; state still updated in memory
    return f"Added {name} the {species}."


def _add_task(
    owner: Owner,
    scheduler: Scheduler,
    pet_name: str,
    description: str,
    time: str,
    duration_minutes: int,
    priority: str = "medium",
    frequency: str = "once",
) -> str:
    """Add a task for a specific pet."""
    pet = owner.find_pet(pet_name)
    if pet is None:
        return f"Pet '{pet_name}' not found. Current pets: {', '.join(p.name for p in owner.pets)}."

    task = Task(
        description=description,
        time=time,
        duration_minutes=duration_minutes,
        priority=priority,
        frequency=frequency,
    )
    pet.add_task(task)
    try:
        owner.save_to_json()
    except Exception:
        pass

    # Proactively check for conflicts after adding
    conflicts = scheduler.detect_conflicts()
    result = f"Added task '{description}' for {pet_name} at {time}."
    if conflicts:
        result += f" Warning: {'; '.join(conflicts)}"
    return result


def _complete_task(
    owner: Owner, scheduler: Scheduler, pet_name: str, task_description: str
) -> str:
    """Mark a task as completed."""
    pet = owner.find_pet(pet_name)
    if pet is None:
        return f"Pet '{pet_name}' not found."

    for task in pet.tasks:
        if task.description.lower() == task_description.lower() and not task.completed:
            next_task = scheduler.mark_task_complete(task)
            try:
                owner.save_to_json()
            except Exception:
                pass
            result = f"Completed '{task_description}' for {pet_name}."
            if next_task:
                result += f" Next occurrence scheduled for {next_task.due_date}."
            return result

    return f"No pending task '{task_description}' found for {pet_name}."


def _get_schedule(scheduler: Scheduler) -> str:
    """Get today's schedule."""
    schedule = scheduler.generate_schedule()
    if not schedule:
        return "No tasks scheduled for today."

    lines = [f"Today's schedule ({date.today()}):", ""]
    for i, task in enumerate(schedule, 1):
        lines.append(f"  {i}. {task}")
    return "\n".join(lines)


def _get_pet_tasks(owner: Owner, pet_name: str, pending_only: bool = False) -> str:
    """Get tasks for a specific pet."""
    pet = owner.find_pet(pet_name)
    if pet is None:
        return f"Pet '{pet_name}' not found."

    tasks = pet.get_pending_tasks() if pending_only else pet.tasks
    if not tasks:
        label = "pending tasks" if pending_only else "tasks"
        return f"No {label} for {pet_name}."

    lines = [f"Tasks for {pet_name}:", ""]
    for i, task in enumerate(tasks, 1):
        lines.append(f"  {i}. {task}")
    return "\n".join(lines)


def _detect_conflicts(scheduler: Scheduler) -> str:
    """Check for scheduling conflicts."""
    conflicts = scheduler.detect_conflicts()
    if not conflicts:
        return "No scheduling conflicts detected."
    return "Conflicts found:\n" + "\n".join(f"  - {c}" for c in conflicts)


def _suggest_time_slot(scheduler: Scheduler, duration_minutes: int) -> str:
    """Find the next available time slot."""
    slot = scheduler.find_next_available_slot(duration_minutes)
    if slot is None:
        return f"No available {duration_minutes}-minute slot found between 07:00 and 21:00."
    return f"Suggested time slot: {slot} ({duration_minutes} minutes available)."
