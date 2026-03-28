"""
PawPal+ Streamlit App
Interactive pet care management UI connected to the backend logic layer.
"""

import streamlit as st
from datetime import date
from pawpal_system import Task, Pet, Owner, Scheduler


# --- Page config ---
st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")
st.title("🐾 PawPal+")
st.caption("A smart pet care management system")

# --- Session state initialization ---
if "owner" not in st.session_state:
    st.session_state.owner = None
if "scheduler" not in st.session_state:
    st.session_state.scheduler = None


# ============================================================
# Section 1: Owner Setup
# ============================================================
st.subheader("Owner Setup")

owner_name = st.text_input("Owner name", value="Jordan")

if st.session_state.owner is None or st.session_state.owner.name != owner_name:
    st.session_state.owner = Owner(name=owner_name)
    st.session_state.scheduler = Scheduler(owner=st.session_state.owner)

owner = st.session_state.owner
scheduler = st.session_state.scheduler

st.divider()


# ============================================================
# Section 2: Manage Pets
# ============================================================
st.subheader("Manage Pets")

col1, col2 = st.columns(2)
with col1:
    new_pet_name = st.text_input("Pet name", value="Mochi")
with col2:
    new_pet_species = st.selectbox("Species", ["dog", "cat", "bird", "hamster", "other"])

if st.button("Add Pet"):
    if new_pet_name.strip():
        # Check for duplicate pet name
        if owner.find_pet(new_pet_name) is not None:
            st.warning(f"A pet named '{new_pet_name}' already exists.")
        else:
            owner.add_pet(Pet(name=new_pet_name, species=new_pet_species))
            st.success(f"Added {new_pet_name} the {new_pet_species}!")
    else:
        st.warning("Please enter a pet name.")

if owner.pets:
    st.markdown("**Your pets:**")
    for pet in owner.pets:
        st.write(f"- {pet}")
else:
    st.info("No pets yet. Add one above!")

st.divider()


# ============================================================
# Section 3: Add Tasks
# ============================================================
st.subheader("Add Tasks")

if not owner.pets:
    st.info("Add a pet first before scheduling tasks.")
else:
    pet_names = [p.name for p in owner.pets]

    col1, col2 = st.columns(2)
    with col1:
        task_pet = st.selectbox("Assign to pet", pet_names)
        task_desc = st.text_input("Task description", value="Morning walk")
        task_time = st.text_input("Time (HH:MM)", value="08:00")
    with col2:
        task_duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=20)
        task_priority = st.selectbox("Priority", ["high", "medium", "low"])
        task_frequency = st.selectbox("Frequency", ["once", "daily", "weekly"])

    if st.button("Add Task"):
        if task_desc.strip() and task_time.strip():
            pet = owner.find_pet(task_pet)
            if pet:
                new_task = Task(
                    description=task_desc,
                    time=task_time,
                    duration_minutes=task_duration,
                    priority=task_priority,
                    frequency=task_frequency,
                    due_date=date.today(),
                )
                pet.add_task(new_task)
                st.success(f"Added '{task_desc}' for {task_pet} at {task_time}.")
        else:
            st.warning("Please fill in task description and time.")

st.divider()


# ============================================================
# Section 4: Today's Schedule
# ============================================================
st.subheader("Today's Schedule")

if st.button("Generate Schedule"):
    schedule = scheduler.generate_schedule()

    if not schedule:
        st.info("No tasks scheduled for today.")
    else:
        # Show conflict warnings first
        conflicts = scheduler.detect_conflicts()
        for warning in conflicts:
            st.warning(f"⚠️ {warning}")

        # Build schedule table
        schedule_data = []
        for task in schedule:
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "")
            schedule_data.append({
                "Time": task.time,
                "Task": task.description,
                "Pet": task.pet_name,
                "Duration": f"{task.duration_minutes} min",
                "Priority": f"{priority_emoji} {task.priority}",
                "Frequency": task.frequency,
            })
        st.table(schedule_data)

        # Explanation of the schedule
        with st.expander("Why this order?"):
            st.markdown(
                "The schedule is sorted by **priority** first (high → medium → low), "
                "then by **time** within the same priority level. "
                "This ensures urgent tasks are always attended to first."
            )

st.divider()


# ============================================================
# Section 5: Manage & Complete Tasks
# ============================================================
st.subheader("Manage Tasks")

all_pending = scheduler.filter_by_status(completed=False)

if not all_pending:
    st.info("No pending tasks.")
else:
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        filter_pet = st.selectbox(
            "Filter by pet",
            ["All"] + [p.name for p in owner.pets],
            key="filter_pet",
        )
    with filter_col2:
        filter_status = st.selectbox(
            "Filter by status",
            ["Pending", "Completed", "All"],
            key="filter_status",
        )

    # Apply filters
    if filter_pet != "All":
        display_tasks = scheduler.filter_by_pet(filter_pet)
    else:
        display_tasks = scheduler.get_all_tasks()

    if filter_status == "Pending":
        display_tasks = [t for t in display_tasks if not t.completed]
    elif filter_status == "Completed":
        display_tasks = [t for t in display_tasks if t.completed]

    # Sort for display
    display_tasks = scheduler.sort_by_time(display_tasks)

    if not display_tasks:
        st.info("No tasks match the current filters.")
    else:
        for i, task in enumerate(display_tasks):
            status_icon = "✅" if task.completed else "⬜"
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(task.priority, "")
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(
                    f"{status_icon} **{task.time}** - {task.description} "
                    f"({task.pet_name}) {priority_emoji} {task.duration_minutes}min"
                )
            with col2:
                if not task.completed:
                    if st.button("Complete", key=f"complete_{i}_{task.description}_{task.pet_name}"):
                        next_task = scheduler.mark_task_complete(task)
                        if next_task:
                            st.success(
                                f"Completed! Next '{task.description}' scheduled for {next_task.due_date}."
                            )
                        else:
                            st.success("Task completed!")
                        st.rerun()
