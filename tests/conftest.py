"""Shared pytest fixtures for the PawPal+ test suite."""

import pytest

from pawpal_system import Owner, Pet, Scheduler, Task


@pytest.fixture
def owner(tmp_path) -> Owner:
    """An owner with no pets.

    data_path points into tmp_path because the tool layer autosaves with no
    arguments -- without this, any test that adds a pet writes data.json into
    the repository root.
    """
    return Owner(name="Jordan", data_path=str(tmp_path / "data.json"))


@pytest.fixture
def owner_with_pets(owner: Owner) -> Owner:
    """An owner with one dog (Mochi) and one cat (Luna), neither with tasks."""
    owner.add_pet(Pet(name="Mochi", species="dog"))
    owner.add_pet(Pet(name="Luna", species="cat"))
    return owner


@pytest.fixture
def scheduler(owner_with_pets: Owner) -> Scheduler:
    """A scheduler over two pets that have no tasks yet."""
    return Scheduler(owner=owner_with_pets)


@pytest.fixture
def busy_scheduler(owner_with_pets: Owner) -> Scheduler:
    """A scheduler with four tasks, including a deliberate 08:00 conflict."""
    mochi = owner_with_pets.find_pet("Mochi")
    luna = owner_with_pets.find_pet("Luna")
    mochi.add_task(Task(description="Walk", time="07:30", duration_minutes=30, priority="high"))
    mochi.add_task(Task(description="Feed Mochi", time="08:00", duration_minutes=10, priority="high"))
    luna.add_task(Task(description="Feed Luna", time="08:00", duration_minutes=10, priority="medium"))
    luna.add_task(Task(description="Play", time="14:00", duration_minutes=20, priority="low"))
    return Scheduler(owner=owner_with_pets)
