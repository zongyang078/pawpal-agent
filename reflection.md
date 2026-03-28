# PawPal+ Project Reflection

## 1. System Design

**a. Initial design**

- Briefly describe your initial UML design.
- What classes did you include, and what responsibilities did you assign to each?

My initial UML design included four classes: `Task`, `Pet`, `Owner`, and `Scheduler`. Task is a dataclass holding all task-related data (description, time, duration, priority, frequency, status, due date). Pet stores pet details and manages a list of Task objects. Owner manages multiple pets and provides an aggregate view of all tasks. Scheduler acts as the orchestration layer — it doesn't own any data but queries the Owner to sort, filter, detect conflicts, and generate schedules.

I chose this separation so the data layer (Task, Pet, Owner) stays clean and the algorithmic logic lives entirely in Scheduler. This makes it easier to test each piece in isolation.


**b. Design changes**

- Did your design change during implementation?
- If yes, describe at least one change and why you made it.

Yes. During implementation I added a `pet_name` attribute to the Task class. Originally, tasks only existed inside Pet objects, so there was no need to track which pet they belonged to. But when the Scheduler collects all tasks across pets for conflict detection and filtering, it needs to know which pet each task came from. Adding `pet_name` directly to Task (auto-set in `Pet.add_task()`) was simpler than passing pet references around.

I also added `Owner.find_pet()` which was not in the original UML. The Scheduler needs this to add recurring task occurrences back to the correct pet after completion.


---

## 2. Scheduling Logic and Tradeoffs

**a. Constraints and priorities**

- What constraints does your scheduler consider (for example: time, priority, preferences)?
- How did you decide which constraints mattered most?

The scheduler considers three main constraints: priority level (high/medium/low), scheduled time (HH:MM), and task frequency (once/daily/weekly). Priority was weighted numerically (high=3, medium=2, low=1) so sorting works cleanly with Python's `sorted()`. The schedule sorts by priority first (descending), then by time (ascending) within the same priority — this ensures urgent tasks always surface at the top while maintaining a logical time flow.

I decided priority should outweigh time because a pet owner would rather see "give medication at 10:00" before "play session at 09:00" if the medication is high priority.


**b. Tradeoffs**

- Describe one tradeoff your scheduler makes.
- Why is that tradeoff reasonable for this scenario?

The conflict detection only checks for exact time matches (same HH:MM string) rather than checking for overlapping durations. For example, a 60-minute task starting at 08:00 and a 10-minute task at 08:30 won't be flagged as conflicting, even though they overlap.

This tradeoff is reasonable because: (1) most pet care tasks in this scenario are short and scheduled at fixed times, (2) implementing duration-aware overlap checking would require converting times to datetime objects and comparing ranges, adding complexity that isn't justified for a daily planner MVP. The exact-match approach catches the most common case (two tasks at the same time) with minimal code.


---

## 3. AI Collaboration

**a. How you used AI**

- How did you use AI tools during this project (for example: design brainstorming, debugging, refactoring)?
- What kinds of prompts or questions were most helpful?

I used AI tools throughout the project in several ways:

- **Design brainstorming**: I described the pet care scenario and asked the AI to help draft a Mermaid.js class diagram with four classes and their relationships. This helped me quickly visualize the system structure before writing any code.
- **Scaffolding**: I used AI to generate class skeletons with type hints and docstrings based on the UML, then filled in the logic incrementally.
- **Debugging**: When the conflict detection wasn't grouping tasks correctly, I asked the AI to review my logic and it suggested using a dictionary to group tasks by time slot instead of nested loops.
- **Test generation**: I described the behaviors I wanted to test and the AI helped draft pytest functions, including edge cases I hadn't considered (like an empty pet with no tasks).

The most helpful prompts were specific ones like "Based on my pawpal_system.py, how should the Scheduler retrieve all tasks from the Owner's pets?" rather than vague ones like "help me code."


**b. Judgment and verification**

- Describe one moment where you did not accept an AI suggestion as-is.
- How did you evaluate or verify what the AI suggested?

The AI initially suggested using `datetime.strptime()` to parse the HH:MM time strings for sorting. I rejected this because simple string comparison already works correctly for "HH:MM" format (e.g., "07:30" < "08:00" < "14:00" lexicographically). Converting to datetime objects on every sort call would add unnecessary overhead and complexity for no functional benefit.

I verified by testing with out-of-order times in `main.py` and confirming the sorted output was correct using plain string comparison.


---

## 4. Testing and Verification

**a. What you tested**

- What behaviors did you test?
- Why were these tests important?

The test suite covers 15 test cases across four test classes:

- Task: mark_complete status change, idempotent completion, daily/weekly recurrence, one-time task returns None.
- Pet: add_task count increase, auto pet_name assignment, remove by description, remove nonexistent, get_pending filters correctly.
- Owner: add_pet, get_all_tasks aggregation across pets.
- Scheduler: sort_by_time chronological order, sort_by_priority descending, filter_by_pet, filter_by_status, conflict detection true/false cases, mark_complete recurring auto-creation, generate_schedule pending-only, empty pet edge case.

These tests are important because they verify both the "happy path" (normal usage) and edge cases (empty data, duplicate operations) that could cause subtle bugs in the Streamlit UI.


**b. Confidence**

- How confident are you that your scheduler works correctly?
- What edge cases would you test next if you had more time?

I'm fairly confident the scheduler works correctly for the intended use case — confidence level 4/5. All core behaviors pass automated tests.

Edge cases I would test next with more time: tasks with overlapping time ranges (not just exact matches), tasks spanning midnight, very large task lists (performance), and concurrent modifications to the same pet's task list.


---

## 5. Reflection

**a. What went well**

- What part of this project are you most satisfied with?

I'm most satisfied with the clean separation between the data layer and the scheduling logic. The Scheduler never modifies data directly — it always goes through Owner/Pet methods. This made testing straightforward because I could test each class independently, and the Streamlit integration was simple since I just called Scheduler methods from the UI.


**b. What you would improve**

- If you had another iteration, what would you improve or redesign?

If I had another iteration, I would:

1. Add duration-aware conflict detection (checking overlapping time ranges, not just exact matches).
2. Implement data persistence with JSON so pets and tasks survive between Streamlit sessions.
3. Add a "suggested schedule" feature that automatically fills open time slots based on pet needs and owner availability.


**c. Key takeaway**

- What is one important thing you learned about designing systems or working with AI on this project?

The most important thing I learned is that AI is most effective when you already have a clear design in mind. Using AI to generate a UML diagram before writing code forced me to think about class responsibilities and relationships first. When I then asked AI to scaffold code from the UML, the output was much more aligned with what I needed compared to asking "build me a pet scheduler" from scratch. Being the "lead architect" means making design decisions yourself and using AI as a fast executor — not the other way around.

