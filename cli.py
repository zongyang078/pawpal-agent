"""Command-line entry point for PawPal+.

Exposes the agent and the scheduler without requiring Streamlit:

    python cli.py ask "How often should I bathe my dog?"
    python cli.py chat
    python cli.py schedule
    python cli.py pets
    python cli.py seed --data demo.json

Runs in rule-based mode unless OPENAI_API_KEY or ANTHROPIC_API_KEY is set.
"""

import argparse
import os
import sys
from datetime import date

from agent import PawPalAgent
from pawpal_system import Owner, Pet, Scheduler, Task

DEFAULT_DATA_PATH = "data.json"


def load_owner(path: str) -> Owner:
    """Load the owner graph from disk, or start a fresh one bound to that path."""
    return Owner.load_from_json(path) or Owner(name="Pet Parent", data_path=path)


def build_agent(owner: Owner) -> PawPalAgent:
    """Wire up an agent. Provider and key come from the environment."""
    return PawPalAgent(owner=owner)


def print_response(agent: PawPalAgent, message: str, *, show_trace: bool) -> None:
    """Run one turn through the agent and print the result."""
    response = agent.process(message)
    print(response.message)

    if show_trace and response.tool_calls_made:
        print("\n  trace:", file=sys.stderr)
        for tc in response.tool_calls_made:
            print(f"    {tc['name']}({tc.get('args', {})})", file=sys.stderr)
        print(f"    confidence: {response.confidence:.2f}", file=sys.stderr)

    for warning in response.guardrail_warnings:
        print(f"  ! {warning}", file=sys.stderr)


def cmd_ask(args: argparse.Namespace) -> int:
    """Answer a single question and exit."""
    owner = load_owner(args.data)
    agent = build_agent(owner)
    print_response(agent, args.message, show_trace=args.trace)
    owner.save_to_json()
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Run an interactive session until EOF or 'exit'."""
    owner = load_owner(args.data)
    agent = build_agent(owner)
    mode = f"{agent.api_provider}:{agent.model}" if agent.use_llm else "rule-based"
    print(f"PawPal+ ({mode}). Ctrl-D or 'exit' to quit.\n")

    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        print_response(agent, message, show_trace=args.trace)
        print()

    owner.save_to_json()
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    """Print today's schedule, plus any conflicts."""
    owner = load_owner(args.data)
    scheduler = Scheduler(owner=owner)

    schedule = scheduler.generate_schedule()
    if not schedule:
        print("No tasks scheduled for today.")
    else:
        print(f"Today's schedule ({date.today()}):")
        for i, task in enumerate(schedule, 1):
            print(f"  {i}. {task}")

    conflicts = scheduler.detect_conflicts()
    if conflicts:
        print("\nConflicts:")
        for conflict in conflicts:
            print(f"  - {conflict}")
    return 0


def cmd_pets(args: argparse.Namespace) -> int:
    """List registered pets and their pending task counts."""
    owner = load_owner(args.data)
    if not owner.pets:
        print("No pets registered. Try: python cli.py ask \"Add Mochi, a dog\"")
        return 0
    print(f"{owner.name}'s pets:")
    for pet in owner.pets:
        print(f"  {pet.name} ({pet.species}) - {len(pet.get_pending_tasks())} pending")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    """Write a sample owner/pet/task graph, for demos and manual testing."""
    if os.path.exists(args.data) and not args.force:
        print(f"{args.data} already exists; pass --force to overwrite.", file=sys.stderr)
        return 1

    owner = Owner(name="Jordan")
    mochi = Pet(name="Mochi", species="dog")
    luna = Pet(name="Luna", species="cat")
    owner.add_pet(mochi)
    owner.add_pet(luna)

    mochi.add_task(Task(description="Morning walk", time="07:30", duration_minutes=30,
                        priority="high", frequency="daily"))
    mochi.add_task(Task(description="Breakfast feeding", time="08:00", duration_minutes=10,
                        priority="high", frequency="daily"))
    mochi.add_task(Task(description="Flea medication", time="09:00", duration_minutes=5,
                        priority="medium", frequency="weekly"))
    # Deliberately collides with Mochi's 08:00 feeding so `schedule` shows a conflict.
    luna.add_task(Task(description="Breakfast feeding", time="08:00", duration_minutes=10,
                       priority="high", frequency="daily"))
    luna.add_task(Task(description="Vet appointment", time="10:30", duration_minutes=60,
                       priority="high", frequency="once"))
    luna.add_task(Task(description="Play session", time="14:00", duration_minutes=20,
                       priority="medium", frequency="daily"))

    owner.save_to_json(args.data)
    print(f"Wrote sample data for {len(owner.pets)} pets to {args.data}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pawpal", description="PawPal+ pet care assistant.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                        help=f"path to the JSON data file (default: {DEFAULT_DATA_PATH})")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="ask the agent a single question")
    ask.add_argument("message", help="what to ask")
    ask.add_argument("--trace", action="store_true", help="print tool calls to stderr")
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="start an interactive session")
    chat.add_argument("--trace", action="store_true", help="print tool calls to stderr")
    chat.set_defaults(func=cmd_chat)

    sub.add_parser("schedule", help="print today's schedule").set_defaults(func=cmd_schedule)
    sub.add_parser("pets", help="list registered pets").set_defaults(func=cmd_pets)

    seed = sub.add_parser("seed", help="write a sample data file")
    seed.add_argument("--force", action="store_true", help="overwrite an existing data file")
    seed.set_defaults(func=cmd_seed)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
