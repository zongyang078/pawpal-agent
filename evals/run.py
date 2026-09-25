"""Evaluation harness.

    python -m evals.run                    # all suites
    python -m evals.run --suite intent     # one suite
    python -m evals.run --show-errors      # list every failing case
    python -m evals.run --json report.json # machine-readable output
    python -m evals.run --fail-under 0.8   # non-zero exit if a headline metric drops

Runs entirely offline against the rule-based path: no API key, no network, so
the numbers are reproducible and safe to gate CI on.
"""

import argparse
import json
import sys
from pathlib import Path

from agent import PawPalAgent
from evals.metrics import (
    BinaryReport,
    ClassificationReport,
    RetrievalReport,
    binary_report,
    classification_report,
    retrieval_report,
)
from guardrails import check_emergency, check_toxic_food_mention, check_vet_referral
from knowledge_base import KnowledgeBase
from pawpal_system import Owner

DATASETS = Path(__file__).parent / "datasets"


def load(name: str) -> list[dict]:
    with open(DATASETS / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Suites ---


def eval_intent() -> ClassificationReport:
    """How often keyword dispatch picks the intent a human would."""
    agent = PawPalAgent(owner=Owner(name="Jordan"), use_llm=False)
    return classification_report([
        (case["utterance"], case["intent"], agent._detect_intent(case["utterance"]))
        for case in load("intents")
    ])


def eval_retrieval() -> RetrievalReport:
    """Whether the right care document surfaces in the top 3."""
    kb = KnowledgeBase()
    return retrieval_report([
        (
            case["query"],
            case["relevant"],
            [doc.title for _, doc in kb.rank(case["query"], top_k=3)],
        )
        for case in load("retrieval")
    ])


def eval_guardrails() -> dict[str, BinaryReport]:
    """Whether each safety check fires exactly when it should."""
    buckets: dict[str, list[tuple[str, bool, bool]]] = {
        "emergency": [], "referral": [], "toxic": []
    }

    for case in load("guardrails"):
        check = case["check"]
        if check == "emergency":
            actual = not check_emergency(case["message"]).passed
            buckets[check].append((case["message"], case["expected"], actual))
        elif check == "referral":
            actual = bool(check_vet_referral(case["message"]).warnings)
            buckets[check].append((case["message"], case["expected"], actual))
        else:
            actual = not check_toxic_food_mention(
                case["response"], case["species"]
            ).passed
            buckets[check].append((case["response"], case["expected"], actual))

    return {name: binary_report(items) for name, items in buckets.items()}


# --- Reporting ---


def pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def print_intent(report: ClassificationReport, show_errors: bool) -> None:
    print(f"\nINTENT DETECTION  ({sum(c['support'] for c in report.per_class.values())} cases)")
    print(f"  accuracy {pct(report.accuracy)}\n")
    print(f"  {'intent':<18}{'prec':>7}{'recall':>8}{'f1':>7}{'n':>5}")
    for label, m in sorted(report.per_class.items(), key=lambda kv: -kv[1]["support"]):
        if not m["support"]:
            continue
        print(f"  {label:<18}{pct(m['precision']):>7}{pct(m['recall']):>8}"
              f"{pct(m['f1']):>7}{m['support']:>5}")
    if show_errors and report.errors:
        print(f"\n  {len(report.errors)} misclassified:")
        for text, gold, pred in report.errors:
            print(f"    {text!r}\n      expected {gold}, got {pred}")


def print_retrieval(report: RetrievalReport, show_errors: bool) -> None:
    print("\nRETRIEVAL")
    print(f"  recall@1 {pct(report.recall_at_1)}   recall@3 {pct(report.recall_at_3)}"
          f"   precision {pct(report.precision)}   MRR {report.mrr:.3f}")
    if show_errors and report.misses:
        print(f"\n  {len(report.misses)} queries with no relevant document in the top 3:")
        for query, relevant, ranked in report.misses:
            print(f"    {query!r}\n      wanted {relevant}\n      got    {ranked}")


def print_guardrails(reports: dict[str, BinaryReport], show_errors: bool) -> None:
    print("\nGUARDRAILS")
    print(f"  {'check':<12}{'recall':>8}{'prec':>8}{'FPR':>8}{'n':>5}")
    for name, r in reports.items():
        n = (r.true_positives + r.false_positives
             + r.true_negatives + r.false_negatives)
        print(f"  {name:<12}{pct(r.recall):>8}{pct(r.precision):>8}"
              f"{pct(r.false_positive_rate):>8}{n:>5}")

    for name, r in reports.items():
        misses = [e for e in r.errors if e[1]]        # should have fired
        spurious = [e for e in r.errors if not e[1]]  # should have stayed quiet
        if misses:
            print(f"\n  {name}: {len(misses)} MISSED (unsafe)")
            for text, _, _ in misses if show_errors else misses[:3]:
                print(f"    {text!r}")
        if spurious and show_errors:
            print(f"\n  {name}: {len(spurious)} false alarms")
            for text, _, _ in spurious:
                print(f"    {text!r}")


def to_dict(intent, retrieval, guardrails) -> dict:
    return {
        "intent": {
            "accuracy": intent.accuracy,
            "per_class": intent.per_class,
            "errors": [
                {"input": i, "expected": g, "got": p} for i, g, p in intent.errors
            ],
        },
        "retrieval": {
            "recall_at_1": retrieval.recall_at_1,
            "recall_at_3": retrieval.recall_at_3,
            "precision": retrieval.precision,
            "mrr": retrieval.mrr,
            "misses": [
                {"query": q, "expected": rel, "got": got}
                for q, rel, got in retrieval.misses
            ],
        },
        "guardrails": {
            name: {
                "recall": r.recall,
                "precision": r.precision,
                "false_positive_rate": r.false_positive_rate,
                "accuracy": r.accuracy,
                "counts": {
                    "tp": r.true_positives, "fp": r.false_positives,
                    "tn": r.true_negatives, "fn": r.false_negatives,
                },
                "errors": [
                    {"input": t, "expected": e, "got": a} for t, e, a in r.errors
                ],
            }
            for name, r in guardrails.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals", description=__doc__.split("\n")[0])
    parser.add_argument("--suite", choices=["intent", "retrieval", "guardrails"],
                        action="append", help="run only these suites (repeatable)")
    parser.add_argument("--show-errors", action="store_true",
                        help="print every failing case, not just a sample")
    parser.add_argument("--json", metavar="PATH", help="also write the report as JSON")
    parser.add_argument("--fail-under", type=float, metavar="X",
                        help="exit 1 if any headline metric is below X (0-1)")
    args = parser.parse_args(argv)

    suites = set(args.suite or ["intent", "retrieval", "guardrails"])
    intent = eval_intent() if "intent" in suites else None
    retrieval = eval_retrieval() if "retrieval" in suites else None
    guardrails = eval_guardrails() if "guardrails" in suites else None

    if intent:
        print_intent(intent, args.show_errors)
    if retrieval:
        print_retrieval(retrieval, args.show_errors)
    if guardrails:
        print_guardrails(guardrails, args.show_errors)

    if args.json:
        Path(args.json).write_text(
            json.dumps(to_dict(intent, retrieval, guardrails), indent=2),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    if args.fail_under is not None:
        headline = {}
        if intent:
            headline["intent accuracy"] = intent.accuracy
        if retrieval:
            headline["retrieval recall@3"] = retrieval.recall_at_3
        if guardrails:
            for name, r in guardrails.items():
                headline[f"{name} recall"] = r.recall
        below = {k: v for k, v in headline.items() if v < args.fail_under}
        if below:
            print(f"\nFAIL: below {args.fail_under}")
            for name, value in below.items():
                print(f"  {name}: {value:.3f}")
            return 1
        print(f"\nOK: all headline metrics >= {args.fail_under}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
