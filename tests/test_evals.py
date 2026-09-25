"""Tests for the evaluation harness.

The metrics decide whether a change to the system counts as an improvement, so
their arithmetic is checked against hand-computed cases.
"""

import json

import pytest

from evals.metrics import binary_report, classification_report, retrieval_report
from evals.run import DATASETS, eval_guardrails, eval_intent, eval_retrieval, load, main


class TestClassificationReport:
    def test_perfect_predictions(self):
        report = classification_report([("a", "x", "x"), ("b", "y", "y")])
        assert report.accuracy == 1.0
        assert report.errors == []
        assert report.per_class["x"]["f1"] == 1.0

    def test_hand_computed_precision_and_recall(self):
        # "x" predicted 3 times, right twice -> precision 2/3.
        # "x" is the gold label 2 times, both caught -> recall 1.0.
        report = classification_report([
            ("1", "x", "x"), ("2", "x", "x"), ("3", "y", "x"), ("4", "y", "y"),
        ])
        assert report.accuracy == 0.75
        assert report.per_class["x"]["precision"] == pytest.approx(2 / 3)
        assert report.per_class["x"]["recall"] == 1.0
        assert report.per_class["y"]["recall"] == 0.5
        assert report.per_class["x"]["support"] == 2

    def test_never_predicted_label_scores_zero_not_missing(self):
        report = classification_report([("1", "x", "y"), ("2", "x", "y")])
        assert report.per_class["x"]["precision"] == 0.0
        assert report.per_class["x"]["recall"] == 0.0
        assert report.per_class["x"]["f1"] == 0.0

    def test_errors_record_what_went_wrong(self):
        report = classification_report([("hello", "greet", "farewell")])
        assert report.errors == [("hello", "greet", "farewell")]

    def test_confusion_matrix_counts(self):
        report = classification_report([("1", "x", "y"), ("2", "x", "x")])
        assert report.confusion["x"]["y"] == 1
        assert report.confusion["x"]["x"] == 1

    def test_empty_input(self):
        assert classification_report([]).accuracy == 0.0


class TestBinaryReport:
    def test_counts_all_four_cells(self):
        report = binary_report([
            ("a", True, True), ("b", True, False),
            ("c", False, True), ("d", False, False),
        ])
        assert (report.true_positives, report.false_negatives,
                report.false_positives, report.true_negatives) == (1, 1, 1, 1)
        assert report.recall == 0.5
        assert report.precision == 0.5
        assert report.false_positive_rate == 0.5
        assert report.accuracy == 0.5

    def test_errors_separate_misses_from_false_alarms(self):
        report = binary_report([("missed", True, False), ("spurious", False, True)])
        assert [e for e in report.errors if e[1]] == [("missed", True, False)]
        assert [e for e in report.errors if not e[1]] == [("spurious", False, True)]

    def test_rates_are_zero_when_undefined(self):
        empty = binary_report([])
        assert empty.recall == 0.0
        assert empty.precision == 0.0
        assert empty.false_positive_rate == 0.0


class TestRetrievalReport:
    def test_hit_at_rank_one(self):
        report = retrieval_report([("q", ["A"], ["A", "B", "C"])])
        assert report.recall_at_1 == 1.0
        assert report.recall_at_3 == 1.0
        assert report.mrr == 1.0

    def test_hit_at_rank_three(self):
        report = retrieval_report([("q", ["C"], ["A", "B", "C"])])
        assert report.recall_at_1 == 0.0
        assert report.recall_at_3 == 1.0
        assert report.mrr == pytest.approx(1 / 3)

    def test_complete_miss(self):
        report = retrieval_report([("q", ["Z"], ["A", "B", "C"])])
        assert report.recall_at_3 == 0.0
        assert report.mrr == 0.0
        assert report.misses == [("q", ["Z"], ["A", "B", "C"])]

    def test_any_relevant_document_counts_as_a_hit(self):
        report = retrieval_report([("q", ["B", "Z"], ["A", "B"])])
        assert report.recall_at_3 == 1.0
        assert report.mrr == 0.5

    def test_averages_across_queries(self):
        report = retrieval_report([
            ("q1", ["A"], ["A"]),
            ("q2", ["Z"], ["A"]),
        ])
        assert report.recall_at_1 == 0.5
        assert report.mrr == 0.5

    def test_empty_input(self):
        assert retrieval_report([]).mrr == 0.0


class TestDatasets:
    """The datasets are the ground truth; a malformed one silently skews scores."""

    @pytest.mark.parametrize("name", ["intents", "retrieval", "guardrails"])
    def test_every_line_parses(self, name):
        assert len(load(name)) > 0

    def test_intent_labels_are_known(self):
        from agent import INTENTS

        for case in load("intents"):
            assert case["intent"] in INTENTS, case

    def test_retrieval_targets_exist_in_the_corpus(self):
        from knowledge_base import KnowledgeBase

        titles = {doc.title for doc in KnowledgeBase().documents}
        for case in load("retrieval"):
            for wanted in case["relevant"]:
                assert wanted in titles, f"{wanted!r} is not a document title"

    def test_guardrail_cases_are_well_formed(self):
        for case in load("guardrails"):
            assert case["check"] in {"emergency", "referral", "toxic"}
            assert isinstance(case["expected"], bool)
            if case["check"] == "toxic":
                assert "response" in case and "species" in case
            else:
                assert "message" in case

    def test_both_outcomes_are_represented(self):
        """A suite of only positives cannot detect a check that always fires."""
        by_check: dict[str, set] = {}
        for case in load("guardrails"):
            by_check.setdefault(case["check"], set()).add(case["expected"])
        for check, outcomes in by_check.items():
            assert outcomes == {True, False}, f"{check} is missing a polarity"

    def test_datasets_directory_is_where_run_expects(self):
        assert DATASETS.is_dir()


class TestSuites:
    """The suites run offline and produce metrics in range."""

    def test_intent_suite(self):
        report = eval_intent()
        assert 0.0 <= report.accuracy <= 1.0

    def test_retrieval_suite(self):
        report = eval_retrieval()
        assert 0.0 <= report.recall_at_1 <= report.recall_at_3 <= 1.0

    def test_guardrail_suite_covers_all_three_checks(self):
        reports = eval_guardrails()
        assert set(reports) == {"emergency", "referral", "toxic"}

    def test_cli_runs_and_writes_json(self, tmp_path, capsys):
        out = tmp_path / "report.json"
        assert main(["--json", str(out)]) == 0

        report = json.loads(out.read_text())
        assert "accuracy" in report["intent"]
        assert "recall_at_3" in report["retrieval"]
        assert "emergency" in report["guardrails"]
        assert "INTENT DETECTION" in capsys.readouterr().out

    def test_fail_under_gates_on_the_headline_metrics(self, capsys):
        assert main(["--suite", "guardrails", "--fail-under", "0.9"]) == 0
        assert main(["--suite", "intent", "--fail-under", "0.99"]) == 1
        assert "FAIL" in capsys.readouterr().out

    def test_single_suite_runs_alone(self, capsys):
        assert main(["--suite", "retrieval"]) == 0
        out = capsys.readouterr().out
        assert "RETRIEVAL" in out
        assert "INTENT DETECTION" not in out


class TestRetrievalPrecision:
    """Precision was added after recall alone missed a cross-species ranker."""

    def test_all_relevant_scores_one(self):
        assert retrieval_report([("q", ["A", "B"], ["A", "B"])]).precision == 1.0

    def test_partially_relevant(self):
        report = retrieval_report([("q", ["A"], ["A", "B", "C"])])
        assert report.recall_at_1 == 1.0, "recall cannot see the two bad results"
        assert report.precision == pytest.approx(1 / 3)

    def test_empty_result_set_scores_zero(self):
        """Returning nothing must not look like perfect precision."""
        assert retrieval_report([("q", ["A"], [])]).precision == 0.0
