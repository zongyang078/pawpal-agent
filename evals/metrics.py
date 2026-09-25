"""Metrics for the evaluation suites.

Plain functions over label lists -- no framework, so a reader can check the
arithmetic against the definitions they already know.
"""

from dataclasses import dataclass, field


@dataclass
class ClassificationReport:
    """Per-class precision/recall/F1 plus overall accuracy."""

    accuracy: float
    per_class: dict[str, dict[str, float]]
    confusion: dict[str, dict[str, int]]
    errors: list[tuple[str, str, str]] = field(default_factory=list)  # input, gold, pred


def classification_report(
    items: list[tuple[str, str, str]]
) -> ClassificationReport:
    """Score (input, gold, predicted) triples.

    Precision is over predictions of a label, recall over its true instances;
    both are defined as 0.0 when the denominator is empty rather than skipped,
    so a label the model never predicts shows up as a zero instead of vanishing.
    """
    labels = sorted({gold for _, gold, _ in items} | {pred for _, _, pred in items})
    confusion = {g: {p: 0 for p in labels} for g in labels}
    for _, gold, pred in items:
        confusion[gold][pred] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(confusion[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    correct = sum(1 for _, gold, pred in items if gold == pred)
    return ClassificationReport(
        accuracy=correct / len(items) if items else 0.0,
        per_class=per_class,
        confusion=confusion,
        errors=[(i, g, p) for i, g, p in items if g != p],
    )


@dataclass
class BinaryReport:
    """Counts and rates for a fire / don't-fire decision."""

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    errors: list[tuple[str, bool, bool]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        """Of the cases that should fire, how many did. Misses are unsafe."""
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def precision(self) -> float:
        actual = self.true_positives + self.false_positives
        return self.true_positives / actual if actual else 0.0

    @property
    def false_positive_rate(self) -> float:
        """Of the cases that should stay quiet, how many fired anyway."""
        actual = self.false_positives + self.true_negatives
        return self.false_positives / actual if actual else 0.0

    @property
    def accuracy(self) -> float:
        total = (self.true_positives + self.false_positives
                 + self.true_negatives + self.false_negatives)
        correct = self.true_positives + self.true_negatives
        return correct / total if total else 0.0


def binary_report(items: list[tuple[str, bool, bool]]) -> BinaryReport:
    """Score (input, expected, actual) triples."""
    report = BinaryReport(0, 0, 0, 0)
    for text, expected, actual in items:
        if expected and actual:
            report.true_positives += 1
        elif expected and not actual:
            report.false_negatives += 1
        elif not expected and actual:
            report.false_positives += 1
        else:
            report.true_negatives += 1
        if expected != actual:
            report.errors.append((text, expected, actual))
    return report


@dataclass
class RetrievalReport:
    """Ranking quality over a set of queries."""

    recall_at_1: float
    recall_at_3: float
    mrr: float
    misses: list[tuple[str, list[str], list[str]]] = field(default_factory=list)


def retrieval_report(
    items: list[tuple[str, list[str], list[str]]]
) -> RetrievalReport:
    """Score (query, relevant titles, ranked titles) triples.

    A query counts as hit@k if any relevant document appears in the top k.
    Reciprocal rank is 1/position of the first relevant document, 0 if absent.
    """
    if not items:
        return RetrievalReport(0.0, 0.0, 0.0)

    hits_1 = hits_3 = 0
    reciprocal_ranks = []
    misses = []

    for query, relevant, ranked in items:
        relevant_set = set(relevant)
        position = next(
            (i + 1 for i, title in enumerate(ranked) if title in relevant_set), None
        )
        if position == 1:
            hits_1 += 1
        if position is not None and position <= 3:
            hits_3 += 1
        else:
            misses.append((query, relevant, ranked))
        reciprocal_ranks.append(1 / position if position else 0.0)

    n = len(items)
    return RetrievalReport(
        recall_at_1=hits_1 / n,
        recall_at_3=hits_3 / n,
        mrr=sum(reciprocal_ranks) / n,
        misses=misses,
    )
