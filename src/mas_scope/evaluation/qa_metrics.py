"""QA metrics for no-memory baselines."""

from __future__ import annotations

from collections import Counter

from mas_scope.tools.answer_parser import prediction_for_scoring
from mas_scope.tools.text_normalization import normalize_answer


def exact_match(prediction, answer) -> float:
    return float(normalize_answer(prediction) == normalize_answer(answer))


def token_f1(prediction, answer) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(answer).split()
    if normalize_answer(prediction) in {"yes", "no", "noanswer"} and normalize_answer(prediction) != normalize_answer(answer):
        return 0.0
    if normalize_answer(answer) in {"yes", "no", "noanswer"} and normalize_answer(prediction) != normalize_answer(answer):
        return 0.0
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(prediction, target: dict, dataset_name: str | None = None) -> dict:
    if not target or ("answer" not in target and "normalized_answer" not in target):
        return {"scored": False}
    answer = target.get("normalized_answer") or target.get("answer")
    scoring_prediction = prediction_for_scoring(prediction, target, dataset_name)
    metrics = {
        "scored": True,
        "exact_match": exact_match(scoring_prediction, answer),
        "token_f1": token_f1(scoring_prediction, answer),
    }
    if scoring_prediction != prediction:
        metrics["raw_prediction"] = "" if prediction is None else str(prediction)
        metrics["scoring_prediction"] = scoring_prediction
    if target.get("normalized_answer") is not None:
        metrics["yes_no_accuracy"] = exact_match(scoring_prediction, target["normalized_answer"])
    return metrics
