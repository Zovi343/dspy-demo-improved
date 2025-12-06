import dspy
import polars as pl

from notebooks.utils.dtos import Acquisition, Merger, Other

def metric(gold_val, pred_val, trace=None):  # trace unused but required for DSPy
    """
    Define a DSPy metric for the optimizer.
    This one calculates an exact match score.
    """
    # Handle None/empty equivalence
    if gold_val in [None, []] and pred_val in [None, []]:
        return 1
    return 1 if gold_val == pred_val else 0


def validate_answer(
    example: dspy.Example,
    pred: Merger | Acquisition | Other,
    trace=None, # type: ignore # trace unused but required for DSPy
) -> float:
    """DSPy-compatible metric function for evaluating extraction results."""
    expected : Merger | Acquisition | Other = example.expected_output # type: ignore

    # Type mismatch or Other type handling
    if not isinstance(pred, type(expected)): # type: ignore
        return 0.0
    if isinstance(expected, Other):
        return 1.0

    # Compare all model fields using Pydantic's field info
    expected_dict = expected.model_dump() # type: ignore
    pred_dict = pred.model_dump()

    # Calculate field-level accuracy
    matches = [metric(expected_dict[field], pred_dict[field]) for field in expected_dict]

    return sum(matches) / len(matches) if matches else 0.0


def validate_answer_with_feedback(
    example: dspy.Example,
    pred: Merger | Acquisition | Other,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> dspy.Prediction:
    """Metric for GEPA that mirrors validate_answer while emitting per-field feedback and score"""
    expected = example.expected_output

    # If incorrectly classified, score is instantly zero
    if not isinstance(pred, type(expected)):
        feedback = f"Article {example.article_id}): expected {type(expected).__name__}, got {type(pred).__name__}"
        return dspy.Prediction(score=0.0, feedback=feedback)

    # If other, no fields to compare, and score is 1.0
    if isinstance(expected, Other):
        feedback = f"✅ Article {example.article_id}): correctly classified as Other"
        return dspy.Prediction(score=1.0, feedback=feedback)

    # If the classification is not "Other", score on all model fields using Pydantic field info
    expected_dict = expected.model_dump()
    pred_dict = pred.model_dump()

    scored_fields = 0
    matches = 0

    feedback = f"Feedback for Article {example.article_id}:\n"
    for field, gold_value in expected_dict.items():
        if field in ["article_id"]:
            continue

        predicted_value = pred_dict.get(field)
        field_score = metric(gold_value, predicted_value)
        scored_fields += 1
        matches += field_score

        if field_score:
            feedback += f"  ✅ {field}: correctly extracted {gold_value!r}\n"
        else:
            feedback += "  ❌ " + f"{field}: expected {gold_value!r}, got {predicted_value!r}\n"

    score = matches / scored_fields if scored_fields else 0.0

    return dspy.Prediction(score=score, feedback=feedback)