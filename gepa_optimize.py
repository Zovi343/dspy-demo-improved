import os

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.teleprompt.gepa.gepa_utils import ReflectiveExample
from gepa.core.adapter import ProposalFn

from evaluate import (
    create_dspy_examples,
    validate_answer,
    validate_answer_with_feedback,
)
from extract import Extract

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


class GenerateWordLimitedInstruction(dspy.Signature):
    """Given a current instruction and feedback examples, generate an improved instruction with word limit constraints."""

    current_instruction = dspy.InputField(desc="The current instruction that needs improvement")
    feedback_summary = dspy.InputField(
        desc="Feedback from examples that might include both positive and negative cases"
    )
    max_words = dspy.InputField(desc="Maximum number of words allowed in the new instruction")

    improved_instruction = dspy.OutputField(
        desc="A new instruction that fixes the issues while staying under the max_words limit"
    )


class WordLimitProposer(ProposalFn):
    def __init__(self, max_words: int = 1000):
        self.max_words = max_words
        self.instruction_improver = dspy.ChainOfThought(GenerateWordLimitedInstruction)

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[ReflectiveExample]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        updated_components = {}

        for component_name in components_to_update:
            if component_name not in candidate or component_name not in reflective_dataset:
                continue

            current_instruction = candidate[component_name]
            component_examples = reflective_dataset[component_name]

            # Create feedback summary
            feedback_text = "\n".join(
                [
                    f"Example {i + 1}: {ex.get('Feedback', 'No feedback')}"
                    for i, ex in enumerate(
                        component_examples
                    )  # Limit examples to prevent context overflow
                ]
            )

            # Use the module to improve the instruction
            result = self.instruction_improver(
                current_instruction=current_instruction,
                feedback_summary=feedback_text,
                max_words=self.max_words,
            )

            updated_components[component_name] = result.improved_instruction

        return updated_components


if __name__ == "__main__":
    # Using OpenRouter. Switch to another LLM provider as needed
    task_lm = dspy.LM(
        model="openrouter/google/gemini-2.5-flash-lite",
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        max_tokens=16_000,
    )
    dspy.configure(lm=task_lm, adapter=BAMLAdapter())

    reflection_lm = dspy.LM(
        model="openrouter/openai/gpt-4.1",
        api_base="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        max_tokens=32_000,
        temperature=1.0,
    )

    # Start with baseline module
    baseline_extract = Extract()

    # Create Examples for training/testing
    train_ids = [2, 5, 6, 1, 7, 4, 10, 9, 11, 3, 12, 13, 8, 14]
    train_set = create_dspy_examples(num_sentences=5, article_ids=train_ids)

    val_ids = [33, 29, 24, 34, 30, 25, 35, 31, 26, 36]
    val_set = create_dspy_examples(num_sentences=5, article_ids=val_ids)

    test_ids = [17, 15, 18, 19, 16, 20, 22, 21, 28, 23]
    test_set = create_dspy_examples(num_sentences=5, article_ids=test_ids)

    # Use with DSPy optimizer
    optimizer = dspy.GEPA(
        metric=validate_answer_with_feedback,
        auto="light",
        num_threads=32,
        track_stats=True,
        use_merge=False,
        reflection_lm=reflection_lm,
        instruction_proposer=WordLimitProposer(max_words=1000),
    )
    optimized_extract = optimizer.compile(baseline_extract, trainset=train_set, valset=val_set)

    # Save optimized module for later use
    optimized_extract.save("./optimized_extract_gepa.json")

    # Evaluate baseline vs optimized performance on test set
    evaluator = dspy.Evaluate(metric=validate_answer, devset=test_set)

    print("Evaluating baseline module performance")
    baseline_score = evaluator(baseline_extract)

    print("Evaluating GEPA-optimized module performance")
    optimized_score = evaluator(optimized_extract)
