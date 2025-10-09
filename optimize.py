import os

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter

from evaluate import create_dspy_examples, validate_answer
from extract import Extract

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Using OpenRouter. Switch to another LLM provider as needed
lm = dspy.LM(
    model="openrouter/google/gemini-2.5-flash-lite",
    api_base="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    max_tokens=10_000,
)
dspy.configure(lm=lm, adapter=BAMLAdapter())

# Start with baseline module
baseline_extract = Extract()

# Define IDs for train/test splits and create Examples for training/testing
train_ids = [2, 5, 6, 1, 7, 4, 10, 9, 11, 3, 12, 13, 8, 14]
train_set = create_dspy_examples(num_sentences=5, article_ids=train_ids)

test_ids = [17, 15, 18, 19, 16, 20, 22, 21, 28, 23]
test_set = create_dspy_examples(num_sentences=5, article_ids=test_ids)

# Use with DSPy optimizer
optimizer = dspy.BootstrapFewShot(
    metric=validate_answer,
    max_bootstrapped_demos=12,
)
optimized_extract = optimizer.compile(baseline_extract, trainset=train_set)

# Save optimized module for later use
optimized_extract.save("./optimized_extract.json")

# Evaluate baseline vs optimized performance on test set
evaluator = dspy.Evaluate(metric=validate_answer, devset=test_set)

print("Evaluating baseline module performance")
baseline_score = evaluator(baseline_extract)

print("Evaluating optimized module performance")
optimized_score = evaluator(optimized_extract)
