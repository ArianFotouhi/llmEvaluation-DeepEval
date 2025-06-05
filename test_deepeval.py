import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from llm_evaluation.llm_interface import MyLlama

# ✅ Hardcoded config
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "data/correctness.csv"

# ✅ Load test data
def load_data(path):
    df = pd.read_csv(path)
    cases = []
    for _, row in df.iterrows():
        prompt = row.get("prompt", "")
        expected = row.get("completion", "")
        context = row.get("context", None)
        if isinstance(context, str) and context.strip():
            context = [context]
        else:
            context = None
        cases.append((prompt, expected, context))
    return cases

TEST_CASES = load_data(DATASET_PATH)
LLM_INSTANCE = MyLlama(model_name=MODEL_NAME)

@pytest.mark.parametrize("prompt,expected_output,context", TEST_CASES)
def test_llm_answer_relevancy(prompt, expected_output, context):
    actual_output = LLM_INSTANCE.generate_rude_response(prompt)

    metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=context
    )

    assert_test(test_case, [metric])
