from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import BiasMetric, GEval, HallucinationMetric
from llm_evaluation.data_etl import DataETL


class LLMTestEvaluator:
    def __init__(self, correctness_path, bias_path, hallucination_path, llm, threshold=0.5):
        self.llm = llm

        self.bias_metric = BiasMetric(threshold=threshold)
        self.hallucination_metric = HallucinationMetric(threshold=threshold)
        self.correctness_metric = GEval(
            name="Correctness",
            evaluation_steps=[
                "Make sure that information is correct factually",
                "You should also heavily penalize omission of detail"
            ],
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT
            ]
        )

        self.correctness_data = DataETL(correctness_path).load_data()
        self.bias_data = DataETL(bias_path).load_data()
        self.hallucination_data = DataETL(hallucination_path).load_data()

    def evaluate_correctness(self):
        print("\n--- Running Correctness Evaluation ---")
        for i, row in enumerate(self.correctness_data):
            input_text = row['prompt']
            expected_output = row['completion']
            actual_output = self.llm.generate_rude_response(input_text)

            test_case = LLMTestCase(input=input_text, actual_output=actual_output, expected_output=expected_output)
            self.correctness_metric.measure(test_case)
            print(f"[{i+1}] Score: {self.correctness_metric.score} | Reason: {self.correctness_metric.reason}")

    def evaluate_bias(self):
        print("\n--- Running Bias Evaluation ---")
        for i, row in enumerate(self.bias_data):
            input_text = row['prompt']
            actual_output = self.llm.generate_rude_response(input_text)

            test_case = LLMTestCase(input=input_text, actual_output=actual_output)
            self.bias_metric.measure(test_case)
            print(f"[{i+1}] Score: {self.bias_metric.score} | Reason: {self.bias_metric.reason}")

    def evaluate_hallucination(self):
        print("\n--- Running Hallucination Evaluation ---")
        for i, row in enumerate(self.hallucination_data):
            input_text = row['prompt']
            raw_context = row.get('context', '')
            context = [raw_context] if raw_context else None
            actual_output = self.llm.generate_rude_response(input_text)

            test_case = LLMTestCase(input=input_text, actual_output=actual_output, context=context)
            self.hallucination_metric.measure(test_case)
            print(f"[{i+1}] Score: {self.hallucination_metric.score} | Reason: {self.hallucination_metric.reason}")
