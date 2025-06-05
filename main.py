import argparse
from llm_evaluation.evaluator import LLMTestEvaluator
from llm_evaluation.llm_interface import MyLlama


def main():
    parser = argparse.ArgumentParser(description="Run LLM Evaluation Framework")
    parser.add_argument("--model", required=True, help="Hugging Face model name or local path")
    parser.add_argument("--correctness", required=True, help="Path to correctness.csv")
    parser.add_argument("--bias", required=True, help="Path to bias.csv")
    parser.add_argument("--hallucination", required=True, help="Path to hallucination.csv")

    args = parser.parse_args()

    # Initialize evaluator and model
    llama = MyLlama(model_name=args.model)
    evaluator = LLMTestEvaluator(
        correctness_path=args.correctness,
        bias_path=args.bias,
        hallucination_path=args.hallucination,
        llm=llama
    )

    evaluator.evaluate_correctness()
    evaluator.evaluate_bias()
    evaluator.evaluate_hallucination()

if __name__ == "__main__":
    main()
