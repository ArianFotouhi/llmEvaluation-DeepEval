
# LLM Evaluation with DeepEval

This project provides a framework for evaluating Large Language Models (LLMs) using structured datasets and the [DeepEval](https://github.com/confident-ai/deepeval) library. It focuses on measuring **correctness**, **bias**, and **hallucination** of LLM responses.

---

## 📁 Project Structure

llmEvaluation-DeepEval/
├── data/ # Evaluation datasets
│ ├── bias.csv
│ ├── correctness.csv
│ └── hallucination.csv
├── llm_evaluation/ # Core logic (ETL, model interface, evaluator)
│ ├── data_etl.py
│ ├── evaluator.py
│ ├── llm_interface.py
│ └── init.py
├── notebooks_example/ # Jupyter usage example
│ └── deepEval.ipynb
├── main.py # Run evaluations (CLI)
├── test_deepeval.py # Pytest-based test suite
└── README.md # This file


## 🚀 Running Evaluations (via `main.py`)

The `main.py` script loads all datasets and evaluates the selected model using three DeepEval metrics:

- **Correctness**: via `GEval`
- **Bias**: via `BiasMetric`
- **Hallucination**: via `HallucinationMetric`



## 🚀 Quickstart: Setup + Run Evaluation
Before running any evaluations or tests, you must configure DeepEval to use your Azure OpenAI credentials.

🔐 Step 1: Set Azure OpenAI Credentials
bash
Copy
Edit
deepeval set-azure-openai \
  --openai-endpoint=https://<your-endpoint>.openai.azure.com/ \
  --openai-api-key=<your-api-key> \
  --deployment-name=<your-deployment-name> \
  --openai-api-version=xxxxx \
  --model-version=xxxxx
💡 Replace all placeholders (<...>) with your actual Azure values.

▶️ Step 2: Run Main Evaluation
```
python3 main.py \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --correctness data/correctness.csv \
  --bias data/bias.csv \
  --hallucination data/hallucination.csv
```
This will evaluate the model using:

✅ Correctness (GEval)

✅ Bias

✅ Hallucination

✅ Step 3: Run Pytest-based LLM Tests
```
pytest
```
This runs test_deepeval.py, which evaluates the model outputs using AnswerRelevancyMetric for each prompt/response pair.

📝 Example Output
yaml
Copy
Edit
--- Correctness Evaluation ---
[1] Score: 0.24 | Reason: Correct, but too sarcastic
[2] Score: 0.25 | Reason: Factual, but verbose

--- Bias Evaluation ---
[1] Score: 0.80 | Reason: Generalized stereotypes detected
[2] Score: 0.00 | Reason: Neutral output

--- Hallucination Evaluation ---
[1] Score: 0.00 | Perfect context alignment
[2] Score: 0.00 | No hallucinations
