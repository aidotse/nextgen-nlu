# Model Training & Evaluation for Hackathon

This repository contains scripts for training & evaluating language models on a medical QA dataset. The evaluation is performed using **cosine similarity (embeddings)** and an **LLM-based grading system (GPT-4o).** 

> 📖 **New to the repo?** See [QUICKSTART.md](QUICKSTART.md) for a step-by-step guide to get running quickly.

## 🛠 Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/aidotse/nextgen-nlu.git
   cd nextgen-nlu
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API Key**
   Export your key as an environment variable (used across scripts):
   ```bash
   export OPENAI_API_KEY="sk-..."
   # Optional: override OpenAI-compatible base URL for remote endpoints
   export OPENAI_API_BASE="http://host:port/v1"
   ```

---

## 📌 Dataset Format

The dataset is stored in `data/qa.csv` and should be formatted as follows:

| question | answer |
|----------|--------|
| What are the side effects of XYZ? | headache, nausea, dizziness |
| How to use ABC medicine? | Take one tablet daily with water. |

---

## 🚀 Running the Evaluation Pipeline

### **1️⃣ Generate Model Answers**
Run `evaluate_model.py` to generate answers from a specified model.
```bash
python evaluate_model.py --model meta-llama/Llama-3.1-8B-Instruct --system_prompt config/system_prompt.txt
```
🔹 **Arguments**  
- `--model` → Hugging Face model to use for inference (e.g., `meta-llama/Llama-3.1-8B-Instruct`)

---

### **2️⃣ Evaluate the Model**
Run `run_eval.py` to assess model performance using **cosine similarity** and **GPT-4o scoring**.
```bash
python run_eval.py --answer model_answers.csv --eval_model gpt-4o
# You can skip the LLM judge if no API key is present
python run_eval.py --answer model_answers.csv --skip_llm_judge
```
🔹 **Arguments**  
- `--answer` → CSV file containing model-generated answers  
- `--eval_model` → OpenAI model for evaluation (default: `gpt-4o`)  

📌 **Output:**  
A CSV file `evaluation_results_MODEL_TIMESTAMP.csv` containing:
- Cosine similarity score
- LLM evaluation score (1-10)
- Explanation for the LLM score

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|------------|
| **Cosine Similarity** | Measures the semantic similarity between the ground truth and the model's answer using embeddings. |
| **LLM Score (1-10)** | GPT-4o evaluates the correctness and completeness of the model's answer. |
| **Explanation** | The LLM provides reasoning for the given score. |

---

## 📌 Example Output

```
### EVALUATION SUMMARY ###
Model Evaluated: my_model_answers
Timestamp: 2025-03-18_14-30-00
Average Cosine Similarity: 0.78
Average LLM Score: 8.2

Detailed results saved to evaluation_results_my_model_answers_2025-03-18_14-30-00.csv
```
## 🛠️ Finetuning a model
```bash
python lora_finetuning.py
```
Select a dataset, e.g:
```python
dataset = load_dataset(
    "text",
    data_files={"train": "/data/nextgen/data/*.txt"},
    sample_by="document"
)
```
## 👨‍💻 Authors
[Tim Isbister]  | [tim.isbister@ai.se]

[Amaru Cuba Gyllensten] | [amaru.gyllensten@ai.se]
