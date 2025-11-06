# Quick Start Guide

## 🚀 Initial Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables
```bash
# Required for evaluation (LLM judge and embeddings)
export OPENAI_API_KEY="sk-your-key-here"

# Optional: For remote model endpoints (hackathon scripts)
export OPENAI_API_BASE="http://185.248.53.82:35711/v1"
```

### 3. Verify Your Data
Make sure `data/qa.csv` exists with your questions and ground truth answers:
```bash
head data/qa.csv
```

---

## 📝 Basic Workflow

### Option A: Using Local Hugging Face Model

**Step 1: Generate Answers**
```bash
python evaluate_model.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --system_prompt config/system_prompt.txt
```

This creates a file like: `answers_Meta-Llama-3-1-8B-Instruct_system_prompt_2025-01-XX_XX-XX-XX.csv`

**Step 2: Evaluate Answers**
```bash
python run_eval.py \
  --answer answers_Meta-Llama-3-1-8B-Instruct_system_prompt_2025-01-XX_XX-XX-XX.csv \
  --eval_model gpt-4o
```

**Without LLM Judge (cosine similarity only):**
```bash
python run_eval.py \
  --answer answers_Meta-Llama-3-1-8B-Instruct_system_prompt_2025-01-XX_XX-XX-XX.csv \
  --skip_llm_judge
```

---

### Option B: Using Remote API (Hackathon)

Specify your remote endpoint (host:port) in one of two ways:

1) Environment variables (recommended, persistent across runs):
```bash
export OPENAI_API_BASE="http://YOUR_HOST:PORT/v1"
export OPENAI_API_KEY="sk-..."
```

2) CLI flags (override env for that command only):
```bash
# ... add to the command: --api_base http://YOUR_HOST:PORT/v1 --api_key sk-...
```

**Step 1: Generate Answers (No RAG)**
```bash
cd hackathon
python model_eval_api.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --system_prompt ../config/system_prompt.txt \
  --api_base http://YOUR_HOST:PORT/v1 \
  --api_key sk-...
```

**Step 2: Generate Answers (With RAG)**
```bash
cd hackathon
python model_eval_api_rag.py \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --system_prompt ../config/system_prompt.txt \
  --top_k 5 \
  --api_base http://YOUR_HOST:PORT/v1 \
  --api_key sk-... \
  --embeddings_api_key sk-...   # if different; else omit to use OPENAI_API_KEY
```

**Step 3: Evaluate**
```bash
cd hackathon
python get_scores.py --answer answers_*.csv
```

---

## 🔧 Common Tasks

### Generate Synthetic QA Dataset
```bash
python generate_raq_dataset.py \
  --input_glob "/data/nextgen/data/*.txt" \
  --output_csv data/qa_pairs.csv \
  --model gpt-4o-2024-08-06
```

### Fine-tune with LoRA
```bash
# Edit paths in lora_finetuning.py first, then:
python lora_finetuning.py
```

### Merge LoRA Adapter
```bash
# Edit paths in merge_lora.py first, then:
python merge_lora.py
```

---

## 📊 Understanding Outputs

### Answer Files
- Format: `answers_{model}_{prompt}_{timestamp}.csv`
- Columns: `question`, `answer`, `system_prompt_file`

### Evaluation Files
- Format: `evaluation_results_{model}_{timestamp}.csv`
- Columns: `question`, `ground_truth`, `model_answer`, `cosine_similarity`, `llm_score`, `llm_explanation`
- Last row contains average scores

---

## ⚠️ Troubleshooting

**"No module named 'sentence_transformers'"**
```bash
pip install -r requirements.txt
```

**"OPENAI_API_KEY not set"**
- Set the environment variable: `export OPENAI_API_KEY="sk-..."`
- Or use `--skip_llm_judge` to skip LLM evaluation

**"CUDA out of memory" (local models)**
- Use a smaller model or reduce batch size
- Or use remote API endpoints instead

**"Connection refused" (remote API)**
- Check `OPENAI_API_BASE` is correct
- Or use `--api_base` flag: `--api_base http://your-endpoint:port/v1`

