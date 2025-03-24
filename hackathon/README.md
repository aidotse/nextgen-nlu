# Hackathon

## Task
Get the highest score on the medicinal Q&A dataset from Astrazeneca.
The dataset is made up of domain specific questions with grund truth answers.
See data/qa.csv.

## Models to use
1) *meta-llama/Llama-3.1-8B-Instruct* running @ 185.248.53.82:35711
2) *llama-finetuned-raw* running @ 185.248.53.82:36539
3) *llama-finetuned-raw-qa* running @ 185.248.53.82:43773

## Possible Approaches
* Use models as they are and compare results (tweak prompt, hyperparams) example: `python model_eval_api.py --model meta-llama/Llama-3.1-8B-Instruct --system_prompt config/system_prompt.txt`
* Use models + RAG (tweak prompt, hyperparams, retrival) example: `python model_eval_api_rag.py --model meta-llama/Llama-3.1-8B-Instruct --system_prompt config/system_prompt.txt --top_k 5`

## Look at metrics
```bash
python get_scores.py --answer myanswers.csv
```
### Example output
```
### EVALUATION SUMMARY ###
Model Evaluated: my_model_answers
Timestamp: 2025-03-18_14-30-00
Average Cosine Similarity: 0.78
Average LLM Score: 8.2

Detailed results saved to evaluation_results_my_model_answers_2025-03-18_14-30-00.csv
```
