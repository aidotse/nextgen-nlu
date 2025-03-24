# Hackathon 2025-03-24

## Challenge
Get the highest score on the medicinal Q&A dataset from Astrazeneca.
The dataset is made up of domain specific questions with grund truth answers.
See `data/qa.csv`.

## Models to test
1) `meta-llama/Llama-3.1-8B-Instruct` running @ `185.248.53.82:35711` (this is the vanilla model from meta)
2) `llama-finetuned-raw` running @ `185.248.53.82:36539` (this is a lora trained model on raw pdfs)
3) `llama-finetuned-raw-qa` running @ `185.248.53.82:43773` (another lora trained model on raw pdfs + synthethic instruct QA data)
   
Please change the `openai_api_base` url to the endpoint where the model is running in `model_eval_api*.py`

## Possible Approaches
* Use models as they are and compare results (tweak prompt, hyperparams) example: 
```bash
python model_eval_api.py --model meta-llama/Llama-3.1-8B-Instruct --system_prompt config/system_prompt.txt
```
* Use models + RAG (tweak prompt, hyperparams, retrival) example:
```bash 
python model_eval_api_rag.py --model meta-llama/Llama-3.1-8B-Instruct --system_prompt config/system_prompt.txt --top_k 5
```
The RAG approach has all the embeddings precomputed from openai stored at edgelab @ `/mnt/data/rag_data/med_qa_embeddings.csv` where the embedding is computed for the `question` field.
```bash
head -n 1 med_qa_embeddings.csv
question,answer,openai_embedding,openai_model_used,local_embedding
```
```bash
wc -l med_qa_embeddings.csv
15826 med_qa_embeddings.csv
```
We have 15826 entries in the retrieval index.

Feel free to open up and edit `model_eval_api.py` or `model_eval_api_rag.py` to improve the performance.

## Look at metrics from the model_eval scripts
```bash
python get_scores.py --answer myanswers.csv
```
### Example output
```python
### EVALUATION SUMMARY ###
Model Evaluated: my_model_answers
Timestamp: 2025-03-18_14-30-00
Average Cosine Similarity: 0.78
Average LLM Score: 8.2

Detailed results saved to evaluation_results_my_model_answers_2025-03-18_14-30-00.csv
```

# Setup
```
pip install -r requirements.txt
```
To run the `get_scores.py` with LLM as a judge. You need the openAI key.
```python
openai.api_key = "YOUR_OPENAI_API_KEY"
```
Ask Tim or Mauricio.
