import argparse
import openai
import pandas as pd
import numpy as np
import json
import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

openai.api_key = "YOUR_OPENAI_API_KEY"

local_embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
openai_failed = False

def get_embedding(text, use_openai=True, openai_model="text-embedding-3-large"):
    global openai_failed
    if use_openai and not openai_failed:
        try:
            client = openai.OpenAI()
            response = client.embeddings.create(input=text, model=openai_model)
            return np.array(response.data[0].embedding), openai_model
        except Exception:
            print(f"⚠️ OpenAI embedding model {openai_model} failed. Switching to local embeddings.")
            openai_failed = True
    return np.array(local_embedding_model.encode(text)), "sentence-transformers/all-mpnet-base-v2"

def get_llm_judge_score(question, ground_truth, model_answer, eval_model="gpt-4o"):
    prompt = f"""
You are an expert AI evaluator. Your task is to assess the accuracy of a given model's answer.

### Question:
{question}

### Ground truth answer:
{ground_truth}

### Model-generated answer:
{model_answer}

You must **only return a valid JSON object** with this structure:
{{
    "score": X,  # A number between 1-10
    "explanation": "Your reasoning"
}}
"""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=eval_model,
        messages=[
            {"role": "system", "content": "You are a strict but fair AI evaluator. You must **only** return valid JSON, nothing else."},
            {"role": "user", "content": prompt}
        ]
    )
    try:
        llm_response = response.choices[0].message.content.strip()
        if llm_response.startswith("```"):
            llm_response = llm_response.strip("`")
            if llm_response.lower().startswith("json"):
                llm_response = llm_response[4:].strip()
        result = json.loads(llm_response)
        return result.get("score", 1), result.get("explanation", "No explanation provided.")
    except (json.JSONDecodeError, AttributeError) as e:
        return 1, f"Critical Error: {str(e)}. Raw response: {llm_response[:200]}"

def evaluate_model(answer_file, eval_model="gpt-4o"):
    qa_data = pd.read_csv("data/qa.csv")
    model_answers = pd.read_csv(answer_file)
    if len(qa_data) != len(model_answers):
        raise ValueError("Mismatch in number of questions and model answers!")
    model_name = answer_file.replace(".csv", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results = []
    cosine_sims = []
    llm_scores = []
    embedding_model_used = None
    for idx, row in tqdm(qa_data.iterrows(), total=len(qa_data), desc="Evaluating QA"):
        question = row["question"]
        ground_truth = row["answer"]
        model_answer = model_answers.iloc[idx]["answer"]
        truth_emb, emb_model = get_embedding(ground_truth)
        model_emb, _ = get_embedding(model_answer)
        embedding_model_used = emb_model
        sim_score = cosine_similarity([truth_emb], [model_emb])[0][0]
        cosine_sims.append(sim_score)
        llm_score, explanation = get_llm_judge_score(question, ground_truth, model_answer, eval_model)
        llm_scores.append(llm_score)
        results.append({
            "timestamp": timestamp,
            "model_name": model_name,
            "embedding_model": embedding_model_used,
            "question": question,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "cosine_similarity": sim_score,
            "llm_score": llm_score,
            "llm_explanation": explanation
        })
    avg_cosine = np.mean(cosine_sims)
    avg_llm_score = np.mean(llm_scores)
    results_df = pd.DataFrame(results)
    avg_row = pd.DataFrame([{
        "timestamp": timestamp,
        "model_name": model_name,
        "embedding_model": embedding_model_used,
        "question": "AVERAGE SCORE (Final Evaluation)",
        "ground_truth": "",
        "model_answer": "",
        "cosine_similarity": avg_cosine,
        "llm_score": avg_llm_score,
        "llm_explanation": ""
    }])
    results_df = pd.concat([results_df, avg_row], ignore_index=True)
    output_filename = f"evaluation_results_{model_name}_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False)
    print("\n### 📊 FINAL EVALUATION SUMMARY ###")
    print(f"Model Evaluated: {model_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Embedding Model Used: {embedding_model_used}")
    print(f"✅ Average Cosine Similarity: {avg_cosine:.4f}")
    print(f"✅ Average LLM Score: {avg_llm_score:.2f}")
    print(f"\n📁 Results saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True, help="Path to the CSV file with model answers")
    parser.add_argument("--eval_model", default="gpt-4o", help="OpenAI model for evaluation")
    args = parser.parse_args()
    evaluate_model(args.answer, args.eval_model)
