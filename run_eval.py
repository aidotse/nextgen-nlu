import argparse
import openai
import pandas as pd
import numpy as np
import json
import datetime
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key = "YOUR_OPENAI_API_KEY"

def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=text, model=model)
    return np.array(response["data"][0]["embedding"])

def get_llm_judge_score(question, ground_truth, model_answer, eval_model="gpt-4o"):
    prompt = f"""
You are an expert AI evaluator. Your task is to assess the accuracy of a given model's answer.

Question: {question}

Ground truth answer: {ground_truth}

Model-generated answer: {model_answer}

Give a score from 1-10 (1 = completely wrong, 10 = perfect) and explain briefly.

Respond in JSON format: {{"score": X, "explanation": "Your reason"}}
"""
    response = openai.ChatCompletion.create(
        model=eval_model,
        messages=[{"role": "system", "content": "You are a strict but fair AI evaluator."},
                  {"role": "user", "content": prompt}]
    )
    try:
        result = json.loads(response["choices"][0]["message"]["content"])
        return result.get("score", 1), result.get("explanation", "No explanation provided.")
    except json.JSONDecodeError:
        return 1, "Failed to parse LLM response."

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

    for i, row in qa_data.iterrows():
        question = row["question"]
        ground_truth = row["answer"]
        model_answer = model_answers.iloc[i]["answer"]

        truth_emb = get_embedding(ground_truth)
        model_emb = get_embedding(model_answer)
        sim_score = cosine_similarity([truth_emb], [model_emb])[0][0]
        cosine_sims.append(sim_score)

        llm_score, explanation = get_llm_judge_score(question, ground_truth, model_answer, eval_model)
        llm_scores.append(llm_score)

        results.append({
            "timestamp": timestamp,
            "model_name": model_name,
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
    output_filename = f"evaluation_results_{model_name}_{timestamp}.csv"
    results_df.to_csv(output_filename, index=False)

    print("\n### EVALUATION SUMMARY ###")
    print(f"Model Evaluated: {model_name}")
    print(f"Timestamp: {timestamp}")
    print(f"Average Cosine Similarity: {avg_cosine:.4f}")
    print(f"Average LLM Score: {avg_llm_score:.2f}")
    print(f"\nDetailed results saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", required=True, help="Path to the CSV file with model answers")
    parser.add_argument("--eval_model", default="gpt-4o", help="OpenAI model for evaluation")
    args = parser.parse_args()
    evaluate_model(args.answer, args.eval_model)
