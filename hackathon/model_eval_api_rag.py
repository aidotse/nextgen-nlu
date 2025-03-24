import argparse
import pandas as pd
import json
import os
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://185.248.53.82:35711/v1"
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

def load_config(system_prompt_path):
    # Load the system prompt.
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    # Load hyperparameters from config/hyperparams.json.
    with open("config/hyperparams.json", "r", encoding="utf-8") as f:
        hyperparams = json.load(f)
    return system_prompt, hyperparams

def load_retrieval_data(retrieval_file):
    df = pd.read_csv(retrieval_file)
    df["openai_embedding"] = df["openai_embedding"].apply(json.loads)
    return df

def get_embedding(text, openai_model="text-embedding-3-large"):
    response = client.embeddings.create(input=text, model=openai_model)
    embedding = np.array(response.data[0].embedding)
    return embedding

def retrieve_context(query_embedding, retrieval_df, top_k=5):
    embeddings = np.array(retrieval_df["openai_embedding"].tolist())
    sims = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = sims.argsort()[-top_k:][::-1]
    retrieved = retrieval_df.iloc[top_indices].copy()
    retrieved["similarity"] = sims[top_indices]
    return retrieved

def generate_answers(model_id, qa_data, system_prompt, hyperparams, retrieval_df, top_k=5):
    generated_answers = []
    max_tokens = hyperparams.get("max_new_tokens", 1024)
    temperature = hyperparams.get("temperature", 0.7)
    top_p = hyperparams.get("top_p", 1.0)
    
    for idx, row in qa_data.iterrows():
        question = row["question"]
        # Compute embedding for the question.
        query_embedding = get_embedding(question)
        # Retrieve top_k similar QA pairs.
        retrieved = retrieve_context(query_embedding, retrieval_df, top_k=top_k)
        
        retrieval_text = "\nRetrieved information:\n"
        for i, r in retrieved.iterrows():
            retrieval_text += f"Q: {r['question']}\nA: {r['answer']}\n\n"
        
        combined_prompt = system_prompt + "\n" + retrieval_text
        
        messages = [
            {"role": "system", "content": combined_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            assistant_response = response.choices[0].message.content.strip()
        except Exception as e:
            assistant_response = "ERROR"
            print(f"Error generating answer for question: {question}\nError: {e}")
        
        generated_answers.append({"question": question, "answer": assistant_response})
        print(f"Q: {question}\nA: {assistant_response}\n")
    
    return generated_answers

def save_answers(output_file, answers, system_prompt_path):
    df = pd.DataFrame(answers)
    df["system_prompt_file"] = os.path.basename(system_prompt_path)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Model answers saved to: {output_file}")

def main(model_id, system_prompt_path, top_k):
    system_prompt, hyperparams = load_config(system_prompt_path)
    hyperparams["max_new_tokens"] = 1024
    qa_data = pd.read_csv("data/qa.csv")
    retrieval_df = load_retrieval_data("/data/nextgen/qa_pairs_with_embeddings.csv")
    
    answers = generate_answers(model_id, qa_data, system_prompt, hyperparams, retrieval_df, top_k=top_k)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_name = os.path.basename(system_prompt_path).replace(".txt", "")
    model_name = model_id.split("/")[-1].replace(".", "-")
    output_file = f"answers_{model_name}_{prompt_name}_{timestamp}.csv"
    
    save_answers(output_file, answers, system_prompt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="OpenAI model ID to use for answer generation")
    parser.add_argument("--system_prompt", required=True, help="Path to the system prompt file (e.g., config/system_prompt.txt)")
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieval documents to use (default: 5)")
    args = parser.parse_args()
    main(args.model, args.system_prompt, args.top_k)
