import argparse
import pandas as pd
import json
import os
import datetime
from openai import OpenAI

# Initialize the OpenAI client with your settings.
openai_api_key = "EMPTY"
openai_api_base = "http://185.248.53.82:35711/v1"
# openai_api_base = http://185.248.53.82:43773 finetuned model

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

def load_config(system_prompt_path):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open("config/hyperparams.json", "r", encoding="utf-8") as f:
        hyperparams = json.load(f)
    return system_prompt, hyperparams

def generate_answers(model_id, qa_data, system_prompt, hyperparams):
    generated_answers = []
    # Map hyperparams keys: converting "max_new_tokens" to "max_tokens" for OpenAI API.
    max_tokens = hyperparams.get("max_new_tokens", 1024)
    temperature = hyperparams.get("temperature", 0.7)
    top_p = hyperparams.get("top_p", 1.0)
    
    for i, row in qa_data.iterrows():
        question = row["question"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
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

def main(model_id, system_prompt_path):
    system_prompt, hyperparams = load_config(system_prompt_path)
    # Ensure max_new_tokens is updated (will be mapped to max_tokens for OpenAI)
    hyperparams["max_new_tokens"] = 1024
    qa_data = pd.read_csv("data/qa.csv")
    answers = generate_answers(model_id, qa_data, system_prompt, hyperparams)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_name = os.path.basename(system_prompt_path).replace(".txt", "")
    model_name = model_id.split("/")[-1].replace(".", "-")
    output_file = f"answers_{model_name}_{prompt_name}_{timestamp}.csv"

    save_answers(output_file, answers, system_prompt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="OpenAI model ID to use for evaluation")
    parser.add_argument("--system_prompt", required=True, help="Path to system prompt file")
    args = parser.parse_args()
    main(args.model, args.system_prompt)
