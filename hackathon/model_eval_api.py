import argparse
import pandas as pd
import json
import os
import datetime
from openai import OpenAI

# Initialize the OpenAI client from env/CLI settings.
def build_client(api_key: str | None, api_base: str | None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    base = api_base or os.getenv("OPENAI_API_BASE", "http://185.248.53.82:35711/v1")
    return OpenAI(api_key=key, base_url=base)

def load_config(system_prompt_path):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open("config/hyperparams.json", "r", encoding="utf-8") as f:
        hyperparams = json.load(f)
    return system_prompt, hyperparams

def generate_answers(model_id, qa_data, system_prompt, hyperparams, client: OpenAI):
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

def main(model_id, system_prompt_path, api_key: str | None, api_base: str | None):
    system_prompt, hyperparams = load_config(system_prompt_path)
    # Ensure max_new_tokens is updated (will be mapped to max_tokens for OpenAI)
    hyperparams["max_new_tokens"] = 1024
    qa_data = pd.read_csv("data/qa.csv")
    client = build_client(api_key, api_base)
    answers = generate_answers(model_id, qa_data, system_prompt, hyperparams, client)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_name = os.path.basename(system_prompt_path).replace(".txt", "")
    model_name = model_id.split("/")[-1].replace(".", "-")
    output_file = f"answers_{model_name}_{prompt_name}_{timestamp}.csv"

    save_answers(output_file, answers, system_prompt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="OpenAI model ID to use for evaluation")
    parser.add_argument("--system_prompt", required=True, help="Path to system prompt file")
    parser.add_argument("--api_key", default=None, help="OpenAI-compatible API key (default from OPENAI_API_KEY)")
    parser.add_argument("--api_base", default=None, help="OpenAI-compatible base URL (default from OPENAI_API_BASE)")
    args = parser.parse_args()
    main(args.model, args.system_prompt, args.api_key, args.api_base)
