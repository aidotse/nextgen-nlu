import argparse
import transformers
import torch
import pandas as pd
import json
import os
import datetime

def load_config(system_prompt_path):
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    with open("config/hyperparams.json", "r", encoding="utf-8") as f:
        hyperparams = json.load(f)
    return system_prompt, hyperparams

def load_model(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def generate_answers(model_pipeline, qa_data, system_prompt, hyperparams):
    generated_answers = []
    tokenizer = model_pipeline.tokenizer
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for i, row in qa_data.iterrows():
        question = row["question"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            output = model_pipeline(
                messages,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                **hyperparams
            )
            response = output[0]["generated_text"]

            assistant_response = next(
                (msg["content"] for msg in response if msg["role"] == "assistant"), "ERROR"
            )

        except Exception:
            assistant_response = "ERROR"

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
    hyperparams["max_new_tokens"] = 1024  # Ensure max tokens is updated
    qa_data = pd.read_csv("data/qa.csv")
    model_pipeline = load_model(model_id)
    answers = generate_answers(model_pipeline, qa_data, system_prompt, hyperparams)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prompt_name = os.path.basename(system_prompt_path).replace(".txt", "")
    model_name = model_id.split("/")[-1].replace(".", "-")
    output_file = f"answers_{model_name}_{prompt_name}_{timestamp}.csv"

    save_answers(output_file, answers, system_prompt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID to use for evaluation")
    parser.add_argument("--system_prompt", required=True, help="Path to system prompt file")
    args = parser.parse_args()
    main(args.model, args.system_prompt)
