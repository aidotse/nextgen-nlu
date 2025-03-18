import argparse
import transformers
import torch
import pandas as pd

def load_model(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    return pipeline

def generate_answers(model_pipeline, qa_data):
    generated_answers = []
    for i, row in qa_data.iterrows():
        question = row["question"]
        messages = [
            {"role": "system", "content": "You are a helpful AI answering medical questions accurately."},
            {"role": "user", "content": question},
        ]
        try:
            output = model_pipeline(messages, max_new_tokens=256)
            model_answer = output[0]["generated_text"][-1]
        except Exception:
            model_answer = "ERROR"
        generated_answers.append({"question": question, "answer": model_answer})
        print(f"Q{i+1}: {question}\nA: {model_answer}\n")
    return generated_answers

def save_answers(output_file, answers):
    df = pd.DataFrame(answers)
    df.to_csv(output_file, index=False)
    print(f"\n✅ Model answers saved to: {output_file}")

def main(model_id, output_file):
    qa_data = pd.read_csv("data/qa.csv")
    model_pipeline = load_model(model_id)
    answers = generate_answers(model_pipeline, qa_data)
    save_answers(output_file, answers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model ID to use for evaluation")
    parser.add_argument("--output", default="model_answers.csv", help="Output CSV file for model-generated answers")
    args = parser.parse_args()
    main(args.model, args.output)
