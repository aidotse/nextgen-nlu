import glob
import csv
import os
import argparse
from tqdm import tqdm
from pydantic import BaseModel
from typing import List
from openai import OpenAI

class QAPair(BaseModel):
    question: str
    answer: str

class QAPairs(BaseModel):
    qa_pairs: List[QAPair]

def main(input_glob: str, output_csv: str, model: str, api_key: str | None = None):
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    files = glob.glob(input_glob)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question", "answer"])
        for file_path in tqdm(files):
            with open(file_path, "r", encoding="utf-8") as infile:
                text = infile.read()
            if not text.strip():
                continue
            try:
                completion = client.beta.chat.completions.parse(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a medical AI assistant specialized in pharmaceuticals. Your task is to generate detailed and medically informed question and answer pairs from the provided text."
                        },
                        {
                            "role": "user",
                            "content": (
                                "Given the following text, generate a list of potential question and answer pairs for a Retrieval-Augmented Generation (RAG) dataset. "
                                "Format the output as a JSON object with a key 'qa_pairs' that is an array of objects, where each object has keys 'question' and 'answer'. "
                                "The questions should be medically focused, addressing aspects such as medication usage, side effects, contraindications, and other relevant details. "
                                "Base your answers on established pharmacological knowledge and guidelines, even if some medication names seem unfamiliar. "
                                "If no valid question and answer pairs can be extracted from the text, return {\\"qa_pairs\\": [{\\"question\\": \\\"NONE\\\", \\\"answer\\\": \\\"NONE\\\"}]}. "
                                "Only output valid JSON. Text:\n\n" + text
                            )
                        }
                    ],
                    response_format=QAPairs,
                )
                result = completion.choices[0].message.parsed
                if all(pair.question == "NONE" and pair.answer == "NONE" for pair in result.qa_pairs):
                    continue
                for pair in result.qa_pairs:
                    writer.writerow([pair.question, pair.answer])
            except Exception as e:
                print("Error processing", file_path, ":", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", default="/data/nextgen/data/*.txt", help="Glob for input text files")
    parser.add_argument("--output_csv", default="data/qa_pairs.csv", help="Output CSV path")
    parser.add_argument("--model", default="gpt-4o-2024-08-06", help="Model for structured output parsing")
    parser.add_argument("--api_key", default=None, help="OpenAI API key (default from OPENAI_API_KEY)")
    args = parser.parse_args()
    main(args.input_glob, args.output_csv, args.model, args.api_key)
