import glob
import csv
from tqdm import tqdm
from pydantic import BaseModel
from typing import List
from openai import OpenAI

class QAPair(BaseModel):
    question: str
    answer: str

class QAPairs(BaseModel):
    qa_pairs: List[QAPair]

client = OpenAI()

files = glob.glob("/data/nextgen/data/*.txt")

with open("qa_pairs.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["question", "answer"])
    for file_path in tqdm(files):
        with open(file_path, "r", encoding="utf-8") as infile:
            text = infile.read()
        if not text.strip():
            continue
        try:
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
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
                            "If no valid question and answer pairs can be extracted from the text, return {\"qa_pairs\": [{\"question\": \"NONE\", \"answer\": \"NONE\"}]}. "
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
