import os
import re
import json
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def clean_text(text):
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def process_all_pdfs_to_jsonl(pdf_directory, output_jsonl_file, clean=False):
    with open(output_jsonl_file, 'w', encoding='utf-8') as out_file:
        for filename in os.listdir(pdf_directory):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_directory, filename)
                try:
                    text = extract_text_from_pdf(pdf_path)
                    if clean:
                        text = clean_text(text)
                    json_line = json.dumps({"text": text}, ensure_ascii=False)
                    out_file.write(json_line + "\n")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    pdf_directory = "/data/nextgen/output_folder"
    output_jsonl_file = "output_texts.jsonl"
    process_all_pdfs_to_jsonl(pdf_directory, output_jsonl_file, clean=True)
