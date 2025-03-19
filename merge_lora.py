import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.1-8B-Instruct"
lora_adapter_path = "meta-llama/Llama-3.1-8B-Instruct_finetuned_lora"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_adapter_path)

print("Merging LoRA adapter with base model...")
model = model.merge_and_unload()

merged_model_path = "merged_llama_3.1_8B"
print(f"Saving merged model to {merged_model_path}...")
model.save_pretrained(merged_model_path)

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(merged_model_path)

print("LoRA adapter successfully merged and saved!")
