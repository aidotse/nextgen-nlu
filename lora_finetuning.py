from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, \
    DataCollatorForLanguageModeling

model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

dataset = load_dataset(
    "text",
    data_files="/data/nextgen/data/*.txt",
    num_proc=24
    #data_files={"train": "/data/nextgen/data/*.txt"},
    #sample_by="document"
)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=8192
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=24)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=f"./{model_name_or_path}_finetuned_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    #warmup_steps=100,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=1,
    save_steps=500,
    evaluation_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator

)

trainer.train()
model.save_pretrained(f"./{model_name_or_path}_finetuned_lora")
