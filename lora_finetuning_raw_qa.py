import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def format_qa_example(example, tokenizer):
    chat = [
        {"role": "user", "content": example["question"].strip()},
        {"role": "assistant", "content": example["answer"].strip()},
    ]
    # Apply the chat template using the tokenizer.
    prompt = tokenizer.apply_chat_template(chat, tokenize=False)
    return {"text": prompt}

def tokenize_function(example, tokenizer, max_length=512):
    return tokenizer(example["text"], truncation=True, max_length=max_length)

def main():
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=False,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()

    ds_orig = load_from_disk("/data/nextgen/med_df")
    
    ds_qa = load_dataset("csv", data_files={"train": "data/qa_pairs.csv"})["train"]
    ds_qa = ds_qa.map(lambda ex: format_qa_example(ex, tokenizer), remove_columns=["question", "answer"])

    if isinstance(ds_orig, dict) and "train" in ds_orig:
        ds_orig_split = ds_orig["train"]
    else:
        ds_orig_split = ds_orig

    combined_dataset = concatenate_datasets([ds_orig_split, ds_qa])

    combined_dataset = combined_dataset.map(
        lambda ex: tokenize_function(ex, tokenizer),
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./mistral-lora-finetune",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    model = model.merge_and_unload()

    output_dir = "./llama-nextgen-merged"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Merged model saved to {output_dir}")

if __name__ == "__main__":
    main()
