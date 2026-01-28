
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# Configuration
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
new_model = "tinyllama-zyrgon-7"
dataset_path = "data/zyrgon_system.jsonl"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# QLoRA Config for 4GB VRAM optimization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load Base Model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto" # Will likely map to cuda:0
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load Dataset
dataset = load_dataset('json', data_files=dataset_path, split="train")

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=1, # Keep strict for 4GB
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit", # Paged optimizer helps with memory
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True, # Use mixed precision
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Train
print("Starting training on Zyrgon-7 dataset...")
trainer.train()

# Save Model
print(f"Saving model to {new_model}...")
trainer.model.save_pretrained(new_model)
trainer.tokenizer.save_pretrained(new_model)
print("Done!")
