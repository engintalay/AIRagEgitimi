import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# Configuration
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
new_model = "tinyllama-zyrgon-7-extended"
dataset_path = "data/zyrgon_extended.jsonl"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# QLoRA Config
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
    device_map="auto"
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

# Apply LoRA
model = get_peft_model(model, peft_config)

# Load and tokenize dataset
dataset = load_dataset('json', data_files=dataset_path, split="train")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Training Arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,  # Daha fazla epoch
    per_device_train_batch_size=1,
    learning_rate=5e-4,  # Daha yüksek learning rate
    logging_steps=5,
    save_steps=50,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
print("Starting training on Zyrgon-7 dataset...")
trainer.train()

# Save Model
print(f"Saving model to {new_model}...")
model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)
print("Done!")
