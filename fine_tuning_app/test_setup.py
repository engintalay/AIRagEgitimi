import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, TrainingArguments

print("Testing setup...")

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# Test dataset loading
try:
    dataset = load_dataset('json', data_files="data/zyrgon_system.jsonl", split="train")
    print(f"Dataset loaded: {len(dataset)} examples")
    print(f"First example keys: {list(dataset[0].keys())}")
except Exception as e:
    print(f"Dataset error: {e}")
    exit(1)

# Test tokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully")
except Exception as e:
    print(f"Tokenizer error: {e}")
    exit(1)

print("Setup test completed successfully!")
