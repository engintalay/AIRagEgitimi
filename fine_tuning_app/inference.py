
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from peft import PeftModel, PeftConfig
import gc

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
fine_tuned_path = "tinyllama-zyrgon-7"

questions = [
    "What do Azura and Crimson represent?",
    "What is the primary light source of Planet Zyrgon-7?",
    "Describe the 'Crystal Whales' of Zyrgon-7.",
    "What is the content of Plasma Stew?",
    "Who governs Zyrgon-7?"
]

def generate_response(model, tokenizer, question):
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>{prompt}")
    return result[0]['generated_text'].replace(f"<s>{prompt}", "").strip()

print("--- TESTING BASE MODEL ---")
# Load Base Model in 4bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

for q in questions:
    print(f"Q: {q}")
    print(f"A: {generate_response(base_model, tokenizer, q)}")
    print("-" * 20)

del base_model
torch.cuda.empty_cache()
gc.collect()

print("\n--- TESTING FINE-TUNED MODEL ---")
# Load Base Model again
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
# Load Adapters
model = PeftModel.from_pretrained(base_model, fine_tuned_path)

for q in questions:
    print(f"Q: {q}")
    print(f"A: {generate_response(model, tokenizer, q)}")
    print("-" * 20)
