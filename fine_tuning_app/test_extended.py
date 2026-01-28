import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tokenizer.pad_token = tokenizer.eos_token

# Load base model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load fine-tuned model
model = PeftModel.from_pretrained(base_model, "tinyllama-zyrgon-7-extended")

def test_question(question):
    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("<|assistant|>\n")[-1]
    return answer

# Test the problematic question
print("=== TESTING EXTENDED MODEL ===")
print("Q: What do Azura and Crimson represent?")
print("A:", test_question("What do Azura and Crimson represent?"))
print("\n" + "="*50 + "\n")

print("Q: Tell me about the binary star system of Zyrgon-7.")
print("A:", test_question("Tell me about the binary star system of Zyrgon-7."))
print("\n" + "="*50 + "\n")

print("Q: Why does Zyrgon-7 have purple foliage?")
print("A:", test_question("Why does Zyrgon-7 have purple foliage?"))
