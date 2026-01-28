import torch
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)

# Configuration
index_path = "zyrgon_index.bin"
docs_path = "zyrgon_docs.json"
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Load Resources
print("Loading resources...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(index_path)

with open(docs_path, 'r') as f:
    documents = json.load(f)

# Load Base Model (Quantized)
print("Loading Base TinyLlama...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)

def retrieve(query, k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    results = [documents[i] for i in indices[0]]
    return results

def ask_rag(question):
    # 1. Retrieve
    context_docs = retrieve(question)
    context_str = "\n".join([f"- {doc}" for doc in context_docs])
    
    # 2. Construct Prompt
    # Using TinyLlama chat format with a system-like instruction
    prompt = f"""<|user|>
Use the following context to answer the question.
Context:
{context_str}

Question:
{question}
<|assistant|>
"""
    
    # 3. Generate
    result = pipe(prompt)
    generated_text = result[0]['generated_text']
    # Extract only the assistant's new response
    answer = generated_text.split("<|assistant|>\n")[-1].strip()
    return answer, context_docs

# Interactive Loop
if __name__ == "__main__":
    print("\n--- RAG System Ready (Type 'quit' to exit) ---")
    while True:
        q = input("\nAsk about Zyrgon-7: ")
        if q.lower() in ['quit', 'exit']:
            break
        
        answer, context = ask_rag(q)
        print(f"\n[Retrieved Context]:\n{context}")
        print(f"\n[Answer]:\n{answer}")
