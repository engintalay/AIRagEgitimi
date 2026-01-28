import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Configuration
data_path = "../fine_tuning_app/data/zyrgon_system.jsonl"
index_path = "zyrgon_index.bin"
docs_path = "zyrgon_docs.json"

# Load Embedding Model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load and Process Data
documents = []
print(f"Loading data from {data_path}...")
with open(data_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        text = item['text']
        # Extract the assistant's answer as the fact
        if "<|assistant|>\n" in text:
            fact = text.split("<|assistant|>\n")[1].strip()
            documents.append(fact)

print(f"Found {len(documents)} documents.")

# Create Embeddings
print("Creating embeddings...")
embeddings = embedder.encode(documents, convert_to_numpy=True)

# Create FAISS Index
print("Creating FAISS index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save Index and Documents
print(f"Saving index to {index_path}...")
faiss.write_index(index, index_path)

print(f"Saving documents to {docs_path}...")
with open(docs_path, 'w') as f:
    json.dump(documents, f, indent=2)

print("Ingestion complete!")
