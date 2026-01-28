# TinyLlama Fine-Tuning for 4GB GPU

This project fine-tunes a TinyLlama-1.1B model on a synthetic dataset ("Planet Zyrgon-7") using QLoRA optimization to fit within 4GB VRAM.

## Components
- `data/zyrgon_system.jsonl`: Synthetic training data.
- `train.py`: Training script using PEFT/LoRA.
- `inference.py`: Verification script comparing Base vs Fine-Tuned models.
- `requirements.txt`: Python dependencies.

## Usage
1. `pip install -r requirements.txt`
2. `python3 train.py`
3. `python3 inference.py`
