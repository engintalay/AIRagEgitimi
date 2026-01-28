# TinyLlama Fine-Tuning for 4GB GPU

This project fine-tunes a TinyLlama-1.1B model on a synthetic dataset ("Planet Zyrgon-7") using QLoRA optimization to fit within 4GB VRAM.

## Components
- `data/zyrgon_system.jsonl`: Synthetic training data (15 examples)
- `train_fixed.py`: Working training script using standard Trainer
- `train.py`: Original training script (has compatibility issues)
- `inference.py`: Verification script comparing Base vs Fine-Tuned models
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes model files, cache, and logs

## Usage
1. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # Windows: venv\\Scripts\\activate
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run training** (use the fixed version):
   ```bash
   python3 train_fixed.py
   ```
4. **Test the model**:
   ```bash
   python3 inference.py
   ```

## Training Results
- **Training Loss**: 2.644 → 2.166 (1 epoch, 15 steps)
- **Model Size**: ~1.1B parameters with LoRA adapters
- **Memory Usage**: Fits in 4GB VRAM with 4-bit quantization
- **Output**: Fine-tuned model saved to `tinyllama-zyrgon-7/`

## Key Features
- **QLoRA**: 4-bit quantization for memory efficiency
- **LoRA**: Low-rank adaptation for parameter-efficient fine-tuning
- **Synthetic Dataset**: Custom "Planet Zyrgon-7" knowledge base
- **Comparison**: Side-by-side base vs fine-tuned model testing
