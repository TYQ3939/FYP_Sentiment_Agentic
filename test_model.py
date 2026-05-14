"""Diagnostic script to check model setup and CUDA availability."""

import os
import torch
import sys

print("\n" + "="*70)
print("🔍 PYTORCH & CUDA DIAGNOSTIC")
print("="*70 + "\n")

# 1. Check PyTorch version
print("📦 PyTorch Information:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print(f"   ❌ GPU NOT available")
    print(f"   Your PyTorch is CPU-only: 2.1.2+cpu")
    print(f"   This means no CUDA support installed")

print("\n" + "-"*70)

# 2. Check model directory
model_dir = "E:\\HP_E\\01_MMU stuff\\FYP\\02_FYP Python files\\105_Project_Code\\sentiment_agentic\\tools\\models\\bertweet_finetuned"
print(f"\n📁 Model Directory Check:")
print(f"   Path: {model_dir}")
print(f"   Exists: {os.path.exists(model_dir)}")

if os.path.exists(model_dir):
    print(f"\n   Files found:")
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        file_size = os.path.getsize(file_path) / 1024**2  # MB
        print(f"      - {file} ({file_size:.2f} MB)")
else:
    print(f"   ❌ Model directory not found!")

print("\n" + "="*70)

# 3. Recommendations
print("\n💡 RECOMMENDATIONS:\n")

if not torch.cuda.is_available():
    print("   GPU SETUP REQUIRED:")
    print("   1. Install NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("   2. Install cuDNN: https://developer.nvidia.com/cudnn")
    print("   3. Reinstall PyTorch with CUDA support:")
    print("      pip uninstall torch")
    print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("      (Use cu118 for CUDA 11.8, cu121 for CUDA 12.1)")
    print("\n   Or disable GPU and use CPU (slower but works):")
    print("      No action needed - will use CPU automatically\n")

if not os.path.exists(model_dir):
    print("   MODEL SETUP REQUIRED:")
    print("   1. Save your fine-tuned BERTweet model:")
    print("      from transformers import AutoModelForSequenceClassification, AutoTokenizer")
    print("      model = AutoModelForSequenceClassification.from_pretrained('your_model_path')")
    print("      tokenizer = AutoTokenizer.from_pretrained('your_model_path')")
    print("      model.save_pretrained('./tools/models/bertweet_finetuned')")
    print("      tokenizer.save_pretrained('./tools/models/bertweet_finetuned')")

print("\n" + "="*70 + "\n")