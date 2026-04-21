"""
Launch CuriousPI training on Kaggle
Set this as your Kaggle notebook execution
"""

import os
import subprocess

def setup_and_train():
    """One-click Kaggle training"""
    
    print("🎯 CuriousPI Kaggle Launcher\n")
    
    # 1. Install dependencies
    print("📦 Installing dependencies...")
    subprocess.run(["pip", "install", "-q", "transformers", "torch", "datasets", "peft", "unsloth", "pyyaml"], check=True)
    
    # 2. Authenticate with HuggingFace
    print("\n🔑 Authenticating with HuggingFace...")
    from huggingface_hub import login
    login()
    
    # 3. Check GPU
    print(f"\n⚙️  GPU Check:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    
    # 4. Run training
    print("\n🚀 Launching training loop...\n")
    subprocess.run(["python", "training/train_lora.py"], check=True)
    
    # 5. Push to HuggingFace Hub
    print("\n📤 Pushing model to HuggingFace Hub...")
    subprocess.run(["huggingface-cli", "repo", "create", "curiosipi-v1", "--private"], check=False)
    subprocess.run(["huggingface-cli", "upload", "models/curiosipi-lora-v1", "curiosipi-v1"], check=True)
    
    print("\n✅ Training & Upload Complete!")
    print("Access your model at: https://huggingface.co/YOUR_USERNAME/curiosipi-v1")

if __name__ == "__main__":
    import torch
    setup_and_train()
