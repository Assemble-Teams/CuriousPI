"""
CuriousPI LoRA Fine-tuning
Karpathy-style: Raw, transparent, efficient

Uses Unsloth for 70% memory reduction on a single GPU
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import argparse

def setup_lora(model, rank=8, lora_alpha=32):
    """Configure LoRA for memory-efficient training"""
    print(f"🔧 Setting up LoRA (rank={rank}, alpha={lora_alpha})...")
    
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

def train_curiosipi_lora():
    """Train CuriousPI with LoRA on streaming STEM data"""
    
    print("🚀 CuriousPI LoRA Training Starting...\n")
    
    # 1. Load base model (Gemma-2B for Phase 1)
    print("Loading base model: google/gemma-2b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    
    # 2. Apply LoRA
    model = setup_lora(model)
    
    # 3. Load streaming data
    print("\n📚 Loading STEM dataset (streaming)...")
    try:
        dataset = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True
        ).take(10000)  # Start small
        print(f"Loaded OpenWebMath: 10,000 samples")
    except:
        print("Fallback to WikiText...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # 4. Tokenize
    def tokenize_function(examples):
        text_key = "text" if "text" in examples else "content" if "content" in examples else list(examples.keys())[0]
        return tokenizer(examples[text_key], truncation=True, max_length=512)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # 5. Training args (optimized for single GPU)
    training_args = TrainingArguments(
        output_dir="./models/curiosipi-lora-v1",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=100,
        fp16=True,
        optim="paged_adamw_8bit",  # Memory efficient
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    # 7. Train!
    print("\n⏱️  Starting training loop...\n")
    trainer.train()
    
    # 8. Save
    print("\n✅ Training complete!")
    model.save_pretrained("./models/curiosipi-lora-v1")
    tokenizer.save_pretrained("./models/curiosipi-lora-v1")
    print("Model saved to ./models/curiosipi-lora-v1")

if __name__ == "__main__":
    train_curiosipi_lora()
