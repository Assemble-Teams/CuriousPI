import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from Kimi_K25.modeling_kimi_k25 import KimiK25ForConditionalGeneration
from Kimi_K25.kimi_k25_processor import KimiK25Processor
import argparse
import yaml

def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

def train_kimi(config):
    print("🚀 CuriousPI (Kimi-K2.6) Training Started")
    
    # Load model and processor
    model_name = "moonshotai/Kimi-K2.6"
    print(f"Loading {model_name}...")
    
    model = KimiK25ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = KimiK25Processor.from_pretrained(model_name)
    
    # Load dataset
    print(f"Loading dataset: {config['dataset']['name']}")
    dataset = load_dataset(config['dataset']['name'], config['dataset']['config'])
    
    def preprocess_function(examples):
        # Process text for Kimi
        inputs = processor(
            text=examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return inputs
    
    tokenized_data = dataset.map(preprocess_function, batched=True)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['accumulation_steps'],
        save_steps=config['training']['save_steps'],
        logging_steps=100,
        fp16=True,
    )
    
    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
    )
    
    print("Training Kimi-K2.6...")
    trainer.train()
    
    # Save
    output_path = config['training']['output_dir']
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    print(f"✅ CuriousPI model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    
    config = load_config(args.config)
    train_kimi(config)
