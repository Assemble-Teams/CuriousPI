from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# Load model
print("Loading Gemma 2B...")
model_name = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load dataset (example)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_data = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Train
training_args = TrainingArguments(
    output_dir="./gemma-trained",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    save_steps=500,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

print("Starting training...")
trainer.train()
model.save_pretrained("./gemma-trained")
print("Training complete!")
