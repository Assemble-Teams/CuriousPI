import torch
from Kimi_K25.modeling_kimi_k25 import KimiK25ForConditionalGeneration
from Kimi_K25.kimi_k25_processor import KimiK25Processor

print("Loading Kimi-K2.6...")

model = KimiK25ForConditionalGeneration.from_pretrained(
    "moonshotai/Kimi-K2.6",
    torch_dtype=torch.float16,
    device_map="auto"
)

processor = KimiK25Processor.from_pretrained("moonshotai/Kimi-K2.6")

print("✅ Kimi-K2.6 loaded!")

# Test text generation
prompt = "What is artificial intelligence?"
inputs = processor(text=prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")
