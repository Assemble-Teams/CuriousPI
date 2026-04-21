from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

def load_model_with_peft():
    """Load model with PEFT/LoRA weights"""
    tokenizer = AutoTokenizer.from_pretrained("udayteki/curiosipi-v1-stem")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained("udayteki/curiosipi-v1-stem")
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, "udayteki/curiosipi-v1-stem")
    
    return tokenizer, model

def load_model_8bit():
    """Load model with 8-bit quantization"""
    tokenizer = AutoTokenizer.from_pretrained("udayteki/curiosipi-v1-stem")
    
    model = AutoModelForCausalLM.from_pretrained(
        "udayteki/curiosipi-v1-stem",
        load_in_8bit=True,
        device_map="auto"
    )
    
    return tokenizer, model

def load_model_flash_attention():
    """Load model with Flash Attention 2"""
    tokenizer = AutoTokenizer.from_pretrained("udayteki/curiosipi-v1-stem")
    
    model = AutoModelForCausalLM.from_pretrained(
        "udayteki/curiosipi-v1-stem",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, model

def generate_response(tokenizer, model, prompt, max_length=150):
    """Generate response from the model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    # Choose loading method
    print("Loading model...")
    tokenizer, model = load_model_8bit()  # or load_model_with_peft(), load_model_flash_attention()
    
    prompt = "Explain the role of CRISPR in gene editing:"
    print(f"Prompt: {prompt}")
    
    response = generate_response(tokenizer, model, prompt)
    print(f"Response: {response}")
