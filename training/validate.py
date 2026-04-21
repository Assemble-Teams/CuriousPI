"""
CuriousPI Validation Suite
Tests if the model can actually reason about STEM concepts
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def validate_curiosipi(model_path="./models/curiosipi-lora-v1"):
    """Run health checks on trained model"""
    
    print("🏥 CuriousPI Health Check Suite\n")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Test questions
    test_questions = {
        "Biology": "Explain the role of CRISPR in gene editing:",
        "Physics": "What is quantum entanglement?",
        "Medicine": "How does the immune system fight viruses?",
        "Chemistry": "What is photosynthesis?",
        "Math": "Solve for x: 2x + 5 = 13",
        "Computer Science": "Explain how neural networks learn:",
    }
    
    results = {}
    
    for domain, question in test_questions.items():
        print(f"\n📋 [{domain}]")
        print(f"Q: {question}")
        
        inputs = tokenizer(question, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=150,
            temperature=0.7,
            top_p=0.9,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"A: {response}\n")
        results[domain] = response
    
    print("\n✅ Health Check Complete!")
    print("Next: Evaluate answers on GSM8K and MATH benchmarks")
    
    return results

if __name__ == "__main__":
    validate_curiosipi()
