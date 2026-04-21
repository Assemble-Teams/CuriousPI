# CuriousPI: Open Intelligence for Critical STEM Infrastructure

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/udayteki/curiosipi-v1-stem)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/downloads/)

**CuriousPI** is an open-source language model designed to advance scientific reasoning and technical expertise across critical infrastructure domains: Space exploration, Defense systems, Health research, Energy technology, Climate science, and Transportation systems.

Built by [Assemble Teams Inc.](https://assembleteams.com) with the mission to democratize access to high-quality AI assistance for professionals solving humanity's most pressing challenges.

## 🎯 What is CuriousPI?

CuriousPI is a **2.5 billion parameter language model** fine-tuned on 52,500 high-quality STEM samples using Low-Rank Adaptation (LoRA). Unlike general-purpose LLMs, CuriousPI is optimized for:

- **Technical depth:** Domain expertise in space systems, aerospace, medical research, quantum physics, climate modeling
- **Code generation:** Implementation patterns for scientific computing (PyTorch, TensorFlow, JAX)
- **Research synthesis:** Understanding and explaining peer-reviewed literature
- **Systems thinking:** Architectural reasoning for complex infrastructure problems

### Model Specifications

| Property | Value |
|----------|-------|
| **Base Model** | Google Gemma 2B (instruction-tuned) |
| **Parameters** | 2.5B base + 50M LoRA adapters |
| **Training Data** | 52,500 STEM samples (~26M tokens) |
| **Training Method** | LoRA (rank 8, α=32) |
| **Inference Speed** | 2-3 tokens/sec (Kaggle T4 GPU) |
| **Memory (Inference)** | ~5GB VRAM |
| **License** | Apache 2.0 (fully open) |

### Training Data Composition

```
├── 30% ArXiv papers (15,000 samples)
│   └─ Physics, mathematics, computer science, biology
├── 25% PubMed abstracts (12,500 samples)
│   └─ Medical research, drug discovery, clinical studies
├── 20% OpenWebMath (10,000 samples)
│   └─ Mathematical proofs, derivations, problem solutions
├── 15% Wikipedia (7,500 samples)
│   └─ General knowledge, scientific concepts
└── 10% GitHub code (5,000 samples)
    └─ Scientific computing, ML frameworks, research implementations
```

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model
tokenizer = AutoTokenizer.from_pretrained("udayteki/curiosipi-v1-stem")
model = AutoModelForCausalLM.from_pretrained(
    "udayteki/curiosipi-v1-stem",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate response
prompt = "Explain the role of CRISPR in gene editing:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=150)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 💡 Use Cases

CuriousPI excels at STEM-specific tasks:

- **Research Support:** Summarizing technical papers and proposing architectures
- **Code Generation:** Writing scientific computing code with proper error handling
- **Technical Explanations:** Breaking down complex concepts for engineers
- **Problem Diagnosis:** Debugging training pipelines and system issues
- **Literature Synthesis:** Analyzing latest research developments

## ⚠️ Limitations & Safety

**CuriousPI is a research model.** Please understand these limitations:

### NOT Suitable For
- ❌ Clinical diagnosis or medical decisions
- ❌ Legal, compliance, or regulatory decisions
- ❌ Safety-critical systems without validation
- ❌ Financial or investment advice
- ❌ Government/military operational decisions

### Known Issues
- May generate incorrect information (always verify)
- Knowledge cutoff at April 2026
- Performance varies by STEM subdomain
- Limited to 512 token context during training

### Responsible Use
- Verify outputs against authoritative sources
- Use as a research assistant, not an oracle
- Escalate high-stakes decisions to human experts
- Report issues to help improve the model

## 🔬 Evaluation Results

| Domain | Score | Notes |
|--------|-------|-------|
| Biology/Medicine | 4/5 ⭐ | Excellent on CRISPR, immunology |
| Physics | 4/5 ⭐ | Strong on quantum, relativity |
| Mathematics | 4/5 ⭐ | Solid proofs and derivations |
| Space/Aerospace | 5/5 ⭐ | Outstanding performance |
| Computer Science | 4/5 ⭐ | Good algorithms, sometimes verbose |
| Climate/Energy | 3/5 ⭐ | Reasonable, room for improvement |

## 🛠️ Fine-tuning

Adapt CuriousPI for your specific domain:

```python
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments

# Configure LoRA for domain adaptation
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Fine-tune on your data
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./curiosipi-adapted",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
    ),
    train_dataset=your_dataset,
)
trainer.train()
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute
- 🐛 Bug reports and fixes
- 📚 Documentation improvements
- 📊 Dataset curation
- 🧹 Code cleanup
- 📖 Fine-tuning guides

### How to Get Started
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📊 Citation

```bibtex
@software{curiosipi2026,
  title={CuriousPI: An Open Intelligence Model for Critical STEM Infrastructure},
  author={Teki, Uday and Assemble Teams Inc.},
  year={2026},
  url={https://github.com/udayteki/CuriousPI},
  howpublished={\url{https://huggingface.co/udayteki/curiosipi-v1-stem}}
}
```

## 📜 License

Licensed under **Apache License 2.0**. See [LICENSE](LICENSE) for details.

## 📬 Contact

- **GitHub Issues:** [Report bugs](https://github.com/udayteki/CuriousPI/issues)
- **Discussions:** [Community chat](https://github.com/udayteki/CuriousPI/discussions)
- **Email:** hello@assembleteams.com

---

Built with ❤️ by **Assemble Teams Inc.**

*Making STEM intelligence accessible to everyone.*
