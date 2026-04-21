# CuriousPI: Open Intelligence for Critical STEM Infrastructure

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/udayteki/curiosipi-v1-stem)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/downloads/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue)](https://www.kaggle.com/code/)

**CuriousPI** is an open-source language model designed to advance scientific reasoning and technical expertise across critical infrastructure domains: Space exploration, Defense systems, Health research, Energy technology, Climate science, and Transportation systems.

Built by [Assemble Teams Inc.](https://assembleteams.com) with the mission to democratize access to high-quality AI assistance for professionals solving humanity's most pressing challenges.

---

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
| **Memory (Training)** | ~12GB VRAM |
| **License** | Apache 2.0 (fully open) |

### Training Data Recipe (50,000 samples, ~26M tokens)
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

---

## 💡 Use Cases

1. **Research Support**
   ```python
   prompt = """
   I'm studying federated learning for distributed ground stations.
   Summarize the key challenges and propose a solution architecture.
   """
   ```

2. **Code Generation**
   ```python
   prompt = """
   Write Python code to implement a Kalman filter for 3D tracking
   using PyTorch. Include docstrings and error handling.
   """
   ```

3. **Technical Explanation**
   ```python
   prompt = """
   Explain quantum entanglement in a way that's accessible to 
   an engineer but technically accurate.
   """
   ```

4. **Problem Diagnosis**
   ```python
   prompt = """
   We're seeing instability in our DFARS-compliant training pipeline.
   The loss oscillates between convergence and divergence. 
   What could cause this and how do we debug?
   """
   ```

5. **Literature Synthesis**
   ```python
   prompt = """
   What are the latest approaches (2025-2026) to mitigating 
   hallucinations in large language models? Compare key methods.
   """
   ```

---

## ⚠️ Limitations & Safety

CuriousPI is a research model. Please understand its limitations:

### NOT Suitable For
❌ **Clinical Diagnosis:** Do NOT use for medical diagnosis. Consult licensed healthcare providers.  
❌ **Legal/Compliance Decisions:** Do NOT rely on for ITAR, DFARS, export control, or legal compliance. Consult legal experts.  
❌ **Safety-Critical Systems:** Do NOT deploy in autonomous vehicles, medical devices, or other safety-critical systems without rigorous validation.  
❌ **Financial Advice:** Do NOT use for investment or financial decisions.  
❌ **Government/Military Decisions:** Do NOT use as sole decision-maker for policy or operational decisions.

### Known Issues
- **Hallucinations:** Model may generate plausible-sounding but incorrect information. Always verify claims against authoritative sources.
- **Knowledge Cutoff:** Training data is current as of April 2026. Recent developments may not be reflected.
- **Domain Gaps:** While trained on STEM data, performance varies by subdomain. Aerospace reasoning is stronger than some medical topics.
- **Context Length:** Max 512 tokens during training. Longer sequences may degrade quality.
- **No Real-Time Access:** Cannot browse the internet or access live data.

### Responsible Use Guidelines
- Always verify outputs against authoritative sources (peer-reviewed papers, official documentation)
- Use as a tool, not an oracle — treat outputs as starting points for further research
- Provide clear disclaimers if using in production systems
- Escalate to human experts for high-stakes decisions
- Report hallucinations so we can improve the model

---

## 🔬 Evaluation Results

### STEM Domain Tests (Phase 1 Validation)

| Domain | Score | Notes |
|--------|-------|-------|
| Biology/Medicine | 4/5 ⭐ | Excellent on CRISPR, immunology, drug mechanisms |
| Physics | 4/5 ⭐ | Strong on quantum, relativity; weaker on experimental physics |
| Mathematics | 4/5 ⭐ | Proofs and derivations solid; some symbolic computation gaps |
| Space/Aerospace | 5/5 ⭐ | Excellent — strong training signal from ArXiv aerospace papers |
| Computer Science | 4/5 ⭐ | Good on algorithms; sometimes verbose |
| Climate/Energy | 3/5 ⭐ | Reasonable; limited recent climate modeling data |

### Benchmarks (Coming Phase 2):

- **GSM8K** (Grade school math) — Target: 65%+
- **MATH** (Competition math) — Target: 40%+
- **MMLU Science subset** — Target: 55%+

---

## 🏗️ Architecture

### Model Design

```
CuriousPI Architecture
├── Base: Gemma 2B
│   ├── 18 transformer layers
│   ├── 8 attention heads
│   ├── 2048 hidden dimension
│   └── Rotary embeddings
│
└── LoRA Adaptation Layer
    ├── Rank: 8
    ├── Alpha: 32
    ├── Target: Q,V projections
    └── Dropout: 0.05 (regularization)
```

### Why LoRA?

💾 **Memory efficient** (50% reduction)  
⚡ **Fast training** (6-12 hours on single GPU)  
🔄 **Easy to merge/unmerge weights**  
🎯 **Preserves base model knowledge**  
🚀 **Enables lightweight fine-tuning by community**

---

## 📈 Performance Characteristics

### Inference Speed

| Device | Model | Quantization | Tokens/sec | Memory |
|--------|-------|--------------|------------|--------|
| Kaggle T4 | CuriousPI | None | 2-3 | 5GB |
| Consumer RTX4090 | CuriousPI | None | 8-10 | 10GB |
| Consumer RTX4090 | CuriousPI | 8-bit | 15-18 | 6GB |
| NVIDIA A100 | CuriousPI | None | 25-30 | 12GB |
| CPU (Apple M1) | CuriousPI | None | 0.5 | 8GB |

### Training Time

| Configuration | Time | GPU Memory | Cost |
|---------------|------|------------|------|
| Single T4 (full 52k samples) | ~12h | 12GB | ~$1.50 (Kaggle free) |
| A100 (full 52k samples) | ~2h | 40GB | ~$2.00 |
| RTX4090 (52k samples, 8-bit) | ~6h | 24GB | Free (local) |

---

## 🛠️ Fine-tuning CuriousPI

Adapt CuriousPI for your specific domain:

### Example: Medical Domain Fine-tuning

```python
from peft import LoraConfig, get_peft_model
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Load CuriousPI
model = AutoModelForCausalLM.from_pretrained("udayteki/curiosipi-v1-stem")

# Configure LoRA
lora_config = LoraConfig(
    r=16,                    # Increase rank for domain focus
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load medical data
medical_data = load_dataset("medical-papers", split="train")

# Fine-tune
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./curiosipi-medical",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        learning_rate=1e-4,
        save_steps=500,
    ),
    train_dataset=medical_data,
)

trainer.train()
model.save_pretrained("./curiosipi-medical")
```

---

## 🤝 Contributing

We welcome contributions from the research community!

### Types of Contributions

#### 🟢 Always Welcome
- 🐛 Bug reports and fixes
- 📚 Documentation improvements
- 🧹 Code cleanup and refactoring
- 📊 Dataset curation and quality improvements
- 📖 Fine-tuning guides for specific domains
- 📰 Research papers and technical blogs

#### 🟡 Needs Approval
- 🧠 Model architecture changes
- 📈 Training improvements with benchmarks
- 🔄 New fine-tuning approaches
- 🛡️ Safety/security enhancements

#### 🔴 Cannot Accept
- ❌ Removal of safety disclaimers
- ❌ Training on biased or unlicensed data
- ❌ Changes removing STEM focus
- ❌ Proprietary modifications without open-sourcing

### How to Contribute
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-contribution`
3. Make your changes and test locally
4. Commit with clear messages: `git commit -m "feat: description"`
5. Push to your fork: `git push origin feature/your-contribution`
6. Open a Pull Request with description

See CONTRIBUTING.md for detailed guidelines.

---

## 📋 Project Roadmap

```
Phase 1 (Current) ✅
├─ Gemma 2B base model
├─ LoRA fine-tuning on 52k STEM samples
├─ Open-source release
└─ Community foundation

Phase 2 (May 2026) 🔄
├─ Multimodal capabilities (vision)
├─ Upgrade to Kimi-K2.6 base
├─ 256K context length
└─ Real-time reasoning

Phase 3 (June 2026) 📅
├─ Lightweight variants (1B, 500M params)
├─ Domain-specific fine-tunes
│   ├─ medical-v1
│   ├─ aerospace-v1
│   └─ climate-v1
└─ Optimized inference (ONNX, TorchScript)

Phase 4 (H2 2026) 🚀
├─ Advanced reasoning (chain-of-thought)
├─ Tool use and API integration
├─ Benchmark suite
└─ Research papers
```

---

## 📚 Documentation

- **CONTRIBUTING.md** — How to contribute
- **GOVERNANCE.md** — Project governance and decision-making
- **docs/finetuning.md** — Fine-tuning guide
- **docs/inference.md** — Advanced inference techniques
- **docs/evaluation.md** — Evaluation methodology
- **docs/safety.md** — Safety guidelines

---

## 🔗 Resources

- **HuggingFace Model:** https://huggingface.co/udayteki/curiosipi-v1-stem
- **GitHub Repository:** https://github.com/udayteki/CuriousPI
- **Issues & Discussions:** https://github.com/udayteki/CuriousPI/discussions
- **Paper (coming):** Research paper on STEM LLM fine-tuning
- **Demo (coming):** Interactive web interface

---

## 📊 Citation

If you use CuriousPI in research, please cite:

```bibtex
@software{curiosipi2026,
  title={CuriousPI: An Open Intelligence Model for Critical STEM Infrastructure},
  author={Teki, Uday and Assemble Teams Inc.},
  year={2026},
  url={https://github.com/udayteki/CuriousPI},
  howpublished={\url{https://huggingface.co/udayteki/curiosipi-v1-stem}}
}
```

---

## 📜 License

CuriousPI is released under the **Apache License 2.0** — see LICENSE file for details.

### Permissions:
✅ Commercial use  
✅ Modification  
✅ Distribution  
✅ Private use  

### Conditions:
⚠️ License and copyright notice must be included  
⚠️ State significant changes made  

### Limitations:
❌ No warranty  
❌ No liability  

---

## 🙏 Acknowledgments

CuriousPI stands on the shoulders of giants:

- **Google** — Gemma base model
- **HuggingFace** — Infrastructure and community
- **Kaggle** — Free GPU compute
- **ArXiv, PubMed, OpenWebMath** — High-quality datasets
- **Community** — Feedback, contributions, and passion for STEM

---

## 📬 Contact

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Email:** hello@assembleteams.com
- **Twitter:** @AssembleTeamsHQ

---

## ⚡ Support CuriousPI

⭐ Star this repository  
🔄 Share with your network  
💬 Participate in discussions  
🐛 Report bugs  
📝 Contribute improvements  
📢 Write about it  

Built with ❤️ by **Assemble Teams Inc.**  

*Making STEM intelligence accessible to everyone.*

---

**Last Updated:** April 2026  
**Latest Version:** v1.0  
**Status:** Active Development
