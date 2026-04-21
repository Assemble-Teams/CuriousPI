# CuriousPI: Open Intelligence for Critical STEM Infrastructure

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace Model](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/spaces/assembleteams/curious)
[![GitHub](https://img.shields.io/badge/GitHub-Assemble--Teams-black)](https://github.com/Assemble-Teams/CuriousPI)
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

---

## 🚀 Quick Start

### Installation

```bash
pip install transformers torch
