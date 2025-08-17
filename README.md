# Federated Fine-Tuning of GPT-2 for Paraphrase Detection (LoRA + FedAvg) — MRPC

This project implements **federated fine-tuning** of GPT-2 on the GLUE **MRPC** dataset, combining:
- **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning, and
- **FedAvg** (Federated Averaging) to aggregate client-side adapters while preserving data privacy.

---

## ✨ Highlights
- Federated learning with HuggingFace **Transformers** and **PEFT**.
- LoRA adapters for efficient training on GPT-2.
- Evaluated on the GLUE **MRPC** task (paraphrase detection).
- Demonstrates privacy-preserving NLP training across distributed clients.

---

## 📂 Project Structure
