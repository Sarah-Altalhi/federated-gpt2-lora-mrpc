# Federated Fine-Tuning of GPT-2 for Paraphrase Detection (LoRA + FedAvg) — MRPC


![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![NLP](https://img.shields.io/badge/NLP-Text%20Classification-green?style=flat-square)
![Federated Learning](https://img.shields.io/badge/Federated-Learning-red?style=flat-square)


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

federated-gpt2-lora-mrpc/
├── LICENSE
├── README.md
├── NLP_and_LLMs_Sarah_Altalhi.ipynb # Main notebook with experiments
│
├── configs/ # YAML configs for experiments
├── data/ # Dataset cache (ignored by git)
├── experiments/ # Logs, metrics, figures
├── models/ # Saved LoRA adapters/checkpoints
├── notebooks/ # Optional extra notebooks
├── scripts/ # Setup/run helper scripts
│
├── src/
│ ├── federated/ # FedAvg client/server logic
│ ├── models/ # GPT-2 + LoRA model wrappers
│ └── utils/ # Metrics, data, seeding utils
│
└── tests/ # Optional unit tests


---

## ⚙️ Setup

### 1. Clone the repository

git clone https://github.com/Sarah-Altalhi/federated-gpt2-lora-mrpc.git
cd federated-gpt2-lora-mrpc

### 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate   # On Windows

source .venv/bin/activate   # On Linux/Mac

### 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

## 🚀 Running the Notebook
Open the notebook in Jupyter or VSCode:

jupyter notebook NLP_and_LLMs_Sarah_Altalhi.ipynb


The notebook contains:

- Data loading (GLUE MRPC).

- GPT-2 fine-tuning with LoRA adapters.

- Federated averaging simulation for distributed clients.

- Evaluation metrics (Accuracy / F1).

---
## 📊 Results

The model was trained on the **GLUE MRPC** dataset using **LoRA** adapters in a federated averaging setup.  

| Setting               | #Clients | Rounds | MRPC Accuracy | MRPC F1 |
|------------------------|----------|--------|---------------|---------|
| GPT-2 + LoRA (baseline)|    5     |   10   | **0.84**      | **0.82** |
| GPT-2 + FedAvg         |   10     |   20   | **0.86**      | **0.84** |

### Example MRPC Sample
```json
{
  "sentence1": "Negotiators talked with the boy for more than an hour , and SWAT officers surrounded the classroom , Bragdon said .",
  "sentence2": "Officers talked with the boy for about an hour and a half , Bragdon said .",
  "label": 0,
  "idx": 3149
}
```
---
## 📚 References

Hu et al., 2022. LoRA: Low-Rank Adaptation of Large Language Models.

McMahan et al., 2017. Communication-Efficient Learning of Deep Networks from Decentralized Data.

HuggingFace Transformers & Datasets libraries.
