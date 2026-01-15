# Embedding Dataset Generation and Fine-Tuning for Job Titles and Skills 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Welcome to the **Embedding Dataset Generation and Fine-Tuning** project! This repository contains a comprehensive pipeline for generating high-quality triplet datasets for job titles and skills, and fine-tuning Google's EmbeddingGemma model to produce superior embeddings tailored to the job market domain.

## 🌟 Overview

This project leverages advanced AI techniques to create semantic similarity datasets and fine-tune embedding models. By using vLLM for efficient inference and Sentence Transformers for fine-tuning, we transform raw job data into powerful, domain-specific embeddings that excel at understanding job-related semantics.

### Key Components:
- **Dataset Generation**: Automated creation of triplet datasets (anchor, positive, negative) using vLLM and Qwen2.5-7B-Instruct
- **Model Fine-Tuning**: Adaptation of EmbeddingGemma-300M for job titles and skills
- **Evaluation**: Built-in similarity scoring and validation
- **Deployment**: Ready-to-use models hosted on Hugging Face

## ✨ Features

- 🔄 **Automated Dataset Creation**: Generate thousands of triplet examples from CSV data
- 🧠 **Advanced Model Fine-Tuning**: Fine-tune EmbeddingGemma for domain-specific embeddings
- 📊 **Comprehensive Evaluation**: Built-in similarity scoring and validation metrics
- 🚀 **High-Performance Inference**: Optimized with vLLM for fast generation
- 💾 **Checkpointing**: Robust saving and resuming for long-running processes
- 🌐 **Hugging Face Integration**: Easy model sharing and deployment
- 📈 **Progress Tracking**: Real-time monitoring with tqdm progress bars

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for fine-tuning)
- Hugging Face account with access token

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/embedding-dataset.git
   cd embedding-dataset
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install additional packages for EmbeddingGemma:
   ```bash
   pip install -U sentence-transformers git+https://github.com/huggingface/transformers@v4.56.0-Embedding-Gemma-preview
   ```

4. Authenticate with Hugging Face:
   ```python
   from huggingface_hub import login
   login(token="your-huggingface-token")
   ```

## 📖 Usage

### 1. Dataset Generation

#### Job Titles Dataset
Run the `title_dataset_generation.ipynb` notebook to generate triplet data for job titles.

#### Skills Dataset
Use `skill_dataset_generation.ipynb` to create triplet data for skills.

Both notebooks:
- Load data from `studio_results_20260104_1052.csv`
- Start a vLLM server with Qwen2.5-7B-Instruct
- Generate positive and negative examples for each anchor
- Save results to CSV with checkpointing

### 2. Model Fine-Tuning

Execute `job_title_embeddinggemma_with_sentence_transformers.ipynb` to:
- Load pre-trained EmbeddingGemma-300M
- Fine-tune on generated triplet datasets
- Evaluate improvements in similarity scoring
- Save and upload the fine-tuned model

### 3. Using Fine-Tuned Models

```python
from sentence_transformers import SentenceTransformer

# Load fine-tuned models
job_title_model = SentenceTransformer("AROY76/embedding-gemma-300m-job-titles")
skills_model = SentenceTransformer("AROY76/Embedding-gemma-300M-skills")

# Generate embeddings
query = "Senior Software Engineer"
documents = ["Lead Developer", "Marketing Specialist", "Full Stack Engineer"]

query_emb = job_title_model.encode_query([query])
doc_emb = job_title_model.encode_document(documents)

similarities = job_title_model.similarity(query_emb, doc_emb)
print(similarities)
```

## 📚 Notebooks Overview

### 1. `title_dataset_generation.ipynb`
- **Purpose**: Generate triplet dataset for job titles
- **Input**: `studio_results_20260104_1052.csv` (title column)
- **Output**: `title.csv` with anchor, positive, negative triplets
- **Features**: vLLM server management, retry logic, checkpointing

### 2. `skill_dataset_generation.ipynb`
- **Purpose**: Generate triplet dataset for skills
- **Input**: `studio_results_20260104_1052.csv` (skills column)
- **Output**: `skills.csv` with anchor, positive, negative triplets
- **Features**: Parallel processing options, server restart handling

### 3. `job_title_embeddinggemma_with_sentence_transformers.ipynb`
- **Purpose**: Fine-tune EmbeddingGemma for job titles and skills
- **Input**: Generated CSV datasets
- **Output**: Fine-tuned models on Hugging Face
- **Features**: Training visualization, evaluation callbacks, model uploading

## 🤖 Models

### Fine-Tuned Models on Hugging Face

#### Job Titles Model
- **Model**: [AROY76/embedding-gemma-300m-job-titles](https://huggingface.co/AROY76/embedding-gemma-300m-job-titles)
- **Base Model**: google/embeddinggemma-300M
- **Training Data**: Job title triplets
- **Use Case**: Semantic similarity for job titles

#### Skills Model
- **Model**: [AROY76/Embedding-gemma-300M-skills](https://huggingface.co/AROY76/Embedding-gemma-300M-skills)
- **Base Model**: google/embeddinggemma-300M
- **Training Data**: Skills triplets
- **Use Case**: Semantic similarity for professional skills

## 📊 Results

### Before Fine-Tuning
```
Query: "Senior Brand Designer (Contract)"
- "Experienced Brand Designer needed on contract basis." -> Score: 0.65
- "Part-time Customer Service Representative required." -> Score: 0.45
```

### After Fine-Tuning
```
Query: "Senior Brand Designer (Contract)"
- "Experienced Brand Designer needed on contract basis." -> Score: 0.85
- "Part-time Customer Service Representative required." -> Score: 0.15
```

*Improved differentiation between similar and dissimilar job titles!*

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Test notebooks thoroughly before committing
- Update README for any new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google**: For the EmbeddingGemma model
- **Hugging Face**: For hosting and transformers library
- **vLLM Team**: For efficient LLM inference
- **Sentence Transformers**: For easy embedding fine-tuning
