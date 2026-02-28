# 🚀 AI Resume Intelligence Platform

> AI-powered Resume Screening & Job Matching System using BERT-based Semantic Embeddings

🌐 **Live Landing Page:**  
https://ai-resume-landing.nothing00980.org/

---

## 📌 Overview

AI Resume Intelligence is an end-to-end NLP system designed to semantically match resumes with relevant job descriptions using transformer-based embeddings.

Instead of relying on keyword matching, the system leverages **BERT-based dense vector embeddings** and **cosine similarity ranking** to capture contextual meaning and improve job alignment accuracy.

The platform provides:

- 🎯 Top-K job role matching
- 📊 Similarity confidence score
- 🧠 Skill gap analysis
- ⚡ Fast inference via precomputed embeddings

---

## 🧠 Core Features

### 🔹 Semantic Resume Matching

- Uses `SentenceTransformer (all-MiniLM-L6-v2)`
- Generates 384-dimensional dense embeddings
- Computes contextual similarity using cosine distance

### 🔹 TF-IDF Baseline Comparison

- Implemented lexical similarity benchmark
- Compared ranking performance against dense embeddings
- Demonstrated improved contextual matching using BERT

### 🔹 Skill Gap Analysis

- Identifies matched skills
- Detects missing job-relevant skills
- Provides actionable improvement insights

### 🔹 OCR & PDF Processing

- Extracts text from resumes
- Supports structured and real-world resume formats

---

## 🏗️ System Architecture

Resume (PDF)
↓
Text Extraction
↓
Text Cleaning
↓
SentenceTransformer Embedding
↓
Cosine Similarity Ranking
↓
Top-K Job Matches
↓
Skill Gap Analysis

---

## 📂 Dataset Structure

### 🧾 Resume Dataset

- Multi-domain resumes
- Categorized across technical fields:
  - AI / ML
  - Data Science
  - .NET Development
  - Backend / Full Stack
  - Other Tech Roles

### 💼 Job Description Dataset

Structured JSON format containing:

- JobID
- Title
- Experience Level
- Skills
- Responsibilities
- Keywords

---

## ⚙️ Tech Stack

| Component         | Technology                           |
| ----------------- | ------------------------------------ |
| NLP Model         | SentenceTransformers (MiniLM - BERT) |
| Similarity Metric | Cosine Similarity                    |
| Baseline Model    | TF-IDF (Scikit-learn)                |
| Backend           | Python                               |
| UI (ML Engine)    | Streamlit                            |
| Landing Page      | TailwindCSS + Vercel                 |
| Hosting           | Streamlit Cloud + Vercel             |
| Domain            | Cloudflare                           |

---

## 🚀 Live Deployment Architecture

Cloudflare (DNS + SSL)
↓
Vercel (Premium Landing Page)
↓
Streamlit Cloud (AI Engine)
↓
Resume Analysis

---

## 📊 Performance Optimization

- Precomputed job embeddings
- Model loading caching
- Lightweight MiniLM transformer
- Top-K ranking instead of full pair scoring

---

## 🧪 Local Development Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/ai-resume-matcher.git
cd ai-resume-matcher


```

### 2️⃣ Install Dependencies

- pip install -r requirements.txt

### 3️⃣ Run Streamlit App

- streamlit run app.py

## 📁 Project Structure

ai-resume-matcher/
│
├── app.py
├── utils.py
├── job_embeddings.npy
├── data/
│ └── description/
│ └── job_dataset.json
├── requirements.txt
└── README.md

## 🎯 Engineering Highlights

- Modular codebase (UI separated from logic)
- Semantic search architecture
- TF-IDF baseline benchmarking
- Production-ready deployment pipeline
- Custom domain integration
- SaaS-grade landing page
- Clean and scalable system design

---

## 🔮 Future Improvements

- Cross-Encoder re-ranking
- Fine-tuned domain-specific embedding model
- Section-aware resume matching
- Vector database integration (FAISS)
- LLM-powered resume improvement suggestions
- Recruiter analytics dashboard

---

## 👨‍💻 Author

**Yuvraj Bhati**  
AI / ML Engineer

🌐 **Live Demo:**  
https://ai-resume-landing.nothing00980.org/

---

## ⭐ Support

If you found this project interesting or useful, consider giving it a star ⭐ on GitHub.
