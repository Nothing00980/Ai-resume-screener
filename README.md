# 🚀 AI Resume Intelligence Platform

> AI-powered Resume Screening & Job Matching System using BERT-based Semantic Embeddings

🌐 **Live Landing Page:**  
https://ai-resume-landing.nothing00980.org/

---

## 📌 Overview

AI Resume Intelligence is an end-to-end NLP system designed to semantically match resumes with relevant job descriptions using transformer-based embeddings.

Instead of relying on keyword matching, the system leverages **BERT-based dense vector embeddings** and **cosine similarity ranking** to capture contextual meaning and improve job alignment accuracy.

🔶 The system now also includes a **Domain Classification model** trained on thousands of resumes to first identify the candidate’s career domain before performing semantic job matching.  
This significantly improves matching accuracy and reduces irrelevant role recommendations.

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

### 🔶 Domain-Aware Job Filtering

- Trained a **Logistic Regression domain classifier**
- Built using **SentenceTransformer embeddings of 10K+ resumes**
- Predicts candidate domain before job matching
- Reduces search space and improves recommendation precision

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

### 🔶 ATS Resume Scoring

- Computes a **composite ATS score (0–100)** for each job match
- Combines:
  - Semantic similarity score
  - Skill coverage ratio
  - Resume completeness signal
- Provides a recruiter-style **resume evaluation metric**

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
🔶 Domain Classification
↓
Domain-based Job Filtering
↓
Cosine Similarity Ranking
↓
Top-K Job Matches
↓
Skill Gap Analysis
↓
🔶 ATS Resume Score

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

| Component                | Technology                           |
| ------------------------ | ------------------------------------ | --- |
| NLP Model                | SentenceTransformers (MiniLM - BERT) |
| Similarity Metric        | Cosine Similarity                    |
| 🔶 Domain Classifier     | Logistic Regression (Scikit-learn)   |
| 🔶 Resume Scoring Engine | Custom ATS Scoring Logic             |
| Baseline Model           | TF-IDF (Scikit-learn)                |
| Backend                  | Python                               |
| UI (ML Engine)           | Streamlit                            |
| Landing Page             | TailwindCSS + Vercel                 |
| Hosting                  | Streamlit Cloud + Vercel             |
| Domain                   | Cloudflare                           |     |

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
- 🔶 Domain filtering to reduce similarity search space
- 🔶 Cached transformer model loading for faster inference

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
- 🔶 Domain classification pipeline trained on large resume dataset
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
- 🔶 LLM-powered career guidance (Gemini / OpenAI)
- LLM-powered resume improvement suggestions
- Recruiter analytics dashboard

## 🧠 AI Pipeline Components

The system consists of three intelligent layers:

1️⃣ **Semantic Resume Encoder**  
Uses transformer embeddings to understand resume context.

2️⃣ **Domain Classification Model**  
Predicts the candidate’s primary technical domain using a trained classifier.

3️⃣ **ATS Scoring Engine**  
Ranks job alignment using semantic similarity, skill coverage, and resume completeness.

---

## 👨‍💻 Author

**Yuvraj Bhati**  
AI / ML Engineer

🌐 **Live Demo:**  
https://ai-resume-landing.nothing00980.org/

---

## ⭐ Support

If you found this project interesting or useful, consider giving it a star ⭐ on GitHub.
