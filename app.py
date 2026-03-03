import streamlit as st
import numpy as np
import json
import os
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import (
    extract_text_from_pdf,
    extract_text_ocr,
    clean_text
)

# =====================================================
# 🔧 WINDOWS CONFIG (Modify if needed)
# =====================================================

# Set this ONLY if running locally on Windows
POPPLER_PATH = None   # change if needed
TESSERACT_PATH = None  # change if needed

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# =====================================================
# 🚀 Load Model (Cached)
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# 📂 Load Job Data (Cached)
# =====================================================

@st.cache_data
def load_jobs():
    with open("data/description/job_dataset.json", "r", encoding="utf-8") as f:
        jobs_data = json.load(f)

    job_texts = []
    job_ids = []
    job_titles = []
    job_skills = []

    for job in jobs_data:
        combined_text = (
            f"{job['Title']} "
            f"{' '.join(job['Skills'])} "
            f"{' '.join(job['Responsibilities'])}"
        )
        job_texts.append(clean_text(combined_text))
        job_ids.append(job["JobID"])
        job_titles.append(job["Title"])
        job_skills.append(job["Skills"])

    return job_texts, job_ids, job_titles, job_skills


job_texts, job_ids, job_titles, job_skills = load_jobs()
job_embeddings = np.load("job_embeddings.npy")

# =====================================================
# 🎨 UI
# =====================================================

st.title("🚀 AI Resume Screening & Job Matching System")
st.write("Upload your resume and discover best matching job roles with skill insights.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# =====================================================
# 🔍 Resume Processing
# =====================================================

if uploaded_file is not None:

    with st.spinner("Processing resume..."):

        # Save temp file
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        # Try text extraction first
        resume_text = extract_text_from_pdf("temp.pdf")

        # If empty → use OCR
        if resume_text.strip() == "":
            st.error("⚠ This resume appears to be a scanned PDF. Please upload a text-based PDF resume.")
            st.stop()

        cleaned_resume = clean_text(resume_text)

        # Generate embedding
        resume_embedding = model.encode([cleaned_resume])

        # Compute similarity
        similarity = cosine_similarity(resume_embedding, job_embeddings)[0]

        threshold = 0.60  # adjust if needed

        applicable_indices = [
            i for i, score in enumerate(similarity)
            if score >= threshold
        ]

# Sort descending
        applicable_indices = sorted(
            applicable_indices,
            key=lambda i: similarity[i],
            reverse=True
        )

    st.subheader("🎯 Top Matching Jobs")

    for idx in applicable_indices:

        st.markdown(f"### {job_titles[idx]}")
        score_percent = round(float(similarity[idx]) * 100, 2)

        st.progress(score_percent / 100)
        st.write(f"**Match Score:** {score_percent}%")

        # Skill Gap Analysis
        matched = []
        missing = []

        for skill in job_skills[idx]:
            if skill.lower() in cleaned_resume:
                matched.append(skill)
            else:
                missing.append(skill)

        st.write("✅ **Matched Skills:**", matched if matched else "None detected")
        st.write("❌ **Missing Skills:**", missing if missing else "None")
        st.markdown("---")

    # Clean temp file
    os.remove("temp.pdf")