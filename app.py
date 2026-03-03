import streamlit as st
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_text_from_pdf, clean_text

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# Load Job Data (Cached)
# -----------------------------
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

# -----------------------------
# Helper Functions
# -----------------------------

def classify_match(score):
    if score >= 0.80:
        return "Strong Match", "success"
    elif score >= 0.60:
        return "Moderate Match", "warning"
    else:
        return "Low Match", "error"

def generate_summary(title, score, matched, missing):
    return (
        f"Your profile shows {round(score*100)}% alignment with the {title} role. "
        f"Strong areas include {', '.join(matched[:3]) if matched else 'core fundamentals'}. "
        f"Improving {', '.join(missing[:3]) if missing else 'advanced skills'} would significantly increase alignment."
    )

def generate_suggestions(score, missing):
    suggestions = []

    if any("CI/CD" in s for s in missing):
        suggestions.append("Include CI/CD or deployment pipeline experience.")
    if any("Spark" in s or "Hadoop" in s for s in missing):
        suggestions.append("Add distributed data processing projects (Spark/Hadoop).")
    if any("Reinforcement Learning" in s for s in missing):
        suggestions.append("Include reinforcement learning or advanced experimentation projects.")
    if score < 0.70:
        suggestions.append("Strengthen core technical stack before applying to senior-level roles.")
    if len(missing) > 10:
        suggestions.append("Improve resume skill coverage and highlight relevant tools clearly.")

    return suggestions

# -----------------------------
# UI
# -----------------------------
st.title("🚀 AI Resume Intelligence Platform")
st.write("Upload your resume to receive a structured AI Career Report.")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    resume_text = extract_text_from_pdf("temp.pdf")

    if resume_text.strip() == "":
        st.error("⚠ Please upload a text-based PDF resume.")
        st.stop()

    cleaned_resume = clean_text(resume_text)
    resume_embedding = model.encode([cleaned_resume])
    similarity = cosine_similarity(resume_embedding, job_embeddings)[0]

    # -----------------------------
    # Filter Applicable Roles
    # -----------------------------
    threshold = 0.60

    applicable_indices = [
        i for i, score in enumerate(similarity)
        if score >= threshold
    ]

    applicable_indices = sorted(
        applicable_indices,
        key=lambda i: similarity[i],
        reverse=True
    )

    # Deduplicate by title
    unique_roles = {}
    for idx in applicable_indices:
        title = job_titles[idx]
        if title not in unique_roles:
            unique_roles[title] = idx

    applicable_indices = list(unique_roles.values())

    if not applicable_indices:
        st.warning("No strong role alignment detected. Consider enhancing your resume.")
        st.stop()

    # -----------------------------
    # Global Career Report Overview
    # -----------------------------
    st.markdown("## 🧠 AI Career Report")

    best_idx = applicable_indices[0]
    best_score = similarity[best_idx]

    st.markdown("### 🏆 Best Career Fit")
    st.success(f"{job_titles[best_idx]} ({round(best_score*100,2)}%)")
    st.progress(best_score)

    st.write(f"Total Relevant Roles Found: {len(applicable_indices)}")

    # -----------------------------
    # Accordion Role Analysis
    # -----------------------------
    for idx in applicable_indices:

        score = similarity[idx]
        title = job_titles[idx]

        with st.expander(f"{title} — {round(score*100,2)}% Match"):

            label, label_type = classify_match(score)

            if label_type == "success":
                st.success(label)
            elif label_type == "warning":
                st.warning(label)
            else:
                st.error(label)

            st.progress(score)

            # Skill Matching
            matched = []
            missing = []

            for skill in job_skills[idx]:
                if skill.lower() in cleaned_resume:
                    matched.append(skill)
                else:
                    missing.append(skill)

            # Categorize Gaps
            critical_keywords = [
                "CI/CD", "Spark", "Hadoop",
                "Kubernetes", "MLOps",
                "Reinforcement Learning",
                "LLMs", "Computer Vision"
            ]

            critical_gaps = []
            advanced_gaps = []

            for skill in missing:
                if any(k.lower() in skill.lower() for k in critical_keywords):
                    critical_gaps.append(skill)
                else:
                    advanced_gaps.append(skill)

            # Career Summary
            st.markdown("### 🧠 Career Alignment Summary")
            st.info(generate_summary(title, score, matched, missing))

            # Strength Areas
            if matched:
                st.markdown("### 💪 Strength Areas")
                for skill in matched[:8]:
                    st.write(f"• {skill}")

            # Critical Gaps
            if critical_gaps:
                st.markdown("### 🔴 Critical Skill Gaps")
                for skill in critical_gaps:
                    st.write(f"• {skill}")

            # Advanced Gaps
            if advanced_gaps:
                st.markdown("### 🟡 Advanced Skill Gaps")
                for skill in advanced_gaps[:8]:
                    st.write(f"• {skill}")

            # Suggestions
            suggestions = generate_suggestions(score, missing)

            if suggestions:
                st.markdown("### 🚀 Action Plan to Improve Match")
                for s in suggestions:
                    st.write(f"• {s}")