import streamlit as st
import numpy as np
import json
import os
import pytesseract
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_text_from_pdf, clean_text

# =====================================================
# 🔧 WINDOWS CONFIG (Local Only)
# =====================================================

POPPLER_PATH = None
TESSERACT_PATH = None

if os.name == "nt" and TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# =====================================================
# 🚀 Load Model
# =====================================================

@st.cache_resource
def load_models():

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    classifier = joblib.load("models/domain_classifier.pkl")
    label_encoder = joblib.load("models/label_encoder.pkl")

    job_embeddings = np.load("job_embeddings.npy")

    with open("data/description/job_dataset.json", "r", encoding="utf-8") as f:
        jobs_data = json.load(f)

    return embedding_model, classifier, label_encoder, job_embeddings, jobs_data


model, classifier, label_encoder, job_embeddings, jobs_data = load_models()


def infer_domain(title):

    t = title.lower()

    # AI / Machine Learning
    if "ai" in t or "machine learning" in t or "ml engineer" in t:
        return "AI"

    # Data
    if "data scientist" in t or "data analyst" in t or "data engineer" in t or "bi analyst" in t or "big data" in t:
        return "DataScience"

    # Backend
    if "backend" in t:
        return "Backend"

    # Frontend
    if "frontend" in t:
        return "Frontend"

    # Full stack
    if "full stack" in t:
        return "FullStack"

    # Mobile
    if "android" in t or "ios" in t:
        return "Mobile"

    # Cloud
    if "cloud" in t:
        return "Cloud"

    # DevOps
    if "devops" in t:
        return "DevOps"

    # Cybersecurity
    if "cyber" in t or "security analyst" in t or "ethical hacker" in t:
        return "CyberSecurity"

    # Networking
    if "network" in t:
        return "Networking"

    # Blockchain
    if "blockchain" in t:
        return "Blockchain"

    # AR/VR
    if "ar" in t or "vr" in t:
        return "ARVR"

    # QA
    if "qa engineer" in t or "quality assurance" in t:
        return "QA"

    # Design
    if "ux" in t or "designer" in t:
        return "Design"

    # Marketing
    if "marketing" in t:
        return "Marketing"

    # Content
    if "content writer" in t or "copywriter" in t:
        return "Content"

    # Management
    if "manager" in t:
        return "Management"

    # Game dev
    if "game developer" in t:
        return "GameDev"

    # Fintech
    if "fintech" in t:
        return "Fintech"

    # Programming language specific
    if ".net" in t:
        return "DotNet"

    if "java" in t:
        return "Java"

    if "python" in t:
        return "Python"

    return "General"
# =====================================================
# 📂 Load Job Data
# =====================================================

@st.cache_data
def load_jobs():
    with open("data/description/job_dataset.json", "r", encoding="utf-8") as f:
        jobs_data = json.load(f)

    job_texts = []
    job_ids = []
    job_titles = []
    job_skills = []
    job_domains = []

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

        domain = infer_domain(job["Title"])
        job_domains.append(domain)

    return job_texts, job_ids, job_titles, job_skills, job_domains


job_texts, job_ids, job_titles, job_skills, job_domains = load_jobs()
job_embeddings = np.load("job_embeddings.npy")

# =====================================================
# 🧠 Helper Functions
# =====================================================

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
        f"Improving {', '.join(missing[:3]) if missing else 'advanced capabilities'} would significantly increase alignment."
    )


def generate_suggestions(score, missing):
    suggestions = []

    if any("CI/CD" in s for s in missing):
        suggestions.append("Include CI/CD or deployment pipeline experience.")

    if any("Spark" in s or "Hadoop" in s for s in missing):
        suggestions.append("Add distributed data processing projects (Spark/Hadoop).")

    if any("Reinforcement Learning" in s for s in missing):
        suggestions.append("Include reinforcement learning or advanced ML experimentation.")

    if any("MLOps" in s for s in missing):
        suggestions.append("Demonstrate model deployment and monitoring experience.")

    if score < 0.70:
        suggestions.append("Strengthen core technical stack before applying to senior-level roles.")

    return suggestions



def compute_ats_score(similarity_score, matched_skills, total_skills, resume_length):

    # semantic similarity weight
    semantic_score = similarity_score * 60

    # skill match weight
    if total_skills > 0:
        skill_ratio = len(matched_skills) / total_skills
    else:
        skill_ratio = 0

    skill_score = skill_ratio * 30

    # resume completeness
    if resume_length > 300:
        completeness = 10
    elif resume_length > 150:
        completeness = 7
    else:
        completeness = 4

    ats_score = semantic_score + skill_score + completeness

    return round(ats_score)


# =====================================================
# 🎨 UI
# =====================================================

st.title("🚀 AI Resume Intelligence Platform")
st.write("Upload your resume to receive a structured AI Career Report.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

# =====================================================
# 🔍 Resume Processing
# =====================================================

if uploaded_file is not None:

    with st.spinner("Processing resume..."):

        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        resume_text = extract_text_from_pdf("temp.pdf")

        if resume_text.strip() == "":
            st.error("⚠ Please upload a text-based PDF resume.")
            st.stop()

        cleaned_resume = clean_text(resume_text)

        if len(cleaned_resume.split()) < 50:
            st.error("Resume content is too short for analysis.")
            st.stop()

        resume_embedding = model.encode([cleaned_resume])
        pred = classifier.predict(resume_embedding)
        predicted_domain = label_encoder.inverse_transform(pred)[0]

        st.success(f"Detected Career Domain: **{predicted_domain}**")

          # ---------------------------------------------
        # FILTER JOBS BY DOMAIN
        # ---------------------------------------------

        def normalize_domain(domain):

            domain = domain.lower().replace(" ", "")

            mapping = {
                "datascience": "datascience",
                "ai": "ai",
                "machinelearning": "ai",
                "backend": "backend",
                "frontend": "frontend",
                "fullstack": "fullstack",
                "mobile": "mobile",
                "cloud": "cloud",
                "devops": "devops",
                "cybersecurity": "cybersecurity",
                "dotnet": "dotnet",
                "java": "java",
                "python": "python"
            }

            return mapping.get(domain, domain)


        predicted_domain = normalize_domain(predicted_domain)
        domain_indices = [
            i for i, d in enumerate(job_domains)
            if normalize_domain(d) == predicted_domain
        ]

        if len(domain_indices) == 0:
            st.warning("No jobs found for predicted domain. Running global search.")

            domain_indices = list(range(len(job_titles)))
        filtered_embeddings = job_embeddings[domain_indices]


        similarity = cosine_similarity(resume_embedding, filtered_embeddings)[0]

        threshold = 0.40

        applicable_indices = [
            domain_indices[i]
            for i, score in enumerate(similarity)
            if score >= threshold
        ]

        applicable_indices = sorted(
            applicable_indices,
            key=lambda i: similarity[domain_indices.index(i)],
            reverse=True
        )

        # Remove duplicate titles
        # unique_roles = {}
        # for idx in applicable_indices:
        #     title = job_titles[idx]
        #     if title not in unique_roles:
        #         unique_roles[title] = idx

        # applicable_indices = list(unique_roles.values())

    if not applicable_indices:
        st.warning("No strong role alignment detected. Consider enhancing your resume.")
        os.remove("temp.pdf")
        st.stop()

    # =====================================================
    # 🧠 Global Career Overview
    # =====================================================

    st.markdown("## 🧠 AI Career Report")

    best_idx = applicable_indices[0]
    best_score = similarity[domain_indices.index(best_idx)]

    st.markdown("### 🏆 Best Career Fit")
    st.success(f"{job_titles[best_idx]} ({round(best_score*100,2)}%)")
    st.progress(int(best_score*100))

    st.write(f"Total Relevant Roles Found: {len(applicable_indices)}")

    # =====================================================
    # 📂 Accordion Role Analysis
    # =====================================================

    for idx in applicable_indices:

        score = similarity[domain_indices.index(idx)]
        title = job_titles[idx]

        with st.expander(f"{title} — {round(score*100,2)}% Match"):

            label, label_type = classify_match(score)

            if label_type == "success":
                st.success(label)
            elif label_type == "warning":
                st.warning(label)
            else:
                st.error(label)

            st.progress(int(score*100))

            matched = []
            missing = []

            ats_score = compute_ats_score(
                score,
                matched,
                len(job_skills[idx]),
                len(cleaned_resume.split())
            )

            st.markdown("### 🎯 ATS Resume Score")

            st.metric(
                label="ATS Score",
                value=f"{ats_score}/100"
            )
            st.caption("Score combines semantic match, skill alignment, and resume completeness.")

            for skill in job_skills[idx]:
                if skill.lower() in cleaned_resume:
                    matched.append(skill)
                else:
                    missing.append(skill)

            # Categorize gaps
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

            # Summary
            st.markdown("### 🧠 Career Alignment Summary")
            st.info(generate_summary(title, score, matched, missing))

            # Strengths
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

    os.remove("temp.pdf")