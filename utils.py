import re
import numpy as np
import fitz
import pytesseract
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1️⃣ PDF TEXT EXTRACTION
# =====================================================

def extract_text_from_pdf(pdf_path):
    """
    Extract text from text-based PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def extract_text_ocr(pdf_path, poppler_path=None):
    """
    Extract text from scanned PDF using OCR.
    """
    pages = convert_from_path(pdf_path, poppler_path=poppler_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
    return text


# =====================================================
# 2️⃣ TEXT CLEANING
# =====================================================

def clean_text(text):
    text = re.sub(r"\S+@\S+", "", str(text))
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


# =====================================================
# 3️⃣ MATCHING FUNCTION
# =====================================================

def compute_similarity(resume_embedding, job_embeddings):
    """
    Compute cosine similarity between resume and jobs.
    """
    similarity = cosine_similarity(
        resume_embedding,
        job_embeddings
    )[0]
    return similarity


def get_top_k_matches(similarity_scores, job_ids, job_titles, k=3):
    """
    Return top-k job matches.
    """
    top_indices = np.argsort(-similarity_scores)[:k]

    results = []
    for idx in top_indices:
        results.append({
            "JobID": job_ids[idx],
            "Title": job_titles[idx],
            "Score": float(similarity_scores[idx])
        })

    return results


# =====================================================
# 4️⃣ SKILL GAP ANALYSIS
# =====================================================

def skill_gap_analysis(resume_text, job_skills):
    matched = []
    missing = []

    resume_lower = resume_text.lower()

    for skill in job_skills:
        if skill.lower() in resume_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    return matched, missing


# =====================================================
# 5️⃣ EVALUATION METRICS (OPTIONAL)
# =====================================================

def top_k_accuracy(similarity_matrix, true_indices, k=3):
    top_k = np.argsort(-similarity_matrix, axis=1)[:, :k]
    correct = 0
    for i in range(len(true_indices)):
        if true_indices[i] in top_k[i]:
            correct += 1
    return correct / len(true_indices)


def mean_reciprocal_rank(similarity_matrix, true_indices):
    ranks = []
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    for i in range(len(true_indices)):
        rank = np.where(sorted_indices[i] == true_indices[i])[0][0] + 1
        ranks.append(1 / rank)
    return np.mean(ranks)