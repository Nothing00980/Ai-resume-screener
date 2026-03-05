import os
import glob
import multiprocessing
from tqdm import tqdm
import joblib

from utils import extract_resume_text, clean_text


# ==============================
# CONFIG
# ==============================

DATASET_PATH = "data/resumeall_data"
OUTPUT_DIR = "data"

MIN_TEXT_LENGTH = 200


# ==============================
# Worker Function
# ==============================

def process_resume(args):
    """
    Worker function for multiprocessing
    """
    pdf_path, domain = args

    try:
        text = extract_resume_text(pdf_path)

        if text is None:
            return None

        if len(text) < MIN_TEXT_LENGTH:
            return None

        text = clean_text(text)

        return (text, domain)

    except Exception:
        # silently ignore corrupted PDFs
        return None


# ==============================
# Collect Resume Paths
# ==============================

def collect_resume_tasks(dataset_path):

    tasks = []

    domains = os.listdir(dataset_path)

    for domain in domains:

        domain_path = os.path.join(dataset_path, domain)

        if not os.path.isdir(domain_path):
            continue

        pdf_files = glob.glob(os.path.join(domain_path, "*.pdf"))

        for pdf in pdf_files:
            tasks.append((pdf, domain))

    return tasks


# ==============================
# Main Pipeline
# ==============================

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nScanning dataset folders...\n")

    tasks = collect_resume_tasks(DATASET_PATH)

    print("Total resumes found:", len(tasks))

    # Leave few cores free for OS
    num_workers = max(2, multiprocessing.cpu_count() - 4)

    print("Using CPU cores:", num_workers)

    texts = []
    labels = []

    print("\nStarting parallel resume extraction...\n")

    with multiprocessing.Pool(num_workers) as pool:

        for result in tqdm(
            pool.imap_unordered(process_resume, tasks),
            total=len(tasks),
            desc="Extracting resumes",
            unit="resume"
        ):

            if result is not None:
                text, label = result

                texts.append(text)
                labels.append(label)

    print("\nExtraction completed!")

    print("Valid resumes extracted:", len(texts))

    # ==============================
    # Save results
    # ==============================

    texts_path = os.path.join(OUTPUT_DIR, "resume_texts.pkl")
    labels_path = os.path.join(OUTPUT_DIR, "resume_labels.pkl")

    joblib.dump(texts, texts_path)
    joblib.dump(labels, labels_path)

    print("\nSaved files:")
    print("Texts ->", texts_path)
    print("Labels ->", labels_path)

    print("\nPipeline finished successfully.\n")


# ==============================
# Windows multiprocessing fix
# ==============================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()