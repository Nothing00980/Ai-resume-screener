import re
import fitz
import pytesseract
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# =====================================================
# Extract text from normal PDFs (text-based)
# =====================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    try:
        doc = fitz.open(pdf_path)

        for page in doc:
            text += page.get_text()

    except Exception as e:
        print("PDF extraction failed:", e)

    return text


# =====================================================
# OCR for scanned PDFs
# =====================================================

def extract_text_ocr(pdf_path):

    text = ""

    try:
        images = convert_from_path(pdf_path,poppler_path=r"C:\poppler-25.12.0\Library\bin")

        for img in images:
            text += pytesseract.image_to_string(img)

    except Exception as e:
        print("OCR extraction failed:", e)

    return text


# =====================================================
# Hybrid extraction (text → OCR fallback)
# =====================================================

def extract_resume_text(pdf_path):

    text = extract_text_from_pdf(pdf_path)

    if text.strip() == "":
        text = extract_text_ocr(pdf_path)

    return text


# =====================================================
# Clean resume text
# =====================================================

def clean_text(text):

    text = str(text)

    # remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # remove URLs
    text = re.sub(r"http\S+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.lower().strip()