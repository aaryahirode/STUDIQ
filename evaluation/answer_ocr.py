# evaluation/answer_ocr.py
from extractor.pdf_extractor import extract_text_pdf
from extractor.image_extractor import extract_text_image
import os

def extract_text_from_answer(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_pdf(path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return extract_text_image(path)
    else:
        return ""
