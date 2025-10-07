# extractor/pdf_extractor.py
import fitz  # pymupdf

def extract_text_pdf(path):
    text_parts = []
    doc = fitz.open(path)
    for page in doc:
        text = page.get_text("text")
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)
