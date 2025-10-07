# extractor/image_extractor.py
from PIL import Image
import pytesseract
import easyocr
import os

# If on Windows and Tesseract installed at default path, uncomment and edit:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Try easyocr first (better for many fonts), fallback to pytesseract
_reader = None
def _get_easyocr_reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(['en'], gpu=False)
    return _reader

def extract_text_image(path):
    ext = os.path.splitext(path)[1].lower()
    # load using PIL
    try:
        img = Image.open(path)
    except Exception:
        return ""
    # Use easyocr
    try:
        reader = _get_easyocr_reader()
        res = reader.readtext(path, detail=0)
        joined = "\n".join(res)
        if joined.strip():
            return joined
    except Exception:
        pass
    # Fallback to pytesseract
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        return ""
