# extractor/text_cleaner.py
import re

def clean_text(text):
    if not text:
        return ""
    # basic cleanup
    text = text.replace("\r", "\n")
    text = re.sub(r"\n\s+\n", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # collapse many newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text
