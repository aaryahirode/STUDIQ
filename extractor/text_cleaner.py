# -*- coding: utf-8 -*-
import re

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\r", "\n")
    # Normalize newlines
    text = re.sub(r"\n\s+\n", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = text.replace('\ufeff', '')

    # Encode-decode to remove any non-UTF-8 bytes
    text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')

    # Normalize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Collapse multiple newlines (3 or more) into 2
    text = re.sub(r'\n{2,}', '\n', text)

    # Strip leading/trailing spaces from each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove lines with only numbers (like page numbers)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)

    # Remove headings like "Module 3: NoSQL" or "Lecture 21"
    text = re.sub(r'^(Module\s+\d+:.*|Lecture\s+\d+.*)$', '', text, flags=re.MULTILINE)
    text = re.sub('●', '', text)
    

    # Replace bullets like  or • with dash
    text = re.sub(r'[•]', '-', text)
    text = re.sub(r'[]', '-', text)
    text = re.sub(r'', '-', text)

    # Remove extra spaces
    text = re.sub(r' {2,}', ' ', text)

    # Strip leading/trailing newlines
    text = text.strip()

    return text
