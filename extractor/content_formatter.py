# extractor/content_formatter.py
import os, json
from .pdf_extractor import extract_text_pdf
from .ppt_extractor import extract_text_pptx
from .image_extractor import extract_text_image
from .text_cleaner import clean_text

MATERIALS_DIR = "data/materials"
OUT_DIR = "data/processed_text"

def process_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_text_pdf(path)
    elif ext in [".pptx", ".ppt"]:
        text = extract_text_pptx(path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        text = extract_text_image(path)
    else:
        text = ""
    return clean_text(text)

def process_all_materials(materials_dir=MATERIALS_DIR, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(materials_dir):
        print("No materials found. Run: python main.py download")
        return
    for subj_folder in sorted(os.listdir(materials_dir)):
        subj_path = os.path.join(materials_dir, subj_folder)
        if not os.path.isdir(subj_path):
            continue
        doc = {"subject": subj_folder, "files": []}
        for fname in sorted(os.listdir(subj_path)):
            fpath = os.path.join(subj_path, fname)
            if fname == "metadata.json":
                continue
            if os.path.isfile(fpath):
                print("Extracting", fpath)
                text = process_file(fpath)
                doc["files"].append({"filename": fname, "path": fpath, "text": text})
        out_file = os.path.join(out_dir, f"{subj_folder}.json")
        with open(out_file, "w", encoding="utf-8") as wf:
            json.dump(doc, wf, indent=2, ensure_ascii=False)
        print("Saved processed text to", out_file)
