# nlp_analysis/keypoint_extractor.py
import os, json, re
from sentence_transformers import SentenceTransformer, util
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL = SentenceTransformer(MODEL_NAME)
PROCESSED_DIR = "data/processed_text"
OUT_PATH = "data/keypoints.json"

def split_into_sentences(text):
    # naive split - works reasonably for slides and docs
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if len(s.strip())>10]
    return sents

def top_k_by_centroid(sentences, k=5):
    if not sentences:
        return []
    embeddings = MODEL.encode(sentences, convert_to_tensor=True)
    centroid = embeddings.mean(dim=0)
    sims = util.cos_sim(embeddings, centroid).cpu().numpy().ravel()
    idxs = np.argsort(-sims)[:k]
    return [sentences[i] for i in idxs]

def generate_keypoints(processed_dir=PROCESSED_DIR, out_path=OUT_PATH, top_k=6):
    keypoints = {}
    if not os.path.exists(processed_dir):
        print("No processed text. Run: python main.py extract")
        return
    for fname in sorted(os.listdir(processed_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(processed_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        combined = "\n\n".join([file["text"] for file in data.get("files", []) if file.get("text")])
        sentences = split_into_sentences(combined)
        top = top_k_by_centroid(sentences, k=top_k)
        keypoints[data["subject"]] = top
        print(f"Generated {len(top)} keypoints for subject {data['subject']}")
    with open(out_path, "w", encoding="utf-8") as outf:
        json.dump(keypoints, outf, indent=2, ensure_ascii=False)
    print("Saved keypoints to", out_path)

def get_top_sentences_for_question(question, subject, top_n=5):
    # helper to find best matching sentences for a question
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        kp = json.load(f)
    if subject not in kp:
        return []
    subject_keypoints = kp[subject]
    q_emb = MODEL.encode(question, convert_to_tensor=True)
    s_emb = MODEL.encode(subject_keypoints, convert_to_tensor=True)
    sims = util.cos_sim(q_emb, s_emb).cpu().numpy().ravel()
    idxs = (-sims).argsort()[:top_n]
    return [subject_keypoints[i] for i in idxs if sims[i] > 0.1]
