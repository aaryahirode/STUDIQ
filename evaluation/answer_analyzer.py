# evaluation/answer_analyzer.py
import os, json
from .answer_ocr import extract_text_from_answer
from nlp_analysis.keypoint_extractor import MODEL, OUT_PATH
from sentence_transformers import util

THRESHOLD = 0.60  # similarity threshold to consider a keypoint 'covered'

def evaluate_answer_text(answer_text, subject):
    with open(OUT_PATH, "r", encoding="utf-8") as f:
        kp = json.load(f)
    if subject not in kp:
        return {"error": "No keypoints for subject. Run keypoints generation."}
    keypoints = kp[subject]
    if not keypoints:
        return {"error": "No keypoints found."}

    # Encode keypoints and answer sentences
    kp_emb = MODEL.encode(keypoints, convert_to_tensor=True)
    ans_sents = [s.strip() for s in answer_text.splitlines() if len(s.strip())>10]
    if not ans_sents:
        return {"score": 0.0, "matched": [], "missing": keypoints, "feedback": "No readable answer text found."}
    ans_emb = MODEL.encode(ans_sents, convert_to_tensor=True)

    # For each keypoint, find max similarity with any answer sentence
    sims = util.cos_sim(kp_emb, ans_emb).cpu().numpy()  # shape (num_kp, num_ans_sents)
    matched = []
    missing = []
    for i, kp_text in enumerate(keypoints):
        max_sim = float(sims[i].max())
        if max_sim >= THRESHOLD:
            matched.append({"keypoint": kp_text, "score": max_sim})
        else:
            missing.append({"keypoint": kp_text, "score": max_sim})

    coverage = len(matched) / len(keypoints)
    score_out_of_10 = round(coverage * 10, 2)

    # Basic feedback
    feedback_lines = []
    if coverage == 1.0:
        feedback_lines.append("Excellent — all key points covered.")
    else:
        feedback_lines.append(f"Covered {len(matched)}/{len(keypoints)} key points.")
        feedback_lines.append("Missing or weakly covered points:")
        for m in missing:
            feedback_lines.append(f" - {m['keypoint'][:120]}... (sim {m['score']:.2f})")

    result = {
        "score": score_out_of_10,
        "coverage": coverage,
        "matched": matched,
        "missing": missing,
        "feedback": "\n".join(feedback_lines)
    }
    return result

def evaluate_answer_file(filepath, subject):
    text = extract_text_from_answer(filepath)
    res = evaluate_answer_text(text, subject)
    # save results
    os.makedirs("data/results", exist_ok=True)
    outpath = os.path.join("data/results", f"result_{os.path.basename(filepath)}.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("Saved evaluation result to", outpath)
    print("Score:", res.get("score"))
    print("Feedback:", res.get("feedback")[:400])
    return res
