import json
from fuzzywuzzy import fuzz

def keyword_match_score(expected_keywords, student_text):
    found, missing = [], []
    for kw in expected_keywords:
        if kw.lower() in student_text.lower():
            found.append(kw)
        else:
            missing.append(kw)
    score = len(found) / len(expected_keywords) if expected_keywords else 0
    return round(score, 2), found, missing


def evaluate_student_answer(student_answer_path, model_json_path, output_path):
    with open(student_answer_path, "r", encoding="utf-8") as f:
        student_data = json.load(f)
    student_answer = student_data["answer"]

    with open(model_json_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    evaluations = []

    for file in model_data["files"]:
        keywords = file.get("keywords", [])
        model_text = file.get("model_answer", "")
        keyword_score, found, missing = keyword_match_score(keywords, student_answer)
        fuzzy_score = fuzz.partial_ratio(model_text.lower(), student_answer.lower()) / 100.0
        total_score = round(0.7 * keyword_score + 0.3 * fuzzy_score, 2)

        evaluations.append({
            "topic": file["filename"],
            "keyword_score": keyword_score,
            "fuzzy_score": fuzzy_score,
            "final_score": total_score,
            "keywords_found": found,
            "keywords_missing": missing
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    print(f"✅ Evaluation saved to {output_path}")


if __name__ == "__main__":
    student_path = "data/student_answers/student1_bda.json"
    model_path = "data/processed_text/Big Data Analytics_model.json"
    output_path = "data/results/student1_evaluation.json"
    evaluate_student_answer(student_path, model_path, output_path)
