import json
from gensim.summarization import summarize

def generate_model_answers(input_json, output_json, ratio=0.05):
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for file in data["files"]:
        text = file.get("text", "")
        try:
            file["model_answer"] = summarize(text, ratio=ratio)
        except ValueError:
            file["model_answer"] = text[:500]
        print(f"✅ Summarized: {file['filename']}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Model answers saved to {output_json}")


if __name__ == "__main__":
    generate_model_answers(
        "data/processed_text/Big Data Analytics_keywords.json",
        "data/processed_text/Big Data Analytics_model.json"
    )
