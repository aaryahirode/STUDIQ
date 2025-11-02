import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def generate_keypoints(question, content, generator):
    """
    Uses a local LLM to generate key points for the question from the content.
    """
    prompt = f"""
You are an academic assistant. Read the given course content and extract concise key points
that a student must include to correctly answer the question.

Course content:
{content[:3000]}

Question:
{question}

Output format:
- Point 1
- Point 2
- Point 3
"""

    response = generator(prompt, num_return_sequences=1)
    text = response[0]["generated_text"]
    # Clean output
    if "Output format:" in text:
        text = text.split("Output format:")[-1].strip()
    return text.strip()


def process_materials(input_json_path, output_json_path, generator):
    """
    Loads the LMS data from JSON, separates questions and content, and generates keypoints.
    """
    print(f"📂 Loading data from {input_json_path} ...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    files = data.get("files", [])
    questions = []
    contents = []
    content_files = []

    # Separate question files (QB) from content files
    for item in files:
        filename = item.get("filename", "").lower()
        text = item.get("text", "").strip()
        if not text:
            continue

        if "qb" in filename or "question" in filename:
            # Treat each line as a separate question
            for line in text.split("\n"):
                line = line.strip()
                if line and any(ch.isalpha() for ch in line):
                    questions.append(line)
        else:
            contents.append(text)
            content_files.append(item.get("filename", "unknown"))

    print(f"📘 Found {len(questions)} questions and {len(contents)} content sections")

    # Embed content
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    content_embeddings = embedder.encode(contents, convert_to_tensor=True)

    results = []

    for q in tqdm(questions, desc="🧠 Generating keypoints"):
        q_emb = embedder.encode(q, convert_to_tensor=True)
        sim = util.cos_sim(q_emb, content_embeddings)
        best_idx = int(sim.argmax())
        related_content = contents[best_idx]
        related_file = content_files[best_idx]

        keypoints = generate_keypoints(q, related_content, generator)

        results.append({
            "question": q,
            "related_file": related_file,
            "keypoints": keypoints
        })

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Keypoints generated and saved to {output_json_path}")
