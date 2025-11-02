import json
import re
import os
import spacy
import pytextrank

# Load the spaCy model
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# Add the PyTextRank component to the spaCy pipeline
if "textrank" not in nlp.pipe_names:
    nlp.add_pipe("textrank")

def clean_text(text: str) -> str:
    """Basic cleanup for extracted PDF text."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"http\S+", "", text)
    return text.strip()

def create_knowledge_graph(text: str):
    """
    Extracts ranked keywords and explicit relations (Subject-Verb-Object triplets)
    using pure NLP algorithms.
    """
    if not text or not text.strip():
        return [], []

    doc = nlp(text)
    
    # --- 1. Extract Ranked Keywords using TextRank ---
    consolidated_phrases = {}
    for phrase in doc._.phrases:
        normalized_text = phrase.text.lower()
        if len(normalized_text) < 3:
            continue
        if normalized_text not in consolidated_phrases:
            consolidated_phrases[normalized_text] = {"rank": phrase.rank, "count": phrase.count}
        else:
            consolidated_phrases[normalized_text]["rank"] = max(consolidated_phrases[normalized_text]["rank"], phrase.rank)
            consolidated_phrases[normalized_text]["count"] += phrase.count
            
    ranked_keywords = [{"keyword": key, **value} for key, value in consolidated_phrases.items()]
    ranked_keywords.sort(key=lambda x: x["rank"], reverse=True)

    # --- 2. Extract Relations (Graph Edges) using Dependency Parsing ---
    relations = []
    for sent in doc.sents:
        for token in sent:
            # Find verbs to form the core of the relation
            if token.pos_ == "VERB":
                subjects = [child.text.lower() for child in token.children if child.dep_ == "nsubj"]
                objects = [child.text.lower() for child in token.children if child.dep_ == "dobj"]
                if subjects and objects:
                    for sub in subjects:
                        for obj in objects:
                            relations.append([sub, token.lemma_, obj]) # [Subject, Verb, Object]
    
    return ranked_keywords, relations

def process_all_files(input_json: str, output_dir: str):
    """Generates a knowledge graph file from the source documents."""
    os.makedirs(output_dir, exist_ok=True)
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_data = {"subject": data.get("subject", "Unknown"), "files": []}

    for file in data.get("files", []):
        filename = file.get("filename", "Unknown file")
        print(f"Processing: {filename}...")
        cleaned = clean_text(file.get("text", ""))
        keywords, relations = create_knowledge_graph(cleaned)
        
        output_data["files"].append({
            "filename": filename,
            "ranked_keywords": keywords,
            "relations": relations
        })
        print(f"✅ Processed: {filename} | Found {len(keywords)} keywords and {len(relations)} relations.")

    output_path = os.path.join(output_dir, "knowledge_graph.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Knowledge Graph saved to: {output_path}")

if __name__ == "__main__":
    input_path = "data/processed_text/Big Data Analytics.json"
    output_folder = "./output"
    process_all_files(input_path, output_folder)