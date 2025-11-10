import json
import re
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import wordnet
from transformers import pipeline
import torch
import nltk

# Ensure required NLTK data is available
nltk.download('wordnet')
nltk.download('omw-1.4')


class HybridQASystem:
    """
    Adaptive Q&A System — dynamically builds topic context and retrieves
    document sections based on semantic and structural signals.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading sentence-transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.keyword_data = None
        self.text_data = None
        self.unique_keywords = []
        self.keyword_embeddings = None
        print("✅ Model loaded.")

        # Load summarizer (small offline model)
        print("Loading summarization model (flan-t5-small)...")
        self.summarizer = pipeline(
            "summarization",
            model="google/flan-t5-small",
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Summarizer ready.")

    # ------------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------------

    def _chunk_text(self, text, chunk_size=4):
        """
        Split text into meaningful sentence groups for retrieval.
        Filters out non-linguistic fragments (numbers, equations, etc.)
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        cleaned_sentences = []
        for s in sentences:
            # Skip numeric/math-only or very short lines
            if len(s.strip()) < 40:
                continue
            if re.match(r'^[0-9\W]+$', s.strip()):
                continue
            if re.search(r'[\d=+/<>-]{4,}', s):  # filter equation-like content
                continue
            cleaned_sentences.append(s.strip())

        # Group sentences into overlapping chunks
        chunks = []
        for i in range(0, len(cleaned_sentences), chunk_size - 1):
            chunk = " ".join(cleaned_sentences[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks

    def _expand_question(self, question):
        """Expand question with WordNet synonyms for better recall."""
        words = question.lower().split()
        expanded = set(words)
        for word in words:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    expanded.add(lemma.name().replace('_', ' '))
        return " ".join(expanded)

    def _map_reduce_summarize(self, text, max_chunk_chars=1000):
        """Breaks large text into parts, summarizes each, and combines."""
        if len(text) < 200:
            return text.strip()

        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks, current = [], ""
        for p in paragraphs:
            if len(current) + len(p) < max_chunk_chars:
                current += " " + p
            else:
                chunks.append(current.strip())
                current = p
        if current:
            chunks.append(current.strip())

        partial_summaries = []
        for ch in chunks:
            try:
                summary = self.summarizer(
                    "summarize: " + ch[:max_chunk_chars],
                    max_length=120,
                    min_length=40,
                    do_sample=False
                )[0]["summary_text"].strip()
                partial_summaries.append(summary)
            except Exception:
                partial_summaries.append(ch[:800])

        combined = " ".join(partial_summaries)
        try:
            final = self.summarizer(
                "summarize: " + combined,
                max_length=200,
                min_length=50,
                do_sample=False
            )[0]["summary_text"].strip()
            return final
        except Exception:
            return combined[:1500].strip()

    def _clean_text(self, text):
        """Remove slide-style noise, bullets, redundant headings."""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(Module\s*[:\-]?\s*\d+|Thank You|Puzzle|Activity)", "", text, flags=re.I)
        text = re.sub(r"[\u2022•▪➢]+", "", text)
        return text.strip()

    def _find_enumerations(self, text):
        """
        Automatically detect key structured enumerations in text
        (like 'Volume, Velocity, Variety, Veracity, Value').
        """
        enum_patterns = [
            r"Volume.*Velocity.*Variety.*Veracity.*Value",
            r"(?i)(advantages|disadvantages|applications|features|characteristics|benefits)[\s:.\-]+.{0,300}",
        ]
        matches = []
        for pat in enum_patterns:
            for m in re.finditer(pat, text, flags=re.I | re.S):
                block = text[m.start():m.end() + 600]
                matches.append(block.strip())
        return list(set(matches))

    # ------------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------------

    def load_data(self, keywords_path, text_path):
        """Loads keyword index and text corpus."""
        print(f"Loading data from '{keywords_path}' and '{text_path}'...")
        with open(keywords_path, "r", encoding="utf-8") as f:
            self.keyword_data = json.load(f)
        with open(text_path, "r", encoding="utf-8") as f:
            self.text_data = json.load(f)

        all_kw = {
            kw_obj["keyword"]
            for file_info in self.keyword_data.get("files", [])
            for kw_obj in file_info.get("ranked_keywords", [])
        }
        self.unique_keywords = list(all_kw)

        if self.unique_keywords:
            self.keyword_embeddings = self.model.encode(
                self.unique_keywords, convert_to_tensor=True, show_progress_bar=True
            )
            print(f"✅ Data loaded. Indexed {len(self.unique_keywords)} unique keywords.")
        else:
            print("⚠️ No keywords found in the provided file.")

    def ask_question(self, question, top_n_keywords=8, top_k_chunks=5, expand_window=2):
        """Enhanced Q&A with content filtering and relevance boosting."""
        if not self.unique_keywords:
            return "⚠️ Keyword index is empty."

        expanded_question = self._expand_question(question)
        question_embedding = self.model.encode(expanded_question, convert_to_tensor=True)

        # --- Step 1: Find best document ---
        cos_scores_kw = util.cos_sim(question_embedding, self.keyword_embeddings)[0]
        top_keywords_results = cos_scores_kw.topk(k=min(top_n_keywords, len(self.unique_keywords)))
        relevant_keywords = {self.unique_keywords[idx] for _, idx in zip(top_keywords_results[0], top_keywords_results[1])}

        file_scores = {}
        for file_info in self.keyword_data["files"]:
            score = sum(
                kw_obj.get("rank", 0)
                for kw_obj in file_info.get("ranked_keywords", [])
                if kw_obj.get("keyword") in relevant_keywords
            )
            if score > 0:
                file_scores[file_info["filename"]] = score

        if not file_scores:
            return "❌ No relevant document found."

        best_filename = max(file_scores, key=file_scores.get)
        print(f"📄 Found best document via keywords: '{best_filename}'")

        # --- Step 2: Load and clean text ---
        best_doc_text = ""
        for f in self.text_data.get("files", []):
            if f["filename"] == best_filename:
                best_doc_text = f.get("text", "")
                break
        if not best_doc_text:
            return f"❌ Could not retrieve text for '{best_filename}'."

        best_doc_text = self._clean_text(best_doc_text)

        # --- Step 3: Find conceptual enumerations first ---
        enumerations = self._find_enumerations(best_doc_text)
        if enumerations:
            best_enum = max(enumerations, key=len)
            return self._map_reduce_summarize(best_enum)

        # --- Step 4: Retrieve and rank conceptual chunks ---
        chunks = self._chunk_text(best_doc_text)
        if not chunks:
            return "⚠️ No meaningful chunks found."

        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        cos_scores_chunks = util.cos_sim(question_embedding, chunk_embeddings)[0]
        top_chunks_results = cos_scores_chunks.topk(k=min(top_k_chunks, len(chunks)))

        selected_indices = [int(i) for i in top_chunks_results[1].tolist()]
        combined_text = "\n\n".join(
            " ".join(chunks[max(0, i - expand_window):min(len(chunks), i + expand_window + 1)])
            for i in selected_indices
        )

        # --- Step 5: Summarize robustly ---
        answer = self._map_reduce_summarize(combined_text)
        if len(answer.split()) < 20:
            # fallback if too short
            answer = self._map_reduce_summarize(best_doc_text[:6000])

        return answer.strip()


# ------------------------------------------------------------------------
# Main Runner
# ------------------------------------------------------------------------

if __name__ == "__main__":
    keywords_json_path = "output/knowledge_graph.json"
    source_text_path = "data/processed_text/Big Data Analytics.json"

    qa_system = HybridQASystem()
    qa_system.load_data(keywords_json_path, source_text_path)

    while True:
        user_question = input("\nAsk a question (or type 'quit' to exit): ")
        if user_question.lower() in ["quit", "exit"]:
            print("Exiting...")
            break

        answer = qa_system.ask_question(user_question)
        print("\n💡 Contextual Answer:\n---")
        print(answer)
        print("\n" + "=" * 50)
