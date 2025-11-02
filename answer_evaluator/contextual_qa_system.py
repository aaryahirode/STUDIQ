import json
import re
from sentence_transformers import SentenceTransformer, util

class HybridQASystem:
    """
    A Q&A system that first uses a keyword index to find the best document,
    then uses semantic search within that document to find the precise answer.
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading sentence-transformer model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.keyword_data = None
        self.text_data = None
        self.unique_keywords = []
        self.keyword_embeddings = None
        print("✅ Model loaded.")

    def _chunk_text(self, text, chunk_size=3):
        """Helper function to split text into overlapping sentences for semantic search."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        for i in range(0, len(sentences), chunk_size - 1): # Overlap by 1 sentence
            chunk = " ".join(sentences[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def load_data(self, keywords_path, text_path):
        """Loads both the keyword file and the original text file."""
        print(f"Loading data from '{keywords_path}' and '{text_path}'...")
        with open(keywords_path, 'r', encoding='utf-8') as f:
            self.keyword_data = json.load(f)
        with open(text_path, 'r', encoding='utf-8') as f:
            self.text_data = json.load(f)

        all_kw = {kw_obj['keyword'] for file_info in self.keyword_data.get('files', []) for kw_obj in file_info.get('ranked_keywords', [])}
        self.unique_keywords = list(all_kw)
        
        if self.unique_keywords:
            self.keyword_embeddings = self.model.encode(self.unique_keywords, convert_to_tensor=True, show_progress_bar=True)
            print(f"Data loaded. Indexed {len(self.unique_keywords)} unique keywords.")
        else:
            print("Warning: No keywords found in the provided file.")

    def ask_question(self, question, top_n_keywords=5, top_k_chunks=3):
        """Finds a precise answer using a two-step search process."""
        if not self.unique_keywords:
            return "The keyword index is empty."

        # --- STEP 1: Find the most relevant DOCUMENT using keywords ---
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        cos_scores_kw = util.cos_sim(question_embedding, self.keyword_embeddings)[0]
        top_keywords_results = cos_scores_kw.topk(k=min(top_n_keywords, len(self.unique_keywords)))
        
        relevant_keywords = {self.unique_keywords[idx] for _, idx in zip(top_keywords_results[0], top_keywords_results[1])}
        
        file_scores = {}
        for file_info in self.keyword_data['files']:
            score = sum(kw_obj['rank'] for kw_obj in file_info['ranked_keywords'] if kw_obj['keyword'] in relevant_keywords)
            if score > 0:
                file_scores[file_info['filename']] = score
        
        if not file_scores:
            return "Sorry, I could not find a relevant document for your question."

        best_filename = max(file_scores, key=file_scores.get)
        print(f"\nFound best document via keywords: '{best_filename}'")

        # --- STEP 2: Find the most relevant PARAGRAPHS within that document ---
        best_doc_text = ""
        for file_info in self.text_data['files']:
            if file_info['filename'] == best_filename:
                best_doc_text = file_info['text']
                break
        
        if not best_doc_text:
            return f"Error: Could not retrieve text for '{best_filename}'."

        # Chunk the text of the best document and find the most relevant parts
        chunks = self._chunk_text(best_doc_text)
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        
        cos_scores_chunks = util.cos_sim(question_embedding, chunk_embeddings)[0]
        top_chunks_results = cos_scores_chunks.topk(k=min(top_k_chunks, len(chunks)))

        print("Extracting the most relevant paragraphs from the document...")
        final_answer = ""
        for score, idx in zip(top_chunks_results[0], top_chunks_results[1]):
            final_answer += chunks[idx] + "\n\n"
        
        return final_answer.strip()

if __name__ == "__main__":
    keywords_json_path = "output/ranked_keywords.json"
    source_text_path = "data/processed_text/Big Data Analytics.json"

    qa_system = HybridQASystem()
    qa_system.load_data(keywords_json_path, source_text_path)

    while True:
        user_question = input("\nAsk a question (or type 'quit' to exit): ")
        if user_question.lower() in ['quit', 'exit']:
            print("Exiting...")
            break
        
        answer = qa_system.ask_question(user_question)
        print("\n💡 Contextual Answer:\n---")
        print(answer)
        print("\n" + "="*50)