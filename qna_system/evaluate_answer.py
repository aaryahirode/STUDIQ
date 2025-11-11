import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Download necessary data silently
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
stop_words = set(stopwords.words("english"))


def extract_concepts(reference_text):
    """Extract top key concepts dynamically using TF-IDF and POS tagging."""
    tokens = [w.lower() for w in word_tokenize(reference_text) if w.isalpha()]
    tagged = nltk.pos_tag(tokens)

    filtered = [w for w, t in tagged if (t.startswith("NN") or t.startswith("JJ")) and w not in stop_words]
    bigrams = [" ".join(bg) for bg in ngrams(filtered, 2)]
    trigrams = [" ".join(tg) for tg in ngrams(filtered, 3)]
    combined_phrases = list(set(filtered + bigrams + trigrams))

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([reference_text])
    feature_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

    ranked = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    top_concepts = [term for term, _ in ranked[:12]]

    key_concepts = [c for c in top_concepts if any(c in phrase for phrase in combined_phrases)]
    return key_concepts


def get_context_explanation(concept, reference_text):
    """Find short context snippet from reference where concept occurs."""
    sentences = sent_tokenize(reference_text)
    related = [s for s in sentences if concept.lower() in s.lower()]
    if not related:
        return "no detailed context found"
    snippet = related[0]
    # Clean & shorten to about 8–12 words
    snippet = re.sub(r"\s+", " ", snippet)
    words = snippet.split()
    if len(words) > 12:
        snippet = " ".join(words[:12]) + "..."
    return snippet


def evaluate_answer(reference, student):
    ref_sentences = list(dict.fromkeys(sent_tokenize(reference)))
    stud_sentences = list(dict.fromkeys(sent_tokenize(student)))

    ref_embeddings = model.encode(ref_sentences, convert_to_tensor=True)
    stud_embeddings = model.encode(stud_sentences, convert_to_tensor=True)

    covered, missing = [], []
    used = set()

    for i, ref in enumerate(ref_sentences):
        sim_scores = util.cos_sim(ref_embeddings[i], stud_embeddings)[0]
        best_idx = int(sim_scores.argmax())
        best_score = float(sim_scores[best_idx])

        if best_idx not in used and best_score > 0.55:
            covered.append((ref, round(best_score, 3)))
            used.add(best_idx)
        else:
            missing.append((ref, round(best_score, 3)))

    marks = round((len(covered) / len(ref_sentences)) * 10, 2)

    # ✅ Automatically detect key concepts
    key_concepts = extract_concepts(reference)

    missing_concepts = [kc for kc in key_concepts if kc.lower() not in student.lower()]
    covered_concepts = [kc for kc in key_concepts if kc.lower() in student.lower()]

    # --- Print output ---
    print("\n" + "=" * 70)
    print("✅ Points Covered:")
    for i, (text, score) in enumerate(covered, 1):
        print(f"{i}. {text} (similarity: {score})")

    print("\n❌ Concepts Missed (with brief context):")
    if missing_concepts:
        for i, concept in enumerate(missing_concepts, 1):
            explanation = get_context_explanation(concept, reference)
            print(f"{i}. {concept} → {explanation}")
    else:
        print("All main concepts covered!")

        print("\n💡 What to Add More:")
    if missing_concepts:
        for concept in missing_concepts:
            # find short contextual snippet from the reference
            context_snippet = ""
            for sent in sent_tokenize(reference):
                if concept.lower() in sent.lower():
                    context_snippet = sent.strip()
                    break

            # fallback if no direct context found
            if not context_snippet:
                context_snippet = f"This concept relates to {concept} in the topic."

            # make a natural one-line suggestion dynamically
            suggestion = f"You missed mentioning '{concept}', which relates to: {context_snippet}"

            print(f"- {suggestion}")
    else:
        print("No additional points needed — great coverage!")


    print(f"\n🏁 Final Score: {marks} / 10")
    print("=" * 70 + "\n")


# Example Run
if __name__ == "__main__":
    reference = """Distributed computing is a paradigm where a network of independent computers collaborate to solve complex problems. This approach enhances performance and resilience by distributing tasks across multiple machines, allowing for parallel processing and better resource utilization. Key aspects of distributed systems include coordination, data sharing, and fault tolerance, which are often managed through techniques like message passing and consensus algorithms. Notable frameworks like Hadoop and Apache Spark leverage distributed architecture to process large-scale data efficiently. However, challenges such as managing network delays, ensuring data consistency, and handling system failures still require careful consideration in the design and operation of these systems."""

    student = """Distributed computing involves multiple machines working together to solve problems, improving scalability and fault tolerance. Frameworks like Hadoop and Spark use parallel processing for big data, while handling communication through message passing. Challenges include network delays, data consistency, and system failures that need careful management."""

    evaluate_answer(reference, student)
