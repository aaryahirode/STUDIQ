# local_qna.py
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

def main():
    print("🤖 Local QnA Chat — No Internet, No API Keys")
    print("Type 'exit' to quit.\n")

    # Load small local model
    torch.set_num_threads(6)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    def generate_response(query):
        """Generates long, detailed academic-style answers."""
        base_prompt = f"""
<|system|>
You are a knowledgeable and disciplined AI tutor.
Your task is to write a **single long, well-structured academic answer** for a 10–15 mark university question.
Do not ask questions, do not start new topics, and do not repeat the query.
Stay focused only on explaining the user's question in detail.

Your answer must:
1. Begin with a brief introduction.
2. Explain all major concepts, subtopics, and examples in structured paragraphs or bullet points.
3. Include formulas, architectures, or diagrams (in text form) if relevant.
4. End with a short conclusion.
5. Write final key terms as: Keywords: [term1, term2, term3, ...]
6. Avoid writing anything outside the answer (no system text or repetition).
7. Do not ask or invent new questions. Write only one complete, continuous answer.
<|user|>
{query}
<|assistant|>
"""

        # Prepare input and streamer
        inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1500,  # reduced to prevent runaway loops
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        # Generate text in background thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\n🧠 Bot:", end=" ", flush=True)
        full_output = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            full_output += new_text
        print("\n")

        return full_output.strip()

    # Main chat loop
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting chat...")
            break

        response = generate_response(query)

        # If too short, request continuation safely
        if len(response) < 500:
            print("\n🔁 Continuing explanation...\n")
            response += "\n\nContinue in the same structured manner, completing any missing subtopics."
            generate_response(response)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
