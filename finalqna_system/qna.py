# local_qna.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer
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

    # ⚡ Keep your original pipeline (unchanged)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    context = ""

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        prompt = f"""
        <|system|>
        You are an advanced AI tutor trained to explain complex academic and technical topics clearly and precisely.
        Your goal is to help students understand by:
        - Giving structured, step-by-step, and logically reasoned answers.
        - Using clear examples, definitions, and real-world analogies when helpful.
        - Covering topics related to Computer Science, Big Data Analytics, Distributed Computing, Cloud (AWS), Data Science, Soft Computing, and Disaster Management.
        - If the question is broad, first summarize the key points, then explain each in detail.
        - Keep the tone professional yet easy to follow, like a helpful university mentor.
        - Also at the end give keypoints that must be written in the answer as "Keywords: [,,,]"
        <|user|>
        {query}
        <|assistant|>
        """


        # ✅ Added streaming feature
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        print("\n🧠 Bot:", end=" ", flush=True)
        for new_text in streamer:
            print(new_text, end="", flush=True)
        print("\n")

if __name__ == "__main__":
    print(torch.cuda.is_available())
    main()
