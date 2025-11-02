import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_local_model(model_id="microsoft/Phi-3-mini-4k-instruct", local_dir="models/phi3-mini"):
    """
    Loads model from local cache if available, otherwise downloads and saves it.
    Returns a text-generation pipeline.
    """
    os.makedirs(local_dir, exist_ok=True)

    if not os.listdir(local_dir):
        print(f"🧩 Downloading model '{model_id}' for the first time...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
    else:
        print(f"✅ Using cached model from {local_dir}")
        tokenizer = AutoTokenizer.from_pretrained(local_dir)
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            device_map="auto",
            torch_dtype="auto"
        )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.5,
        device_map="auto"
    )

    return generator
