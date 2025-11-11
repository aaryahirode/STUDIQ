# test.py (debug-enhanced)
import requests, base64, json, os

BASE_URL = "https://c821f8103da7726feb.gradio.live"
API_URL = f"{BASE_URL}/gradio_api/call/predict"
IMAGE_PATH = r"Evaluate.jpeg"

print(f"🛰️ Using endpoint: {API_URL}")

with open(IMAGE_PATH, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "data": [
        {
            "path": None,
            "url": f"data:image/jpeg;base64,{image_b64}",
            "meta": {"_type": "gradio.FileData"},
            "orig_name": os.path.basename(IMAGE_PATH),
            "is_stream": False
        }
    ]
}

headers = {"Content-Type": "application/json"}

res = requests.post(API_URL, headers=headers, data=json.dumps(payload))
print("\nRaw response:\n", res.text)

try:
    event_id = res.json().get("event_id")
    if not event_id:
        print("\n❌ No event_id found. Response:", res.text)
        exit()
except Exception:
    print("\n❌ Invalid JSON response. Maybe link expired.")
    exit()

stream_url = f"{BASE_URL}/gradio_api/call/predict/{event_id}"
print(f"\n📡 Connecting to stream: {stream_url}\n")

try:
    with requests.get(stream_url, stream=True) as resp:
        buffer = ""
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            decoded = chunk.decode("utf-8", errors="ignore")
            buffer += decoded

            # Debug output: show partial stream chunks
            print(decoded.strip())

            while "\n\n" in buffer:
                part, buffer = buffer.split("\n\n", 1)
                if part.startswith("data:"):
                    data = part[len("data:"):].strip()
                    if data == "[DONE]":
                        print("\n⚠️ Stream ended.")
                        break
                    try:
                        json_data = json.loads(data)
                        if "data" in json_data:
                            text = json_data["data"][0]
                            print("\n✅ Recognized Text:\n", text)
                            with open("recognized_text.txt", "w", encoding="utf-8") as f:
                                f.write(text)
                    except Exception:
                        continue

except Exception as e:
    print(f"\n❌ Stream error: {e}")
