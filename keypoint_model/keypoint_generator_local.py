import os
import glob
from utils.model_loader import load_local_model
from utils.keypoint_logic import process_materials

# === CONFIG ===
INPUT_FOLDER = "./data/processed_text"                   # folder containing all JSON files
OUTPUT_FOLDER = "outputs"                    # output folder for generated keypoints
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LOCAL_DIR = "models/phi3-mini"

# === SETUP ===
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === LOAD MODEL (cached if available) ===
generator = load_local_model(model_id=MODEL_ID, local_dir=LOCAL_DIR)

# === FIND ALL INPUT JSON FILES ===
json_files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))

if not json_files:
    print("❌ No JSON files found in 'materials/' folder!")
else:
    print(f"📚 Found {len(json_files)} subject files in {INPUT_FOLDER}\n")

# === PROCESS EACH SUBJECT FILE ===
for file_path in json_files:
    subject_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(OUTPUT_FOLDER, f"keypoints_{subject_name}.json")

    print(f"\n🧠 Generating keypoints for subject: {subject_name}")
    process_materials(file_path, output_path, generator)

print("\n✅ All subjects processed successfully!")
