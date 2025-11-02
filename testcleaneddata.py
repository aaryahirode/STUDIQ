import os
import json

# Get the absolute path to the JSON file in another folder
json_path = os.path.join(os.path.dirname(__file__), '', 'data', 'processed_text/Big Data Analytics.json')

# Normalize path (important for Windows)
json_path = os.path.abspath(json_path)

# Read JSON file as a string
with open(json_path, 'r', encoding='utf-8') as f:
    json_string = f.read()


data = json.loads(json_string)
print("Parsed JSON data:", data["files"][0])

# print("JSON as string:")
# print(json_string)
