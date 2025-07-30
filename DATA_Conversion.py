import os
from datasets import load_dataset

# Get absolute path to this script's folder
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")  # Changed from "slm_project/data" to "data"
os.makedirs(data_dir, exist_ok=True)

corpus_path = os.path.join(data_dir, "corpus.txt")
print("➡ Writing to:", corpus_path)

# Load the dataset
ds = load_dataset("iohadrubin/wikitext-103-raw-v1", split="train")

# Write all lines from the dataset
with open(corpus_path, "w", encoding="utf-8") as f:
    for i, line in enumerate(ds["text"]):
        if i % 1000 == 0:  # Progress indicator
            print(f"Processing line {i}")
        line = line.strip()
        if line:  # Only write non-empty lines
            f.write(line + "\n")

print(f"Corpus created with {len(ds['text'])} lines")
