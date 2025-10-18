import re
import sys
from pathlib import Path

# Aggregated dataset paths
AGGREGATE_TRAIN = Path("data/train.txt")
AGGREGATE_VAL = Path("data/val.txt")

def clean_text(text: str) -> str:

    text = re.sub(r"(?s).*START OF (THE|THIS) PROJECT GUTENBERG.*?\n", "", text)
    text = re.sub(r"(?s)End of (the|this) Project Gutenberg.*", "", text)

    text = text.lower()

    text = re.sub(r"\[.*?\]", " ", text)

    text = re.sub(r"^[a-z\s]+\.", " ", text, flags=re.MULTILINE)

    text = re.sub(r"[^a-z0-9\s.,;:!?'\-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

def split_and_append(filename: Path, train_ratio=0.8):
    """
    Clean the input file, create individual train/val files,
    and append to the main aggregated train/val files.
    """
    book_name = filename.stem  
    processed_path = Path(f"data/processed/{book_name}_clean.txt")
    train_path = Path(f"data/processed/{book_name}_train.txt")
    val_path = Path(f"data/processed/{book_name}_val.txt")

    processed_path.parent.mkdir(parents=True, exist_ok=True)

    raw_text = filename.read_text(encoding="utf-8")
    clean = clean_text(raw_text)
    processed_path.write_text(clean, encoding="utf-8")
    print(f"🧹 Cleaned text saved: {processed_path}")

    tokens = clean.split()
    split_idx = int(len(tokens) * train_ratio)
    train_tokens = " ".join(tokens[:split_idx])
    val_tokens = " ".join(tokens[split_idx:])

    train_path.write_text(train_tokens, encoding="utf-8")
    val_path.write_text(val_tokens, encoding="utf-8")
    print(f"Split into {train_path.name} ({split_idx:,} tokens) and {val_path.name} ({len(tokens)-split_idx:,} tokens)")

    with open(AGGREGATE_TRAIN, "a", encoding="utf-8") as agg_train:
        agg_train.write(" " + train_tokens)
    with open(AGGREGATE_VAL, "a", encoding="utf-8") as agg_val:
        agg_val.write(" " + val_tokens)

    print(f"Appended to {AGGREGATE_TRAIN.name} and {AGGREGATE_VAL.name}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python src/preprocess.py <path_to_raw_text>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"File not found: {input_file}")
        sys.exit(1)

    split_and_append(input_file)

if __name__ == "__main__":
    main()
