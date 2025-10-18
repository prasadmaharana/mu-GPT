import os
import torch
from tokenizer.bpe_tokenizer import BPETokenizer

DATA_DIR = "data"
VOCAB_PATH = os.path.join("tokenizer", "vocab.json")
VOCAB_SIZE = 5000

def tokenize_and_encode():
    train_file = os.path.join(DATA_DIR, "train.txt")
    val_file = os.path.join(DATA_DIR, "val.txt")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError("train.txt or val.txt not found. Run preprocess.py first.")

    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(val_file, "r", encoding="utf-8") as f:
        val_text = f.read()

    tokenizer = BPETokenizer()

    # Build or load vocab
    if os.path.exists(VOCAB_PATH):
        print("Loading existing vocab...")
        tokenizer.load(VOCAB_PATH)
        tokenizer.merge_vocab(train_text + val_text)  # ✅ merge new tokens incrementally
    else:
        print(f"Building vocab (size={VOCAB_SIZE})...")
        tokenizer.build_vocab(train_text, vocab_size=VOCAB_SIZE)

    tokenizer.save(VOCAB_PATH)
    print(f"Saved updated vocab to {VOCAB_PATH}")

    # Encode datasets
    new_train_ids = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    new_val_ids = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    train_ids_path = os.path.join(DATA_DIR, "train_ids.pt")
    val_ids_path = os.path.join(DATA_DIR, "val_ids.pt")

    # Append instead of overwriting
    if os.path.exists(train_ids_path):
        print("Appending to existing train_ids.pt...")
        old_train_ids = torch.load(train_ids_path)
        train_ids = torch.cat([old_train_ids, new_train_ids])
    else:
        train_ids = new_train_ids

    if os.path.exists(val_ids_path):
        print("Appending to existing val_ids.pt...")
        old_val_ids = torch.load(val_ids_path)
        val_ids = torch.cat([old_val_ids, new_val_ids])
    else:
        val_ids = new_val_ids

    torch.save(train_ids, train_ids_path)
    torch.save(val_ids, val_ids_path)

    print(f"train_ids.pt: {len(train_ids)} tokens")
    print(f"val_ids.pt: {len(val_ids)} tokens")
    print(f"Sample decode: {tokenizer.decode(train_ids[:80].tolist())}")

if __name__ == "__main__":
    tokenize_and_encode()
