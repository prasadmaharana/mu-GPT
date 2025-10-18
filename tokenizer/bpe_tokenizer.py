import json
from collections import Counter

class BPETokenizer:
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def build_vocab(self, text, vocab_size=5000):
        # Start with character-level tokens
        vocab = Counter(text)
        vocab = {tuple(ch): freq for ch, freq in vocab.items()}

        # BPE merge operations
        for _ in range(vocab_size - len(vocab)):
            pairs = Counter()
            for word, freq in vocab.items():
                for i in range(len(word)-1):
                    pairs[(word[i], word[i+1])] += freq
            if not pairs:
                break
            best_pair = pairs.most_common(1)[0][0]

            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                        new_word.append(word[i]+word[i+1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab

        # create stoi and itos
        tokens = sorted(set([sub for word in vocab for sub in word]))
        self.stoi = {tok: i for i, tok in enumerate(tokens)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab_size = len(tokens)

    def encode(self, text):
        tokens = []
        for ch in text:
            if ch in self.stoi:
                tokens.append(self.stoi[ch])
            else:
                # handle unknown chars
                tokens.append(self.stoi.get('<unk>', 0))
        return tokens

    def decode(self, token_ids):
        return "".join([self.itos[i] for i in token_ids])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi, "itos": self.itos}, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {int(k): v for k, v in data["itos"].items()}
            self.vocab_size = len(self.stoi)
