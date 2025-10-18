import torch
import json
from config import get_config
from model.gpt import GPT, GPTConfig
from tokenizer.bpe_tokenizer import BPETokenizer

#load and setup configuration
cfg = get_config()
device = cfg["system"]["device"]

#load tokenizer
tokenizer = BPETokenizer()
tokenizer.load(cfg["data"]["vocab_file"])
vocab_size = len(tokenizer.stoi)

#load model
model_cfg = GPTConfig(
    vocab_size=vocab_size,
    block_size=cfg["training"]["block_size"],
    n_embd=cfg["model"]["embed_dim"],
    n_head=cfg["model"]["num_heads"],
    n_layer=cfg["model"]["num_layers"],
    dropout=cfg["model"]["dropout"]
)

model = GPT(model_cfg).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
model.eval()
print("Loaded trained model!")

#generate text
@torch.no_grad()
def top_k_top_p_filter(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    Filter logits using top-k and/or top-p (nucleus) sampling
    """
    top_k = min(top_k, logits.size(-1))  # Safety
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits

@torch.no_grad()
def generate(model, tokenizer, start_text, max_tokens=200, temperature=1.0, top_k=0, top_p=0.0):
    idx = torch.tensor(tokenizer.encode(start_text), dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_tokens):
        idx_cond = idx[:, -cfg["training"]["block_size"]:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        # Apply top-k / top-p filtering
        logits_filtered = top_k_top_p_filter(logits.clone(), top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits_filtered, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return tokenizer.decode(idx[0].tolist())

#example
prompt = "To be, or not to be, "
output = generate(model, tokenizer, prompt, max_tokens=300, temperature=0.8)
print("\n=== Generated Text ===\n")
print(output)
