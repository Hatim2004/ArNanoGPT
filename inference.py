"""
Inference script for Hatim2221/arabic-nanogptv2.
Downloads the model from Hugging Face, rebuilds the architecture,
loads the trained weights, and generates Arabic text.

Usage:
    python inference.py
"""

import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# ---------------------------------------------------------------------------
# 1. Download model files from the Hub
# ---------------------------------------------------------------------------
REPO_ID = "Hatim2221/arabic-nanogptv2"

config_path = hf_hub_download(REPO_ID, "config.json")
weights_path = hf_hub_download(REPO_ID, "pytorch_model.bin")
tokenizer_path = hf_hub_download(REPO_ID, "tokenizer.json")

with open(config_path) as f:
    config = json.load(f)

vocab_size = config["vocab_size"]
n_embd = config["n_embd"]
n_layer = config["n_layer"]
n_head = config["n_head"]
block_size = config["block_size"]
dropout = config["dropout"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
print("config:", config)

# ---------------------------------------------------------------------------
# 2. Load tokenizer
# ---------------------------------------------------------------------------
tokenizer = Tokenizer.from_file(tokenizer_path)
encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)

# ---------------------------------------------------------------------------
# 3. Model architecture (must match training exactly)
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.ffwd = FeedFoward(n_embd)  # unused in forward, kept to match saved checkpoint's keys
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, repetition_penalty=1.3):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            # penalize tokens that already appeared in the recent context
            if repetition_penalty is not None and repetition_penalty != 1.0:
                for token_id in set(idx_cond[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# 4. Instantiate model and load trained weights
# ---------------------------------------------------------------------------
model = BigramLanguageModel(vocab_size)
model.load_state_dict(torch.load(weights_path, map_location=device))
m = model.to(device)
m.eval()  # disable dropout for clean generation

total_params = sum(p.numel() for p in model.parameters())
print(f"Model loaded successfully. Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

# ---------------------------------------------------------------------------
# 5. Generate text
# ---------------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
output_ids = m.generate(
    context,
    max_new_tokens=300,
    temperature=0.7,
    top_k=40,
    repetition_penalty=1.3,
)[0].tolist()

print(decode(output_ids))
