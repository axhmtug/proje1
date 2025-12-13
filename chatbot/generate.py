import torch
import torch.nn as nn
import math
import sentencepiece as spm

# ===========================================================
# MODEL DEFINITION (same as training)
# ===========================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        rms = norm / math.sqrt(x.size(-1))
        return self.weight * x / (rms + self.eps)

def rotary_embedding(q, k, seq_len, dim):
    pos = torch.arange(seq_len, device=q.device).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    angles = torch.einsum('i,j->ij', pos, freqs)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    emb = emb.unsqueeze(0).unsqueeze(2)

    q_embed = torch.cat([
        q[..., :dim//2] * emb[..., dim//2:] + q[..., dim//2:] * emb[..., :dim//2],
        q[..., :dim//2] * emb[..., :dim//2] - q[..., dim//2:] * emb[..., dim//2:]
    ], dim=-1)

    k_embed = torch.cat([
        k[..., :dim//2] * emb[..., dim//2:] + k[..., dim//2:] * emb[..., :dim//2],
        k[..., :dim//2] * emb[..., :dim//2] - k[..., dim//2:] * emb[..., dim//2:]
    ], dim=-1)

    return q_embed, k_embed

class MultiQueryAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, self.head_dim)
        self.v = nn.Linear(dim, self.head_dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q(x).view(B, T, self.heads, self.head_dim)
        k = self.k(x).unsqueeze(2)
        v = self.v(x).unsqueeze(2)

        q, k = rotary_embedding(q, k, T, self.head_dim)

        att = torch.einsum("bthd,bThd->bhtT", q, k) / math.sqrt(self.head_dim)
        att = att.masked_fill(torch.triu(torch.ones(T, T), 1).bool().to(x.device), float('-inf'))
        att = att.softmax(dim=-1)

        out = torch.einsum("bhtT,bThd->bthd", att, v)
        out = out.reshape(B, T, C)
        return self.o(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiQueryAttention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.ff = FeedForward(dim, dim * 4)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, vocab, dim=256, layers=6, heads=4, max_len=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, vocab)

    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        return self.out(h)

# ===========================================================
# LOAD TOKENIZER + MODEL
# ===========================================================

sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
VOCAB = sp.get_piece_size()

model = TinyLM(VOCAB)
model.load_state_dict(torch.load("tinyLM.pt", map_location="cpu"))
model.eval()

# ===========================================================
# GENERATION
# ===========================================================

@torch.no_grad()
def generate(prompt, max_new_tokens=30, temperature=0.75, top_k=50):
    x = torch.tensor(sp.encode(prompt, out_type=int)).unsqueeze(0)

    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]

        if temperature > 0:
            logits /= temperature

        if top_k:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1].unsqueeze(1)] = -1e10

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        x = torch.cat([x, next_id], dim=1)

    return sp.decode(x.squeeze().tolist())


if __name__ == "__main__":
    print(generate(" i feel nervous now  "))
