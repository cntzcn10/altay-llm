"""
==========================================================================
🏔️ ALTAY LLM — Adım 4: Modeli Test Et (Chat)
==========================================================================
Bu script'i Google Colab'de çalıştır:
1. colab.research.google.com'a git
2. File > Upload notebook > Bu dosyayı seç
3. Runtime > Run all (Ctrl+F9)
==========================================================================
"""

# %% [markdown]
# # 🏔️ Altay LLM — Chat Test
# Eğitilmiş modelle konuşma!

# %% [code]
!pip install -q torch transformers huggingface_hub gradio

# %% [code]
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from huggingface_hub import hf_hub_download, login
from google.colab import userdata

# %% [code]
HF_TOKEN = userdata.get("HF_TOKEN") or input("HF token: ")
login(token=HF_TOKEN)
HF_USERNAME = userdata.get("HF_USERNAME") or input("HF kullanıcı adı: ")
MODEL_REPO = f"{HF_USERNAME}/altay-125M"
TOKENIZER_REPO = f"{HF_USERNAME}/altay-tokenizer"

print(f"Model: {MODEL_REPO}")
print(f"Tokenizer: {TOKENIZER_REPO}")

# %% [code]
# Model mimarisi (eğitimdekiyle aynı)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x.float() / rms).to(x.dtype) * self.weight

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (rotate_half(x) * sin)

class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin, mask=None):
        batch, seq, _ = x.shape
        q = self.wq(x).view(batch, seq, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch, seq, self.n_kv_heads, self.head_dim)
        q = apply_rotary_emb(q, cos[:seq], sin[:seq])
        k = apply_rotary_emb(k, cos[:seq], sin[:seq])
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)
        out = nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.wo(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(nn.functional.silu(self.gate(x)) * self.up(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim, n_heads, n_kv_heads):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)

    def forward(self, x, cos, sin, mask=None):
        h = x + self.attention(self.attention_norm(x), cos, sin, mask)
        return h + self.feed_forward(self.ffn_norm(h))

class AltayModel(nn.Module):
    def __init__(self, vocab_size, dim, hidden_dim, n_layers, n_heads, n_kv_heads, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        cos, sin = precompute_freqs_cis(dim // n_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, input_ids):
        h = self.token_embedding(input_ids)
        for layer in self.layers:
            h = layer(h, self.cos, self.sin)
        h = self.norm(h)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.cos.shape[0]:
                input_ids = input_ids[:, -self.cos.shape[0]:]
            logits = self.forward(input_ids)[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cum_probs > top_p
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = 0
                indices_mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_mask] = -float("Inf")

            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

# %% [code]
# Modeli yükle
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_REPO)

checkpoint_path = hf_hub_download(repo_id=MODEL_REPO, filename="pytorch_model.bin", token=HF_TOKEN)

model = AltayModel(
    vocab_size=tokenizer.vocab_size,
    dim=768, hidden_dim=2048,
    n_layers=12, n_heads=12, n_kv_heads=4,
    max_seq_len=2048
)
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.to("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Model yüklendi! Device: {next(model.parameters()).device}")

# %% [code]
# Sohbet fonksiyonu
def chat(prompt, max_tokens=200, temperature=0.7, top_k=50, top_p=0.9):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(next(model.parameters()).device)
    output = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# %% [code]
# Test
test_prompts = [
    "Türkiye'nin başkenti neresidir?",
    "Yapay zeka nedir? Kısaca açıkla.",
    "Merhaba, nasılsın?",
]

for prompt in test_prompts:
    print(f"\n🧑 {prompt}")
    print(f"🤖 {chat(prompt, max_tokens=100)}")

# %% [code]
# İnteraktif chat
print("\n" + "="*50)
print("🏔️ ALTAY LLM ile sohbet et!")
print("Çıkmak için 'exit' yaz.")
print("="*50)

while True:
    user_input = input("\n🧑 Sen: ")
    if user_input.lower() in ["exit", "çıkış", "quit"]:
        print("🤖 Altay: Görüşmek üzere! 🏔️")
        break
    response = chat(user_input, max_tokens=200)
    print(f"🤖 Altay: {response}")
