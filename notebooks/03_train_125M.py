"""
==========================================================================
🏔️ ALTAY LLM — Adım 3: 125M Parametre Model Eğitimi (ANA EĞİTİM)
==========================================================================
Bu script'i Google Colab'de çalıştır:
1. colab.research.google.com'a git
2. File > Upload notebook > Bu dosyayı seç
3. Runtime > Run all (Ctrl+F9)
==========================================================================
"""

# %% [markdown]
# # 🏔️ Altay LLM — 125M Parametre Eğitimi
#
# **Tahmini süre:** ~12 saat (T4 GPU)
# **VRAM kullanımı:** ~8 GB
#
# ⚠️ Colab'in bağlantısı kesilmesin diye browser'ı açık tut!

# %% [code]
# Kütüphaneler
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers datasets tokenizers accelerate huggingface_hub pyyaml

# %% [code]
import os
import sys
import math
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi, login, create_repo
from accelerate import Accelerator
from google.colab import userdata

# %% [code]
# Hugging Face girişi
HF_TOKEN = userdata.get("HF_TOKEN") or input("HF token: ")
login(token=HF_TOKEN)
HF_USERNAME = userdata.get("HF_USERNAME") or input("HF kullanıcı adı: ")
TOKENIZER_REPO = f"{HF_USERNAME}/altay-tokenizer"
MODEL_REPO = f"{HF_USERNAME}/altay-125M"

# Tokenizer'ı yükle
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_REPO)
PAD_ID = tokenizer.pad_token_id or 0
print(f"Tokenizer yüklendi! Vocab: {tokenizer.vocab_size}, Pad ID: {PAD_ID}")

# %% [markdown]
# ## Model Mimarisi

# %% [code]
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
            TransformerBlock(dim, hidden_dim, n_heads, n_kv_heads)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        cos, sin = precompute_freqs_cis(dim // n_heads, max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None):
        h = self.token_embedding(input_ids)
        for layer in self.layers:
            h = layer(h, self.cos, self.sin)
        h = self.norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

# %% [markdown]
# ## Modeli Oluştur

# %% [code]
# 125M parametre konfigürasyonu
VOCAB_SIZE = tokenizer.vocab_size
DIM = 768
HIDDEN_DIM = 2048
N_LAYERS = 12
N_HEADS = 12
N_KV_HEADS = 4
MAX_SEQ_LEN = 2048

model = AltayModel(VOCAB_SIZE, DIM, HIDDEN_DIM, N_LAYERS, N_HEADS, N_KV_HEADS, MAX_SEQ_LEN)

# Parametre sayısını hesapla
total = sum(p.numel() for p in model.parameters())
print(f"🏔️ Altay-125M oluşturuldu!")
print(f"   Parametre: {total:,} ({total/1e6:.1f}M)")
print(f"   Katman: {N_LAYERS} | Boyut: {DIM} | Heads: {N_HEADS} | KV-Heads: {N_KV_HEADS}")

# %% [markdown]
# ## Dataset Pipeline (Streaming)

# %% [code]
print("Türkçe veri yükleniyor...")
tr_data = load_dataset("uonlp/CulturaX", "tr", split="train", streaming=True, trust_remote_code=True)
print("İngilizce veri yükleniyor...")
en_data = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True, trust_remote_code=True)
dataset = interleave_datasets([tr_data, en_data], probabilities=[0.7, 0.3])

def tokenize_fn(examples):
    texts = examples.get("text", "")
    return tokenizer(texts, truncation=True, padding=False, max_length=MAX_SEQ_LEN)

dataset = dataset.map(tokenize_fn, batched=True, batch_size=100, remove_columns=dataset.column_names)
dataset = dataset.with_format("torch")

# %% [markdown]
# ## Eğitim Döngüsü

# %% [code]
# Hiperparametreler
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 3e-4
MIN_LR = 3e-5
WARMUP = 1000
MAX_STEPS = 50000
LOG_STEPS = 10
SAVE_STEPS = 5000
GRAD_CLIP = 1.0

# Accelerator (bf16 otomatik)
accelerator = Accelerator(
    gradient_accumulation_steps=GRAD_ACCUM,
    mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
)
device = accelerator.device
print(f"Device: {device} | BF16: {torch.cuda.is_bf16_supported()}")

# Optimizer
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))

# DataLoader
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)

# Prepare
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# %% [code]
# Eğitim
print("🚀 Eğitim başlıyor! Colab'i kapatma, browser'ı açık tut.")
print(f"Toplam adım: {MAX_STEPS} | Batch: {BATCH_SIZE} | Grad Accum: {GRAD_ACCUM}")
print(f"LR: {LR} | Warmup: {WARMUP}")

model.train()
global_step = 0
total_loss = 0
start_time = time.time()

while global_step < MAX_STEPS:
    for batch in dataloader:
        if global_step >= MAX_STEPS:
            break

        # LR schedule
        if global_step < WARMUP:
            lr = LR * (global_step + 1) / WARMUP
        else:
            progress = (global_step - WARMUP) / (MAX_STEPS - WARMUP)
            lr = MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        with accelerator.accumulate(model):
            input_ids = batch["input_ids"]
            _, loss = model(input_ids, labels=input_ids)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            total_loss += loss.item()

            if global_step % LOG_STEPS == 0:
                avg_loss = total_loss / LOG_STEPS
                elapsed = time.time() - start_time
                tps = BATCH_SIZE * GRAD_ACCUM * MAX_SEQ_LEN * LOG_STEPS / elapsed
                print(f"Step {global_step}/{MAX_STEPS} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | Tok/s: {tps:.0f}")
                total_loss = 0
                start_time = time.time()

            if global_step % SAVE_STEPS == 0:
                print(f"💾 Checkpoint kaydediliyor: step-{global_step}...")
                accelerator.save_state(f"/content/checkpoint-{global_step}")

                # Hugging Face'e yükle
                if accelerator.is_main_process:
                    accelerator.unwrap_model(model).cpu()
                    torch.save(model.state_dict(), f"/content/altay-125M-step-{global_step}.pt")
                    hf_api = HfApi(token=HF_TOKEN)
                    try:
                        hf_api.create_repo(MODEL_REPO, exist_ok=True)
                        hf_api.upload_file(
                            path_or_fileobj=f"/content/altay-125M-step-{global_step}.pt",
                            path_in_repo=f"checkpoint-{global_step}.pt",
                            repo_id=MODEL_REPO,
                        )
                        print(f"✅ Hugging Face'e yüklendi: {MODEL_REPO}")
                    except Exception as e:
                        print(f"Hata: {e}")
                    model.to(device)

print("🎉 Eğitim tamamlandı!")
print(f"Model: https://huggingface.co/{MODEL_REPO}")
