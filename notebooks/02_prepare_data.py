"""
==========================================================================
🏔️ ALTAY LLM — Adım 2: Veri Hazırlık
==========================================================================
Bu script'i Google Colab'de çalıştır:
1. colab.research.google.com'a git
2. File > Upload notebook > Bu dosyayı seç
3. Runtime > Run all (Ctrl+F9)
==========================================================================
"""

# %% [markdown]
# # 🏔️ Altay LLM — Veri Hazırlık
# Streaming dataset pipeline

# %% [code]
!pip install -q datasets transformers huggingface_hub

# %% [code]
import torch
from datasets import load_dataset, interleave_datasets
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login
from google.colab import userdata

# %% [code]
# Hugging Face girişi
HF_TOKEN = userdata.get("HF_TOKEN") or input("HF token: ")
login(token=HF_TOKEN)
HF_USERNAME = userdata.get("HF_USERNAME") or input("HF kullanıcı adı: ")
TOKENIZER_REPO = f"{HF_USERNAME}/altay-tokenizer"

# Tokenizer'ı yükle
tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_REPO)
print(f"Tokenizer yüklendi! Vocab: {tokenizer.vocab_size}")

# %% [markdown]
# ## Dataset Pipeline
# Hugging Face'den streaming ile veri çekme

# %% [code]
def create_dataset(max_seq_len=2048, batch_size=8):
    # Türkçe veri
    print("Türkçe CulturaX yükleniyor...")
    tr_data = load_dataset(
        "uonlp/CulturaX", "tr",
        split="train", streaming=True,
        trust_remote_code=True
    )

    # İngilizce veri
    print("İngilizce FineWeb yükleniyor...")
    en_data = load_dataset(
        "HuggingFaceFW/fineweb",
        split="train", streaming=True,
        trust_remote_code=True
    )

    # Türkçe ağırlıklı interleave (%70 Türkçe, %30 İngilizce)
    dataset = interleave_datasets(
        [tr_data, en_data],
        probabilities=[0.7, 0.3],
        stopping_strategy="first_exhausted"
    )

    def tokenize_fn(examples):
        texts = examples.get("text", examples.get("content", ""))
        if isinstance(texts, str):
            texts = [texts]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_seq_len,
        )
        return {"input_ids": tokenized["input_ids"]}

    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
    )

    return dataset

# %% [code]
# Test - 1 batch al
dataset = create_dataset(max_seq_len=512)
sample = next(iter(dataset))
print(f"Örnek batch: {len(sample['input_ids'])} token")
print(f"Token ID'ler (ilk 20): {sample['input_ids'][:20]}")
print(f"Çözüm: {tokenizer.decode(sample['input_ids'][:30])}")

# %% [code]
print("✅ Veri pipeline'ı hazır!")
print(f"Dataset tipi: {type(dataset)}")
print("Eğitim notebook'una geçebilirsin!")
