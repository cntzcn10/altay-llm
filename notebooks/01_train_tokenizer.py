"""
==========================================================================
🏔️ ALTAY LLM — Adım 1: Tokenizer Eğitimi
==========================================================================
Bu script'i Google Colab'de çalıştır:
1. colab.research.google.com'a git
2. File > Upload notebook > Bu dosyayı seç
3. Runtime > Run all (Ctrl+F9)
==========================================================================
"""

# %% [markdown]
# # 🏔️ Altay LLM — Tokenizer Eğitimi
# Türkçe + İngilizce için BPE tokenizer eğitimi

# %% [code]
# Gerekli kütüphaneleri yükle
!pip install -q tokenizers datasets huggingface_hub sentencepiece

# %% [code]
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast
from huggingface_hub import HfApi, login
from google.colab import userdata

# %% [code]
# Hugging Face token'ı ayarla (kendi token'ını kullan)
HF_TOKEN = userdata.get("HF_TOKEN") or input("Hugging Face token'ını gir: ")
login(token=HF_TOKEN)

# Hugging Face repo ismi
HF_USERNAME = userdata.get("HF_USERNAME") or input("Hugging Face kullanıcı adını gir: ")
REPO_NAME = f"{HF_USERNAME}/altay-tokenizer"

# %% [code]
print(f"Tokenizer Hugging Face'e yüklenecek: {REPO_NAME}")

# %% [markdown]
# ## Veri Toplama
# Türkçe ve İngilizce metinler indiriliyor.

# %% [code]
from datasets import load_dataset

print("Türkçe veri indiriliyor...")
tr_dataset = load_dataset("uonlp/CulturaX", "tr", split="train", streaming=True)
tr_texts = [item["text"] for i, item in enumerate(tr_dataset) if i < 50000]

print("İngilizce veri indiriliyor...")
en_dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
en_texts = [item["text"] for i, item in enumerate(en_dataset) if i < 50000]

all_texts = tr_texts + en_texts
print(f"Toplam {len(all_texts):,} metin toplandı")

# %% [markdown]
# ## BPE Tokenizer Eğitimi

# %% [code]
print("Tokenizer eğitiliyor...")

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    min_frequency=2,
    show_progress=True,
)

tokenizer.train_from_iterator(all_texts, trainer)
print(f"Tokenizer eğitildi! Kelime dağarcığı: {tokenizer.get_vocab_size()}")

# %% [code]
# Hugging Face formatına dönüştür
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    pad_token="[PAD]",
    unk_token="[UNK]",
    bos_token="[BOS]",
    eos_token="[EOS]",
)
hf_tokenizer.save_pretrained("altay-tokenizer")
tokenizer.save("altay-tokenizer/tokenizer.json")

# %% [code]
# Hugging Face'e yükle
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(REPO_NAME, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=REPO_NAME,
        folder_path="altay-tokenizer",
        path_in_repo=".",
    )
    print(f"✅ Tokenizer yüklendi: https://huggingface.co/{REPO_NAME}")
except Exception as e:
    print(f"Hata: {e}")

# %% [code]
# Test
test_texts = [
    "Merhaba dünya! Bu Altay LLM için bir test.",
    "Hello world! This is a test for Altay LLM.",
    "Türkçe ve İngilizce karışık bir metin: yapay zeka, machine learning, derin öğrenme."
]

for text in test_texts:
    encoded = hf_tokenizer.encode(text)
    decoded = hf_tokenizer.decode(encoded)
    print(f"\nOrijinal: {text}")
    print(f"Token ID: {encoded[:20]}...")
    print(f"Çözülmüş: {decoded}")
