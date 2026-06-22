# %% [markdown]
# # 🏔️ ALTAY LLM TURBO — Tek Tıkla Efsane Model
#
# **Ne yapar:** Qwen 2.5 7B'yi Türkçe + İngilizce için fine-tune eder
# **Süre:** ~2 saat (Colab T4 GPU)
# **Yapman gereken:** ⬇️ AŞAĞIDAKİ 3 ADIMI TAKİP ET

# %% [markdown]
# ## 🎯 3 ADIMDA ÇALIŞTIR
#
# 1. **Runtime** → **Run all** (Ctrl+F9) — tıkla
# 2. Açılan kutuya Hugging Face token'ını **yapıştır** → Enter
# 3. **2 saat bekle** — Colab açık kalsın, bitince modelin hazır!
#
# ⚡ Tokenin yoksa: https://huggingface.co/settings/tokens

# %% [code]
# ============================================================
# ADIM 0: Gerekli kütüphaneler ve token
# ============================================================
import os, sys, time, json, math, torch, gc, random
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from huggingface_hub import HfApi, login, create_repo
from google.colab import userdata
from getpass import getpass

try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = getpass("🤗 Hugging Face token'ını yapıştır ve Enter'a bas: ")
    try:
        userdata.set("HF_TOKEN", HF_TOKEN)
    except:
        pass

login(token=HF_TOKEN)
HF_USERNAME = "cntzcn10"
MODEL_REPO = f"{HF_USERNAME}/altay-llm"
print(f"✅ Token alındı, model yüklenecek: {MODEL_REPO}")

# %% [code]
# ============================================================
# ADIM 1: Unsloth + Qwen 2.5 7B
# ============================================================
print("⚡ Unsloth kuruluyor... (1 dk)")
!pip install -q unsloth xformers trl accelerate bitsandbytes

from unsloth import FastLanguageModel

MAX_SEQ_LEN = 4096
print("🚀 Qwen 2.5 7B yükleniyor... (2 dk)")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B",
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"📊 {total/1e9:.2f}B parametre | {trainable/1e6:.2f}M eğitilecek")

# %% [markdown]
# ## Veri Hazırlığı
# OpenAssistant'tan Türkçe + İngilizce konuşmalar alınıyor

# %% [code]
# ============================================================
# ADIM 2: Veri seti
# ============================================================
print("📚 Veri hazırlanıyor... (2 dk)")

ALTAY_SISTEM = "Sen, Altay LLM'sin. Türkçe ve İngilizce her konuda yardımcı olan bilgili ve nazik bir yapay zeka asistanısın."

def format_konusma(mesajlar):
    icerik = [{"role": "system", "content": ALTAY_SISTEM}]
    for m in mesajlar:
        icerik.append({"role": m["role"], "content": m["text"]})
    return tokenizer.apply_chat_template(icerik, tokenize=False)

oa = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)

threads = {}
for item in oa:
    tid = item["message_tree_id"]
    if tid not in threads:
        threads[tid] = []
    threads[tid].append(item)
    if len(threads) >= 6000:
        break

konusmalar = []
random.seed(42)
for tid, mesajlar in threads.items():
    mesajlar.sort(key=lambda x: x["created_date"])
    dil = mesajlar[0].get("lang", "en")
    if dil not in ["tr", "en"]:
        continue
    if dil == "en" and random.random() > 0.15:
        continue
    conv = []
    for m in mesajlar:
        role = "assistant" if m["role"] == "assistant" else "user"
        conv.append({"role": role, "text": m["text"]})
    if 2 <= len(conv) <= 20:
        konusmalar.append(format_konusma(conv))
    if len(konusmalar) >= 3000:
        break

random.shuffle(konusmalar)
split = int(len(konusmalar) * 0.95)

from datasets import Dataset
train_ds = Dataset.from_dict({"text": konusmalar[:split]})
eval_ds = Dataset.from_dict({"text": konusmalar[split:]})
print(f"✅ {len(train_ds)} egitim + {len(eval_ds)} eval konusmasi")

# %% [markdown]
# ## Eğitim Başlıyor!
# ~2 saat sürecek. Colab'i kapatma, sayfayı açık tut.

# %% [code]
# ============================================================
# ADIM 3: EĞİTİM
# ============================================================
print("🔥 EĞİTİM BAŞLIYOR! ~2 SAAT")
print("Colab'i kapatma, sayfa açık kalsın. Bitince seni uyaracağım.")

args = TrainingArguments(
    output_dir="./altay-cikti",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_steps=50,
    save_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    max_grad_norm=0.3,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    dataloader_num_workers=2,
    report_to="none",
    push_to_hub=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
)

trainer.train()
print("🎉 EĞİTİM TAMAMLANDI!")

# %% [code]
# ============================================================
# ADIM 4: Hugging Face'e yükle
# ============================================================
print("💾 Hugging Face'e kaydediliyor...")

model.save_pretrained_merged("./altay-son", tokenizer, save_method="merged_16bit")

api = HfApi(token=HF_TOKEN)
create_repo(MODEL_REPO, exist_ok=True)
api.upload_folder(repo_id=MODEL_REPO, folder_path="./altay-son", path_in_repo=".")
print(f"✅ Model yayında: https://huggingface.co/{MODEL_REPO}")

# %% [code]
# ============================================================
# ADIM 5: Test
# ============================================================
print("🧪 Test ediliyor...")

model, tokenizer = FastLanguageModel.from_pretrained(
    "./altay-son", max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

testler = [
    "Türkiye'nin başkenti neresidir?",
    "Yapay zeka nedir? Kısaca açıklar mısın?",
    "Merhaba! Nasılsın? Biraz kendinden bahseder misin?",
]

for test in testler:
    msg = [{"role": "system", "content": ALTAY_SISTEM}, {"role": "user", "content": test}]
    girdi = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    cikti = model.generate(girdi, max_new_tokens=200, temperature=0.7, top_p=0.9)
    cevap = tokenizer.decode(cikti[0][girdi.shape[1]:], skip_special_tokens=True)
    print(f"\n🧑 {test}")
    print(f"🤖 {cevap}")

# %% [code]
print("="*60)
print("🏔️ ALTAY LLM HAZIR!")
print("="*60)
print(f"\n📦 Model: https://huggingface.co/{MODEL_REPO}")
print(f"💬 Chat (20sn sonra hazır): https://cntzcn10-altay-llm.hf.space")
print(f"\n📂 Kod: https://github.com/cntzcn10/altay-llm")
print("\n🎉 Tebrikler! Kendi LLM'in hazır!")
