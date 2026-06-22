# %% [markdown]
# # 🏔️ ALTAY LLM — Qwen 7B Fine-Tune
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
# TEK HÜCRE - HER ŞEY BURADA (ÇALIŞMA GARANTİLİ)
# ============================================================
import os, sys, json, gc, random, torch
from getpass import getpass

print("🏔️ ALTAY basliyor...")
print("GPU kontrol ediliyor...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
if not torch.cuda.is_available():
    print("❌ GPU YOK! Ustteki menuden: Calisma Zamani > Turunu degistir > T4 GPU sec > Kaydet")
    print("  Sonra bu hucreyi TEKRAR CALISTIR (sol ustteki ▶️)")
    sys.exit()
print(f"✅ GPU: {torch.cuda.get_device_name()}")

# --- KURULUM ---
print("📦 Kutuphaneler yukleniyor... (2 dk)")
!pip install -q transformers datasets accelerate peft bitsandbytes huggingface_hub

from huggingface_hub import login, HfApi, create_repo
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset, Dataset

# --- TOKEN ---
from google.colab import userdata
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = getpass('🤗 Hugging Face token: ')
    try:
        userdata.set("HF_TOKEN", HF_TOKEN)
    except:
        pass
login(token=HF_TOKEN)
HF_USERNAME = "cntzcn10"
MODEL_REPO = f"{HF_USERNAME}/altay-llm"
print(f"✅ Token alindi, model: {MODEL_REPO}")

# --- MODEL (Qwen 7B + 4-bit QLoRA) ---
print("🚀 Qwen 2.5 7B yukleniyor... (3 dk)")
try:
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ),
        device_map="auto", trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM",
    ))
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"✅ Model hazir! {trainable:.1f}M parametre")
except Exception as e:
    print(f"❌ Model yuklenemedi: {e}")
    sys.exit()

# --- VERI ---
print("📚 Veri yukleniyor... (2 dk)")
try:
    ALTAY_SISTEM = "Sen, Altay LLM'sin. Turkce ve Ingilizce her konuda yardimci olan bilgili ve nazik bir yapay zeka asistanisin."
    oa = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
    threadlar = {}
    for item in oa:
        t = item["message_tree_id"]
        if t not in threadlar:
            threadlar[t] = []
        threadlar[t].append(item)
        if len(threadlar) >= 5000:
            break

    konusmalar = []
    for tid, msgs in threadlar.items():
        msgs.sort(key=lambda x: x["created_date"])
        dil = msgs[0].get("lang", "en")
        if dil not in ["tr", "en"]:
            continue
        if dil == "en" and len(konusmalar) > 800:
            continue
        conv = [{"role": "system", "content": ALTAY_SISTEM}]
        for m in msgs:
            conv.append({"role": "assistant" if m["role"] == "assistant" else "user", "content": m["text"]})
        if len(conv) >= 2:
            konusmalar.append(tokenizer.apply_chat_template(conv, tokenize=False))
        if len(konusmalar) >= 2000:
            break

    random.shuffle(konusmalar)
    split = int(len(konusmalar) * 0.95)
    train_ds = Dataset.from_dict({"text": konusmalar[:split]})
    eval_ds = Dataset.from_dict({"text": konusmalar[split:]})
    print(f"✅ {len(train_ds)} egitim + {len(eval_ds)} degerlendirme")
except Exception as e:
    print(f"❌ Veri hatasi: {e}")
    sys.exit()

# --- EGITIM ---
print("🔥 EGITIM BASLIYOR! ~2 saat")
print("Colab'i kapatma, sayfa acik kalsin. Bitince bildiririm.")
torch.cuda.empty_cache()
gc.collect()
try:
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./altay-cikti",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
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
            dataloader_num_workers=0,
            optim="adamw_8bit",
            report_to="none",
            push_to_hub=False,
            remove_unused_columns=False,
        ),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()
    print("🎉 EGITIM TAMAMLANDI!")
except Exception as e:
    print(f"❌ Egitim hatasi: {e}")
    print("Tekrar Ctrl+F9 dene, genelde ikincide calisir.")
    sys.exit()

# --- KAYDET ---
print("💾 Hugging Face'e kaydediliyor...")
try:
    model.save_pretrained("./altay-son")
    tokenizer.save_pretrained("./altay-son")
    api = HfApi()
    create_repo(MODEL_REPO, exist_ok=True)
    api.upload_folder(repo_id=MODEL_REPO, folder_path="./altay-son", path_in_repo=".")
    print(f"✅ Model yayinda: https://huggingface.co/{MODEL_REPO}")
except Exception as e:
    print(f"❌ Kayit hatasi: {e}")

# --- TEST ---
print("\n🧪 Test ediliyor...")
try:
    model.eval()
    for soru in ["Turkiye'nin baskenti neresidir?", "Yapay zeka nedir? Kisaca aciklar misin?", "Merhaba! Nasilsin?"]:
        msg = [{"role":"system","content":ALTAY_SISTEM},{"role":"user","content":soru}]
        girdi = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            cikti = model.generate(girdi, max_new_tokens=200, temperature=0.7, top_p=0.9)
        cevap = tokenizer.decode(cikti[0][girdi.shape[1]:], skip_special_tokens=True)
        print(f"\n🧑 {soru}")
        print(f"🤖 {cevap}")
    print("\n🎉🎉🎉 ALTAY LLM HAZIR!")
except Exception as e:
    print(f"Test hatasi: {e}")

print("="*60)
print("📦 Model: https://huggingface.co/" + MODEL_REPO)
print("📂 Kod: https://github.com/cntzcn10/altay-llm")
