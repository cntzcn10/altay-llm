import os
import torch
import gradio as gr
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AltayConfig
from src.model import AltayForCausalLM
from src.inference import AltayInference


HF_REPO_ID = os.getenv("HF_REPO_ID", "altay-llm/altay-125M")
HF_TOKEN = os.getenv("HF_TOKEN", None)

config = AltayConfig()

model = None
inference_engine = None


def load_model():
    global model, inference_engine
    try:
        checkpoint_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename="pytorch_model.bin",
            token=HF_TOKEN,
        )
        model = AltayForCausalLM(config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        tokenizer = PreTrainedTokenizerFast.from_pretrained(HF_REPO_ID, token=HF_TOKEN)
        inference_engine = AltayInference(model, tokenizer)
        return "Model yüklendi!"
    except Exception as e:
        return f"Model yüklenemedi: {e}"


def predict(message, history, temperature, top_k, top_p, max_tokens):
    if inference_engine is None:
        result = load_model()
        if "yüklenemedi" in result:
            return result

    messages = [{"role": "user", "content": message}]
    response = inference_engine.generate(
        prompt=message,
        max_new_tokens=int(max_tokens),
        temperature=temperature,
        top_k=int(top_k),
        top_p=top_p,
    )
    return response


with gr.Blocks(
    title="Altay LLM",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="emerald"),
) as demo:
    gr.Markdown(
        """
        # 🏔️ Altay LLM

        **İlk Türk Süper LLM** — Tamamen bulut tabanlı, her alanda yetkin.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=predict,
                title="Altay ile Sohbet",
                description="Aşağıya mesajını yaz, Altay LLM sana cevap versin.",
                additional_inputs=[
                    gr.Slider(0.1, 2.0, value=0.7, label="Sıcaklık (Temperature)"),
                    gr.Slider(1, 100, value=50, step=1, label="Top-K"),
                    gr.Slider(0.1, 1.0, value=0.9, label="Top-P"),
                    gr.Slider(50, 512, value=200, step=1, label="Maksimum Token"),
                ],
            )

        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### ⚙️ Kontroller
                - **Sıcaklık:** Düşük = tutarlı, Yüksek = yaratıcı
                - **Top-K:** En olası K token arasından seç
                - **Top-P:** Olasılık birikimine göre seç

                ### 📊 Model Bilgisi
                - Parametre: 125M
                - Mimarı: LLaMA-style
                - Dil: Türkçe + İngilizce

                ### 🔗 Bağlantılar
                - [GitHub](https://github.com/altay-llm/altay-llm)
                - [Hugging Face](https://huggingface.co/altay-llm)
                """
            )

    load_model()

if __name__ == "__main__":
    demo.launch()
