from typing import Optional, List

import torch
from transformers import PreTrainedTokenizerFast

from .config import AltayConfig
from .model import AltayForCausalLM


class AltayInference:
    def __init__(
        self,
        model: AltayForCausalLM,
        tokenizer: PreTrainedTokenizerFast,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        eos_token_id = self.tokenizer.eos_token_id

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def chat(
        self,
        messages: List[dict],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        prompt = self._format_chat_prompt(messages)
        return self.generate(prompt, max_new_tokens, temperature, top_k, top_p)

    @staticmethod
    def _format_chat_prompt(messages: List[dict]) -> str:
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"[SİSTEM] {content}\n"
            elif role == "user":
                prompt += f"[KULLANICI] {content}\n"
            elif role == "assistant":
                prompt += f"[ALTAY] {content}\n"
        prompt += "[ALTAY] "
        return prompt
