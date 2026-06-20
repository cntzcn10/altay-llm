from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AltayConfig:
    vocab_size: int = 32000
    hidden_dim: int = 768
    intermediate_dim: int = 2048
    num_layers: int = 12
    num_attention_heads: int = 12
    num_kv_heads: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    name: str = "altay-125M"

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.num_attention_heads

    @property
    def total_params(self) -> int:
        embedding = self.vocab_size * self.hidden_dim
        per_layer_attn = (
            self.hidden_dim * self.hidden_dim
            + self.hidden_dim * self.num_kv_heads * self.head_dim
            + self.hidden_dim * self.num_kv_heads * self.head_dim
            + self.hidden_dim * self.hidden_dim
        )
        per_layer_ffn = 3 * self.hidden_dim * self.intermediate_dim
        per_layer_norm = 2 * self.hidden_dim
        layers = self.num_layers * (per_layer_attn + per_layer_ffn + per_layer_norm)
        final_norm = self.hidden_dim
        lm_head = self.hidden_dim * self.vocab_size
        return embedding + layers + final_norm + lm_head
