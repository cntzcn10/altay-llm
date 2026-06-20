from typing import List, Optional, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from transformers import PreTrainedTokenizerFast


class AltayTokenizer:
    SPECIAL_TOKENS = {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
    }

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.SPECIAL_TOKENS["unk_token"]))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    def train(self, files: List[str]) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.SPECIAL_TOKENS.values()),
            min_frequency=2,
            show_progress=True,
        )
        self.tokenizer.train(files, trainer)

    def train_from_iterator(self, texts: List[str]) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=list(self.SPECIAL_TOKENS.values()),
            min_frequency=2,
            show_progress=True,
        )
        self.tokenizer.train_from_iterator(texts, trainer)

    def to_huggingface(self) -> PreTrainedTokenizerFast:
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            **self.SPECIAL_TOKENS,
        )
        hf_tokenizer.add_special_tokens(self.SPECIAL_TOKENS)
        return hf_tokenizer

    def save(self, path: str) -> None:
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "AltayTokenizer":
        tokenizer = cls.__new__(cls)
        tokenizer.tokenizer = Tokenizer.from_file(path)
        tokenizer.vocab_size = tokenizer.tokenizer.get_vocab_size()
        return tokenizer
