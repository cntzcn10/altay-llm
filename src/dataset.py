from typing import Iterator, List, Optional
from datasets import load_dataset, interleave_datasets, concatenate_datasets
from transformers import PreTrainedTokenizerFast


class AltayDataset:
    TURKISH_SOURCES = [
        "uonlp/CulturaX",  # Turkish subset
        "wikipedia",        # Turkish Wikipedia
    ]

    ENGLISH_SOURCES = [
        "HuggingFaceFW/fineweb",
        "wikimedia/wikipedia",
    ]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
        mix_ratio: float = 0.5,  # Turkish ratio
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mix_ratio = mix_ratio

    def tokenize_function(self, examples):
        texts = examples.get("text", examples.get("content", [""]))
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=self.max_seq_len,
            return_attention_mask=False,
        )
        return {"input_ids": tokenized["input_ids"]}

    def get_dataset(self, split: str = "train", streaming: bool = True):
        datasets = []
        datasets.append(
            load_dataset(
                "uonlp/CulturaX",
                "tr",
                split=split,
                streaming=streaming,
                trust_remote_code=True,
            )
        )
        fineweb = load_dataset(
            "HuggingFaceFW/fineweb",
            split=split,
            streaming=streaming,
            trust_remote_code=True,
        )
        datasets.append(fineweb)

        if streaming:
            dataset = interleave_datasets(datasets, probabilities=[self.mix_ratio, 1 - self.mix_ratio])
        else:
            dataset = concatenate_datasets(datasets)

        dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return dataset

    @staticmethod
    def get_collate_fn(pad_token_id: int):
        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            max_len = max(len(ids) for ids in input_ids)
            padded = []
            for ids in input_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [pad_token_id] * pad_len)
            import torch
            return {"input_ids": torch.tensor(padded, dtype=torch.long)}
        return collate_fn
