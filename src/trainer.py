import os
import math
import time
import logging
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from accelerate import Accelerator
from huggingface_hub import HfApi, create_repo
from datasets import load_dataset

from .config import AltayConfig
from .model import AltayForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: AltayForCausalLM,
        config: AltayConfig,
        train_config: Dict[str, Any],
        hf_token: Optional[str] = None,
        hf_repo_id: Optional[str] = None,
    ):
        self.model = model
        self.config = config
        self.train_config = train_config
        self.hf_token = hf_token
        self.hf_repo_id = hf_repo_id
        self.accelerator = Accelerator(
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
            mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16",
        )
        self.device = self.accelerator.device

    def save_checkpoint(self, step: int, loss: float) -> None:
        if self.hf_repo_id and self.hf_token:
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                hf_api = HfApi(token=self.hf_token)
                hf_api.upload_folder(
                    repo_id=self.hf_repo_id,
                    folder_path="/tmp/altay-checkpoint",
                    path_in_repo=f"checkpoint-{step}",
                )
                logger.info(f"Checkpoint saved to {self.hf_repo_id}/checkpoint-{step}")

    def train(self, dataset: IterableDataset, pad_token_id: int) -> None:
        model = self.model
        train_config = self.train_config

        optimizer = AdamW(
            model.parameters(),
            lr=train_config["learning_rate"],
            weight_decay=train_config["weight_decay"],
            betas=(0.9, 0.95),
        )

        dataloader = DataLoader(
            dataset,
            batch_size=train_config["batch_size"],
            collate_fn=self._collate_fn(pad_token_id),
            num_workers=0,
            pin_memory=True,
        )

        model, optimizer, dataloader = self.accelerator.prepare(
            model, optimizer, dataloader
        )

        max_steps = train_config["max_steps"]
        warmup_steps = train_config["warmup_steps"]
        log_steps = train_config["log_steps"]
        save_steps = train_config["save_steps"]
        grad_clip = train_config["grad_clip"]
        max_lr = train_config["learning_rate"]
        min_lr = train_config["min_lr"]

        global_step = 0
        total_loss = 0
        start_time = time.time()

        model.train()
        while global_step < max_steps:
            for batch in dataloader:
                if global_step >= max_steps:
                    break

                lr = self._get_lr(global_step, warmup_steps, max_steps, max_lr, min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                with self.accelerator.accumulate(model):
                    input_ids = batch["input_ids"]
                    _, loss = model(input_ids, labels=input_ids)
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), grad_clip)

                    optimizer.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    global_step += 1
                    total_loss += loss.item()

                    if global_step % log_steps == 0:
                        avg_loss = total_loss / log_steps
                        elapsed = time.time() - start_time
                        tokens_per_sec = (
                            train_config["batch_size"]
                            * train_config["gradient_accumulation_steps"]
                            * self.config.max_seq_len
                            * log_steps
                            / elapsed
                        )
                        logger.info(
                            f"Step {global_step}/{max_steps} | Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | Tokens/s: {tokens_per_sec:.0f}"
                        )
                        total_loss = 0
                        start_time = time.time()

                    if global_step % save_steps == 0:
                        self.save_checkpoint(global_step, loss.item())

                        if self.accelerator.is_main_process:
                            self.accelerator.save_state(f"/tmp/altay-checkpoint/checkpoint-{global_step}")

    def _get_lr(self, step: int, warmup: int, max_steps: int, max_lr: float, min_lr: float) -> float:
        if step < warmup:
            return max_lr * (step + 1) / warmup
        progress = (step - warmup) / (max_steps - warmup)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    @staticmethod
    def _collate_fn(pad_token_id: int):
        def collate(batch):
            input_ids = [item["input_ids"] for item in batch]
            max_len = max(len(ids) for ids in input_ids)
            padded = []
            for ids in input_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [pad_token_id] * pad_len)
            return {"input_ids": torch.tensor(padded, dtype=torch.long)}
        return collate
