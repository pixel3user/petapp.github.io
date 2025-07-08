import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


def load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class QwenJsonlDataset(Dataset):
    def __init__(self, records: List[dict], processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        for _ in range(len(self.records)):  # attempt up to N records before failing
            rec = self.records[idx]
            try:
                # Validate conversation structure
                if "conversations" not in rec or len(rec["conversations"]) < 2:
                    raise ValueError("Missing conversation data")

                question = rec["conversations"][0]["value"]
                answer = rec["conversations"][1]["value"]

                image_path = rec["image"]
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image = Image.open(image_path).convert("RGB")

                data = self.processor(
                    text="<image>\n" + question,
                    images=image,
                    do_resize=True,
                    padding=False,
                    return_tensors="pt",
                )

                prompt_ids = data["input_ids"][0]
                pixel_values = data["pixel_values"][0]
                grid = data["image_grid_thw"][0]

                if pixel_values.shape[1] == 0 or grid.numel() == 0:
                    raise ValueError("Invalid patch grid")

                ans_ids = self.processor.tokenizer(
                    answer, add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0]

                input_ids = torch.cat([prompt_ids, ans_ids], dim=0)
                labels = torch.cat([torch.full_like(prompt_ids, -100), ans_ids], dim=0)

                return {
                    "input_ids": input_ids,
                    "labels": labels,
                    "pixel_values": pixel_values,
                    "image_grid_thw": grid,
                }

            except Exception as e:
                print(f"[WARN] Skipping idx={idx}: {e}")
                idx = (idx + 1) % len(self)

        # If we fail N times in a row, crash
        raise RuntimeError("All data records are invalid.")


def collate_fn(processor):
    pad_id = processor.tokenizer.pad_token_id

    def _fn(batch):
        input_ids = pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=pad_id,
        )
        labels = pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=-100,
        )
        attention_mask = input_ids.ne(pad_id)
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        grids = torch.stack([b["image_grid_thw"] for b in batch])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": grids,
        }

    return _fn


def find_linear_layers(model):
    names = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            names.append(n)
    return names


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL on jsonl dataset")
    parser.add_argument("--model", default="qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--grad_accum", type=int, default=1)
    args = parser.parse_args()

    train_file = os.path.join(args.data_dir, "train.jsonl")
    val_file = os.path.join(args.data_dir, "val.jsonl")
    train_records = load_jsonl(train_file)
    val_records = load_jsonl(val_file)

    processor = AutoProcessor.from_pretrained(args.model)

    train_ds = QwenJsonlDataset(train_records, processor)
    val_ds = QwenJsonlDataset(val_records, processor)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16
    )
    # ✅ ADD THIS:
    if torch.cuda.is_available():
        model = model.to("cuda")

    if args.lora_rank > 0:
        target_modules = find_linear_layers(model)
        lora_config = LoraConfig(r=args.lora_rank, target_modules=target_modules)
        model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=1,
        save_steps=200,
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn(processor),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
