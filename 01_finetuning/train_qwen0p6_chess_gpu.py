"""
GPU fine-tuning script for Qwen3-0.6B on the chess move + rationale task.

Usage (single GPU example):
    pip install "torch>=2.3.0" "transformers>=4.45.0" "datasets>=3.0.0" "trl>=0.11.4" "peft>=0.16.0"
    python 01_finetuning/train_qwen0p6_chess_gpu.py --output-dir ./outputs/qwen3-0p6-chess

This trains a LoRA adapter on aicrowd/ChessExplained, saves the adapter, merges it
into the base model, and optionally pushes the merged model to Hugging Face.
"""

import argparse
import os
from typing import Tuple

import torch
import inspect
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def load_splits(train_size: int, eval_size: int, seed: int) -> Tuple[dict, dict]:
    dataset = load_dataset("aicrowd/ChessExplained", split="train")
    dataset = dataset.shuffle(seed=seed)

    train_end = min(train_size, len(dataset))
    train_ds = dataset.select(range(train_end))

    eval_start = train_end
    eval_end = min(eval_start + eval_size, len(dataset))
    if eval_end > eval_start:
        eval_ds = dataset.select(range(eval_start, eval_end))
    else:
        eval_ds = dataset.select(range(min(500, len(dataset))))

    return train_ds, eval_ds


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA SFT for chess on Qwen3-0.6B (GPU).")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=str, default="./outputs/qwen3-0p6-chess")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available.")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--hub-repo-id", type=str, default=None, help="Optional HF repo to push merged model.")
    parser.add_argument("--hub-token", type=str, default=None, help="HF token, otherwise uses cached login.")
    return parser


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_ds, eval_ds = load_splits(args.train_size, args.eval_size, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    eval_kwargs = {}
    if "evaluation_strategy" in TrainingArguments.__init__.__code__.co_varnames:
        eval_kwargs["evaluation_strategy"] = "steps" if args.eval_steps and args.eval_steps > 0 else "no"
        if eval_kwargs["evaluation_strategy"] == "steps":
            eval_kwargs["eval_steps"] = args.eval_steps

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        bf16=args.bf16,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        report_to="none",
        seed=args.seed,
        **eval_kwargs,
    )

    eval_enabled = eval_kwargs.get("evaluation_strategy") == "steps"

    trainer_kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds if eval_enabled else None,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        args=training_args,
    )

    sig = inspect.signature(SFTTrainer.__init__)
    accepted = set(sig.parameters.keys())

    # If dataset_text_field not accepted, fall back to formatting_func
    if "dataset_text_field" not in accepted and "formatting_func" in accepted:
        trainer_kwargs.pop("dataset_text_field", None)
        trainer_kwargs["formatting_func"] = lambda samples: samples["text"]

    filtered_kwargs = {k: v for k, v in trainer_kwargs.items() if k in accepted}

    trainer = SFTTrainer(**filtered_kwargs)

    trainer.train()

    lora_dir = os.path.join(args.output_dir, "lora")
    trainer.save_model(lora_dir)
    tokenizer.save_pretrained(lora_dir)

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    merged_model = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = merged_model.merge_and_unload()

    merged_dir = os.path.join(args.output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)

    if args.hub_repo_id:
        merged_model.push_to_hub(args.hub_repo_id, token=args.hub_token, safe_serialization=True)
        tokenizer.push_to_hub(args.hub_repo_id, token=args.hub_token)


if __name__ == "__main__":
    main()
