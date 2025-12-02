#!/usr/bin/env python3
"""
Compile the chess model for AWS Neuron using optimum-neuron.

Example:
    python3 compile_model.py \
        --model-path /home/ubuntu/environment/ml/qwen/merged_model \
        --output-path /home/ubuntu/environment/ml/qwen/compiled_model \
        --tensor-parallel-size 2 \
        --batch-size 4 \
        --sequence-length 2048 \
        --overwrite
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from optimum.neuron import NeuronModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a causal LM for AWS Neuron / vLLM."
    )
    parser.add_argument(
        "--model-path",
        default="/home/ubuntu/environment/ml/qwen/merged_model",
        help="Base HuggingFace checkpoint to compile.",
    )
    parser.add_argument(
        "--output-path",
        default="/home/ubuntu/environment/ml/qwen/compiled_model",
        help="Directory to write compiled artifacts (should match MODEL_PATH in vllm.sh).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel factor to compile for.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum batch size / concurrent sequences.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=2048,
        help="Maximum sequence length (tokens) to support.",
    )
    parser.add_argument(
        "--auto-cast-type",
        default="bf16",
        help="Computation dtype during compilation (bf16 recommended).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output directory before saving new artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if output_path.exists():
        if args.overwrite:
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )

    print(f"Compiling model from {model_path}")
    print(f"Saving compiled artifacts to {output_path}")
    print("This may take 10-30 minutes depending on model size.")

    model = NeuronModelForCausalLM.from_pretrained(
        str(model_path),
        export=True,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        auto_cast_type=args.auto_cast_type,
    )

    model.save_pretrained(str(output_path))
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "generation_config.json",
        "tokenizer.model",
    ]
    for filename in tokenizer_files:
        src = model_path / filename
        if src.exists():
            shutil.copy2(src, output_path / filename)

    print(f"✓ Model compiled and saved to {output_path}")


if __name__ == "__main__":
    main()
