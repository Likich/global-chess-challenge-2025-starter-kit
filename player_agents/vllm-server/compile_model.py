#!/usr/bin/env python3
"""
Compile the chess model for AWS Neuron using optimum-neuron.
"""
import os
from optimum.neuron import NeuronModelForCausalLM

MODEL_PATH = "/home/ubuntu/chess-model-qwen/"
COMPILED_MODEL_PATH = "/home/ubuntu/traced_model/sharded-Qwen3-chess-tp-2/"

print(f"Compiling model from {MODEL_PATH}")
print(f"Output will be saved to {COMPILED_MODEL_PATH}")
print("This will take 10-30 minutes...")

# Compile the model for Neuron
# This exports and compiles the model for vLLM inference
model = NeuronModelForCausalLM.from_pretrained(
    MODEL_PATH,
    export=True,
    tensor_parallel_size=2,
    batch_size=1,
    sequence_length=4096,
    auto_cast_type="bf16",
)

# Save the compiled model
model.save_pretrained(COMPILED_MODEL_PATH)

print(f"✓ Model compiled and saved to {COMPILED_MODEL_PATH}")
