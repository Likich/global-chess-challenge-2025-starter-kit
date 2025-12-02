#!/usr/bin/env python3
"""
Start vLLM server using Python API with pre-compiled Neuron model.
This bypasses the CLI task inference bug.
"""
import os
import uvicorn
from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.entrypoints.openai.api_server import build_app
from vllm.usage.usage_lib import UsageContext

# Model configuration
MODEL_PATH = "/home/ubuntu/ChessLM_Qwen3_Trainium"

# Engine arguments
engine_args = AsyncEngineArgs(
    model=MODEL_PATH,
    device="neuron",
    tensor_parallel_size=2,
    max_model_len=2048,
    max_num_seqs=1,
    dtype="bfloat16",
    trust_remote_code=False,
    # Override neuron config to specify task
    override_neuron_config={"task": "text-generation"},
)

if __name__ == "__main__":
    print(f"Starting vLLM server with model: {MODEL_PATH}")
    print(f"Engine args: {engine_args}")

    # Create async engine
    engine = AsyncLLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.OPENAI_API_SERVER
    )

    # Build FastAPI app
    app = build_app(engine)

    # Start server
    print("Server starting on http://localhost:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
