#!/bin/bash
set -e

# Activate the Neuron NxD inference environment so python/vLLM are available
source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/vllm.log"

# 1) Point to the locally compiled Qwen3 checkpoint
MODEL_PATH="/home/ubuntu/environment/ml/qwen/compiled_model"

# 2) Make sure we use NxD inference as the Neuron backend in vLLM
export VLLM_NEURON_FRAMEWORK="neuronx-distributed-inference"

# (Optional) Explicitly list plugins, but leaving this unset is also fine since
# optimum_neuron is already being found & loaded.
# export VLLM_PLUGINS="optimum_neuron"

# (Optional) Where to cache compiled artifacts if/when vLLM/Optimum compiles anything new
# export NEURON_COMPILED_ARTIFACTS="/home/ubuntu/neuron-compiled-artifacts/chess-qwen"

echo "Logging to ${LOG_FILE}"
VLLM_RPC_TIMEOUT=100000 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --device neuron \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --dtype bfloat16 \
  --port 8080 \
  --task generate > "${LOG_FILE}" 2>&1 &

PID=$!
echo "vLLM server started with PID $PID"
echo "Server will be available at http://localhost:8080"
echo "To stop: kill $PID"
echo "Tail logs with: tail -f ${LOG_FILE}"
