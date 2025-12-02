# vLLM Server for AWS Neuron/Trainium

This directory contains scripts for deploying a chess-playing LLM on AWS Neuron hardware (Trainium/Inferentia) using vLLM with OpenAI-compatible API.

## Overview

The vLLM deployment provides:
- **Local inference** on AWS Neuron accelerators (no external API costs)
- **OpenAI-compatible API** at `http://localhost:8000/v1`
- **Optimized for Qwen3** chess model with tensor parallelism

## Prerequisites

Follow the [instructions](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance) to launch a trn2.3xlarge instance.

### Software Requirements

```bash
# Install optimum-neuron with vLLM support
pip install optimum-neuron[vllm]==0.3.0
```

That's it! The optimum-neuron package handles all Neuron SDK dependencies and vLLM integration automatically.

## Quick Start

### Step 1: Launch the vLLM Server

```bash
# Edit paths in vllm.sh if needed
bash vllm.sh
```

**What this does:**
- Compiles the model on first run (if needed)
- Starts OpenAI-compatible API server on port 8000
- Runs in background (use `jobs` to check status)
- Logs PID for easy management

### Step 2: Test the Connection

```bash
# Test with curl
curl http://localhost:8000/v1/models

# Test with Python
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='dummy')
response = client.chat.completions.create(
    model='/home/ubuntu/chess-model-qwen',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print(response.choices[0].message.content)
"
```

### Step 3: Use with Chess Agent

```python
from agents import VLLMAgent
from env import ChessEnvironment, StockfishAgent

# Create vLLM agent (connects to localhost:8000)
vllm_agent = VLLMAgent()

# Test connection
if vllm_agent.test_connection():
    print("✓ Connected to vLLM server")

    # Play a game
    env = ChessEnvironment(vllm_agent, StockfishAgent(skill_level=5, depth=10))
    result = env.play_game(verbose=True)
    print(f"Result: {result['result']}")
else:
    print("✗ Failed to connect - is vLLM server running?")
```

Or run games directly:

```bash
python run_game.py --agent1 vllm --agent2 stockfish-skill1-depth2 --verbose
```

## Configuration

### Model Paths

Edit these variables in `vllm.sh`:

```bash
# Source model directory (HuggingFace format)
MODEL_PATH="/home/ubuntu/chess-model-qwen/"

# Compiled model artifacts directory (auto-generated on first run)
COMPILED_MODEL_PATH="/home/ubuntu/traced_model/sharded-Qwen3-chess-tp-4/"
```

### Tensor Parallelism

Configure in `vllm.sh`:

```bash
# Adjust based on available NeuronCores (trn2.3xlarge has 4 cores)
--tensor-parallel-size 4
```

### Environment Variables

Set these in your `.env` file:

```bash
# vLLM server configuration
VLLM_BASE_URL=http://localhost:8000/v1  # Server endpoint
VLLM_MODEL=Qwen3-chess                   # Model name
VLLM_TEMPERATURE=0.1                     # Generation temperature
VLLM_MAX_TOKENS=50                       # Max tokens per response
VLLM_TIMEOUT=30.0                        # Request timeout
```

## 🔧 Management

### Check Server Status

```bash
# Check if server is running
ps aux | grep vllm

# Check logs
tail -f vllm-server/log

# Test endpoint
curl http://localhost:8000/health
```

### Stop the Server

```bash
# Find the PID
ps aux | grep vllm

# Kill the process
kill <PID>

# Or use the PID from startup
# (printed as "vLLM server started with PID <PID>")
```

### Restart the Server

```bash
# Stop existing server
pkill -f "vllm.entrypoints.openai.api_server"

# Start new server
bash vllm.sh
```

## Integration with Chess Environment

The `VLLMAgent` class automatically connects to your local vLLM server:

```python
# Default configuration (localhost:8000)
agent = VLLMAgent()

# Custom configuration
agent = VLLMAgent(
    base_url="http://localhost:8000/v1",
    model="Qwen3-chess",
    temperature=0.1,
    max_tokens=50
)

# Use in games
from env import ChessEnvironment
env = ChessEnvironment(agent, opponent)
result = env.play_game(verbose=True)
```

## Additional Resources

- [Optimum Neuron Documentation](https://huggingface.co/docs/optimum-neuron/index)
- [vLLM Documentation](https://docs.vllm.ai/)
- [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

## Support

For issues:
1. Check logs: `tail -f vllm-server/log`
2. Verify Neuron status: `neuron-ls` and `neuron-top`
3. Test endpoint: `curl http://localhost:8000/v1/models`
4. Review AWS Neuron documentation for hardware-specific issues
