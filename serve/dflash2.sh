#!/usr/bin/env bash
# The DFlash2 variant we tested (faster decode, weaker prefix-cache reuse in our runs).
# DRAFTER_DIR = local copy of incoai/Qwen3.8-27B-DFlash2.
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-nvfp4}
DRAFTER_DIR=${DRAFTER_DIR:-$HOME/models/qwen3.8-27b-dflash2}

docker run -d --name vllm-qwen38-dflash2 --gpus all --ipc=host -p 8000:8000 \
  -v "$MODEL_DIR":/models/qwen3.8-27b-nvfp4:ro \
  -v "$DRAFTER_DIR":/models/qwen3.8-27b-dflash2:ro \
  -v "$HOME/.cache/vllm":/root/.cache/vllm \
  vllm/vllm-openai:v0.32.0 \
  --model /models/qwen3.8-27b-nvfp4 --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 --gpu-memory-utilization 0.60 --kv-cache-dtype fp8 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-chunked-prefill --enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"/models/qwen3.8-27b-dflash2","num_speculative_tokens":5}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,6,8,12,16,24],"max_cudagraph_capture_size":24}'
