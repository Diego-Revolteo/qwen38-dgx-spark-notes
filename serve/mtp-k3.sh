#!/usr/bin/env bash
# What we run in production on one DGX Spark (GB10): Qwen3.8-27B NVFP4, MTP k=3, thinking on.
# MODEL_DIR = local copy of Unsloth's NVFP4 checkpoint of Qwen3.8-27B (includes the MTP head).
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen3.8-27b-nvfp4}

docker run -d --name vllm-qwen38 --gpus all --ipc=host -p 8000:8000 \
  -v "$MODEL_DIR":/models/qwen3.8-27b-nvfp4:ro \
  -v "$HOME/.cache/vllm":/root/.cache/vllm \
  vllm/vllm-openai:v0.32.0 \
  --model /models/qwen3.8-27b-nvfp4 --served-model-name qwen3.8-27b \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 262144 --gpu-memory-utilization 0.60 --kv-cache-dtype fp8 \
  --max-num-seqs 2 --max-num-batched-tokens 8192 \
  --limit-mm-per-prompt '{"image":4,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --enable-chunked-prefill --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}'
