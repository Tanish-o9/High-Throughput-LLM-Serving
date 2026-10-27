#!/usr/bin/env bash
# LLM Serving Lab: run the whole benchmark, one engine at a time.
#
#   bash run_benchmark.sh baseline     # naive HF server
#   bash run_benchmark.sh vllm         # vLLM
#   bash run_benchmark.sh all          # both, one after the other (same machine, same load)
#
# Only ONE server runs at a time, because a 4 GB GPU cannot hold two engines.
# Use two Python environments (vLLM pins its own torch version):
#   PY       = python of the env that has transformers + httpx (baseline server and load test)
#   VLLM_BIN = the `vllm` executable of the vLLM env
#
# Example (WSL2 / Linux):
#   PY=~/venv-base/bin/python VLLM_BIN=~/venv-vllm/bin/vllm bash run_benchmark.sh all
#
# Useful overrides (environment variables):
#   MODEL, PORT, CONC, SCENARIOS, REQS_PER_USER, MAX_LEN, GPU_UTIL, VLLM_EXTRA
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
PORT=${PORT:-8000}
CONC=${CONC:-1,4,16,32}
SCENARIOS=${SCENARIOS:-64:64,512:64,64:128}     # prompt_tokens:output_tokens
REQS_PER_USER=${REQS_PER_USER:-4}
PY=${PY:-python}
VLLM_BIN=${VLLM_BIN:-vllm}
MAX_LEN=${MAX_LEN:-2048}
GPU_UTIL=${GPU_UTIL:-0.80}                      # lower this (0.7) if vLLM says not enough free memory
VLLM_EXTRA=${VLLM_EXTRA:---enforce-eager}       # eager mode saves memory on a small GPU
OUT=${OUT:-serving_results.csv}

SERVER_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  sleep 5                                       # let the GPU memory be released
}
trap stop_server EXIT

wait_for_server() {
  echo "Waiting for server on port $PORT ..."
  for _ in $(seq 1 150); do                     # up to ~5 minutes (first run downloads the model)
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "Server is up."
      return 0
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Server process died. Check the log: server_$1.log"
      return 1
    fi
    sleep 2
  done
  echo "Server did not become ready in time. Check the log: server_$1.log"
  return 1
}

run_load() {
  "$PY" load_test.py --engine "$1" --url "http://localhost:$PORT" --model "$MODEL" \
    --concurrency "$CONC" --scenarios "$SCENARIOS" --reqs-per-user "$REQS_PER_USER" --out "$OUT"
}

bench_baseline() {
  echo "=== Baseline: HF generate() + FastAPI ==="
  "$PY" baseline_server.py --model "$MODEL" --port "$PORT" > server_baseline.log 2>&1 &
  SERVER_PID=$!
  wait_for_server baseline
  run_load baseline
  stop_server
}

bench_vllm() {
  echo "=== vLLM ==="
  # shellcheck disable=SC2086
  "$VLLM_BIN" serve "$MODEL" --port "$PORT" --dtype float16 \
    --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" $VLLM_EXTRA \
    > server_vllm.log 2>&1 &
  SERVER_PID=$!
  wait_for_server vllm
  run_load vllm
  stop_server
}

case "${1:-}" in
  baseline) bench_baseline ;;
  vllm)     bench_vllm ;;
  all)      bench_baseline; bench_vllm ;;
  *) echo "Usage: bash run_benchmark.sh [baseline|vllm|all]"; exit 1 ;;
esac

echo "Done. Results in $OUT. Dashboard: streamlit run app_serving.py"
