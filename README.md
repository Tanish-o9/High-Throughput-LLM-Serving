# LLM Serving Lab: Naive Hugging Face Server vs vLLM

Benchmarking how well an LLM API handles many users at once. A simple one-request-at-a-time Hugging Face server (the baseline) is compared against vLLM (continuous batching and PagedAttention) on the same model, the same GPU and the same load, across concurrency levels and prompt/output lengths.

## TL;DR

| At N concurrent users | Baseline (HF + FastAPI) | vLLM |
|---|---|---|
| Throughput (output tokens/s) | X | X (X times) |
| End-to-end latency p95 | X ms | X ms |
| Time to first token p95 | X ms | X ms |
| Error rate | X% | X% |

> Replace X with your measured values (see [Results](#results)). GPU: X, model: Qwen2.5-0.5B-Instruct.

## Why this project

Training a model happens once. Serving it to many users happens every day, and the GPU is the expensive part. A naive server leaves the GPU mostly idle between tokens and makes users queue. This project measures how much a serving engine improves:

1. Throughput: how many tokens per second the system produces in total.
2. Latency: how long users wait (time to first token and total time, including p95 and p99).
3. Behavior under load: what happens at 1, 4, 16 and 32 simultaneous users.

## How it works

```
Load tester (load_test.py)
   N virtual users, fixed prompts, streaming
              |
              v   same OpenAI-style API: POST /v1/completions
   +---------------------------+---------------------------+
   |  [A] Baseline             |  [B] vLLM                 |
   |  baseline_server.py       |  vllm serve ...           |
   |  HF generate(), a lock    |  continuous batching,     |
   |  runs ONE request at a    |  PagedAttention           |
   |  time                     |                           |
   +---------------------------+---------------------------+
              |
              v
          one GPU
              |
              v
   serving_results.csv  ->  Streamlit dashboard  ->  this report
```

Both servers speak the same API, so the load tester sends byte-identical requests to each.

### Key ideas

- **Prefill vs decode:** the prompt is processed at once (compute-bound), then tokens are generated one by one (memory-bandwidth-bound). During decode the GPU is underused, so serving many requests together is nearly free.
- **Continuous batching:** after every token the scheduler can swap a finished request out and a new one in, so the GPU is never waiting for the slowest request.
- **PagedAttention:** the KV cache is stored in small pages instead of one big block per request, which wastes far less memory and fits more simultaneous requests.
- **Streaming:** tokens are sent as soon as they are produced, so the user sees the first words quickly (low TTFT).

## Metrics

| Metric | Meaning |
|---|---|
| TTFT | Time to first token: when the user first sees output |
| TPOT | Average time per output token after the first |
| E2E latency | Total time for the full answer (p50, p95, p99) |
| Throughput | Output tokens per second across all users |
| Requests/s | Completed requests per second |
| Error rate | Failed or timed-out requests |
| GPU memory | Peak whole-GPU memory (nvidia-smi), see caveat below |

p95 and p99 are reported because averages hide the slow requests.

## Fair benchmark rules used here

- Same model, same dtype (FP16), same GPU, same prompts (seeded) for both engines.
- Every prompt starts with a unique id, so vLLM prefix caching cannot reuse work across requests.
- Fixed output length (`ignore_eos`, temperature 0), so every request generates the same number of tokens.
- Warm-up requests are sent first and not counted.
- Only one server runs at a time. Same machine for both runs.
- Results are saved after every cell, and re-running a cell replaces the old row.

## Experiments

| Scenario (prompt:output tokens) | Why |
|---|---|
| 64:64 | Short chat-style request |
| 512:64 | Long prompt, short answer (prefill-heavy) |
| 64:128 | Short prompt, longer answer (decode-heavy) |

Each scenario is run at 1, 4, 16 and 32 concurrent users. Edit `CONC` and `SCENARIOS` in `run_benchmark.sh` to change them.

## Project structure

```
llm-serving-lab/
├── baseline_server.py        # naive HF generate() server (FastAPI, streaming)
├── load_test.py              # load tester: TTFT, TPOT, p50/p95/p99, throughput, errors
├── run_benchmark.sh          # starts each server, runs the load test, stops it
├── demo_client.py            # streaming demo client
├── app_serving.py            # Streamlit dashboard
├── requirements_serving.txt  # baseline server, load test, dashboard
├── requirements_vllm.txt     # vLLM (separate environment)
├── serving_results.csv       # created by load_test.py
└── server_*.log              # server logs
```

## Setup

vLLM needs an NVIDIA GPU and Linux, and Python 3.10 or newer. On Windows use WSL2 (Ubuntu) with an up-to-date NVIDIA driver on Windows. Use two virtual environments, because vLLM pins its own torch version.

```bash
# Environment 1: baseline server, load test, dashboard
python -m venv ~/venv-base && source ~/venv-base/bin/activate
pip install -r requirements_serving.txt
deactivate

# Environment 2: vLLM
python -m venv ~/venv-vllm && source ~/venv-vllm/bin/activate
pip install -r requirements_vllm.txt
deactivate

nvidia-smi        # check that the GPU is visible
```

## Usage

### 1. Try the baseline server and the demo

```bash
source ~/venv-base/bin/activate
python baseline_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000
# in another terminal:
python demo_client.py "Explain what a GPU does in one paragraph."
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of India is", "max_tokens": 16, "temperature": 0}'
```

### 2. Try vLLM

```bash
~/venv-vllm/bin/vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000 --dtype float16 \
  --max-model-len 2048 --gpu-memory-utilization 0.80 --enforce-eager
python demo_client.py "Explain what a GPU does in one paragraph."
```

### 3. Run the full benchmark

```bash
PY=~/venv-base/bin/python VLLM_BIN=~/venv-vllm/bin/vllm bash run_benchmark.sh all
# or one engine at a time:
PY=~/venv-base/bin/python bash run_benchmark.sh baseline
VLLM_BIN=~/venv-vllm/bin/vllm PY=~/venv-base/bin/python bash run_benchmark.sh vllm
```

Quick smoke test before the real run:

```bash
CONC=1,4 SCENARIOS=64:32 REQS_PER_USER=2 PY=~/venv-base/bin/python bash run_benchmark.sh baseline
rm serving_results.csv
```

### 4. Dashboard

```bash
streamlit run app_serving.py
```

### Settings for a small GPU (4 GB)

| Setting | Value | Reason |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct | About 1 GB in FP16 |
| `--max-model-len` | 2048 | Smaller KV cache |
| `--gpu-memory-utilization` | 0.80 (lower to 0.70 if vLLM reports not enough free memory) | The desktop also uses some VRAM |
| `--enforce-eager` | on | Saves the memory used by CUDA graphs |
| Concurrency | up to 32 | Keeps runs short |

### Troubleshooting

| Problem | Fix |
|---|---|
| vLLM: not enough free GPU memory | Lower `GPU_UTIL` (for example `GPU_UTIL=0.70`), close other GPU apps, or reduce `MAX_LEN` |
| Out of memory in the baseline | Use a smaller model or lower `max_tokens` |
| Next engine fails right after the previous one | The old server may still hold GPU memory. Run `nvidia-smi` and kill leftover python or vllm processes |
| Server never becomes ready | Read `server_baseline.log` or `server_vllm.log` (the first run also downloads the model) |
| Very slow baseline at 32 users | Expected, requests wait in line. Use `REQS_PER_USER=2` to shorten the run |

## Results

> Fill these in after running. Replace X with your measured values. GPU: X, driver: X, vLLM version: X, torch version: X.

### Throughput (output tokens/s), scenario 64:64

| Users | Baseline | vLLM | Speedup |
|---|---|---|---|
| 1 | X | X | Xx |
| 4 | X | X | Xx |
| 16 | X | X | Xx |
| 32 | X | X | Xx |

### Latency at 32 users, scenario 64:64

| | Baseline | vLLM |
|---|---|---|
| TTFT p95 | X ms | X ms |
| E2E p95 | X ms | X ms |
| E2E p99 | X ms | X ms |
| TPOT | X ms | X ms |

### Effect of prompt and output length (32 users)

| Scenario | Baseline tok/s | vLLM tok/s | Speedup |
|---|---|---|---|
| 64:64 | X | X | Xx |
| 512:64 | X | X | Xx |
| 64:128 | X | X | Xx |

## Findings

> Replace with your own observations. Report small or negative differences honestly.

- At 1 user the two engines are close (X vs X tok/s), because there is nothing to batch.
- As concurrency grows, the baseline throughput stays flat (it serves one request at a time) while vLLM keeps scaling, reaching X times higher throughput at 32 users.
- p95 end-to-end latency at 32 users dropped by X% because requests no longer wait in line.
- Long prompts (512 tokens) changed the picture by X, since prefill is compute-heavy.

## Limitations

- Small model (0.5B) on a small GPU. Larger models and GPUs show different, usually bigger, gaps.
- The baseline is intentionally naive (one request at a time). A baseline with static or dynamic batching would be a stronger comparison.
- Closed-loop load (users wait for an answer before sending the next). Real traffic often arrives at random times (open-loop).
- Synthetic prompts, not real user traffic.
- The load generator runs on the same machine as the server and competes for CPU.
- Laptop GPUs throttle with heat. Results can drift between runs, so repeat and report the spread.
- `--enforce-eager` lowers vLLM speed a little in exchange for memory.
- Token timing in the baseline comes from a text streamer, which can group a few tokens into one chunk. TTFT is still measured on the first text received.
- GPU memory numbers cover the whole GPU, and vLLM reserves memory up front.

## Future work

- Add a batched Hugging Face baseline.
- Serve a quantized model (AWQ or GPTQ) in vLLM and measure the extra gain.
- Open-loop (Poisson arrival) load test.
- Package both servers with Docker and run the benchmark in CI.
- Add prefix caching and speculative decoding experiments.
- Try several GPUs or larger models.

## Tech stack

Python, PyTorch, Hugging Face Transformers, FastAPI, vLLM, httpx and asyncio, pandas, Streamlit.

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention* (2023)
- Yu et al., *Orca: A Distributed Serving System for Transformer-Based Generative Models* (2022), the origin of continuous batching
- vLLM documentation: https://docs.vllm.ai

## License

MIT
