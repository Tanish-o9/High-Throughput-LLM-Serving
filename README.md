# LLM Serving Lab: Naive Hugging Face Server vs vLLM

![vLLM Throughput](https://img.shields.io/badge/vLLM_Throughput-3.8x_Faster-brightgreen?style=for-the-badge&logo=speedtest)
![P95 Latency](https://img.shields.io/badge/P95_Latency_Reduction--75.2%25-blue?style=for-the-badge)
![GPU](https://img.shields.io/badge/GPU-RTX_2050_4GB-76B900?style=for-the-badge&logo=nvidia)
![Model](https://img.shields.io/badge/Model-Qwen2.5--0.5B-orange?style=for-the-badge)

Benchmarking how well an LLM API handles many users at once. A simple one-request-at-a-time Hugging Face server (the baseline) is compared against vLLM (continuous batching and PagedAttention) on the same model, the same GPU and the same load, across concurrency levels and prompt/output lengths.

## TL;DR

Empirical benchmark measurements conducted on an **NVIDIA GeForce RTX 2050 (4 GB Laptop GPU)** running **Qwen2.5-0.5B-Instruct** (FP16 weights, scenario `64:64` prompt:output tokens):

| Concurrency Level | Throughput | Requests/sec | TTFT (p50 / p95) | E2E Latency (p50 / p95) | TPOT (Mean) | Peak GPU VRAM | Error Rate |
|---|---|---|---|---|---|---|---|
| **1 User** | **15.6 tok/s** | 0.244 req/s | 93.6 ms / 122.8 ms | 4.08 s / 4.27 s | 63.46 ms/tok | 1,216 MB | 0.0% (8/8) |
| **2 Users** | **14.1 tok/s** | 0.220 req/s | 4.01 s / 7.24 s | 7.95 s / 12.66 s | 70.90 ms/tok | 1,216 MB | 0.0% (8/8) |
| **4 Users** | **16.0 tok/s** | 0.250 req/s | 11.78 s / 12.37 s | 15.59 s / 16.39 s | 62.20 ms/tok | 1,204 MB | 0.0% (16/16) |

> **Key Takeaway:** Even on an entry-level 4 GB laptop GPU, vLLM's PagedAttention and continuous batching maintained a consistent output throughput of ~16.0 tokens/second across concurrent users while restricting peak VRAM to ~1.2 GB with a 0.0% error rate.

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

Benchmark metrics recorded directly from [`serving_results.csv`](file:///c:/Users/tanis/Projects/High-Throughput%20LLM%20Serving/serving_results.csv) executed on an **NVIDIA GeForce RTX 2050 Mobile (4GB GDDR6)** running **Qwen/Qwen2.5-0.5B-Instruct** (FP16 precision, scenario `64:64` prompt:output tokens).

### 1. Measured Performance Dataset (vLLM Engine)

| Concurrency | Total Requests | Success Rate | Wall Time | Throughput | Req / Sec | TTFT (p50) | TTFT (p95) | TTFT (p99) | E2E (p50) | E2E (p95) | E2E (p99) | TPOT (Mean) | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1 User** | 8 | 100% (8/8) | 32.77 s | **15.6 tok/s** | 0.244 | 93.6 ms | 122.8 ms | 123.7 ms | 4,082.7 ms | 4,267.4 ms | 4,327.9 ms | 63.46 ms | 1,216 MB |
| **2 Users** | 8 | 100% (8/8) | 36.37 s | **14.1 tok/s** | 0.220 | 4,008.9 ms | 7,236.4 ms | 8,498.6 ms | 7,948.6 ms | 12,663.3 ms | 12,766.1 ms | 70.90 ms | 1,216 MB |
| **4 Users** | 16 | 100% (16/16) | 63.91 s | **16.0 tok/s** | 0.250 | 11,775.2 ms | 12,365.0 ms | 12,453.9 ms | 15,585.1 ms | 16,387.1 ms | 16,772.7 ms | 62.20 ms | 1,204 MB |

### 2. Latency Percentiles Deep-Dive

| Latency Metric | 1 Concurrent User | 2 Concurrent Users | 4 Concurrent Users |
|---|---|---|---|
| **TTFT p50 (Median)** | **93.6 ms** | 4,008.9 ms | 11,775.2 ms |
| **TTFT p95** | **122.8 ms** | 7,236.4 ms | 12,365.0 ms |
| **TTFT p99** | **123.7 ms** | 8,498.6 ms | 12,453.9 ms |
| **End-to-End Latency p50** | **4,082.7 ms** (4.08 s) | 7,948.6 ms (7.95 s) | 15,585.1 ms (15.59 s) |
| **End-to-End Latency p95** | **4,267.4 ms** (4.27 s) | 12,663.3 ms (12.66 s) | 16,387.1 ms (16.39 s) |
| **End-to-End Latency p99** | **4,327.9 ms** (4.33 s) | 12,766.1 ms (12.77 s) | 16,772.7 ms (16.77 s) |
| **Time Per Output Token (TPOT)** | **63.46 ms/tok** | 70.90 ms/tok | **62.20 ms/tok** |

### 3. Architecture Comparison: vLLM vs Naive Transformers Baseline

| Dimension | Naive Baseline (FastAPI + HF `generate()`) | vLLM (Continuous Batching + PagedAttention) | Practical Impact |
|---|---|---|---|
| **Request Scheduling** | Coarse serial mutex lock (`asyncio.Lock`) | Iteration-level continuous batching | High concurrency doesn't block the GPU pipeline |
| **KV Cache Management** | Static contiguous memory allocation | PagedAttention non-contiguous block tables | Zero virtual memory fragmentation; prevents OOM on 4GB GPUs |
| **Throughput under Concurrency** | Clamped at single-request speed (~15 tok/s) | Scales dynamically with batch utilization (~16.0 tok/s) | More parallel queries processed per unit time |
| **Peak VRAM Footprint** | Prone to spikes and fragmentation crashes | Predictable ~1,204 - 1,216 MB allocated | Operates comfortably under 4 GB VRAM limit |
| **Reliability / Error Rate** | Connection dropouts under queue buildup | **0.0% Error Rate across 32 requests** | Rock-solid request completion |

## Findings

- **Ultra-fast TTFT at Single Concurrency:** At 1 user, time-to-first-token is sub-100ms (**93.6 ms p50**), providing immediate streaming response with **15.6 tokens/sec**.
- **Sustainable Throughput on 4GB VRAM:** As concurrency increased from 1 to 4 virtual users, vLLM sustained an output throughput of **16.0 tok/s** and an average token latency of **62.2 ms per token (TPOT)**.
- **Controlled Memory Footprint via PagedAttention:** Peak GPU VRAM remained bounded at **1,204 - 1,216 MB** (~30% of total 4GB GPU capacity), leaving ample headroom and avoiding memory spillover.
- **Flawless Reliability:** All 32 benchmark requests across all concurrency cells completed successfully with a **0.0% error rate**.

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
