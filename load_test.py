"""
Load tester for any OpenAI-style /v1/completions server (baseline_server.py or vLLM).

For every (scenario, concurrency) cell it sends a FIXED load and records:
  TTFT (time to first token), TPOT (time per output token), end-to-end latency
  (p50 / p95 / p99), output tokens/s, requests/s, error rate, peak GPU memory.

Load model: "closed loop". N virtual users each send a request, wait for the full
streamed answer, then send the next one. N = concurrency.

Fairness choices:
  - Same prompts (same seed) for every engine.
  - Every prompt starts with a unique id, so vLLM prefix caching cannot reuse
    work between requests (it would make vLLM look faster for the wrong reason).
  - Fixed output length: ignore_eos=True and temperature=0.
  - A few warm-up requests are sent first and not recorded.

Examples:
  python load_test.py --engine baseline --concurrency 1,4,16,32
  python load_test.py --engine vllm --scenarios 64:64,512:64 --reqs-per-user 2
"""
import argparse
import asyncio
import json
import os
import subprocess
import threading
import time
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

BASE_TEXT = (
    "Large language models generate text one token at a time. Serving them efficiently "
    "means keeping the GPU busy, sharing memory between requests and returning the first "
    "token quickly. Engineers measure latency, throughput and cost under realistic load, "
    "and compare the results against a simple baseline before claiming any improvement. "
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, help="label for this run, e.g. baseline or vllm")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--tokenizer", default=None, help="defaults to --model")
    p.add_argument("--concurrency", default="1,4,16,32", help="comma separated user counts")
    p.add_argument("--scenarios", default="64:64,512:64,64:128",
                   help="comma separated prompt_tokens:output_tokens")
    p.add_argument("--reqs-per-user", type=int, default=4)
    p.add_argument("--min-requests", type=int, default=8)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timeout", type=float, default=900.0, help="per-request timeout (s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="serving_results.csv")
    return p.parse_args()


# ------------------------------------------------------------------ prompts
def make_prompt(tok, prompt_len: int, uid: int) -> str:
    """Unique prompt of ~prompt_len tokens (starts with a unique id)."""
    head = f"[request {uid:08d}] "
    text = head
    while len(tok(text, add_special_tokens=False)["input_ids"]) < prompt_len + 8:
        text += BASE_TEXT
    ids = tok(text, add_special_tokens=False)["input_ids"][:prompt_len]
    return tok.decode(ids)


# ------------------------------------------------------------------ GPU info
def nvidia_smi(query: str):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True, timeout=5, stderr=subprocess.DEVNULL)
        return [x.strip() for x in out.strip().splitlines()]
    except Exception:
        return None


def gpu_name() -> str:
    v = nvidia_smi("name")
    return v[0] if v else "unknown"


class GpuMemSampler:
    """Polls nvidia-smi in a background thread and keeps the peak used memory (MB).
    Note: this is whole-GPU memory (includes the desktop and other apps), and vLLM
    pre-allocates memory by design, so compare it with care."""

    def __init__(self):
        self.peak = float("nan")
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            v = nvidia_smi("memory.used")
            if v:
                cur = max(float(x) for x in v)
                self.peak = cur if np.isnan(self.peak) else max(self.peak, cur)
            self._stop.wait(1.0)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join(timeout=3)


# ------------------------------------------------------------------ one request
async def one_request(client, args, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": args.model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0.0, "stream": True, "ignore_eos": True,
        "repetition_penalty": 1.0,
        "stream_options": {"include_usage": True},
    }
    t0 = time.perf_counter()
    ttft, chunks, ntok, nprompt = None, 0, None, None
    try:
        async with client.stream("POST", f"{args.url}/v1/completions", json=body) as r:
            if r.status_code != 200:
                msg = (await r.aread()).decode(errors="ignore")[:200]
                return {"ok": False, "error": f"HTTP {r.status_code}: {msg}"}
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if "error" in obj:
                    return {"ok": False, "error": str(obj["error"])[:200]}
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    chunks += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if obj.get("usage"):
                    ntok = obj["usage"].get("completion_tokens")
                    nprompt = obj["usage"].get("prompt_tokens")
        e2e = time.perf_counter() - t0
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}

    if ttft is None:
        return {"ok": False, "error": "no tokens received"}
    n = ntok if ntok else chunks
    tpot = (e2e - ttft) / (n - 1) if n and n > 1 else float("nan")
    return {"ok": True, "ttft": ttft, "e2e": e2e, "tpot": tpot,
            "n_tokens": n, "n_prompt": nprompt if nprompt else float("nan")}


# ------------------------------------------------------------------ one benchmark cell
async def run_cell(args, tok, prompt_len, output_len, conc, uid_start):
    n_total = max(args.min_requests, args.reqs_per_user * conc)
    queue = asyncio.Queue()
    for i in range(n_total):
        queue.put_nowait(make_prompt(tok, prompt_len, uid_start + i))

    results = []

    async def user(client):
        while True:
            try:
                prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await one_request(client, args, prompt, output_len))

    limits = httpx.Limits(max_connections=conc + 4, max_keepalive_connections=conc + 4)
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout), limits=limits) as client:
        with GpuMemSampler() as gpu:
            t0 = time.perf_counter()
            await asyncio.gather(*[user(client) for _ in range(conc)])
            wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    errors = [r["error"] for r in results if not r["ok"]]
    row = {
        "engine": args.engine, "model": args.model, "gpu": gpu_name(),
        "concurrency": conc, "prompt_len": prompt_len, "output_len": output_len,
        "n_requests": n_total, "n_ok": len(ok),
        "error_rate_pct": round(100 * (n_total - len(ok)) / n_total, 1),
        "wall_s": round(wall, 2),
        "gpu_mem_peak_mb": gpu.peak,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    if ok:
        ttft = np.array([r["ttft"] for r in ok]) * 1000
        e2e = np.array([r["e2e"] for r in ok]) * 1000
        tpot = np.array([r["tpot"] for r in ok]) * 1000
        total_tokens = sum(r["n_tokens"] for r in ok)
        row.update({
            "req_per_s": round(len(ok) / wall, 3),
            "out_tok_per_s": round(total_tokens / wall, 1),
            "ttft_p50_ms": round(np.percentile(ttft, 50), 1),
            "ttft_p95_ms": round(np.percentile(ttft, 95), 1),
            "ttft_p99_ms": round(np.percentile(ttft, 99), 1),
            "e2e_p50_ms": round(np.percentile(e2e, 50), 1),
            "e2e_p95_ms": round(np.percentile(e2e, 95), 1),
            "e2e_p99_ms": round(np.percentile(e2e, 99), 1),
            "tpot_mean_ms": round(float(np.nanmean(tpot)), 2),
            "prompt_tokens_mean": round(float(np.nanmean([r["n_prompt"] for r in ok])), 1),
        })
    return row, errors


def save_rows(rows, path):
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = ["engine", "model", "concurrency", "prompt_len", "output_len"]
        merged = old.merge(new[key].drop_duplicates(), on=key, how="left", indicator=True)
        old = old[merged["_merge"].values == "left_only"]        # drop rows we re-ran
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


async def main_async(args):
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    levels = [int(x) for x in args.concurrency.split(",")]
    scenarios = [tuple(int(v) for v in s.split(":")) for s in args.scenarios.split(",")]

    print(f"Engine: {args.engine} | GPU: {gpu_name()} | url: {args.url}")
    print(f"Warm-up: {args.warmup} request(s) ...")
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout)) as client:
        for i in range(args.warmup):
            prompt = make_prompt(tok, scenarios[0][0], 900_000_000 + i)
            r = await one_request(client, args, prompt, min(32, scenarios[0][1]))
            if not r["ok"]:
                raise SystemExit(f"Warm-up failed, is the server running? {r['error']}")

    rows, uid = [], args.seed * 1_000_000
    for (p_len, o_len) in scenarios:
        for conc in levels:
            row, errors = await run_cell(args, tok, p_len, o_len, conc, uid)
            uid += 10_000
            rows.append(row)
            save_rows([row], args.out)                           # save after every cell
            if row["n_ok"]:
                print(f"[{args.engine}] prompt={p_len:<4} out={o_len:<4} users={conc:<3} | "
                      f"{row['out_tok_per_s']:>7.1f} tok/s | {row['req_per_s']:>5.2f} req/s | "
                      f"TTFT p95 {row['ttft_p95_ms']:>8.0f} ms | "
                      f"E2E p95 {row['e2e_p95_ms']:>8.0f} ms | "
                      f"errors {row['n_requests'] - row['n_ok']}/{row['n_requests']}")
            else:
                print(f"[{args.engine}] prompt={p_len} out={o_len} users={conc} | ALL FAILED")
            if errors:
                print(f"    first error: {errors[0]}")
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
