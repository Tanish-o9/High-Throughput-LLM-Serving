"""
Tiny streaming demo client (works with baseline_server.py and vLLM).

    python demo_client.py "Explain what a GPU does in one paragraph."
    python demo_client.py --url http://localhost:8000 --max_tokens 100 "Write a haiku about rain"

Prints the answer as it streams, then the time to first token and tokens/s.
"""
import argparse
import json
import time

import httpx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max_tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    body = {"model": args.model, "prompt": args.prompt, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "stream": True,
            "stream_options": {"include_usage": True}}

    t0 = time.perf_counter()
    ttft, n_tokens = None, None
    with httpx.stream("POST", f"{args.url}/v1/completions", json=body, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            obj = json.loads(payload)
            for ch in obj.get("choices") or []:
                if ch.get("text"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    print(ch["text"], end="", flush=True)
            if obj.get("usage"):
                n_tokens = obj["usage"].get("completion_tokens")

    total = time.perf_counter() - t0
    print()
    if ttft is not None:
        msg = f"\nTTFT: {ttft * 1000:.0f} ms | total: {total:.2f} s"
        if n_tokens:
            msg += f" | {n_tokens} tokens | {n_tokens / total:.1f} tok/s"
        print(msg)


if __name__ == "__main__":
    main()
