"""
Baseline LLM server: plain Hugging Face generate() behind FastAPI.

This is the NAIVE baseline: one request is generated at a time (a lock serializes
the GPU work), no batching. Requests that arrive while the GPU is busy wait in line.
It exposes the same OpenAI-style API as vLLM (POST /v1/completions, with streaming),
so load_test.py can hit both servers with exactly the same requests.

Run:
    python baseline_server.py --model Qwen/Qwen2.5-0.5B-Instruct --port 8000

Quick check:
    curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d '{"prompt": "The capital of India is", "max_tokens": 16, "temperature": 0}'
"""
import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

STATE = {}                      # model, tokenizer, device, name
GEN_LOCK = None                 # asyncio.Lock created at startup: one generation at a time (the "naive" part)


# ------------------------------------------------------------------ model
def load_model(name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    STATE.update(model=model, tok=tok, device=device, name=name)

    # warm-up so the first real request does not pay CUDA start-up cost
    ids = tok("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=8, do_sample=False)
    print(f"Loaded {name} on {device} ({dtype})")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global GEN_LOCK
    GEN_LOCK = asyncio.Lock()
    load_model(os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    yield


app = FastAPI(title="Baseline HF server", lifespan=lifespan)


# ------------------------------------------------------------------ API schema
class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    stream: bool = False
    ignore_eos: bool = False          # keep generating to max_tokens (fixed output length)
    stream_options: Optional[dict] = None


def gen_kwargs(req: CompletionRequest):
    kw = dict(
        max_new_tokens=req.max_tokens,
        min_new_tokens=req.max_tokens if req.ignore_eos else 0,
        repetition_penalty=1.0,       # model's generation_config may set 1.05; keep it neutral
        pad_token_id=STATE["tok"].eos_token_id,
    )
    if req.temperature > 0:
        kw.update(do_sample=True, temperature=req.temperature)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    return kw


def encode(req: CompletionRequest):
    return STATE["tok"](req.prompt, return_tensors="pt").to(STATE["device"])


def usage_dict(prompt_tokens: int, completion_tokens: int):
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# ------------------------------------------------------------------ routes
@app.get("/health")
async def health():
    return {"status": "ok", "model": STATE.get("name")}


@app.get("/v1/models")
async def models():
    return {"data": [{"id": STATE.get("name"), "object": "model"}]}


def run_generate(req: CompletionRequest):
    inputs = encode(req)
    n_prompt = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = STATE["model"].generate(**inputs, **gen_kwargs(req))
    new_ids = out[0, n_prompt:]
    text = STATE["tok"].decode(new_ids, skip_special_tokens=True)
    return text, n_prompt, int(new_ids.shape[0])


async def stream_response(req: CompletionRequest, rid: str):
    async with GEN_LOCK:                                   # wait for our turn on the GPU
        inputs = encode(req)
        n_prompt = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(STATE["tok"], skip_prompt=True, skip_special_tokens=True)
        result = {"n": 0}

        def work():
            try:
                with torch.no_grad():
                    out = STATE["model"].generate(**inputs, streamer=streamer, **gen_kwargs(req))
                result["n"] = int(out.shape[1] - n_prompt)
            except Exception as e:                         # never leave the stream hanging
                result["error"] = str(e)
                streamer.end()

        thread = threading.Thread(target=work, daemon=True)
        thread.start()

        it = iter(streamer)
        while True:
            piece = await asyncio.to_thread(next, it, None)
            if piece is None:
                break
            if piece:
                yield sse({"id": rid, "object": "text_completion", "model": STATE["name"],
                           "choices": [{"index": 0, "text": piece, "finish_reason": None}]})
        await asyncio.to_thread(thread.join)

    if "error" in result:
        yield sse({"error": result["error"]})
    if (req.stream_options or {}).get("include_usage"):
        yield sse({"id": rid, "object": "text_completion", "choices": [],
                   "usage": usage_dict(n_prompt, result["n"])})
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    rid = "cmpl-" + uuid.uuid4().hex[:12]
    if req.stream:
        return StreamingResponse(stream_response(req, rid), media_type="text/event-stream")
    async with GEN_LOCK:
        text, n_prompt, n_new = await asyncio.to_thread(run_generate, req)
    return JSONResponse({
        "id": rid, "object": "text_completion", "created": int(time.time()),
        "model": STATE["name"],
        "choices": [{"index": 0, "text": text, "finish_reason": "length"}],
        "usage": usage_dict(n_prompt, n_new),
    })


# ------------------------------------------------------------------ entry point
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    os.environ["MODEL_NAME"] = args.model
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
