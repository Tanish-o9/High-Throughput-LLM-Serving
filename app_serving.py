"""
High-Throughput LLM Serving Lab: Ultimate Interactive Terminal & Benchmark Studio
================================================================================
A state-of-the-art developer platform for LLM serving:
  1. 💻 Cyber Terminal & AI Playground (Single & Multi-turn Chat, Live Streaming, Telemetry HUD)
  2. ⚡ Live Concurrency Race Simulator (Simulate simultaneous users side-by-side in real-time)
  3. 🚀 In-Dashboard Benchmark Engine (Run load tests with live log stream)
  4. 📊 Performance Visualizer & Head-to-Head Analytics (vLLM vs Naive Baseline)
  5. 🎛️ Server & GPU Hardware Manager (1-click Server Start/Stop, Live RTX 2050 VRAM/Temp HUD)
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime

import httpx
import numpy as np
import pandas as pd
import streamlit as st

RESULTS_FILE = "serving_results.csv"
SERVER_LOG = "server_baseline.log"
SERVER_PID_FILE = "server.pid"

# -----------------------------------------------------------------------------
# 1. Page Configuration & Ultra-Modern Cyber Glassmorphic Theme
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LLM Serving Lab | Neural Studio & Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    code, pre, .terminal-text {
        font-family: 'Fira Code', monospace !important;
    }
    
    /* Hardware HUD Cards */
    .hud-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }
    
    .hud-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        position: relative;
    }
    
    .hud-card .hud-label {
        font-size: 11px;
        font-weight: 700;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    
    .hud-card .hud-value {
        font-family: 'Fira Code', monospace;
        font-size: 22px;
        font-weight: 700;
        color: #f0f6fc;
        margin-top: 4px;
    }
    
    .hud-card .hud-sub {
        font-size: 11px;
        color: #58a6ff;
        margin-top: 2px;
    }
    
    /* Terminal Console */
    .cyber-terminal {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    .cyber-terminal-bar {
        background: #161b22;
        padding: 10px 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #30363d;
    }
    
    .terminal-dots {
        display: flex;
        gap: 8px;
    }
    
    .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    .dot-red { background: #ef4444; }
    .dot-yellow { background: #f59e0b; }
    .dot-green { background: #10b981; }
    
    .terminal-screen {
        padding: 18px;
        min-height: 220px;
        max-height: 480px;
        overflow-y: auto;
        font-family: 'Fira Code', monospace;
        font-size: 14px;
        line-height: 1.6;
        color: #58a6ff;
        white-space: pre-wrap;
    }
    
    .prompt-prefix {
        color: #3fb950;
        font-weight: 600;
    }
    
    /* Concurrency Tile */
    .concurrency-tile {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        font-family: 'Fira Code', monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Hardware & Process Management Utilities
# -----------------------------------------------------------------------------
def get_live_gpu_stats():
    """Fetch live GPU memory, utilization, temperature, and device name."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,name",
             "--format=csv,noheader,nounits"],
            text=True, timeout=2, stderr=subprocess.DEVNULL
        )
        parts = [x.strip() for x in out.strip().split(",")]
        if len(parts) >= 5:
            return {
                "used_mb": int(parts[0]),
                "total_mb": int(parts[1]),
                "util_pct": int(parts[2]),
                "temp_c": int(parts[3]),
                "name": parts[4],
                "ok": True
            }
    except Exception:
        pass
    return {"used_mb": 0, "total_mb": 0, "util_pct": 0, "temp_c": 0, "name": "CPU / No GPU detected", "ok": False}


def check_server_status(url: str):
    """Pings the target LLM server for health and model name."""
    clean_url = url.replace("localhost", "127.0.0.1")
    try:
        t0 = time.perf_counter()
        r = httpx.get(f"{clean_url}/health", timeout=2.0)
        latency = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            return True, r.json().get("model", "Default Model"), latency
    except Exception:
        pass
    return False, None, None


def get_server_pid():
    if os.path.exists(SERVER_PID_FILE):
        try:
            with open(SERVER_PID_FILE, "r") as f:
                pid = int(f.read().strip())
                # verify process is still alive
                if sys.platform == "win32":
                    import ctypes
                    kernel32 = ctypes.windll.kernel32
                    SYNCHRONIZE = 0x00100000
                    h = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
                    if h:
                        kernel32.CloseHandle(h)
                        return pid
                else:
                    os.kill(pid, 0)
                    return pid
        except Exception:
            pass
    return None


def launch_server_bg(port: int = 8000, model: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Launch baseline_server.py in a background process."""
    cmd = [sys.executable, "baseline_server.py", "--model", model, "--port", str(port)]
    log_file = open(SERVER_LOG, "a")
    kwargs = {}
    if sys.platform == "win32":
        # DETACHED_PROCESS (0x8) | CREATE_NEW_PROCESS_GROUP (0x200)
        kwargs["creationflags"] = 0x00000008 | 0x00000200
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, **kwargs)
    with open(SERVER_PID_FILE, "w") as f:
        f.write(str(proc.pid))
    time.sleep(2)
    return proc.pid


def stop_server_bg():
    pid = get_server_pid()
    if pid:
        try:
            if sys.platform == "win32":
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
        if os.path.exists(SERVER_PID_FILE):
            os.remove(SERVER_PID_FILE)


# -----------------------------------------------------------------------------
# 3. Sidebar: Server Control & Hardware HUD
# -----------------------------------------------------------------------------
st.sidebar.markdown("### ⚡ Server & Node Controls")
raw_url = st.sidebar.text_input("Serving Endpoint", value="http://127.0.0.1:8000")
server_url = raw_url.replace("localhost", "127.0.0.1")

is_online, current_model, ping_ms = check_server_status(server_url)

if is_online:
    st.sidebar.success(f"🟢 **ONLINE** ({ping_ms:.0f} ms latency)")
    st.sidebar.caption(f"**Loaded Model:** `{current_model}`")
else:
    st.sidebar.error("🔴 **OFFLINE** — Server not reached")

col_srv1, col_srv2 = st.sidebar.columns(2)
with col_srv1:
    if st.button("▶️ Start Server", use_container_width=True, disabled=is_online):
        with st.spinner("Initializing Hugging Face Baseline Server..."):
            launch_server_bg(8000)
            time.sleep(3)
            st.rerun()

with col_srv2:
    if st.button("⏹️ Stop Server", use_container_width=True, disabled=not is_online and get_server_pid() is None):
        stop_server_bg()
        st.info("Server process terminated.")
        time.sleep(1)
        st.rerun()

if os.path.exists(SERVER_LOG):
    with st.sidebar.expander("📄 View Server Logs"):
        with open(SERVER_LOG, "r") as f:
            lines = f.readlines()
            st.text("".join(lines[-25:]))

st.sidebar.divider()

# Live GPU HUD
gpu_stats = get_live_gpu_stats()
st.sidebar.markdown("### 🖥️ Hardware Telemetry")
if gpu_stats["ok"]:
    st.sidebar.markdown(f"**GPU:** `{gpu_stats['name']}`")
    vram_percent = int((gpu_stats["used_mb"] / gpu_stats["total_mb"]) * 100) if gpu_stats["total_mb"] else 0
    st.sidebar.progress(vram_percent / 100, text=f"VRAM: {gpu_stats['used_mb']} / {gpu_stats['total_mb']} MB ({vram_percent}%)")
    st.sidebar.caption(f"🌡️ Temperature: {gpu_stats['temp_c']}°C | ⚙️ GPU Utilization: {gpu_stats['util_pct']}%")
else:
    st.sidebar.info("NVIDIA GPU stats not available via nvidia-smi.")

# -----------------------------------------------------------------------------
# 4. Top Hardware HUD Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="hud-container">
    <div class="hud-card">
        <div class="hud-label">Serving Engine Status</div>
        <div class="hud-value" style="color: {'#10b981' if is_online else '#ef4444'};">{'ONLINE' if is_online else 'OFFLINE'}</div>
        <div class="hud-sub">{'Ping: ' + str(int(ping_ms)) + ' ms' if is_online else 'Port 8000 Closed'}</div>
    </div>
    <div class="hud-card">
        <div class="hud-label">Hardware Device</div>
        <div class="hud-value" style="font-size: 17px;">{gpu_stats['name'].replace('NVIDIA GeForce ', '')}</div>
        <div class="hud-sub">Driver: CUDA 12.4+</div>
    </div>
    <div class="hud-card">
        <div class="hud-label">VRAM Allocation</div>
        <div class="hud-value">{gpu_stats['used_mb']} <span style="font-size: 14px; color:#8b949e;">/ {gpu_stats['total_mb']} MB</span></div>
        <div class="hud-sub">{(gpu_stats['used_mb']/gpu_stats['total_mb']*100):.1f}% VRAM in use</div>
    </div>
    <div class="hud-card">
        <div class="hud-label">Active Model</div>
        <div class="hud-value" style="font-size: 16px;">{current_model.split('/')[-1] if current_model else 'None'}</div>
        <div class="hud-sub">FP16 Precision (CUDA)</div>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. Core Studio Tabs
# -----------------------------------------------------------------------------
tab_terminal, tab_concurrency, tab_benchmark, tab_analytics = st.tabs([
    "💻 Cyber Terminal & Playground",
    "⚡ Concurrency Race Arena",
    "🚀 In-Dashboard Load Tester",
    "📊 Deep Analytics & Graphs"
])


# =============================================================================
# TAB 1: CYBER TERMINAL & PLAYGROUND
# =============================================================================
with tab_terminal:
    col_input, col_config = st.columns([2.2, 1])

    with col_input:
        presets = {
            "🚀 Quick Matrix Multiplication": "Explain in one clear paragraph why GPUs are 100x faster than CPUs for matrix multiplication.",
            "🐍 Python LRU Cache (Code)": "Write a clean, optimized Python implementation of an LRU Cache with docstring and comments.",
            "🧠 Prefill vs Decode Reasoning": "What is the difference between LLM prefill phase and decode phase, and why is decode memory bandwidth bound?",
            "🎭 AI Humor": "Tell me a short, witty joke about a deep learning engineer trying to deploy a 70B model on a 4GB GPU.",
            "✍️ Custom Prompt": ""
        }
        selected_preset = st.selectbox("🎯 Quick Prompt Templates", list(presets.keys()))
        prompt_text = presets[selected_preset] if selected_preset != "✍️ Custom Prompt" else ""
        
        user_prompt = st.text_area("Interactive Input Terminal Prompt:", value=prompt_text, height=120, placeholder="Type your prompt here...")

    with col_config:
        st.markdown("**Generation Hyperparameters**")
        max_tokens = st.slider("Max Output Tokens", min_value=16, max_value=512, value=128, step=16)
        temperature = st.slider("Sampling Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        force_tokens = st.checkbox("Enforce Fixed Length (`ignore_eos`)", value=False, help="Forces model to generate exact max_tokens (useful for fair latency benchmarks).")

    col_run_btn, col_spacer = st.columns([1, 4])
    with col_run_btn:
        btn_stream = st.button("⚡ Dispatch Stream", type="primary", use_container_width=True)

    # Modern Cyber Terminal Output
    st.markdown("""
    <div class="cyber-terminal">
        <div class="cyber-terminal-bar">
            <div class="terminal-dots">
                <span class="dot dot-red"></span>
                <span class="dot dot-yellow"></span>
                <span class="dot dot-green"></span>
            </div>
            <div style="font-family:'Fira Code'; font-size:12px; color:#8b949e;">root@llm-serving-lab:~# POST /v1/completions (stream=True)</div>
            <div style="font-family:'Fira Code'; font-size:11px; color:#34d399;">● READY</div>
        </div>
    """, unsafe_allow_html=True)

    terminal_content = st.empty()
    terminal_content.markdown("""
        <div class="terminal-screen" style="color: #64748b;">
            <span class="prompt-prefix">user@neural-node:~$</span> Waiting for prompt execution...
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Real-time Telemetry Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    c_ttft = m1.empty()
    c_lat = m2.empty()
    c_tps = m3.empty()
    c_tok = m4.empty()

    c_ttft.markdown('<div class="hud-card"><div class="hud-label">Time to First Token (TTFT)</div><div class="hud-value">-- ms</div><div class="hud-sub">Prompt processing speed</div></div>', unsafe_allow_html=True)
    c_lat.markdown('<div class="hud-card"><div class="hud-label">End-to-End Latency</div><div class="hud-value">-- s</div><div class="hud-sub">Total elapsed response time</div></div>', unsafe_allow_html=True)
    c_tps.markdown('<div class="hud-card"><div class="hud-label">Decode Throughput</div><div class="hud-value">-- tok/s</div><div class="hud-sub">Token generation velocity</div></div>', unsafe_allow_html=True)
    c_tok.markdown('<div class="hud-card"><div class="hud-label">Tokens Emitted</div><div class="hud-value">--</div><div class="hud-sub">Generated output volume</div></div>', unsafe_allow_html=True)

    if btn_stream:
        if not user_prompt.strip():
            st.error("Please enter a prompt first.")
        elif not is_online:
            st.error("Server is OFFLINE. Click '▶️ Start Server' in the sidebar or run `python baseline_server.py` in your terminal.")
        else:
            t0 = time.perf_counter()
            ttft = None
            generated_text = ""
            tokens_count = 0

            req_body = {
                "model": current_model or "Qwen/Qwen2.5-0.5B-Instruct",
                "prompt": user_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "ignore_eos": force_tokens,
                "stream_options": {"include_usage": True}
            }

            try:
                with httpx.stream("POST", f"{server_url}/v1/completions", json=req_body, timeout=300) as r:
                    if r.status_code != 200:
                        st.error(f"HTTP Error {r.status_code}: {r.read().decode('utf-8')}")
                    else:
                        for line in r.iter_lines():
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break
                            obj = json.loads(data_str)
                            choices = obj.get("choices") or []
                            if choices and choices[0].get("text"):
                                token_chunk = choices[0]["text"]
                                if ttft is None:
                                    ttft = time.perf_counter() - t0
                                    c_ttft.markdown(
                                        f'<div class="hud-card"><div class="hud-label">Time to First Token (TTFT)</div><div class="hud-value" style="color:#10b981;">{ttft*1000:.0f} ms</div><div class="hud-sub">Initial latency captured!</div></div>',
                                        unsafe_allow_html=True
                                    )
                                generated_text += token_chunk
                                tokens_count += 1

                                # Live display update
                                safe_out = generated_text.replace("<", "&lt;").replace(">", "&gt;")
                                terminal_content.markdown(f"""
                                    <div class="terminal-screen">
                                        <span class="prompt-prefix">user@neural-node:~$</span> {user_prompt}
                                        <br><br>
                                        <span style="color:#f8fafc;">{safe_out}</span><span class="cursor-blink"></span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            if obj.get("usage"):
                                tokens_count = obj["usage"].get("completion_tokens", tokens_count)

                total_duration = time.perf_counter() - t0
                tok_per_sec = (tokens_count / total_duration) if total_duration > 0 and tokens_count > 0 else 0

                # Final update
                safe_out = generated_text.replace("<", "&lt;").replace(">", "&gt;")
                terminal_content.markdown(f"""
                    <div class="terminal-screen">
                        <span class="prompt-prefix">user@neural-node:~$</span> {user_prompt}
                        <br><br>
                        <span style="color:#f8fafc;">{safe_out}</span>
                        <br><br>
                        <span style="color:#10b981; font-size:12px;">[COMPLETED in {total_duration:.2f}s | TTFT: {(ttft*1000) if ttft else 0:.0f}ms | Speed: {tok_per_sec:.1f} tok/s]</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c_ttft.markdown(f'<div class="hud-card"><div class="hud-label">Time to First Token (TTFT)</div><div class="hud-value">{(ttft*1000) if ttft else 0:.0f} ms</div><div class="hud-sub">Prompt processing speed</div></div>', unsafe_allow_html=True)
                c_lat.markdown(f'<div class="hud-card"><div class="hud-label">End-to-End Latency</div><div class="hud-value">{total_duration:.2f} s</div><div class="hud-sub">Completed successfully</div></div>', unsafe_allow_html=True)
                c_tps.markdown(f'<div class="hud-card"><div class="hud-label">Decode Throughput</div><div class="hud-value" style="color:#10b981;">{tok_per_sec:.1f} tok/s</div><div class="hud-sub">Average token speed</div></div>', unsafe_allow_html=True)
                c_tok.markdown(f'<div class="hud-card"><div class="hud-label">Tokens Emitted</div><div class="hud-value">{tokens_count}</div><div class="hud-sub">Exact tokens delivered</div></div>', unsafe_allow_html=True)

            except Exception as ex:
                st.error(f"Generation error: {ex}")


# =============================================================================
# TAB 2: CONCURRENCY RACE ARENA
# =============================================================================
with tab_concurrency:
    st.header("⚡ Live Concurrency Race Arena")
    st.caption("Send 2, 4, or 8 requests simultaneously to observe how the Naive Baseline server queues requests vs how continuous batching handles concurrent streams.")

    col_race_cfg, col_race_desc = st.columns([1, 2])
    with col_race_cfg:
        num_racers = st.selectbox("Simultaneous Virtual Users (Concurrency)", [2, 4], index=0)
        race_tokens = st.slider("Tokens per user", 32, 128, 64, step=16)
        btn_race = st.button("🏁 Start Concurrency Race", type="primary", use_container_width=True)

    with col_race_desc:
        st.info(
            "💡 **What to observe:**\n"
            "- In **Baseline (`baseline_server.py`)**, requests queue behind a lock (`asyncio.Lock`). User 2's TTFT will be high because it waits for User 1 to finish completely!\n"
            "- In **vLLM**, continuous batching interleaves both requests, so both start outputting tokens almost simultaneously!"
        )

    if btn_race:
        if not is_online:
            st.error("Server is offline. Start the server first.")
        else:
            race_cols = st.columns(num_racers)
            boxes = []
            metrics = []
            for idx in range(num_racers):
                with race_cols[idx]:
                    st.markdown(f"**Virtual User #{idx + 1}**")
                    b = st.empty()
                    b.markdown(f'<div class="concurrency-tile" style="color:#64748b;">Waiting in queue...</div>', unsafe_allow_html=True)
                    boxes.append(b)
                    m = st.empty()
                    metrics.append(m)

            async def racer_task(user_id, box, metric_box):
                t_start = time.perf_counter()
                my_ttft = None
                text_accum = ""
                tok_cnt = 0
                body = {
                    "model": current_model or "Qwen/Qwen2.5-0.5B-Instruct",
                    "prompt": f"Request {user_id}: Write a 3-sentence tip for programmers.",
                    "max_tokens": race_tokens,
                    "temperature": 0.0,
                    "stream": True,
                    "ignore_eos": True
                }
                async with httpx.AsyncClient(timeout=300) as client:
                    async with client.stream("POST", f"{server_url}/v1/completions", json=body) as resp:
                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            obj = json.loads(payload)
                            chs = obj.get("choices") or []
                            if chs and chs[0].get("text"):
                                piece = chs[0]["text"]
                                if my_ttft is None:
                                    my_ttft = time.perf_counter() - t_start
                                text_accum += piece
                                tok_cnt += 1
                                safe_t = text_accum[-150:].replace("<", "&lt;").replace(">", "&gt;")
                                box.markdown(f'<div class="concurrency-tile" style="color:#38bdf8;">{safe_t}▌</div>', unsafe_allow_html=True)
                
                t_total = time.perf_counter() - t_start
                safe_final = text_accum.replace("<", "&lt;").replace(">", "&gt;")
                box.markdown(f'<div class="concurrency-tile" style="color:#10b981;">{safe_final}</div>', unsafe_allow_html=True)
                metric_box.markdown(f"⏱️ **TTFT:** {(my_ttft*1000) if my_ttft else 0:.0f} ms | **E2E:** {t_total:.2f} s")

            async def run_race_all():
                tasks = [racer_task(i + 1, boxes[i], metrics[i]) for i in range(num_racers)]
                await asyncio.gather(*tasks)

            with st.spinner("Race in progress..."):
                asyncio.run(run_race_all())
            st.success("🏁 All concurrent requests finished!")


# =============================================================================
# TAB 3: IN-DASHBOARD LOAD TESTER
# =============================================================================
with tab_benchmark:
    st.header("🚀 High-Throughput Load Tester")
    st.caption("Benchmark your engine across multiple concurrency tiers and write certified metrics to `serving_results.csv`.")

    col_b1, col_b2, col_b3 = st.columns(3)
    with col_b1:
        bm_engine = st.selectbox("Engine Identifier", ["baseline", "vllm"])
        bm_concurrency = st.text_input("Concurrency List", value="1, 2, 4")
    with col_b2:
        bm_scenarios = st.selectbox("Benchmark Scenarios", ["64:64", "512:64", "64:128"])
        bm_reqs_per_user = st.slider("Requests per virtual user", 1, 6, 2)
    with col_b3:
        bm_warmup = st.number_input("Warmup rounds", 0, 5, 2)
        bm_model = st.text_input("Model Path / Tag", value=current_model or "Qwen/Qwen2.5-0.5B-Instruct")

    btn_start_benchmark = st.button("🔥 Execute Benchmark Suite", type="primary")

    if btn_start_benchmark:
        if not is_online:
            st.error("Cannot run benchmark while server is OFFLINE.")
        else:
            benchmark_cmd = [
                sys.executable, "load_test.py",
                "--engine", bm_engine,
                "--url", server_url,
                "--model", bm_model,
                "--concurrency", bm_concurrency.replace(" ", ""),
                "--scenarios", bm_scenarios,
                "--reqs-per-user", str(bm_reqs_per_user),
                "--warmup", str(bm_warmup),
                "--out", RESULTS_FILE
            ]
            st.info(f"Running command: `{' '.join(benchmark_cmd)}`")

            bm_log_area = st.empty()
            bm_log_area.markdown('<div class="cyber-terminal"><div class="terminal-screen">Initializing synthetic workload generator...</div></div>', unsafe_allow_html=True)

            try:
                proc = subprocess.Popen(
                    benchmark_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                accum_lines = []
                for line in proc.stdout:
                    accum_lines.append(line.strip())
                    view = "\n".join(accum_lines[-18:])
                    bm_log_area.markdown(f'<div class="cyber-terminal"><div class="terminal-screen">{view}</div></div>', unsafe_allow_html=True)

                proc.wait()
                if proc.returncode == 0:
                    st.success("✅ Benchmark complete! Check the 'Deep Analytics & Graphs' tab to view results.")
                else:
                    st.error(f"Process terminated with exit code {proc.returncode}")
            except Exception as ex:
                st.error(f"Execution failure: {ex}")


# =============================================================================
# TAB 4: DEEP ANALYTICS & HEAD-TO-HEAD COMPARISON
# =============================================================================
with tab_analytics:
    st.header("📊 Deep Analytics & Engine Comparisons")

    df_results = None
    if os.path.exists(RESULTS_FILE):
        try:
            df_results = pd.read_csv(RESULTS_FILE)
        except Exception:
            pass

    use_sample = False
    if df_results is None or df_results.empty:
        st.warning("No benchmark data in `serving_results.csv` yet.")
        use_sample = st.checkbox("🔮 Show Realistic Demonstration Benchmark Data", value=True)
        if use_sample:
            df_results = pd.DataFrame({
                "engine": ["baseline", "baseline", "baseline", "baseline", "vllm", "vllm", "vllm", "vllm"],
                "model": ["Qwen2.5-0.5B"] * 8,
                "gpu": [gpu_stats["name"]] * 8,
                "concurrency": [1, 2, 4, 8, 1, 2, 4, 8],
                "prompt_len": [64] * 8,
                "output_len": [64] * 8,
                "out_tok_per_s": [39.1, 40.2, 41.5, 41.8, 42.0, 81.4, 162.8, 310.5],
                "e2e_p95_ms": [1710, 3350, 6680, 13400, 1650, 1720, 1890, 2400],
                "ttft_p95_ms": [135, 1780, 5100, 11800, 125, 140, 180, 260],
                "tpot_mean_ms": [25.0, 25.1, 25.3, 25.4, 24.8, 25.0, 25.1, 25.2],
                "req_per_s": [0.61, 0.63, 0.65, 0.65, 0.65, 1.27, 2.54, 4.85],
                "error_rate_pct": [0.0] * 8,
                "gpu_mem_peak_mb": [1820, 1840, 1860, 1880, 3200, 3200, 3200, 3200]
            })

    if df_results is not None and not df_results.empty:
        df_results = df_results.drop_duplicates(
            subset=["engine", "model", "concurrency", "prompt_len", "output_len"], keep="last"
        )

        df_results["scenario"] = df_results["prompt_len"].astype(str) + " prompt : " + df_results["output_len"].astype(str) + " output"
        available_scenarios = sorted(df_results["scenario"].unique())
        selected_scen = st.selectbox("Scenario Filter", available_scenarios)
        filtered_df = df_results[df_results["scenario"] == selected_scen].sort_values("concurrency")
        engines_present = sorted(filtered_df["engine"].unique())

        # Head to Head Comparison Card
        if {"baseline", "vllm"}.issubset(engines_present):
            b_df = filtered_df[filtered_df["engine"] == "baseline"].set_index("concurrency")
            v_df = filtered_df[filtered_df["engine"] == "vllm"].set_index("concurrency")
            common_conc = sorted(set(b_df.index) & set(v_df.index))
            if common_conc:
                st.subheader("⚡ The Verdict: vLLM vs Naive Baseline")
                top_users = common_conc[-1]
                b_top_tps = b_df.loc[top_users, "out_tok_per_s"]
                v_top_tps = v_df.loc[top_users, "out_tok_per_s"]
                multiplier = (v_top_tps / b_top_tps) if b_top_tps > 0 else 0

                b_top_e2e = b_df.loc[top_users, "e2e_p95_ms"]
                v_top_e2e = v_df.loc[top_users, "e2e_p95_ms"]
                lat_reduc = ((1 - v_top_e2e / b_top_e2e) * 100) if b_top_e2e > 0 else 0

                v1, v2, v3 = st.columns(3)
                v1.metric(f"🚀 Throughput Gain ({top_users} users)", f"{multiplier:.1f}x Faster", delta=f"{v_top_tps - b_top_tps:.1f} tok/s")
                v2.metric(f"⏱️ P95 Latency Reduction ({top_users} users)", f"{lat_reduc:.1f}% Lower", delta=f"-{(b_top_e2e - v_top_e2e)/1000:.2f}s")
                v3.metric("Evaluated Concurrency Scale", f"{common_conc[0]} → {top_users} users")

        # Visual Analytics Charts
        st.subheader("📈 Performance Trajectories Across Concurrency")
        
        def render_chart(metric_col, title, unit=""):
            st.markdown(f"**{title}**")
            pvt = filtered_df.pivot_table(index="concurrency", columns="engine", values=metric_col)
            st.line_chart(pvt)

        gc1, gc2 = st.columns(2)
        with gc1:
            render_chart("out_tok_per_s", "⚡ Total Output Throughput (Tokens/sec) — Higher is better")
        with gc2:
            render_chart("e2e_p95_ms", "⏱️ End-to-End Latency p95 (ms) — Lower is better")

        gc3, gc4 = st.columns(2)
        with gc3:
            render_chart("ttft_p95_ms", "🎯 Time to First Token (TTFT) p95 (ms) — Lower is better")
        with gc4:
            render_chart("req_per_s", "📦 Completed Requests / Second — Higher is better")

        with st.expander("📑 Raw Benchmark Data Table"):
            st.dataframe(filtered_df, use_container_width=True)
            st.download_button(
                "📥 Export CSV Results",
                filtered_df.to_csv(index=False),
                "serving_benchmark_results.csv",
                "text/csv"
            )
