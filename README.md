# Scale Your LLM: Hands-On Experiments

## Prerequisites

| Need | Why |
| --- | --- |
| Python 3.10+ and `git` | download + convert scripts |
| Build tools (`build-essential`, `cmake`) | build llama.cpp from source |
| ~5 GB free disk | model weights (FP16 + GGUF variants) |
| Docker + Docker Compose | LABs 04–07 |
| (Optional) NVIDIA GPU + drivers | EXP 6 GPU offload, LAB 04 vLLM |

EXP 0–5 run fine on **CPU only**. A GPU is needed for EXP 6 and LAB 04.

## Initial Setup

### Git Installation

```bash
sudo apt update
sudo apt install -y git
```
Check (prints a version number):

```bash
git --version
```

### Docker Installation

```bash
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```
Check (prints a version number — log out and back in to apply the group change):

```bash
docker --version
```

### Clone the repository
```bash
git clone https://github.com/euler1729/scale-your-llm.git 
```

### Work from the repo root

```bash
cd scale-your-llm/
```
### Create a Python virtual environment

```bash
sudo apt install python3.12-venv
```
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -U pip
```

---

# Part A — GGUF Deep-Dive (Qwen3.5 0.8B)

## EXP 0 — Setup

**Goal:** create a clean Python env and build the llama.cpp toolchain.

```bash
# 1. Python deps for downloading + converting
pip install -U huggingface_hub "numpy<2" sentencepiece safetensors torch

# 2. Clone and build llama.cpp (CPU build)
git clone https://github.com/ggml-org/llama.cpp
pip install -r llama.cpp/requirements.txt
sudo apt install cmake

cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release

# Build with LIMITED parallelism. Plain `-j` (no number) spawns one job per CPU
# core, and each job uses ~1–2 GB RAM — that can exhaust memory and freeze the
# machine. Cap it: use 1 jobs (safe), or `nproc/1`. Lower it further if it lags.
cmake --build llama.cpp/build -j 1

# 4. Verify the build
./llama.cpp/build/bin/llama-cli --version
```

**Check:** the version command prints a build number. Binaries now live in
`llama.cpp/build/bin/` (`llama-cli`, `llama-quantize`, `llama-perplexity`, …).

---

## EXP 1 — Download the model

**Goal:** pull the FP16 HuggingFace weights locally.

```bash
huggingface-cli download Qwen/Qwen3.5-0.8B-Base \
  --local-dir models/qwen

ls -lh models/qwen
```

**Check:** you see `model.safetensors` (~1.6 GB), `config.json`, and the
tokenizer files. This is the full-precision model — large and framework-bound.

---

## EXP 2 — Convert HF → GGUF (FP16)

**Goal:** repackage the model into a single portable GGUF file.

```bash
python llama.cpp/convert_hf_to_gguf.py models/qwen \
  --outfile models/qwen-f16.gguf \
  --outtype f16

ls -lh models/qwen-f16.gguf
```

**Check:** one self-contained `qwen-f16.gguf` (~1.6 GB). Weights +
tokenizer + metadata are now in **one file** — no Python/framework needed to load it.

---

## EXP 3 — Quantize → Q4_K_M

**Goal:** shrink the model ~4× with k-quant mixed precision.

```bash
./llama.cpp/build/bin/llama-quantize \
  models/qwen-f16.gguf \
  models/qwen-Q4_K_M.gguf \
  Q4_K_M

ls -lh models/*.gguf
```

**Check:** size drops from ~1.6 GB → **~0.5 GB**.

**Reading the quant name** `Q4_K_M`:
- `Q4` → 4-bit weights
- `K`  → k-quant (block-wise, smarter bit allocation)
- `M`  → medium mix (some tensors kept at higher precision)

`Q4_K_M` is the **safe default**: best size/quality trade-off for most models.

---

## EXP 4 — Inference

**Goal:** run the quantized model and read the speed.

```bash
./llama.cpp/build/bin/llama-cli \
  -m models/qwen-Q4_K_M.gguf \
  -p "Explain what GGUF is in one sentence." \
  -n 128 -no-cnv
```

**Check:** the model generates text, then prints a timing footer. Note the
**eval tokens/sec** — that's your CPU throughput baseline for EXP 6.

---

## EXP 5 — Perplexity (quality measurement)

**Goal:** quantify the quality cost of quantization.

```bash
# Get a small evaluation set (WikiText-2 raw test split)
curl -L -o wikitext-2-raw-v1.zip \
  https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip
unzip -o wikitext-2-raw-v1.zip

# Perplexity of the quantized model
./llama.cpp/build/bin/llama-perplexity \
  -m models/qwen-Q4_K_M.gguf \
  -f wikitext-2-raw/wiki.test.raw

# (Optional) Compare against the FP16 baseline
./llama.cpp/build/bin/llama-perplexity \
  -m models/qwen-f16.gguf \
  -f wikitext-2-raw/wiki.test.raw
```

**Check:** the final `PPL` for Q4_K_M is only marginally higher than FP16
(typically **<2%**) — a 4× smaller model for a tiny quality cost.

---

## EXP 6 (BONUS) — GPU Offloading

**Goal:** move layers onto the GPU and compare throughput. *Requires NVIDIA GPU.*

```bash
# Rebuild llama.cpp with CUDA support (limited parallelism — see EXP 0 note)
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build llama.cpp/build -j 2

# Offload ALL layers to GPU (-ngl 99)
./llama.cpp/build/bin/llama-cli \
  -m models/qwen-Q4_K_M.gguf \
  -p "Explain what GGUF is in one sentence." \
  -n 128 -no-cnv -ngl 99

# Limited VRAM? Offload only some layers (CPU+GPU split)
./llama.cpp/build/bin/llama-cli \
  -m models/qwen-Q4_K_M.gguf \
  -p "Explain what GGUF is in one sentence." \
  -n 128 -no-cnv -ngl 16
```

**Check:** compare eval tokens/sec to EXP 4. Full offload (`-ngl 99`) should be
dramatically faster than CPU. `-ngl N` controls how many layers go to the GPU.

---

# Part B — Scalable Deployment

## LAB 04 — GPU vLLM Container

**Goal:** serve Qwen3.5 through vLLM (PagedAttention + continuous batching) over
an OpenAI-compatible API. *Requires NVIDIA GPU + Container Toolkit.*

```bash
# Verify GPU passthrough into Docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

# Launch the vLLM server
docker run --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-0.8B-Base

# In another terminal — test the endpoint
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3.5-0.8B-Base","prompt":"Hello, my name is","max_tokens":32}'
```

**Check:** the server logs show the model loaded and the curl returns a JSON
completion.

---

## LAB 05 — Stress Test & 429s

**Goal:** push concurrent load and watch batching + backpressure.

Save as `stress.py`:

```python
import asyncio, time, httpx

URL = "http://localhost:8000/v1/completions"
PAYLOAD = {
    "model": "Qwen/Qwen3.5-0.8B-Base",
    "prompt": "Write one sentence about GGUF.",
    "max_tokens": 64,
}
N = 100  # concurrent requests

async def one(client, i):
    r = await client.post(URL, json=PAYLOAD, timeout=60)
    return r.status_code

async def main():
    t0 = time.time()
    async with httpx.AsyncClient() as c:
        codes = await asyncio.gather(*(one(c, i) for i in range(N)))
    print(f"{N} reqs in {time.time()-t0:.1f}s | "
          f"200s={codes.count(200)} 429s={codes.count(429)}")

asyncio.run(main())
```

```bash
pip install httpx
python stress.py
```

**Check:** requests complete quickly thanks to **continuous batching**. Push
`N` higher (e.g. 500) to start seeing `429` throttling — that's backpressure.

---

## LAB 06 — Async Client (decouple inference)

**Goal:** show that an async client overlaps requests instead of blocking.

Save as `async_client.py`:

```python
import asyncio, time, httpx

URL = "http://localhost:8000/v1/completions"
PROMPTS = [f"Give fact #{i} about LLM quantization." for i in range(20)]

def payload(p):
    return {"model": "Qwen/Qwen3.5-0.8B-Base",
            "prompt": p, "max_tokens": 48}

async def main():
    t0 = time.time()
    async with httpx.AsyncClient(timeout=60) as c:
        tasks = [c.post(URL, json=payload(p)) for p in PROMPTS]
        results = await asyncio.gather(*tasks)
    print(f"{len(results)} concurrent requests in {time.time()-t0:.1f}s")

asyncio.run(main())
```

```bash
python async_client.py
```

**Check:** 20 prompts finish in roughly the time of the slowest single request,
not the sum — client and server pipeline them concurrently.

---

## LAB 07 — Celery + Redis Worker Scaling

**Goal:** decouple request intake from inference with a queue, then scale workers
horizontally under load.

`tasks.py`:

```python
import os, httpx
from celery import Celery

app = Celery("infer", broker=os.environ["REDIS_URL"], backend=os.environ["REDIS_URL"])
VLLM = os.environ.get("VLLM_URL", "http://host.docker.internal:8000/v1/completions")

@app.task
def generate(prompt):
    r = httpx.post(VLLM, timeout=120, json={
        "model": "Qwen/Qwen3.5-0.8B-Base",
        "prompt": prompt, "max_tokens": 64,
    })
    return r.json()["choices"][0]["text"]
```

`docker-compose.yml`:

```yaml
services:
  redis:
    image: redis:7
    ports: ["6379:6379"]

  worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=1
    environment:
      REDIS_URL: redis://redis:6379/0
      VLLM_URL: http://host.docker.internal:8000/v1/completions
    depends_on: [redis]
```

`Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN pip install celery redis httpx
COPY tasks.py .
```

```bash
# Start the queue with one worker
docker compose up --build -d

# Enqueue jobs
docker compose exec worker python -c \
  "from tasks import generate; [generate.delay(f'Fact {i} about GGUF') for i in range(50)]"

# Scale workers horizontally and watch jobs redistribute
docker compose up -d --scale worker=4
docker compose logs -f worker
```

**Check:** with 1 worker the 50 jobs drain slowly; after `--scale worker=4`
they finish ~4× faster as the queue spreads across workers.

---

## Recap & Resources

**What you did**
- Quantized a real model: FP16 → GGUF → Q4_K_M (~1.6 GB → ~0.5 GB).
- Ran CPU inference and measured quality with perplexity (<2% loss).
- Offloaded layers to GPU and served at scale with vLLM, async clients, and Celery.

**Key takeaways**
- Quantization shrinks models 4–8× with <2% quality loss.
- GGUF = portable, CPU-friendly, **single-file** format.
- `Q4_K_M` is the safe default. GPTQ/AWQ shine on GPU; bitsandbytes is easiest in HF.

**Next steps**
- One-command GGUF:

```bash
ollama run tinyllama
```
- Homework: quantize a 3B model (Qwen2.5-3B or Phi-3).
- Explore **imatrix** quantization for better low-bit quality.
- Docs: https://github.com/ggml-org/llama.cpp
