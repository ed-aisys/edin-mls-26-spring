# PyLet Example — Two LLMs Debating (SLURM Edition)

Deploy two `Qwen/Qwen3.5-2B` instances on a SLURM GPU cluster with PyLet. One plays a **Python fan**, the other plays a **Rust evangelist**. They debate back and forth automatically.

## Architecture

```
Login Node (gala2)            Compute Node (saxa, 2× GPU)
┌──────────────┐              ┌──────────────────────────┐
│  pylet head  │◄────────────►│     pylet worker         │
│  (port 8000) │              │  GPU 0 → SGLang python-fan│
│              │              │  GPU 1 → SGLang rust-fan  │
│  debate.py   │              └──────────────────────────┘
└──────────────┘
```

- **Head** runs on the login node (no GPU needed, just the scheduler).
- **Worker** runs on a SLURM-allocated compute node with 2 GPUs.
- Each SGLang instance gets 1 GPU via PyLet's automatic allocation.

## Prerequisites

```bash
pip install uv
uv pip install pylet
uv pip install sglang==0.5.6
```

If you do not have cuda toolkit installed (i.e., no CUDA or NVCC found), please do:
```bash
conda install conda-forge::cuda-toolkit==12.8.1
echo 'export LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH' >> ~/.bashrc
```

## Step-by-step

### 1. Start the head node (Login Node — Terminal 1)

```bash
pylet start
```

This starts the PyLet scheduler on `hastings:8000`.

### 2. Submit the SLURM worker (Login Node — Terminal 2)

```bash
cd edin-mls-26-spring/pylet_example
sbatch start_worker.sh
```

This allocates 2 GPUs on `saxa` and starts a PyLet worker that registers with the head.

Check that the worker registered:

```bash
pylet list-workers
```

You should see 1 worker with 2 GPU units available.

### 3. Deploy two SGLang instances (Login Node — Terminal 2)

```bash
# Python fan
pylet submit \
  'python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --reasoning-parser qwen3 --host 0.0.0.0 --port $PORT --cuda-graph-max-bs 1' \
  --name python-fan --gpu-units 1

# Rust evangelist
pylet submit \
  'python -m sglang.launch_server --model-path Qwen/Qwen3-1.7B --reasoning-parser qwen3 --host 0.0.0.0 --port $PORT --cuda-graph-max-bs 1' \
  --name rust-fan --gpu-units 1
```

Wait for both to be ready:

```bash
pylet get-endpoint --name python-fan   # → e.g. saxa:15600
pylet get-endpoint --name rust-fan     # → e.g. saxa:15601
```

### 4. Run the debate (Login Node — Terminal 2)

```bash
python debate.py <python-fan-endpoint> <rust-fan-endpoint>

# Example:
python debate.py saxa:15600 saxa:15601
```

### Example output

```
🎤 Topic: Which language is better for building distributed systems?

🐍 Python Fan: Python's async/await plus libraries like FastAPI and Ray make
distributed systems a breeze. Why fight the borrow checker when you could ship
your prototype before lunch?

🦀 Rust Fan: Because that "prototype" will segfault in production at 3am.
Rust's ownership model catches concurrency bugs at compile time. Sleep well,
Pythonista.

🐍 Python Fan: Bold words from someone who spent 3 hours satisfying the
borrow checker for a simple HTTP handler. Meanwhile, my 10-line FastAPI
server is already serving traffic.

🦀 Rust Fan: And when that traffic hits 100k req/s, your GIL will be crying.
Rust + tokio handles that with zero-cost abstractions and zero garbage
collection pauses.

...
```

### 5. Clean up

```bash
pylet delete --name python-fan -y
pylet delete --name rust-fan -y
scancel --name pylet-worker   # release the SLURM GPUs
```

---

## Files

| File | Purpose |
|---|---|
| `debate.py` | Python vs Rust debate script (uses OpenAI client) |
| `start_worker.sh` | SLURM batch script — launches pylet worker with 2 GPUs |
| `README.md` | This file |

## What this shows

| PyLet Feature | How it's used |
|---|---|
| **Head on login node** | Scheduler runs without GPU, accessible from anywhere |
| **Worker via SLURM** | `sbatch` allocates GPUs; pylet worker manages them |
| **Multi-instance** | Two SGLang servers running simultaneously, 1 GPU each |
| **Auto port** | `$PORT` → each server gets a unique port |
| **Service discovery** | `get-endpoint` finds both by name |
| **Standard API** | Both expose OpenAI-compatible endpoints |

Same pattern scales to any multi-model SLURM setup: RAG pipelines, model A/B testing, ensemble inference.
