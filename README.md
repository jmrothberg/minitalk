# MiniTalk

**Terminal chatbot for [NVIDIA Jetson Orin Nano Super](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) (8 GB unified memory).**

MiniTalk is built for boards where CPU, GPU, and the desktop all share one small RAM pool. It runs from a **plain terminal**, memory-maps NF4 weights from disk, uses the **Orin CUDA GPU** when the driver can JIT the NF4 kernel, and falls back to ARM NEON + OpenMP.

> Default model folder layout: put Hugging Face–style dumps under `~/Transformers/` (or set `MINITALK_TRANSFORMERS`). At startup MiniTalk lists them and you pick by number.

## Why a terminal on Orin Nano Super?

| Constraint | How MiniTalk handles it |
|------------|-------------------------|
| 8 GB shared RAM | NF4 mmap, chunked `lm_head`, memory-clamped KV cache |
| Desktop apps eat RAM | Prefer a TTY / single terminal; close browsers if needed |
| Long model windows (128k / 256k) | Read `max_position_embeddings` from `config.json`; allocate only an 8GB-safe working KV |
| Growing chats | Auto-compact past turns to ~**10%** then **5%** of the working window |

A full 128k–256k KV cache does **not** fit in 8 GB. MiniTalk keeps the model’s declared window for RoPE/limits, then clamps the **working** context so the machine stays alive.

## Start in a terminal

```bash
cd /path/to/Minitalk
python3 minitalk.py
```

You will see something like:

```
MiniTalk — Jetson Orin Nano Super (8 GB)
Models in /home/jrothberg/Transformers
------------------------------------------------------------
  [1] MiniCPM5-2B                  ctx  131072  4.7GB  (needs-NF4-export)
  [2] MiniCPM5-2B-NF4              ctx  131072  2.0GB  (NF4, default)
------------------------------------------------------------
Model [2]:
```

Type the number (or press Enter for the default NF4 model). Wait until the UI says the model is mapped.

### Useful launches

```bash
# Skip the picker
python3 minitalk.py --model MiniCPM5-2B-NF4

# Absolute path
python3 minitalk.py --model /home/jrothberg/Transformers/MiniCPM5-2B-NF4

# One-shot reply (no TUI)
python3 minitalk.py --model MiniCPM5-2B-NF4 --once "Hello" --max-new 64

# Force CPU (skip GPU)
python3 minitalk.py --cpu

# Kernel self-test (CPU + CUDA)
python3 minitalk.py --self-test
```

### Flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--transformers DIR` | `$MINITALK_TRANSFORMERS` or `~/Transformers` | Model zoo to list |
| `--model NAME\|PATH` | picker | Skip interactive list |
| `--pick` | off | Force the number picker |
| `--max-seq N` | auto from `config.json`, clamped for 8GB | Working KV length |
| `--max-new N` | `128` | Max new tokens |
| `--cpu` | off | Disable Orin GPU path |
| `--self-test` | | Check NF4 kernels |

Local Python deps live in `.pydeps/` (tokenizers, textual). `minitalk.py` adds that path automatically. Native kernels (`.so`) are built with `gcc` on first run.

## Chat commands

| Command | Action |
|---------|--------|
| `/new` | New chat (drops turns **and** compacted facts) |
| `/clear` | Clear transcript/turns; **keep** facts |
| `/compact` | Compact now |
| `/facts` | Show major facts |
| `/temp` `/topp` `/max` `/repeat` | Sampling |
| `/think` | Toggle think block |
| `/sys TEXT` | System prompt |
| `/help` `/quit` | Help / exit |
| `Esc` | Stop generation |

## Compaction (5% / 10%)

Inspired by long-horizon compaction work (e.g. relevance-filtered history at ~**5%** matching full-history quality — [arXiv:2610.14952](https://arxiv.org/abs/2610.14952); early trigger + keep recent turns — Bud ContextCompact / CompactionBench):

1. Trigger near **70%** of the **working** KV window (leave room for the next turn).
2. Keep the newest turns verbatim.
3. Fold older turns into a short **Major facts** list in the system prompt.
4. Size that past block to **≤10%** of the working window; if still tight, shrink to **≤5%**.

`/new` clears facts. `/facts` inspects them.

## Context window from model JSON

MiniTalk reads, in order:

`max_position_embeddings` → `max_sequence_length` → `n_positions` → `seq_length` → `model_max_length`

If none are present, it assumes **256k** (`262144`). The **working KV** is then clamped from free RAM so the Orin does not OOM.

## GPU (Orin CUDA)

On startup MiniTalk tries:

1. **CUDA driver API** + `nf4_gemv.ptx` (`sm_87`) via `libcuda.so` — no full CUDA toolkit required to *run*.
2. If that fails → **CPU NEON + OpenMP** (`nf4_gemv.c`).

The HUD shows `cuda` or `cpu`. First GPU use uploads each NF4 matrix once and reuses it (unified memory).

```bash
python3 minitalk.py --self-test
# backend: cuda  Orin GPU ready (CUDA driver / sm_87)
```

## Project structure

```
Minitalk/
├── README.md          # this file
├── minitalk.py        # model picker, runtime, Textual UI, compaction
├── nf4_gemv.c         # ARM NEON + OpenMP NF4 / f16 matvecs
├── nf4_cuda.c         # CUDA driver-API host (loads PTX, caches weights)
├── nf4_gemv.ptx       # Orin sm_87 NF4 GEMV kernel
├── .gitignore
└── .pydeps/           # local pip packages (not committed)
```

| Piece | Role |
|-------|------|
| Model picker | Lists `Transformers/*` with `config.json`; select by number |
| `Kernel` | Prefers Orin GPU; CPU fallback |
| `WeightStore` / `NF4Matrix` | mmap NF4 shards; scales cached in RAM |
| `MiniCPM` | Llama-style decode with RoPE + KV |
| `ChatEngine` | `/new`, 5%/10% compaction, sampling |
| `MiniTalkApp` | Terminal UI |

### Models (not in this repo)

Place checkpoints under the transformers directory, for example:

```
Transformers/
├── MiniCPM5-2B/         # full bf16 — too heavy for 8GB as-is
└── MiniCPM5-2B-NF4/     # bitsandbytes NF4 — default for Orin
```

Only **NF4** dumps (bitsandbytes `quant_state`) are loadable today.

## License / credit

Local inference helper for Jetson. Model weights remain under their original licenses (e.g. MiniCPM / OpenBMB).
