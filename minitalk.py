#!/usr/bin/env python3
"""MiniTalk: terminal chat for Jetson Orin Nano Super (8 GB unified memory).

Designed for NVIDIA Jetson Orin Nano Super / Orin Nano with tight RAM:
NF4 weights stay memory-mapped, the lm_head is read in chunks, and the Orin
GPU is used via the CUDA driver API when available (CPU NEON fallback).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import mmap
import os
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, ".pydeps"))

import numpy as np

DEFAULT_TRANSFORMERS = os.path.expanduser(
    os.environ.get("MINITALK_TRANSFORMERS", "/home/jrothberg/Transformers")
)
DEFAULT_MODEL_NAME = "MiniCPM5-2B-NF4"
# Logical default when a model config has no max_position_embeddings.
DEFAULT_MODEL_WINDOW = 262144  # 256k
SYSTEM_PROMPT = (
    "You are MiniTalk, a helpful assistant running locally on a Jetson Orin Nano Super. "
    "Answer clearly and keep replies concise unless the user asks for detail."
)
# Keep this many recent messages (user+assistant) after compaction.
COMPACT_KEEP_RECENT = 4
# Compact early enough to leave room for the next turn (Bud ContextCompact / CompactionBench).
COMPACT_TRIGGER_RATIO = 0.70
# Retain compacted past history at ~10% of the working window; fall back to 5%
# (relevance-filtered ~5% matched full-history quality in arXiv:2610.14952).
COMPACT_BUDGET_RATIOS = (0.10, 0.05)


def meminfo():
    """Return (available_mb, total_mb) from /proc/meminfo."""
    avail = total = 0
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) // 1024
            elif line.startswith("MemTotal:"):
                total = int(line.split()[1]) // 1024
    return avail, total


def rss_mb():
    with open("/proc/self/status", "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return 0


class Kernel:
    """NF4 matvec: prefer Orin GPU (CUDA driver PTX), else ARM NEON + OpenMP."""

    def __init__(self, root, prefer_gpu=True):
        self.backend = "cpu"
        self.gpu_note = ""
        self.cuda = None
        cores = os.cpu_count() or 4
        os.environ.setdefault("OMP_NUM_THREADS", str(cores))
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

        cpu_src = os.path.join(root, "nf4_gemv.c")
        cpu_so = os.path.join(root, "nf4_gemv.so")
        if not os.path.exists(cpu_so) or os.path.getmtime(cpu_so) < os.path.getmtime(cpu_src):
            subprocess.check_call(
                ["gcc", "-O3", "-mcpu=native", "-fopenmp", "-fPIC", "-shared", "-o", cpu_so, cpu_src, "-lm"]
            )
        self.lib = ctypes.CDLL(cpu_so)
        self.lib.nf4_linear.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.nf4_linear.restype = ctypes.c_int
        self.lib.f16_linear.argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.f16_linear.restype = ctypes.c_int

        if prefer_gpu:
            cuda_src = os.path.join(root, "nf4_cuda.c")
            cuda_so = os.path.join(root, "nf4_cuda.so")
            ptx = os.path.join(root, "nf4_gemv.ptx")
            try:
                if (
                    not os.path.exists(cuda_so)
                    or os.path.getmtime(cuda_so) < os.path.getmtime(cuda_src)
                    or (os.path.exists(ptx) and os.path.getmtime(cuda_so) < os.path.getmtime(ptx))
                ):
                    subprocess.check_call(
                        ["gcc", "-O2", "-fPIC", "-shared", "-o", cuda_so, cuda_src, "-ldl"]
                    )
                cuda = ctypes.CDLL(cuda_so)
                cuda.cuda_nf4_init.restype = ctypes.c_int
                cuda.cuda_nf4_available.restype = ctypes.c_int
                cuda.cuda_nf4_last_error.restype = ctypes.c_char_p
                cuda.cuda_nf4_linear.argtypes = self.lib.nf4_linear.argtypes
                cuda.cuda_nf4_linear.restype = ctypes.c_int
                if cuda.cuda_nf4_init() == 0 and cuda.cuda_nf4_available():
                    self.cuda = cuda
                    self.backend = "cuda"
                    self.gpu_note = (cuda.cuda_nf4_last_error() or b"").decode()
                else:
                    self.gpu_note = (cuda.cuda_nf4_last_error() or b"cuda init failed").decode()
            except Exception as exc:
                self.gpu_note = f"cuda unavailable: {exc}"

    def nf4(self, packed, scales, x, y, codebook):
        batch, cols = x.shape
        out_rows = y.shape[1] if y.ndim == 2 else y.shape[0]
        y_ptr = y
        use_cuda = self.cuda is not None and batch == 1 and y.ndim == 2
        if use_cuda:
            # GPU path expects a flat [rows] output for batch 1.
            flat = np.ascontiguousarray(y_ptr.reshape(-1))
            rc = self.cuda.cuda_nf4_linear(
                packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(out_rows),
                int(cols),
                1,
                codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            if rc == 0:
                y_ptr.reshape(-1)[:] = flat
                return
            # One failed launch → stay on CPU for the rest of the session.
            self.gpu_note = (self.cuda.cuda_nf4_last_error() or b"cuda launch failed").decode()
            self.cuda = None
            self.backend = "cpu"
        rc = self.lib.nf4_linear(
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            scales.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(out_rows),
            int(cols),
            int(batch if y.ndim == 2 else 1),
            codebook.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if rc != 0:
            raise RuntimeError(f"nf4_linear failed ({rc})")

    def f16(self, weights, x, y):
        rows = y.shape[-1] if y.ndim == 2 else y.shape[0]
        cols = x.shape[-1]
        batch = x.shape[0]
        rc = self.lib.f16_linear(
            weights.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(rows),
            int(cols),
            int(batch),
        )
        if rc != 0:
            raise RuntimeError(f"f16_linear failed ({rc})")


KERNEL = None


def kernel(prefer_gpu=True):
    global KERNEL
    if KERNEL is None:
        KERNEL = Kernel(ROOT, prefer_gpu=prefer_gpu)
    return KERNEL


class WeightStore:
    """Read safetensors shards. Small quant state is copied; packed weights stay mapped."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "model.safetensors.index.json"), "r", encoding="utf-8") as handle:
            self.weight_map = json.load(handle)["weight_map"]
        self.headers = {}
        self._fd = {}
        self._mm = {}

    def _header(self, filename):
        cached = self.headers.get(filename)
        if cached is not None:
            return cached
        path = os.path.join(self.folder, filename)
        with open(path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        self.headers[filename] = (data_start, header)
        return self.headers[filename]

    def _open(self, filename):
        if filename in self._mm:
            return self._fd[filename], self._mm[filename]
        path = os.path.join(self.folder, filename)
        fd = os.open(path, os.O_RDONLY)
        # Random advice keeps the 535MB embedding / lm_head files from being readahead in full.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        size = os.fstat(fd).st_size
        mapped = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        self._fd[filename] = fd
        self._mm[filename] = mapped
        return fd, mapped

    def release(self, filename, start, length):
        fd = self._fd.get(filename)
        if fd is not None and length > 0:
            os.posix_fadvise(fd, start, length, os.POSIX_FADV_DONTNEED)

    def read_copied(self, name, dtype):
        filename = self.weight_map[name]
        data_start, header = self._header(filename)
        meta = header[name]
        begin, end = meta["data_offsets"]
        file_off = data_start + begin
        nbytes = end - begin
        _fd, mapped = self._open(filename)
        blob = bytes(mapped[file_off : file_off + nbytes])
        self.release(filename, file_off, nbytes)
        array = np.frombuffer(blob, dtype=dtype)
        shape = meta["shape"]
        if shape == [array.size] or shape == [array.size, 1]:
            array = array.reshape(-1)
        else:
            array = array.reshape(shape)
        return np.array(array, copy=True)

    def view_u8(self, name):
        filename = self.weight_map[name]
        data_start, header = self._header(filename)
        meta = header[name]
        begin, end = meta["data_offsets"]
        file_off = data_start + begin
        nbytes = end - begin
        _fd, mapped = self._open(filename)
        view = np.frombuffer(mapped, dtype=np.uint8, count=nbytes, offset=file_off)
        return filename, file_off, view

    def view_f16(self, name):
        filename = self.weight_map[name]
        data_start, header = self._header(filename)
        meta = header[name]
        begin, end = meta["data_offsets"]
        file_off = data_start + begin
        count = int(np.prod(meta["shape"]))
        _fd, mapped = self._open(filename)
        flat = np.frombuffer(mapped, dtype=np.float16, count=count, offset=file_off)
        matrix = flat.reshape(meta["shape"])
        return filename, file_off, matrix


class NF4Matrix:
    """One bitsandbytes NF4 weight: uint8 payload on disk, scales in RAM."""

    def __init__(self, store, prefix):
        state = json.loads(store.read_copied(prefix + ".quant_state.bitsandbytes__nf4", np.uint8).tobytes())
        if state.get("quant_type") != "nf4" or state.get("blocksize") != 64:
            raise RuntimeError(f"unsupported quant state for {prefix}: {state}")
        self.rows, self.cols = state["shape"]
        self.nested_blocksize = int(state["nested_blocksize"])
        self.offset = float(state["nested_offset"])
        self.codebook = np.ascontiguousarray(store.read_copied(prefix + ".quant_map", np.float32))
        self.nested_code = np.ascontiguousarray(store.read_copied(prefix + ".nested_quant_map", np.float32))
        self.nested_absmax = np.ascontiguousarray(store.read_copied(prefix + ".nested_absmax", np.float32))
        self.absmax_u8 = np.ascontiguousarray(store.read_copied(prefix + ".absmax", np.uint8))
        self.filename, self.file_off, self.packed = store.view_u8(prefix)
        self.store = store
        expect = self.rows * self.cols // 2
        if self.packed.size != expect:
            raise RuntimeError(f"{prefix} packed size {self.packed.size} != {expect}")
        if self.absmax_u8.size != self.rows * self.cols // 64:
            raise RuntimeError(f"{prefix} absmax size mismatch")
        self._scales = np.empty(self.absmax_u8.size, dtype=np.float32)
        self._scales_ready = False

    def scales(self):
        """Undo the nested 8-bit absmax quantization. Cached: one float per 64 weights."""
        if self._scales_ready:
            return self._scales
        n = self.absmax_u8.size
        block = self.nested_blocksize
        blocks = n // block
        view = self._scales[: blocks * block].reshape(blocks, block)
        codes = self.nested_code[self.absmax_u8[: blocks * block]].reshape(blocks, block)
        view[:] = codes
        view *= self.nested_absmax[:blocks][:, None]
        view += self.offset
        if blocks * block < n:
            tail = self.absmax_u8[blocks * block :]
            self._scales[blocks * block :] = self.nested_code[tail] * self.nested_absmax[-1] + self.offset
        self._scales_ready = True
        return self._scales

    def apply(self, x):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[1] != self.cols:
            raise RuntimeError(f"NF4 input {x.shape[1]} != cols {self.cols}")
        y = np.empty((x.shape[0], self.rows), dtype=np.float32)
        kernel().nf4(self.packed, self.scales(), x, y, self.codebook)
        # Leave the packed pages cached. They are ~1GB total and the desktop
        # still has room; dropping them makes every token wait on the NVMe.
        return y


class F16Matrix:
    """Unquantized float16 matrix (embeddings and lm_head). lm_head is applied in row chunks."""

    def __init__(self, store, name):
        self.store = store
        self.filename, self.file_off, self.matrix = store.view_f16(name)
        self.rows, self.cols = self.matrix.shape

    def gather(self, token_ids):
        rows = np.asarray(self.matrix[np.asarray(token_ids, dtype=np.int64)], dtype=np.float32)
        return np.ascontiguousarray(rows)

    def apply_chunked(self, x, chunk_rows=2048):
        """Chunked so the 535MB lm_head is never fully resident."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] != 1:
            raise RuntimeError("lm_head is applied to the last token only")
        if x.shape[1] != self.cols:
            raise RuntimeError(f"f16 input {x.shape[1]} != cols {self.cols}")
        y = np.empty(self.rows, dtype=np.float32)
        flat = self.matrix.reshape(-1)
        for row0 in range(0, self.rows, chunk_rows):
            count = min(chunk_rows, self.rows - row0)
            weights = np.ascontiguousarray(flat[row0 * self.cols : (row0 + count) * self.cols])
            # ascontiguousarray on a strided mmap slice copies just this chunk.
            weights_u16 = weights.view(np.uint16)
            part = y[row0 : row0 + count]
            kernel().f16(weights_u16, x, part)
            byte_off = self.file_off + row0 * self.cols * 2
            self.store.release(self.filename, byte_off, count * self.cols * 2)
        return y.reshape(1, -1)


def rmsnorm(x, weight, eps):
    var = np.mean(x * x, axis=-1, keepdims=True)
    return x * np.reciprocal(np.sqrt(var + eps)) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rope(x, cos, sin):
    # x: [batch, heads, dim], cos/sin: [batch, dim]
    return x * cos[:, None, :] + rotate_half(x) * sin[:, None, :]


class LlamaLayer:
    def __init__(self, index, store, hidden, n_heads, n_kv, head_dim):
        self.index = index
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        base = f"model.layers.{index}"
        self.in_norm = store.read_copied(base + ".input_layernorm.weight", np.float16).astype(np.float32)
        self.post_norm = store.read_copied(base + ".post_attention_layernorm.weight", np.float16).astype(np.float32)
        self.q = NF4Matrix(store, base + ".self_attn.q_proj.weight")
        self.k = NF4Matrix(store, base + ".self_attn.k_proj.weight")
        self.v = NF4Matrix(store, base + ".self_attn.v_proj.weight")
        self.o = NF4Matrix(store, base + ".self_attn.o_proj.weight")
        self.gate = NF4Matrix(store, base + ".mlp.gate_proj.weight")
        self.up = NF4Matrix(store, base + ".mlp.up_proj.weight")
        self.down = NF4Matrix(store, base + ".mlp.down_proj.weight")

    def __call__(self, x, pos, model):
        hidden = rmsnorm(x, self.in_norm, model.eps)
        batch = hidden.shape[0]
        q = self.q.apply(hidden).reshape(batch, self.n_heads, self.head_dim)
        k = self.k.apply(hidden).reshape(batch, self.n_kv, self.head_dim)
        v = self.v.apply(hidden).reshape(batch, self.n_kv, self.head_dim)
        cos = model.cos[pos]
        sin = model.sin[pos]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        start = int(pos[0])
        end = start + batch
        model.k_cache[self.index, start:end] = k
        model.v_cache[self.index, start:end] = v
        attn = self._attend(q, model.k_cache[self.index, :end], model.v_cache[self.index, :end], start)
        x = x + self.o.apply(attn)
        hidden = rmsnorm(x, self.post_norm, model.eps)
        gate = self.gate.apply(hidden)
        up = self.up.apply(hidden)
        x = x + self.down.apply(silu(gate) * up)
        return x

    def _attend(self, q, k, v, pos0):
        # q [B, heads, d], k/v [T, kv, d]
        batch, n_heads, dim = q.shape
        total = k.shape[0]
        group = n_heads // k.shape[1]
        k_rep = np.repeat(k, group, axis=1)
        v_rep = np.repeat(v, group, axis=1)
        scale = 1.0 / math.sqrt(dim)
        scores = np.einsum("bhd,thd->bht", q, k_rep) * scale
        qpos = pos0 + np.arange(batch)
        kpos = np.arange(total)
        future = kpos[None, :] > qpos[:, None]
        if np.any(future):
            scores = np.where(future[:, None, :], -1.0e9, scores)
        scores -= scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=-1, keepdims=True)
        out = np.einsum("bht,thd->bhd", weights, v_rep)
        return out.reshape(batch, n_heads * dim)


class MiniCPM:
    def __init__(self, folder, max_seq, progress=None):
        cfg = read_model_config(folder)
        self.folder = folder
        self.cfg = cfg
        self.hidden = int(cfg["hidden_size"])
        self.n_layers = int(cfg["num_hidden_layers"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg.get("head_dim") or self.hidden // self.n_heads)
        self.eps = float(cfg["rms_norm_eps"])
        self.rope_theta = float(cfg["rope_theta"])
        self.vocab = int(cfg["vocab_size"])
        eos = cfg["eos_token_id"]
        self.eos_ids = set(eos if isinstance(eos, list) else [eos])
        # model_window = what the checkpoint claims; max_seq = KV we actually allocate.
        self.model_window = model_context_window(cfg)
        self.max_seq = int(max_seq)
        self.store = WeightStore(folder)
        if progress:
            progress("weights", "embeddings")
        self.embed = F16Matrix(self.store, "model.embed_tokens.weight")
        self.lm_head = F16Matrix(self.store, "lm_head.weight")
        self.norm = self.store.read_copied("model.norm.weight", np.float16).astype(np.float32)
        self.layers = []
        for index in range(self.n_layers):
            if progress:
                progress("weights", f"layer {index + 1}/{self.n_layers}")
            self.layers.append(LlamaLayer(index, self.store, self.hidden, self.n_heads, self.n_kv, self.head_dim))
        self.cos, self.sin = self._rope_table(max_seq)
        self.k_cache = np.empty((self.n_layers, max_seq, self.n_kv, self.head_dim), dtype=np.float32)
        self.v_cache = np.empty((self.n_layers, max_seq, self.n_kv, self.head_dim), dtype=np.float32)
        self.cache_len = 0

    def _rope_table(self, max_seq):
        half = self.head_dim // 2
        inv = 1.0 / (self.rope_theta ** (np.arange(0, half, dtype=np.float32) / self.head_dim))
        freqs = np.outer(np.arange(max_seq, dtype=np.float32), inv)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    def forward(self, token_ids, progress=None, stop=None):
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 1:
            raise RuntimeError("token ids must be a 1D list")
        if self.cache_len + token_ids.size > self.max_seq:
            raise RuntimeError("context is full")
        x = self.embed.gather(token_ids)
        pos = np.arange(self.cache_len, self.cache_len + token_ids.size)
        for layer in self.layers:
            if stop is not None and stop.is_set():
                return None
            if progress:
                progress("layer", f"{layer.index + 1}/{self.n_layers}")
            x = layer(x, pos, self)
        self.cache_len = int(pos[-1]) + 1
        hidden = rmsnorm(x[-1:], self.norm, self.eps)
        if progress:
            progress("layer", "lm head")
        return self.lm_head.apply_chunked(hidden)


def sample_token(logits, temperature, top_p, recent_ids, repeat_penalty):
    scores = np.array(logits, dtype=np.float32, copy=True).reshape(-1)
    if repeat_penalty != 1.0 and recent_ids:
        seen = np.unique(np.asarray(recent_ids[-256:], dtype=np.int64))
        positive = scores[seen] > 0
        scores[seen[positive]] /= repeat_penalty
        scores[seen[~positive]] *= repeat_penalty
    if temperature <= 1e-5:
        return int(np.argmax(scores))
    scores /= temperature
    scores -= np.max(scores)
    if top_p < 1.0:
        order = np.argsort(scores)[::-1]
        ordered = scores[order]
        probs = np.exp(ordered)
        probs /= np.sum(probs)
        keep = int(np.searchsorted(np.cumsum(probs), top_p)) + 1
        ordered[keep:] = -1.0e9
        scores[order] = ordered
        scores -= np.max(scores)
    probs = np.exp(scores)
    probs /= np.sum(probs)
    return int(np.random.choice(probs.size, p=probs))


def _fact_lines_from_text(role, content, limit=4):
    """Pull short durable lines from a turn. Prefer names, numbers, and decisions."""
    lines = []
    for raw in content.replace("\r", "\n").split("\n"):
        line = " ".join(raw.strip().split())
        if not line or line.startswith("<") or line.startswith("/"):
            continue
        # Strip think blocks from assistant text before mining facts.
        if "<think>" in line or "</think>" in line:
            continue
        lower = line.lower()
        keep = False
        if role == "user":
            keep = True
        elif any(ch.isdigit() for ch in line):
            keep = True
        elif any(
            key in lower
            for key in (
                "my name",
                "i am",
                "i'm",
                "i live",
                "prefer",
                "remember",
                "decision",
                "we agreed",
                "goal",
                "project",
            )
        ):
            keep = True
        if not keep:
            continue
        if len(line) > 160:
            line = line[:157] + "..."
        prefix = "User" if role == "user" else "Note"
        lines.append(f"- {prefix}: {line}")
        if len(lines) >= limit:
            break
    return lines


class ChatEngine:
    def __init__(self, model, tokenizer, template, max_new):
        self.model = model
        self.tokenizer = tokenizer
        self.template = template
        self.max_new = max_new
        self.temperature = 0.7
        self.top_p = 0.9
        self.repeat_penalty = 1.05
        self.thinking = False
        self.base_system = SYSTEM_PROMPT
        self.facts = []
        self.messages = [{"role": "system", "content": self._system_text()}]

    def _system_text(self):
        if not self.facts:
            return self.base_system
        # Compacted facts ride in the system prompt so later turns stay short.
        body = "\n".join(self.facts[-24:])
        return f"{self.base_system}\n\n## Major facts (compacted)\n{body}"

    def _rebuild_system(self):
        self.messages[0] = {"role": "system", "content": self._system_text()}

    def set_system(self, text):
        self.base_system = text.strip() or SYSTEM_PROMPT
        self._rebuild_system()

    def clear(self):
        """Wipe turns but keep the current system prompt and compacted facts."""
        self.messages = [{"role": "system", "content": self._system_text()}]

    def new_chat(self):
        """Start a fresh conversation. Drops turns and compacted facts."""
        self.facts = []
        self.messages = [{"role": "system", "content": self._system_text()}]
        self.model.cache_len = 0

    def prompt_tokens(self):
        text = self.template.render(
            messages=self.messages,
            bos_token="<s>",
            eos_token="</s>",
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            tools=None,
        )
        return len(self.tokenizer.encode(text).ids)

    def _window(self):
        return max(64, int(getattr(self.model, "max_seq", 512)))

    def _fact_token_budget(self, ratio):
        # Budget for the compacted "past" block (not counting the live recent turns).
        return max(32, int(self._window() * ratio))

    def should_compact(self):
        return self.prompt_tokens() >= int(self._window() * COMPACT_TRIGGER_RATIO)

    def _facts_tokens(self):
        if not self.facts:
            return 0
        return len(self.tokenizer.encode("\n".join(self.facts)).ids)

    def _trim_facts_to_budget(self, budget_tokens):
        while self.facts and self._facts_tokens() > budget_tokens:
            self.facts.pop(0)

    def compact(self, keep_recent=COMPACT_KEEP_RECENT, force=False):
        """Fold older turns into major facts; keep newest turns verbatim.

        Target size for the compacted past is 10% of the working window, then
        5% if still over the trigger (arXiv:2610.14952; Bud ContextCompact).
        """
        if len(self.messages) <= 1 + keep_recent and not force:
            return "nothing to compact"
        older = self.messages[1:-keep_recent] if keep_recent else self.messages[1:]
        recent = self.messages[-keep_recent:] if keep_recent else []
        if not older and not force:
            return "nothing to compact"
        added = 0
        for msg in older:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role not in ("user", "assistant") or not content:
                continue
            for line in _fact_lines_from_text(role, content, limit=2):
                if line not in self.facts:
                    self.facts.append(line)
                    added += 1

        used_ratio = COMPACT_BUDGET_RATIOS[0]
        for ratio in COMPACT_BUDGET_RATIOS:
            used_ratio = ratio
            self._trim_facts_to_budget(self._fact_token_budget(ratio))
            self.messages = [{"role": "system", "content": self._system_text()}] + recent
            self.model.cache_len = 0
            if self.prompt_tokens() < int(self._window() * COMPACT_TRIGGER_RATIO):
                break
        return (
            f"compacted +{added} · past≤{int(used_ratio * 100)}% window "
            f"({self._facts_tokens()} tok) · kept {len(recent)} turns · {len(self.facts)} facts"
        )

    def _prompt_ids(self):
        limit = self.model.max_seq - 1
        # Compact before the window fills so major facts survive.
        if self.should_compact():
            self.compact()
        while True:
            text = self.template.render(
                messages=self.messages,
                bos_token="<s>",
                eos_token="</s>",
                add_generation_prompt=True,
                enable_thinking=self.thinking,
                tools=None,
            )
            ids = self.tokenizer.encode(text).ids
            if len(ids) <= limit or len(self.messages) <= 2:
                if len(ids) > limit:
                    ids = ids[-limit:]
                return ids
            # Still too long: compact harder, then drop oldest remaining turn.
            if len(self.messages) > 1 + COMPACT_KEEP_RECENT:
                self.compact(keep_recent=max(2, COMPACT_KEEP_RECENT // 2), force=True)
                continue
            del self.messages[1]
            if len(self.messages) > 1 and self.messages[1]["role"] == "assistant":
                del self.messages[1]

    def reply(self, user_text, on_token=None, on_status=None, stop=None):
        self.messages.append({"role": "user", "content": user_text})
        if self.should_compact():
            note = self.compact()
            if on_status:
                on_status(note)
        ids = self._prompt_ids()
        # Each turn rebuilds the cache from the full prompt. Context stays short on purpose.
        self.model.cache_len = 0
        started = time.perf_counter()
        if on_status:
            on_status(f"prefill {len(ids)} tokens")
        logits = self.model.forward(ids, progress=lambda _p, detail: on_status and on_status(f"prefill {detail}"), stop=stop)
        if logits is None or (stop is not None and stop.is_set()):
            self.messages.pop()
            return ""
        produced = []
        piece_ids = []
        t0 = time.perf_counter()
        for _ in range(self.max_new):
            if stop is not None and stop.is_set():
                break
            if self.model.cache_len >= self.model.max_seq:
                break
            nxt = sample_token(logits[0], self.temperature, self.top_p, ids + produced, self.repeat_penalty)
            if nxt in self.model.eos_ids:
                break
            produced.append(nxt)
            piece_ids.append(nxt)
            if on_token:
                on_token(self.tokenizer.decode(piece_ids, skip_special_tokens=True))
            logits = self.model.forward([nxt], stop=stop)
            if on_status:
                elapsed = time.perf_counter() - t0
                rate = len(produced) / elapsed if elapsed > 0 else 0.0
                on_status(f"{len(produced)} tok · {rate:.2f} tok/s · rss {rss_mb()}MB")
            if logits is None:
                break
        text = self.tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
        if not text:
            self.messages.pop()
            return ""
        self.messages.append({"role": "assistant", "content": text})
        if on_status:
            elapsed = time.perf_counter() - started
            on_status(f"done {len(produced)} tok in {elapsed:.1f}s")
        return text


def load_tokenizer(folder):
    from tokenizers import Tokenizer
    from jinja2 import Environment

    tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
    # The chat template already writes <s>. The tokenizer would add a second one.
    tokenizer.post_processor = None
    with open(os.path.join(folder, "chat_template.jinja"), "r", encoding="utf-8") as handle:
        template = Environment().from_string(handle.read())
    return tokenizer, template


def read_model_config(folder):
    path = os.path.join(folder, "config.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def model_context_window(cfg):
    """Prefer the model's declared window; else 256k."""
    for key in (
        "max_position_embeddings",
        "max_sequence_length",
        "n_positions",
        "seq_length",
        "model_max_length",
    ):
        if key in cfg and cfg[key]:
            return int(cfg[key])
    return DEFAULT_MODEL_WINDOW


def is_nf4_model(folder, cfg=None):
    cfg = cfg or read_model_config(folder)
    qc = cfg.get("quantization_config") or {}
    if qc.get("load_in_4bit") or qc.get("bnb_4bit_quant_type") == "nf4":
        return True
    # Heuristic: bitsandbytes NF4 shards expose quant_state keys in the index.
    index_path = os.path.join(folder, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            wm = json.load(handle).get("weight_map", {})
        return any("bitsandbytes__nf4" in name for name in wm)
    return False


def memory_safe_working_seq(cfg, model_window, avail_mb=None):
    """KV cache must fit in the Orin 8GB pool. Full 128k/256k will not."""
    if avail_mb is None:
        avail_mb, _ = meminfo()
    layers = int(cfg.get("num_hidden_layers") or 32)
    n_kv = int(cfg.get("num_key_value_heads") or cfg.get("num_attention_heads") or 8)
    head_dim = int(cfg.get("head_dim") or (int(cfg.get("hidden_size", 2048)) // int(cfg.get("num_attention_heads", 16))))
    # bytes ≈ layers * seq * n_kv * head_dim * 4bytes * 2(K+V)
    bytes_per_token = layers * n_kv * head_dim * 8
    # Leave headroom for OS, mapped weights, and desktop.
    budget_mb = max(256, int(avail_mb * 0.35))
    by_ram = max(256, (budget_mb * 1024 * 1024) // max(1, bytes_per_token))
    return int(min(model_window, by_ram, 8192))


def list_transformer_models(transformers_dir):
    """Return sorted model folders under Transformers/ with metadata."""
    models = []
    if not os.path.isdir(transformers_dir):
        return models
    for name in sorted(os.listdir(transformers_dir)):
        folder = os.path.join(transformers_dir, name)
        cfg_path = os.path.join(folder, "config.json")
        if not os.path.isdir(folder) or not os.path.exists(cfg_path):
            continue
        try:
            cfg = read_model_config(folder)
        except Exception:
            continue
        window = model_context_window(cfg)
        nf4 = is_nf4_model(folder, cfg)
        size_gb = 0.0
        try:
            total = 0
            for root, _dirs, files in os.walk(folder):
                for fn in files:
                    if fn.endswith(".safetensors"):
                        total += os.path.getsize(os.path.join(root, fn))
            size_gb = total / (1024 ** 3)
        except OSError:
            pass
        models.append(
            {
                "name": name,
                "path": folder,
                "window": window,
                "nf4": nf4,
                "size_gb": size_gb,
                "arch": (cfg.get("architectures") or ["?"])[0],
                "default": name == DEFAULT_MODEL_NAME or name.endswith("-NF4"),
            }
        )
    # Prefer NF4 defaults first in numbering when ties — keep alpha order but mark default.
    return models


def pick_model_interactive(transformers_dir, default_path=None):
    models = list_transformer_models(transformers_dir)
    if not models:
        raise SystemExit(f"No models with config.json found in {transformers_dir}")
    default_idx = 0
    if default_path:
        for i, m in enumerate(models):
            if os.path.abspath(m["path"]) == os.path.abspath(default_path):
                default_idx = i
                break
    else:
        for i, m in enumerate(models):
            if m["nf4"]:
                default_idx = i
                break

    print()
    print(f"MiniTalk — Jetson Orin Nano Super (8 GB)")
    print(f"Models in {transformers_dir}")
    print("-" * 60)
    for i, m in enumerate(models, 1):
        tag = []
        if m["nf4"]:
            tag.append("NF4")
        else:
            tag.append("needs-NF4-export")
        if i - 1 == default_idx:
            tag.append("default")
        label = ", ".join(tag)
        print(
            f"  [{i}] {m['name']:<28}  ctx {m['window']:>7}  "
            f"{m['size_gb']:.1f}GB  ({label})"
        )
    print("-" * 60)
    print("Enter a number to select the model (NF4 recommended on 8GB).")
    while True:
        raw = input(f"Model [{default_idx + 1}]: ").strip()
        if not raw:
            choice = default_idx
            break
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            choice = int(raw) - 1
            break
        print(f"  type 1..{len(models)}")
    selected = models[choice]
    if not selected["nf4"]:
        print(
            f"WARNING: {selected['name']} is not an NF4 dump. "
            "MiniTalk currently loads bitsandbytes NF4 shards only."
        )
        cont = input("Continue anyway? [y/N]: ").strip().lower()
        if cont not in ("y", "yes"):
            raise SystemExit("Pick an NF4 model (name ends with -NF4).")
    return selected


def self_test():
    """Check CPU/GPU NF4 kernels against NumPy, then time one real projection."""
    kn = kernel(prefer_gpu=True)
    print(f"backend: {kn.backend}  {kn.gpu_note}")
    code = np.linspace(-1, 1, 16, dtype=np.float32)
    rows, cols, batch = 4, 128, 3
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, size=rows * (cols // 2), dtype=np.uint8)
    scales = rng.random(rows * (cols // 64), dtype=np.float32).astype(np.float32) * 0.05
    x = rng.standard_normal((batch, cols), dtype=np.float32).astype(np.float32)
    y = np.empty((batch, rows), dtype=np.float32)
    kn.nf4(packed, scales, x, y, code)
    # Also exercise the CUDA batch-1 path when present.
    if kn.backend == "cuda":
        y1 = np.empty((1, rows), dtype=np.float32)
        kn.nf4(packed, scales, x[:1], y1, code)
    ref = np.zeros((batch, rows), dtype=np.float32)
    for r in range(rows):
        block = packed[r * (cols // 2) : (r + 1) * (cols // 2)]
        weights = np.empty(cols, dtype=np.float32)
        for i, byte in enumerate(block):
            scale = scales[r * (cols // 64) + (2 * i) // 64]
            weights[2 * i] = code[byte >> 4] * scale
            weights[2 * i + 1] = code[byte & 15] * scale
        ref[:, r] = x @ weights
    err = float(np.max(np.abs(y - ref)))
    print(f"kernel max abs error {err:.3e}")
    if err > 1e-4:
        raise SystemExit("NF4 kernel does not match NumPy")

    store = WeightStore(os.path.join(DEFAULT_TRANSFORMERS, DEFAULT_MODEL_NAME))
    gate = NF4Matrix(store, "model.layers.0.mlp.gate_proj.weight")
    vec = np.random.default_rng(1).standard_normal((1, gate.cols)).astype(np.float32)
    gate.apply(vec)
    t0 = time.perf_counter()
    gate.apply(vec)
    dt = time.perf_counter() - t0
    scales = gate.scales()
    print(
        f"gate_proj {gate.rows}x{gate.cols} {dt * 1000:.1f} ms  "
        f"scale mean {scales.mean():.4f}  rss {rss_mb()}MB  free {meminfo()[0]}MB"
    )
    print("self-test ok")


def run_once(model_path, max_seq, max_new, temperature, prompt, prefer_gpu=True):
    def status(message):
        print(f"\r{message:<48}", end="", file=sys.stderr, flush=True)

    kn = kernel(prefer_gpu=prefer_gpu)
    print(f"backend {kn.backend}: {kn.gpu_note or 'CPU NEON'}", file=sys.stderr)
    print("loading", model_path, file=sys.stderr)
    model = MiniCPM(model_path, max_seq, progress=lambda _p, detail: status(f"load {detail}"))
    tokenizer, template = load_tokenizer(model_path)
    engine = ChatEngine(model, tokenizer, template, max_new)
    engine.temperature = temperature

    def on_token(text):
        status(f"gen {len(text)} chars")

    print(file=sys.stderr)
    text = engine.reply(prompt, on_token=on_token, on_status=status)
    print(file=sys.stderr)
    print(text)


def run_tui(model_path, max_seq, max_new, model_window=None, prefer_gpu=True):
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Footer, Header, Input, RichLog, Static
    from textual import work

    class MiniTalkApp(App):
        TITLE = "MiniTalk"
        SUB_TITLE = "MiniCPM5-2B-NF4"
        CSS = """
        Screen { background: #070b14; }
        Header { background: #10182c; color: #9be7ff; }
        Footer { background: #10182c; color: #9be7ff; }
        #transcript {
            height: 1fr;
            margin: 0 1;
            background: #070b14;
            border: round #1a2744;
            padding: 0 1;
        }
        #live {
            height: auto;
            max-height: 14;
            margin: 0 1;
            padding: 0 1;
            color: #d7e2f2;
            background: #0c1424;
        }
        #hud {
            height: 1;
            background: #10182c;
            color: #8ea0c0;
            padding: 0 1;
        }
        #composer {
            margin: 0 1 1 1;
            border: tall #1d8a9a;
            background: #0e1526;
        }
        """
        BINDINGS = [
            Binding("ctrl+q", "quit", "Quit"),
            Binding("escape", "stop", "Stop"),
            Binding("ctrl+l", "clear", "Clear"),
        ]

        def __init__(self, model_path, max_seq, max_new, model_window=None, prefer_gpu=True):
            super().__init__()
            self.model_path = model_path
            self.max_seq = max_seq
            self.model_window = model_window or max_seq
            self.max_new = max_new
            self.prefer_gpu = prefer_gpu
            self.engine = None
            self.backend = "…"
            self.busy = False
            self.stop_flag = threading.Event()
            self.rates = deque(maxlen=24)
            self._spin = 0
            self._status = "starting"

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            yield RichLog(id="transcript", wrap=True, markup=True, highlight=False, auto_scroll=True)
            yield Static("", id="live")
            yield Static("", id="hud")
            yield Input(placeholder="Message MiniCPM    /help for commands", id="composer")
            yield Footer()

        def on_mount(self):
            self.query_one("#live", Static).display = False
            avail, total = meminfo()
            log = self.query_one("#transcript", RichLog)
            log.write(
                Panel(
                    Text.assemble(
                        ("MiniTalk\n", "bold cyan"),
                        "For NVIDIA Jetson Orin Nano Super — 8 GB unified memory.\n",
                        "Run from a terminal. Weights stay mapped; GPU used when CUDA inits.\n",
                        (
                            f"Model window {self.model_window} · working KV {self.max_seq} · "
                            f"RAM free {avail}/{total} MB\n",
                            "dim",
                        ),
                        "Esc stops · /new fresh chat · /compact keeps ~5–10% past · /help",
                    ),
                    border_style="cyan",
                    title="minitalk · orin nano",
                )
            )
            self._set_hud("mapping shards")
            self.set_interval(0.15, self._pulse)
            self._load()

        def _pulse(self):
            if not self.busy:
                return
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            self._spin = (self._spin + 1) % len(frames)
            self._paint_hud(frames[self._spin] + " " + self._status)

        def _set_hud(self, text):
            self._status = text
            self._paint_hud(text)

        def _paint_hud(self, text):
            avail, total = meminfo()
            used = max(total - avail, 0)
            cells = 10
            filled = int(round(cells * used / total)) if total else 0
            bar = "█" * filled + "░" * (cells - filled)
            engine = self.engine
            temp = engine.temperature if engine else 0.7
            think = "think" if engine and engine.thinking else "direct"
            spark = ""
            if self.rates:
                lo, hi = min(self.rates), max(self.rates)
                blocks = "▁▂▃▄▅▆▇█"
                chars = []
                for rate in self.rates:
                    if hi <= lo:
                        chars.append(blocks[0])
                    else:
                        chars.append(blocks[int((rate - lo) / (hi - lo) * (len(blocks) - 1))])
                spark = "  " + "".join(chars)
            self.query_one("#hud", Static).update(
                f"{text}   ram {bar} {used}/{total}MB   {self.backend}   "
                f"{temp:.2f} {think}   kv {self.max_seq}/{self.model_window}{spark}"
            )

        @work(thread=True, exclusive=True, group="load", exit_on_error=False)
        def _load(self):
            try:
                def progress(_phase, detail):
                    self.app.call_from_thread(self._set_hud, f"load {detail}")

                kn = kernel(prefer_gpu=self.prefer_gpu)
                model = MiniCPM(self.model_path, self.max_seq, progress=progress)
                tokenizer, template = load_tokenizer(self.model_path)
                engine = ChatEngine(model, tokenizer, template, self.max_new)
                self.app.call_from_thread(self._loaded, engine, kn)
            except Exception:
                self.app.call_from_thread(self._fail, traceback.format_exc())

        def _loaded(self, engine, kn):
            self.engine = engine
            self.backend = kn.backend
            self.model_window = getattr(engine.model, "model_window", self.model_window)
            note = kn.gpu_note or "CPU NEON+OpenMP"
            self._set_hud("ready")
            self.query_one("#transcript", RichLog).write(
                Text(f"Model mapped · backend {kn.backend} · {note}", style="green")
            )
            self.query_one("#composer", Input).focus()

        def _fail(self, message):
            self.query_one("#transcript", RichLog).write(Panel(message, title="error", border_style="red"))
            self._set_hud("failed")

        def on_input_submitted(self, event: Input.Submitted):
            text = event.value.strip()
            event.input.value = ""
            if not text:
                return
            if text.startswith("/"):
                self._command(text)
                return
            if self.engine is None:
                self._set_hud("still loading the model")
                event.input.value = text
                return
            if self.busy:
                self._set_hud("still generating — Esc stops")
                event.input.value = text
                return
            self._ask(text)

        def _command(self, text):
            parts = text.split(maxsplit=1)
            name = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            log = self.query_one("#transcript", RichLog)
            if name in ("/quit", "/exit"):
                self.exit()
                return
            if name == "/help":
                log.write(
                    Panel(
                        "/new            start a new chat (drops turns and facts)\n"
                        "/clear          wipe the transcript; keep compacted facts\n"
                        "/compact        fold older turns into major facts now\n"
                        "/facts          show the compacted major-facts list\n"
                        "/temp 0.7       sampling temperature (0 = greedy)\n"
                        "/topp 0.9       nucleus cutoff\n"
                        "/max 128        max new tokens\n"
                        "/repeat 1.05    repetition penalty\n"
                        "/think          toggle the think block\n"
                        "/sys TEXT       replace the system prompt\n"
                        "/quit           leave\n"
                        "Esc             stop the current reply\n"
                        "\n"
                        "Auto-compact near 70% of the working KV window.\n"
                        "Past history is kept at ~10% of the window, then 5%\n"
                        "if still tight (keeps names, prefs, decisions).",
                        title="commands",
                        border_style="cyan",
                    )
                )
                return
            if self.engine is None:
                self._set_hud("model is not ready")
                return
            if name in ("/new", "/reset"):
                self.action_new()
                return
            if name == "/clear":
                self.action_clear()
                return
            if name == "/compact":
                note = self.engine.compact(force=True)
                log.write(Text(note, style="yellow"))
                self._set_hud(note)
                return
            if name == "/facts":
                facts = self.engine.facts
                if not facts:
                    log.write(Text("No compacted facts yet.", style="dim"))
                else:
                    log.write(
                        Panel(
                            "\n".join(facts),
                            title=f"major facts ({len(facts)})",
                            border_style="yellow",
                        )
                    )
                return
            if name == "/temp":
                self.engine.temperature = float(arg)
                self._set_hud(f"temperature {self.engine.temperature}")
                return
            if name == "/topp":
                self.engine.top_p = min(1.0, max(0.05, float(arg)))
                self._set_hud(f"top_p {self.engine.top_p}")
                return
            if name == "/max":
                self.engine.max_new = max(1, int(arg))
                self._set_hud(f"max new tokens {self.engine.max_new}")
                return
            if name == "/repeat":
                self.engine.repeat_penalty = float(arg)
                self._set_hud(f"repeat penalty {self.engine.repeat_penalty}")
                return
            if name == "/think":
                self.engine.thinking = not self.engine.thinking
                state = "on" if self.engine.thinking else "off"
                self._set_hud(f"thinking {state}")
                return
            if name == "/sys":
                self.engine.set_system(arg)
                log.write(Text("System prompt updated.", style="yellow"))
                return
            log.write(Text(f"Unknown command {name}. Try /help.", style="red"))

        def action_new(self):
            if self.busy:
                self.stop_flag.set()
            if self.engine:
                self.engine.new_chat()
            self.query_one("#transcript", RichLog).clear()
            self.query_one("#live", Static).update("")
            self.query_one("#live", Static).display = False
            self.query_one("#transcript", RichLog).write(
                Text("New chat. Compacted facts cleared.", style="green")
            )
            self._set_hud("new chat")

        def action_clear(self):
            if self.busy:
                self.stop_flag.set()
            if self.engine:
                self.engine.clear()
            self.query_one("#transcript", RichLog).clear()
            self.query_one("#live", Static).update("")
            self.query_one("#live", Static).display = False
            self._set_hud("cleared")

        def action_stop(self):
            if self.busy:
                self.stop_flag.set()
                self._set_hud("stopping")

        def _ask(self, text):
            log = self.query_one("#transcript", RichLog)
            log.write(Panel(Text(text), title="you", border_style="cyan", title_align="left"))
            live = self.query_one("#live", Static)
            live.display = True
            live.update(Text("…", style="dim"))
            self.busy = True
            self.stop_flag.clear()
            self.rates.clear()
            self._generate(text)

        @work(thread=True, exclusive=True, group="gen", exit_on_error=False)
        def _generate(self, text):
            try:
                last_len = 0

                def on_token(partial):
                    nonlocal last_len
                    if len(partial) != last_len:
                        last_len = len(partial)
                        self.app.call_from_thread(self._show_partial, partial)

                def on_status(message):
                    if "tok/s" in message:
                        try:
                            rate = float(message.split("tok/s")[0].split()[-1])
                            self.rates.append(rate)
                        except ValueError:
                            pass
                    self.app.call_from_thread(self._set_hud, message)

                answer = self.engine.reply(text, on_token=on_token, on_status=on_status, stop=self.stop_flag)
                self.app.call_from_thread(self._finish, answer)
            except Exception:
                self.app.call_from_thread(self._finish_error, traceback.format_exc())

        def _show_partial(self, text):
            body = Text(text)
            self.query_one("#live", Static).update(Panel(body, title="minicpm", border_style="green"))

        def _finish(self, answer):
            live = self.query_one("#live", Static)
            live.update("")
            live.display = False
            self.busy = False
            if not answer:
                self._set_hud("stopped" if self.stop_flag.is_set() else "empty reply")
                return
            body = _render_answer(answer, Markdown, Text)
            self.query_one("#transcript", RichLog).write(
                Panel(body, title="minicpm", border_style="green", title_align="left")
            )
            self._set_hud("ready")

        def _finish_error(self, message):
            self.busy = False
            self.query_one("#live", Static).display = False
            self.query_one("#transcript", RichLog).write(Panel(message, title="error", border_style="red"))
            self._set_hud("error")

    MiniTalkApp(model_path, max_seq, max_new, model_window=model_window, prefer_gpu=prefer_gpu).run()


def _render_answer(answer, markdown_cls, text_cls):
    """Show a think block dimmed when the model emitted one, and markdown the reply."""
    visible = answer
    thought = ""
    if "</think>" in answer:
        thought, visible = answer.split("</think>", 1)
        thought = thought.replace("<think>", "").strip()
        visible = visible.strip()
    parts = []
    if thought:
        parts.append(text_cls(thought + "\n", style="dim italic"))
    try:
        parts.append(markdown_cls(visible or answer))
    except Exception:
        parts.append(text_cls(visible or answer))
    if len(parts) == 1:
        return parts[0]
    from rich.console import Group

    return Group(*parts)


def main():
    parser = argparse.ArgumentParser(
        description="MiniTalk — terminal LLM chat for Jetson Orin Nano Super (8GB)"
    )
    parser.add_argument(
        "--transformers",
        default=DEFAULT_TRANSFORMERS,
        help="directory of local HuggingFace-style models",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model folder or name under --transformers (skips the picker)",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=None,
        help="working KV length (default: from config.json, clamped for 8GB)",
    )
    parser.add_argument("--max-new", type=int, default=128, help="max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="temperature for --once")
    parser.add_argument("--once", metavar="PROMPT", help="print one reply and exit")
    parser.add_argument("--self-test", action="store_true", help="check NF4 CPU/GPU kernels and exit")
    parser.add_argument("--cpu", action="store_true", help="force CPU NEON (skip Orin GPU)")
    parser.add_argument(
        "--pick",
        action="store_true",
        help="force interactive model list even if --model is set",
    )
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    prefer_gpu = not args.cpu
    model_path = args.model
    if model_path and not os.path.isabs(model_path) and not os.path.isdir(model_path):
        cand = os.path.join(args.transformers, model_path)
        if os.path.isdir(cand):
            model_path = cand

    need_pick = args.pick or not model_path
    if need_pick:
        if not sys.stdin.isatty() and not args.once:
            # Non-interactive: fall back to default NF4 if present.
            fallback = os.path.join(args.transformers, DEFAULT_MODEL_NAME)
            if os.path.isdir(fallback):
                model_path = fallback
                print(f"Using default model {model_path}", file=sys.stderr)
            else:
                raise SystemExit("No TTY for model picker and no --model given")
        else:
            selected = pick_model_interactive(args.transformers, default_path=model_path)
            model_path = selected["path"]
            model_window = selected["window"]
    else:
        model_window = model_context_window(read_model_config(model_path))

    cfg = read_model_config(model_path)
    model_window = model_context_window(cfg)
    if args.max_seq is not None:
        max_seq = min(args.max_seq, model_window)
    else:
        max_seq = memory_safe_working_seq(cfg, model_window)
    print(
        f"model={os.path.basename(model_path)}  config_window={model_window}  "
        f"working_kv={max_seq}  (8GB-safe clamp)",
        file=sys.stderr,
    )

    if args.once:
        run_once(model_path, max_seq, args.max_new, args.temp, args.once, prefer_gpu=prefer_gpu)
        return
    run_tui(model_path, max_seq, args.max_new, model_window=model_window, prefer_gpu=prefer_gpu)


if __name__ == "__main__":
    main()
