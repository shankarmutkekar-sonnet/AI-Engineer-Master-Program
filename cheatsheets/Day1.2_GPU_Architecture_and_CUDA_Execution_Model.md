# Day 1 — GPU Architecture and the CUDA Execution Model

**Program:** AI Systems Engineer Master Program — Phase 0 (Compressed Track)
**Topic:** GPU Architecture and CUDA Execution Model
**Format:** Diagnostic-driven deep dive

---

## 1. The Motivating Example

```python
import torch
a = torch.randn(4096, 4096, device='cuda')
b = torch.randn(4096, 4096, device='cuda')
c = torch.matmul(a, b)
torch.cuda.synchronize()
```

Everything in this lesson explains, in order, what actually happens between the `torch.matmul(a, b)` call and the GPU finishing the computation.

---

## 2. The Host Side: What `torch.matmul` Actually Does

When you call `torch.matmul(a, b)` on CUDA tensors, the call is **asynchronous**. The CPU does not wait for the GPU to finish. Instead:

- The CPU enqueues a lightweight **launch packet** onto the GPU's work queue and immediately moves on to the next line of Python.
- This launch packet does **not** contain the actual tensor data (that's already sitting in GPU memory since both tensors were created with `device='cuda'`). It contains:
  - A reference to **which compiled GPU kernel** to run
  - **Pointers** (memory addresses) to where `a`, `b`, and `c` live in GPU memory
  - **Launch configuration** (how many blocks/threads to use)
  - **Which stream** the work is enqueued on

This is why launching GPU work is cheap and fast from the CPU's perspective — you're shipping a tiny instruction packet, not megabytes of data.

Because the call is async, if you want the CPU to actually wait for the GPU to finish (e.g., before timing the operation, or reading the result), you need an explicit barrier: `torch.cuda.synchronize()`.

**Implicit sync:** Operations like `print(c)` also force a wait — because printing requires reading the actual computed values, which live in GPU memory, PyTorch triggers a sync internally before copying the data back. The difference is that `synchronize()` is an *explicit, visible* barrier you control, while relying on `print()`, `.item()`, or `.cpu()` to sync is considered bad practice — the stall is invisible in your code and makes profiling/debugging performance harder.

---

## 3. Kernel Dispatch: How PyTorch Picks the Actual Code to Run

PyTorch does not write GPU code on the fly. `torch.matmul` is routed through PyTorch's **dispatcher (ATen)**, which decides which backend implementation to call based on:

- **Device** (`cuda`)
- **Dtype** (`float32`, `float16`, `bfloat16`, etc.)
- **Operation** (dense matmul, in this case)

For dense matrix multiplication on NVIDIA GPUs, this almost always routes to **cuBLAS** (or cuBLASLt / CUTLASS-based kernels in some paths). cuBLAS ships **many pre-compiled kernel variants** tuned for different shapes and dtypes, and it picks (via heuristics, or short autotuning benchmarks) which specific kernel/tile size to actually launch.

### Why shape and dtype change which kernel gets picked

Modern NVIDIA GPUs (Volta architecture onward: V100, A100, H100, etc.) have **two distinct categories of compute hardware**:

| Hardware | What it is | Behavior |
|---|---|---|
| **CUDA cores** | General-purpose ALUs | ~1 multiply-add per cycle per core; work at fp32 (and fp64) |
| **Tensor Cores** | Specialized matrix-multiply silicon | Do a whole small matrix multiply-accumulate (e.g. a tile) in one shot — but only accept 16-bit-class inputs |

So kernel selection is really answering: **can this matmul be routed through Tensor Cores, or does it have to fall back to plain CUDA cores?**

- **`float32` tensors** → runs on CUDA cores at full precision, **or** — if `torch.backends.cuda.matmul.allow_tf32 = True` (often default) — gets silently downcast internally to **TF32** and routed through Tensor Cores anyway, trading a little precision for a lot of speed.
- **`float16` / `bfloat16` tensors** → dispatch straight to Tensor Core kernels — full speed, no compromise needed.
- **Shape matters** because Tensor Cores operate on fixed tile sizes. If matrix dimensions don't divide cleanly into those tiles, cuBLAS may need padding or a different tiling/kernel strategy — part of why cuBLAS ships so many kernel variants and sometimes autotunes per exact shape.

**Key mental model:** it's not "16-bit numbers are inherently faster." It's that **there is a specialized fast lane on the chip (Tensor Cores), and only 16-bit-shaped traffic is allowed onto it.** fp32 has to take the slower general-purpose road (CUDA cores) unless explicitly downcast (e.g. via TF32).

---

## 4. Number Formats: fp32 vs fp16 vs bf16

A floating point number is stored using a fixed number of bits split into three parts:

- **Sign** — positive or negative
- **Exponent** — controls *range* (how big/small a number can get)
- **Mantissa** — controls *precision* (how many significant digits are kept)

| Format | Total bits | Sign | Exponent bits | Mantissa bits | Range | Precision |
|---|---|---|---|---|---|---|
| fp64 | 64 | 1 | 11 | 52 | huge | very high |
| fp32 | 32 | 1 | 8 | 23 | huge | high |
| fp16 | 16 | 1 | 5 | 10 | small | medium |
| bf16 | 16 | 1 | 8 | 7 | huge (same as fp32) | low |
| tf32 | 19 (internal only) | 1 | 8 | 10 | huge | medium |

### fp16 vs bf16 — the actual difference

Both use 16 bits total, but split them differently:

- **fp16**: more mantissa bits (10) → better precision, but only 5 exponent bits → **small range**. Gradients that get very large or very small during training can **overflow/underflow to infinity or zero**, silently breaking training. This was a real, historically common failure mode.
- **bf16** ("Brain Floating Point 16," developed by Google Brain specifically for deep learning): copies fp32's exponent size exactly (8 bits) → **same wide range as fp32**, but only 7 mantissa bits → much less precise about the exact value within that range.

**The trade-off in one line:** bf16 says *"I'd rather lose precision than lose range."* That's why it was purpose-built for deep learning, where avoiding overflow/underflow during training matters more than exact decimal precision.

### Why bf16 is faster than fp32 (hardware routing, not just memory)

It's not simply "fewer bits = faster." The real chain is:

1. Tensor Cores are physically separate silicon from CUDA cores.
2. Tensor Cores are wired to execute a whole small matrix-multiply operation in one shot — but **only** if inputs are in a 16-bit-class format.
3. fp32 numbers are too "wide" to feed into that circuit directly (unless downcast via TF32).
4. So bf16 traffic gets access to the fast lane (Tensor Cores); fp32 is stuck on the general-purpose road (CUDA cores) — that's the actual source of the speedup, not the bit-width alone.

(Note: fewer bits also reduces memory bandwidth pressure — more numbers fit per cache line / transfer — but that's a separate benefit from the hardware-routing one above.)

---

## 5. Execution Hierarchy: Thread → Warp → Block → Grid → SM

Once a kernel is launched, work is distributed across the GPU's hardware using a specific hierarchy:

```
Thread  →  Warp (32 threads, scheduled together)  →  Block (many warps)  →  Grid (many blocks)
```

- **Thread** — the smallest unit of execution; runs one "lane" of the kernel's code.
- **Warp** — a group of **32 threads** that are scheduled and executed together in lockstep.
- **Block** — a group of many warps.
- **Grid** — the full set of blocks launched for a kernel call.

**SM (Streaming Multiprocessor)** is **hardware**, not a level of the thread hierarchy itself. It's a physical execution unit on the GPU chip. **Blocks get assigned to SMs**, and the SM schedules and executes the warps within those blocks. A single SM can hold multiple resident blocks simultaneously if there's enough register/shared-memory capacity. Many SMs operate in parallel across the chip — that parallel operation across SMs is where the GPU's actual massive parallelism comes from.

So the GPU does **not** run "all threads at the exact same instant" in some undifferentiated mass — it's a structured, scheduled distribution of blocks onto SMs, and warps onto each SM's execution units.

---

## 6. CUDA Streams

### The problem streams solve

By default, all `.cuda()` operations you launch are queued **in the order you wrote them** and executed **one after another** on the GPU — even though each individual launch is async from the CPU's perspective. This default queue is the **default stream**.

### What a stream actually is

A stream is simply an **ordered queue of GPU work**. Operations on the *same* stream execute in the order they were enqueued — guaranteed. Operations on *different* streams have **no ordering guarantee relative to each other**, meaning the GPU is free to run them **concurrently** if hardware capacity allows.

### Example

```python
# Default stream — sequential, even though the ops are unrelated
a = torch.randn(1000, 1000, device='cuda')
b = a @ a          # runs
c = a + a          # waits for the matmul to finish first, unnecessarily

# Two streams — can now overlap in time
s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

with torch.cuda.stream(s1):
    c = a @ a       # enqueued on stream 1

with torch.cuda.stream(s2):
    d = a + a       # enqueued on stream 2, can run concurrently with stream 1
```

### Real production use case: overlapping data transfer with compute

CPU→GPU data transfer runs on a **separate copy engine**, distinct from the compute SMs. A common real-world pattern: put the *next* batch's CPU→GPU copy on one stream while the *current* batch is still computing on the default stream — so the GPU is never idle waiting for data. This is what `DataLoader(..., pin_memory=True)` combined with `.to(device, non_blocking=True)` sets up under the hood.

### Why single-stream (strict ordering) is the default

PyTorch cannot automatically know which of your operations depend on each other's outputs. If unrelated ops ran concurrently by default with no ordering guarantee, an operation could read data before a prior operation finished writing it — producing wrong results or errors. **Correctness by default, concurrency as an explicit opt-in** — you only step outside the default when you, the programmer, know two pieces of work are genuinely independent.

### Synchronization granularity

- `torch.cuda.synchronize()` — waits for **all** streams to finish.
- `stream.synchronize()` — waits for just **one specific stream** to finish. Useful when two streams are doing independent work but need to rejoin at a specific point (e.g., "wait for this stream's cast to bf16 to finish before the other stream's matmul reads it").

---

## Cheat Sheet — Quick Revision

**Async launch:**
- `torch.matmul(...)` on CUDA tensors is async; CPU enqueues a launch packet (kernel ref + memory pointers + launch config + stream) and moves on.
- `torch.cuda.synchronize()` = explicit CPU-blocks-until-GPU-done barrier.
- Implicit syncs happen too — e.g. `print(tensor)`, `.item()`, `.cpu()` — but these are invisible/bad-practice sync points.

**Kernel dispatch:**
- ATen dispatcher routes op + device + dtype → backend library.
- Dense matmul on CUDA → cuBLAS (or cuBLASLt/CUTLASS).
- cuBLAS picks specific kernel variant based on shape + dtype (heuristics or autotuning).

**CUDA cores vs Tensor Cores:**
- CUDA cores = general ALU, ~1 multiply-add/cycle, handles fp32/fp64.
- Tensor Cores = specialized matmul silicon, does a tile-sized matmul in one shot, only accepts 16-bit-class inputs.
- fp32 → CUDA cores (or Tensor Cores via TF32 downcast if `allow_tf32=True`).
- fp16/bf16 → straight to Tensor Cores.

**Number formats (bits: sign / exponent / mantissa):**

| Format | Bits | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|---|
| fp32 | 32 | 8 | 23 | huge | high |
| fp16 | 16 | 5 | 10 | small | medium |
| bf16 | 16 | 8 | 7 | huge | low |
| tf32 | 19 | 8 | 10 | huge | medium |

- Exponent bits → range. Mantissa bits → precision.
- fp16: good precision, narrow range → risk of overflow/underflow in training.
- bf16: fp32's range, less precision → built for deep learning by Google Brain.
- bf16 faster than fp32 because it fits Tensor Cores' 16-bit input requirement — a hardware routing advantage, not just "fewer bits = faster."

**Execution hierarchy:**
```
Thread → Warp (32 threads) → Block (many warps) → Grid (many blocks)
```
- SM (Streaming Multiprocessor) = physical hardware, not a hierarchy level.
- Blocks get assigned to SMs; SM schedules the warps within its assigned blocks.
- Many SMs run in parallel across the chip = the GPU's real parallelism.

**CUDA Streams:**
- Stream = ordered queue of GPU work.
- Same stream → strict execution order guaranteed.
- Different streams → no ordering guarantee → can run concurrently.
- Default stream = everything sequential (safety default, since PyTorch can't infer dependencies automatically).
- Common real use: overlap next-batch CPU→GPU copy (copy engine) with current-batch compute (SMs) via `pin_memory=True` + `non_blocking=True`.
- `torch.cuda.synchronize()` = wait for all streams. `stream.synchronize()` = wait for one stream only.

---

*Day 1 of the AI Systems Engineer Master Program — Phase 0 (Compressed Track). Next: Day 2 — Tensor Operations and Memory Layout.*
