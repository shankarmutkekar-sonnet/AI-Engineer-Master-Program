# Day 1 — GPU Architecture and the CUDA Execution Model

Phase 0, AI Systems Engineer Master Program

## Tensors and memory layout

A tensor is a flat block of memory plus metadata that tells you how to read that block as having shape. There's no such thing as 2D or 3D memory on real hardware — RAM and GPU memory are both linear address spaces. A matrix `[[1,2,3],[4,5,6]]` with shape (2,3) is stored as `[1,2,3,4,5,6]`.

Strides are the metadata that maps a logical index back to a position in that flat array. They're computed right to left:

```
stride[last] = 1
stride[i] = stride[i+1] * shape[i+1]
```

For shape (2, 3, 4), that gives strides (12, 4, 1). To find element `x[1, 2, 3]`, the computer just does `1*12 + 2*4 + 3*1 = 23` and reads position 23 out of the flat array.

A tensor is contiguous when its actual strides match what you'd get computing them fresh for its current shape. `transpose()` and `permute()` only rewrite the shape/stride metadata, so they're nearly free, but they usually break contiguity — the shape says one thing, the physical memory order says another. `.view()` needs contiguous memory to work, since it reinterprets the flat buffer directly. `.contiguous()` forces a real copy, laying the data out fresh, which costs memory bandwidth — something that matters on GPUs where bandwidth is often the actual bottleneck.

## Why GPUs exist

A CPU core is built to run one instruction stream fast, with deep pipelines, branch prediction, and large caches. A GPU instead uses thousands of small, simple cores and gives up per-core cleverness for raw parallel throughput. Something like matmul is millions of independent multiply-adds — no output element depends on another — which is exactly the kind of workload GPUs are built for.

## Hardware hierarchy: GPU, SM, warp, thread

- GPU: the whole chip, containing many Streaming Multiprocessors (an H100 has roughly 130).
- SM (Streaming Multiprocessor): a cluster of CUDA cores with its own registers and shared memory/L1 cache.
- Thread: the smallest unit of work — one core, one task.
- Warp: 32 threads bundled together and forced to execute the same instruction at the same time, just on different data. This is SIMT — single instruction, multiple threads.

An SM can have several warps in flight and switches between them to hide memory latency, so idle waiting is rare — if one warp is stalled waiting on memory, the SM just runs a different warp that's ready.

Warp divergence happens when threads within the same warp hit different branches of an `if/else`. Since all 32 threads must execute the same instruction at once, the hardware runs one branch while the other threads sit idle, then runs the other branch while the first set sits idle. This doesn't produce wrong answers, but it can roughly halve throughput, or worse with more branches.

Warp size is fixed at 32 by NVIDIA, going back to the original Tesla architecture in 2006, and has stayed that way for compatibility since. It's not derived from a formula — it's a hardware tradeoff between having groups too small (scheduling overhead dominates) and too large (more wasted idle time when threads diverge).

CUDA itself is not hardware. It's the toolkit and language NVIDIA provides for writing the kernels — the functions — that run on the GPU.

## Threads: Python threading, asyncio, and CUDA threads

These three share a word but aren't related.

Python's `threading` module is limited by the GIL (Global Interpreter Lock), so only one thread executes Python bytecode at a time. Multiple threads exist, but they take turns; this mainly helps when a thread is waiting on something external, like a network call, since it can hand off control while idle.

`asyncio` is single-threaded. One thread juggles multiple tasks by voluntarily pausing at await points and picking up other work in the meantime. It's cooperative concurrency, not parallelism.

CUDA threads run on real, separate hardware lanes simultaneously — genuine parallel execution, with none of Python's single-lock bottleneck.

### Why the GIL exists

Every Python object carries a reference count, and when it hits zero the object is freed immediately — this is how Python does most of its garbage collection. Incrementing or decrementing that count isn't atomic at the CPU level; it's a read, modify, write. If two OS threads updated the same object's refcount at the same time, you'd get race conditions that could crash the interpreter or leak memory.

The fix Guido van Rossum chose in the early 1990s, when single-core CPUs were standard, was a single lock around the whole interpreter: only one thread can run Python bytecode, and therefore touch refcounts, at once. That's the GIL. It's stuck around since because a huge ecosystem of C extensions, including NumPy, was built assuming its guarantees, and removing it risks breaking that or requires rethinking memory management from scratch. PEP 703, free-threaded Python, is the active effort to remove it, available as an experimental build starting in Python 3.13, but not yet the default.

This is also why PyTorch and NumPy release the GIL internally during heavy C or CUDA computation — the actual kernel runs outside the GIL's reach, so GPU work isn't bottlenecked by Python's single-thread rule.

## Kernel launches and streams

Calling `torch.matmul(a, b)` doesn't run the computation and wait. The host (CPU) tells the device (GPU) which kernel to run, on what data, with how many threads — and that's the kernel launch. It's queued, not blocking. The Python call returns almost instantly while the GPU works in the background. This is asynchronous execution.

That means code like:

```python
c = torch.matmul(a, b)
print("done")
```

can print "done" before the GPU has actually finished computing `c`. PyTorch handles correctness for you — it blocks automatically the moment you try to read `c`'s values — but naive timing around a matmul with `time.time()` will only measure the launch, not the computation, unless you force a synchronization point.

A CUDA stream is a queue of GPU work executed in order within that stream. By default PyTorch uses a single stream, so operations happen in the order you issue them. Multiple streams let independent work run concurrently — for example overlapping data transfer for the next batch with compute on the current one, a real technique for hiding I/O latency.

To benchmark correctly:

```python
torch.cuda.synchronize()
start = time.time()
c = torch.matmul(a, b)
torch.cuda.synchronize()
end = time.time()
print(end - start)
```

Reading `c` (`c.cpu()`, `print(c)`) also forces an implicit sync, but explicit `synchronize()` calls are safer for benchmarking since it's easy to accidentally sync on the wrong tensor and think you measured something you didn't.

---

## Interview questions — answers

**1. A PyTorch operation "returns instantly" on GPU but the result actually took 200ms to compute. Why, and how do you measure it correctly?**

Kernel launches are asynchronous. The CPU queues the work on the GPU and moves on immediately without waiting for it to finish, so the Python call returns as soon as the launch is queued, not when the computation completes. To measure real compute time, you need an explicit synchronization point — `torch.cuda.synchronize()` before and after the operation — or you need to force a read of the result, which implicitly blocks until the GPU is done. Without that, `time.time()` around the call only measures how long it took to queue the work, not how long the GPU spent computing it.

**2. A colleague's kernel has `if (thread_id % 2 == 0)` and runs 2x slower than expected. Why, in terms of warp mechanics?**

All 32 threads in a warp execute the same instruction at the same time, on different data — that's the SIMT model. When threads within a single warp take different branches of an if/else, the hardware can't actually run both branches simultaneously. It runs the "true" branch while the threads that should take the "false" branch sit idle, then runs the "false" branch while the first group sits idle. Correctness is preserved, but half the warp's threads are doing nothing at any given moment, which is exactly why performance drops by roughly half with a two-way branch like this one.

**3. `x.permute(2, 0, 1)` followed by `.view(-1)` throws a runtime error. Why, and what's the one-line fix?**

`permute()` only rewrites shape and stride metadata — it doesn't move any data in memory. After the permute, the tensor's strides no longer match what you'd get computing them fresh for the new shape, so the tensor is non-contiguous: the logical order implied by the new shape doesn't match the physical order of the data in memory. `.view()` requires contiguous memory because it reinterprets the flat buffer directly without doing any data movement, so it fails on a non-contiguous tensor. The fix is to call `.contiguous()` before `.view()`, which forces an actual copy into the correct physical order: `x.permute(2, 0, 1).contiguous().view(-1)`.

**4. "Python's asyncio gives true parallelism for CPU-bound work, just like CUDA threads." What's wrong with that?**

Asyncio is single-threaded. There's exactly one thread of execution, and it juggles multiple tasks by cooperatively pausing at await points and switching to other pending work — nothing is actually running at the same instant as anything else. This helps with I/O-bound work, where a task spends most of its time waiting rather than computing, but it does nothing for CPU-bound work, since the single thread still has to run each computation start to finish before switching. CUDA threads, by contrast, run on separate physical hardware lanes and genuinely execute at the same time. Asyncio is concurrency without parallelism; CUDA threads are real parallelism.

**5. Why is warp size fixed at 32 instead of being dynamic per kernel?**

It's a fixed hardware property of the SM's scheduler, not something software chooses. The scheduler issues one instruction per cycle to a fixed number of ALU lanes, and that number is baked into the silicon design. Making it dynamic would mean redesigning the scheduling hardware for every kernel, which isn't how GPUs are built. The value itself is a design tradeoff: too small a group and scheduling overhead dominates, too large a group and warp divergence wastes more idle capacity whenever threads disagree on a branch. NVIDIA picked 32 for the original Tesla architecture in 2006 and kept it for consistency and compatibility across every generation since.
