"""
PyTorch Primer Scratch Script
Day 2 pause -> PyTorch fundamentals
Run locally with: python pytorch_primer.py

Sections are added incrementally as we cover them in the session.
Currently covers: Section 1 (Tensor object) + Section 2 (creation, dtypes, devices)
"""

import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print()

# ---------------------------------------------------------------------------
# SECTION 1: Tensor as a Python object (Storage vs metadata — you know this)
# ---------------------------------------------------------------------------
print("=" * 60)
print("SECTION 1: Tensor object basics")
print("=" * 60)

x = torch.tensor([1, 2, 3])
print("x:", x)
print("x.shape:", x.shape)
print("x.dtype:", x.dtype)
print("x.device:", x.device)
print("x.stride():", x.stride())

# reshape shares storage when possible
y = x.reshape(3, 1)
print("\ny = x.reshape(3, 1):", y.tolist())
print("Same storage as x?", x.storage().data_ptr() == y.storage().data_ptr())

# ---------------------------------------------------------------------------
# SECTION 2: Creation, dtypes, devices
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SECTION 2: Creation, dtypes, devices")
print("=" * 60)

a = torch.zeros(3, 4)
b = torch.ones(3, 4)
c = torch.randn(3, 4)
d = torch.arange(10)
print("zeros shape:", a.shape)
print("randn sample:\n", c)

# dtypes
f32 = torch.tensor([1, 2, 3], dtype=torch.float32)
f16 = torch.tensor([1, 2, 3], dtype=torch.float16)
bf16 = torch.tensor([1, 2, 3], dtype=torch.bfloat16)
i64 = torch.tensor([1, 2, 3], dtype=torch.int64)
print("\nf32.dtype:", f32.dtype)
print("bf16.dtype:", bf16.dtype)

# casting
casted = f32.to(torch.bfloat16)
print("casted dtype:", casted.dtype)

# device-agnostic pattern (works with or without a GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nResolved device:", device)

x_cpu = torch.tensor([1, 2, 3])
x_dev = x_cpu.to(device)   # returns a NEW tensor, must reassign
print("x_cpu.device:", x_cpu.device)
print("x_dev.device:", x_dev.device)

# .item() pulls a python scalar out of a single-element tensor
scalar_tensor = torch.tensor(3.14)
print("\n.item() ->", scalar_tensor.item(), type(scalar_tensor.item()))

print("\nDone. Sections 4-7 will be appended here as we cover them.")

# ---------------------------------------------------------------------------
# SECTION 3: Indexing, slicing, reshaping, broadcasting
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SECTION 3: Indexing, slicing, broadcasting")
print("=" * 60)

x = torch.arange(24).reshape(4, 6)
print("x:\n", x)
print("x[0]:", x[0].tolist())
print("x[:, 0]:", x[:, 0].tolist())
print("x[1:3, 2:4]:\n", x[1:3, 2:4])

# basic slicing = view, shares storage
y = x[1:3]
y[0, 0] = -1
print("\nAfter mutating y (a slice of x), x is also changed:")
print("x[1,0] is now:", x[1, 0].item())

# fancy indexing = copy, not a view
z = x[[0, 2, 3]]
print("\nFancy-indexed z shares storage with x?",
      z.storage().data_ptr() == x.storage().data_ptr())

# unsqueeze / squeeze
v = torch.randn(5)
v_batched = v.unsqueeze(0)
print("\nv shape:", v.shape, "-> unsqueeze(0) shape:", v_batched.shape)
print("squeeze back:", v_batched.squeeze(0).shape)

# broadcasting
a = torch.ones(4, 1, 3)
b = torch.ones(2, 3)
print("\nBroadcast (4,1,3) + (2,3) -> shape:", (a + b).shape)

p = torch.arange(8).reshape(8, 1)
q = torch.arange(5).reshape(1, 5)
print("Broadcast (8,1) + (1,5) -> shape:", (p + q).shape)