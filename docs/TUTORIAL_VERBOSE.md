# Tutorial: Implementing GPU Kernels for GLM-ASR with Triton (Verbose Teaching Edition)

A step-by-step guide to completing the GPU kernel implementations for the
GLM-ASR speech-to-text model using OpenAI Triton.

**Who this document is for:** You know Python and basic linear algebra (vectors,
matrices, dot products). You have never written a GPU kernel. By the end of this
tutorial you will understand how to write, test, and optimize custom GPU programs
using Triton.

---

## Prerequisites

- Python 3.11+
- NVIDIA GPU (Blackwell recommended, Hopper/Ampere also work)
- CUDA Toolkit 13.x
- PyTorch 2.10+
- Triton 3.6+

---

## 1. Environment Setup

### Option A: Using the setup script (recommended for cluster)
```bash
cd edin-mls-26-spring
source utils/setup-triton.sh
```

### Option B: Manual pip install
```bash
pip install torch triton numpy transformers datasets huggingface_hub safetensors accelerate soundfile
```

### Verify GPU access
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Verify the baseline works FIRST
```bash
cd hw1-asr
./benchmark.sh glm_asr_triton_example
```
You should see `Accuracy: 100.0%` and `Status: PASS`. If this fails, fix your
environment before writing any code.

---

## 2. Understanding the Project Structure

```
hw1-asr/
  glm_asr_triton_template/    <- YOUR WORK (fill in TODOs)
    layers.py                  <- 6 kernels to implement + layer classes
    attention.py               <- 3 kernels to implement
    rope.py                    <- 1 kernel to implement
    __init__.py                <- Configuration (backend, fusion flags)
    model.py                   <- DO NOT MODIFY (stock generate, no KV cache)
    conv.py                    <- DO NOT MODIFY
    weight_loader.py           <- DO NOT MODIFY

  glm_asr_triton_example/      <- REFERENCE (study this)
    layers.py                  <- Working implementations
    attention.py               <- Working implementations
    rope.py                    <- Working implementations
```

**Important:** Per GUIDE.md, you must NOT modify `model.py`, `weight_loader.py`, or `conv.py`.

**Key model.py facts (origin/main):**
- Encoder MLP uses plain `self.fc1(x) -> gelu(x) -> self.fc2(x)` -- NOT the `EncoderMLP` class
- Projector uses plain `self.linear_1(x) -> self.act(x) -> self.linear_2(x)` -- NOT `LinearGELU`
- Only has stock `generate()` -- O(n^2) decode, no KV cache
- `EncoderMLP` and `LinearGELU` classes exist in layers.py but are NOT used by model.py

---

## 3. Triton Kernel Basics

### 3.1 What Is a GPU Kernel, and Why Do We Need One?

Before we write a single line of Triton code, let us answer the foundational
question: **what is a GPU kernel?**

A normal Python function runs on the **CPU** (Central Processing Unit). The CPU
is very good at doing one thing at a time very fast, and it can handle complex
branching logic. When you write `for i in range(1000000): y[i] = x[i] * 2`, the
CPU processes those one million multiplications **one after another**,
sequentially.

A GPU (Graphics Processing Unit) is a completely different beast. It has
thousands of tiny cores, each individually slower than a CPU core, but working
**all at once**. A GPU kernel is a function that you write once, and the GPU runs
**thousands of copies simultaneously**, each operating on a different piece of
data.

> **Think of it this way:** Imagine you need to stamp 10,000 envelopes. The CPU
> approach is one very fast employee stamping them one by one. The GPU approach
> is hiring 10,000 slower employees and having them each stamp one envelope at
> the same time. For repetitive, uniform tasks, the GPU approach finishes much
> faster.

A **kernel** is the function that each of those "employees" executes. You write
it once, and the GPU hardware takes care of launching thousands of copies in
parallel.

### 3.2 Why Triton Instead of Raw CUDA?

NVIDIA's native GPU programming language is CUDA C/C++. It is extremely powerful
but also extremely verbose and error-prone. You have to manually manage thread
indices, shared memory allocation, memory alignment, warp divergence, and dozens
of other low-level details.

**Triton** is a Python-embedded language created by OpenAI that lets you write
GPU kernels in a Python-like syntax. Triton operates at the **block** level
rather than the **thread** level. Instead of thinking about individual threads,
you think about blocks of data. Triton's compiler handles the thread-level
details for you.

> **Think of it this way:** CUDA is like building a house brick by brick. Triton
> is like building with prefabricated wall panels -- you work at a higher level
> of abstraction, and the factory (compiler) handles the bricks.

### 3.3 The Anatomy of a Triton Kernel

Every Triton kernel follows this pattern:

```python
@triton.jit
def my_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    # 1. Get block ID
    pid = tl.program_id(0)

    # 2. Compute element offsets for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Create bounds mask
    mask = offs < N

    # 4. Load data
    x = tl.load(input_ptr + offs, mask=mask, other=0.0)

    # 5. Compute
    y = x * 2.0  # your operation here

    # 6. Store result
    tl.store(output_ptr + offs, y, mask=mask)
```

Let us walk through every line in detail.

#### Line 1: `@triton.jit`

This decorator tells Triton: "This is a GPU kernel, not a normal Python
function. Compile it for the GPU." When Python encounters this decorator, it
does not run the function on the CPU. Instead, Triton's just-in-time (JIT)
compiler translates the function body into GPU machine code the first time it is
called.

#### Line 2: `def my_kernel(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr)`

The function parameters define the kernel's interface:

- **`input_ptr` and `output_ptr`**: These are **pointers** -- memory addresses
  on the GPU. When you pass a PyTorch tensor to a kernel, you actually pass
  `tensor.data_ptr()`, which is the address of the tensor's first element in GPU
  memory. Think of it as a street address; the kernel uses this address plus an
  offset to find each specific element.

- **`N`**: The total number of elements. This is a plain integer. The kernel
  needs this to know when to stop (so it does not read past the end of the
  array).

- **`BLOCK_SIZE: tl.constexpr`**: This is a **compile-time constant**. The
  `tl.constexpr` annotation tells Triton that this value is fixed before the
  kernel runs. Triton uses it to decide how much memory to allocate, how to
  unroll loops, and how to optimize register usage. Common values are powers of
  2 like 128, 256, 512, or 1024.

#### Line 3: `pid = tl.program_id(0)`

This is one of the most important concepts.

**`tl.program_id(0)`** answers the question: **"Which block am I?"**

When you launch a kernel, you tell the GPU to run many copies of it in parallel.
Each copy is called a **program instance** (or "block" in CUDA terminology). The
GPU assigns each instance a unique ID number, starting from 0.

The `0` argument means "give me the ID along axis 0." For 1D problems (like
element-wise operations on a flat array), you only need axis 0. For 2D problems
(like matrix multiplication), you also use `tl.program_id(1)` for the second
axis.

> **Think of it this way:** Imagine a long assembly line with numbered stations:
> Station 0, Station 1, Station 2, and so on. Each station processes one chunk
> of the input. `tl.program_id(0)` is each station looking at its own number
> plate to figure out "I am station 7, so I process chunk 7."

#### Line 4: `offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`

This line computes which specific elements this block is responsible for.

**`tl.arange(0, BLOCK_SIZE)`** creates a vector of consecutive integers:
`[0, 1, 2, ..., BLOCK_SIZE-1]`. This is the **within-block offset** -- it
answers "which elements within my block do I handle?"

By multiplying `pid * BLOCK_SIZE` and adding the range, you get the global
element indices. For example, if `BLOCK_SIZE=4` and `pid=2`:

```
offs = 2 * 4 + [0, 1, 2, 3] = [8, 9, 10, 11]
```

So block 2 processes elements 8, 9, 10, and 11.

> **Think of it this way:** `tl.program_id()` is "which page of the book am I
> reading?" and `tl.arange()` is "which lines on that page?" Together they give
> you the exact line numbers in the whole book.

#### Line 5: `mask = offs < N`

This is **masking**, and it is crucial for correctness.

The problem: data arrays rarely divide evenly by your BLOCK_SIZE. If you have
`N=10` elements and `BLOCK_SIZE=4`, you need 3 blocks:
- Block 0: elements [0, 1, 2, 3] -- all valid
- Block 1: elements [4, 5, 6, 7] -- all valid
- Block 2: elements [8, 9, 10, 11] -- elements 10 and 11 DO NOT EXIST

Without a mask, block 2 would try to read from memory addresses that belong to
some other data (or are completely unallocated), causing garbage results or
crashes.

The mask is a boolean vector. For block 2 in our example:
```
offs  = [8, 9, 10, 11]
N     = 10
mask  = [True, True, False, False]
```

Every subsequent `tl.load` and `tl.store` uses this mask to skip the invalid
elements.

> **Think of it this way:** You are reading a book that has 10 lines per page,
> but the last page only has 6 lines. The mask is you checking "is there
> actually a line here, or am I past the end of the text?" before trying to read.

#### Line 6: `x = tl.load(input_ptr + offs, mask=mask, other=0.0)`

This loads data from GPU global memory into registers.

- **`input_ptr + offs`**: Pointer arithmetic. `input_ptr` is the base address;
  adding `offs` gives you the addresses of the specific elements this block
  needs.
- **`mask=mask`**: Only load elements where the mask is True.
- **`other=0.0`**: For masked-out positions, use 0.0 instead of whatever garbage
  is in memory. This is important because even masked-out values will
  participate in vector operations (Triton processes the whole block); they just
  need to be harmless values.

#### Line 7: `y = x * 2.0`

This is where your actual computation happens. In this toy example, we double
every element. In real kernels, this could be a normalization formula, an
activation function, a dot product, or anything else.

The key insight: this single line operates on the **entire block at once**. If
BLOCK_SIZE is 256, then `x * 2.0` multiplies 256 elements simultaneously. This
is how GPU parallelism works in Triton -- you write scalar-looking code, but it
executes on vectors.

#### Line 8: `tl.store(output_ptr + offs, y, mask=mask)`

This writes results back to GPU global memory. Same pointer arithmetic and
masking as the load. Only valid positions get written; invalid ones are skipped.

### 3.4 How a Kernel Gets Launched

On the Python (CPU) side, you launch a kernel by specifying a **grid** -- how
many blocks to create:

```python
grid = (triton.cdiv(N, BLOCK_SIZE),)  # ceiling division
my_kernel[grid](input_ptr, output_ptr, N, BLOCK_SIZE=256)
```

`triton.cdiv(N, BLOCK_SIZE)` computes the ceiling of N/BLOCK_SIZE. If N=1000
and BLOCK_SIZE=256, the grid is (4,), meaning 4 blocks: blocks 0-3 will process
elements 0-255, 256-511, 512-767, and 768-1023 respectively. Block 3 will have
its mask set to False for elements 1000-1023.

### 3.5 Summary of Key Concepts

| Concept | What it answers | Analogy |
|---------|----------------|---------|
| `tl.program_id(axis)` | Which block am I? | Which station on the assembly line? |
| `tl.arange(0, N)` | Which elements within this block? | Which items at my station? |
| mask = `offs < N` | Is this element valid? | Am I past the end of the data? |
| `tl.load/tl.store` | Read/write GPU memory | Fetching/returning items from the warehouse |
| `tl.constexpr` | Compile-time constant | A setting that is fixed before the factory starts |

---

## 4. Implementing Each Kernel (Recommended Order)

### Phase 1: Element-wise Operations

Element-wise operations are the simplest class of GPU kernels. Each output
element depends only on the corresponding input element (no neighbors, no
reductions across elements). This makes them perfectly parallel and the ideal
starting point for learning.

#### 4.1 SiLU Kernel (simplest -- start here)

##### What problem does SiLU solve?

Neural networks need **activation functions** -- nonlinear transformations
applied between linear layers. Without them, stacking multiple linear layers
would just produce another linear layer (a matrix times a matrix is still a
matrix). Activation functions introduce the nonlinearity that allows neural
networks to learn complex patterns.

SiLU (Sigmoid Linear Unit), also called "Swish," is a modern activation
function used in the decoder part of this model. Its formula is:

```
y = x * sigmoid(x) = x / (1 + exp(-x))
```

For positive x, SiLU behaves like a slightly smoothed version of ReLU. For
negative x, it allows a small negative signal through (unlike ReLU, which clips
everything negative to zero). This helps with gradient flow during training.

##### Why do we need a GPU kernel for this?

You could just write `y = x * torch.sigmoid(x)` in PyTorch. That would work,
but it would launch **two separate GPU operations**: one for the sigmoid and one
for the element-wise multiply. Each operation reads data from GPU memory and
writes it back. Memory access is slow compared to computation.

A custom kernel does both operations in a **single pass**: read x once, compute
sigmoid and multiply, write y once. This cuts memory traffic in half.

> **Think of it this way:** Imagine you are a librarian. If someone asks you to
> first highlight all the nouns in a book, then separately underline all the
> highlighted words, you would walk to the shelf twice and scan every page twice.
> A fused approach is doing both in one pass through the book.

##### How does the GPU parallelism work?

SiLU is element-wise: element 0 of the output depends only on element 0 of the
input, element 1 depends only on element 1, and so on. There are zero
dependencies between elements, so all of them can be computed simultaneously.

We divide the input into blocks of BLOCK_SIZE elements. Each GPU block processes
one chunk. If the input has 100,000 elements and BLOCK_SIZE=1024, we launch
about 98 blocks that all run in parallel.

##### Line-by-line walkthrough

```python
@triton.jit
def silu_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid
    tl.store(y_ptr + offs, y, mask=mask)
```

- **`pid = tl.program_id(0)`**: Get this block's ID.
- **`offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)`**: Compute the global
  indices of the elements this block handles.
- **`mask = offs < n_elements`**: Create a boolean mask so we do not read/write
  past the end of the array.
- **`x = tl.load(...).to(tl.float32)`**: Load this block's chunk of input data,
  and **cast to float32**. The `.to(tl.float32)` is important: the input might
  be stored in float16 (half precision) to save memory, but `exp()` needs the
  extra precision of float32 to avoid overflow/underflow. This is a pattern you
  will see in every kernel.
- **`sigmoid = 1.0 / (1.0 + tl.exp(-x))`**: The sigmoid function. `tl.exp` is
  the GPU's hardware exponential instruction.
- **`y = x * sigmoid`**: The SiLU formula: multiply input by its own sigmoid.
- **`tl.store(y_ptr + offs, y, mask=mask)`**: Write the result back. Only valid
  positions (where mask is True) are written.

**Grid:** `(ceil(n_elements / BLOCK_SIZE),)` -- one block per chunk of elements.
One-dimensional grid because this is a flat, element-wise operation.

#### 4.2 GELU Kernel

##### What problem does GELU solve?

GELU (Gaussian Error Linear Unit) is another activation function, used in the
encoder part of this model. It is defined mathematically as:

```
GELU(x) = x * Phi(x)
```

where Phi(x) is the cumulative distribution function of the standard normal
distribution. In practice, computing the exact CDF is expensive, so we use the
**tanh approximation**:

```
y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

This looks complicated, but it is still an element-wise operation: each output
depends only on the corresponding input.

##### Why a separate kernel from SiLU?

SiLU uses `x * sigmoid(x)`. GELU uses a completely different formula involving
tanh and a cubic term. They serve similar purposes (activation functions) but
have different mathematical formulations and are used in different parts of the
model. The encoder uses GELU; the decoder uses SiLU.

##### Line-by-line walkthrough

```python
sqrt_2_over_pi = 0.7978845608028654
x3 = x * x * x
inner = sqrt_2_over_pi * (x + 0.044715 * x3)
y = x * 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
```

- **`sqrt_2_over_pi = 0.7978845608028654`**: A precomputed constant. Computing
  `sqrt(2/pi)` every time would waste cycles.
- **`x3 = x * x * x`**: The cubic term. Note we write `x * x * x` rather than
  `x ** 3` because explicit multiplication is more efficient on GPU hardware.
- **`inner = sqrt_2_over_pi * (x + 0.044715 * x3)`**: The argument to tanh.
  The magic number 0.044715 makes this approximation accurate to within 0.01%
  of the true GELU.
- **`tl.extra.cuda.libdevice.tanh(inner)`**: The `tanh` function. Unlike `exp`
  which Triton provides directly, `tanh` comes from NVIDIA's `libdevice` math
  library. This is just an implementation detail -- it works exactly like you
  would expect.
- **`y = x * 0.5 * (1.0 + ...)`**: Combining everything with the outer
  multiplication.

**Grid:** `(ceil(n_elements / BLOCK_SIZE),)` -- same pattern as SiLU. Element-
wise means 1D grid.

### Phase 2: Reductions

Reductions are a step up in complexity. Instead of each output depending on
exactly one input, a reduction combines **many** inputs into fewer outputs. For
example, computing the mean of a row requires reading all elements in that row.

#### 4.3 RMSNorm Kernel

##### What problem does RMSNorm solve?

Neural networks suffer from a problem called **internal covariate shift**: as
data flows through many layers, the distribution of values (their scale and
mean) can drift wildly, making training unstable and inference noisy.

**Normalization layers** fix this by rescaling each hidden-state vector to have
a consistent magnitude. RMSNorm (Root Mean Square Normalization) is a
lightweight normalization used in the **text decoder** part of this model:

```
y = x / sqrt(mean(x^2) + eps) * weight
```

It computes the root-mean-square of the input vector, divides by it (to get
unit magnitude), and then scales by a learned weight vector.

##### Why does this need a GPU kernel?

RMSNorm involves a **reduction**: computing `mean(x^2)` requires summing over
all elements in a row. In PyTorch, this would be multiple operations: square,
mean, rsqrt, multiply, multiply. Each operation is a separate GPU kernel
launch with its own memory read/write cycle. A fused Triton kernel does the
entire normalization in one pass.

##### How does the GPU parallelism work?

The input is a 2D matrix of shape `(num_rows, hidden_size)`. Each row is an
independent vector that gets normalized independently. So we launch one block
per row -- all rows are processed in parallel.

Within each block, we need to sum across all elements in the row (a reduction).
Triton's `tl.sum()` handles this efficiently using parallel tree reduction
internally.

> **Think of it this way:** You have a spreadsheet with 1000 rows, each with
> 512 numbers. You need to normalize each row by its own statistics. You hire
> 1000 assistants; each one takes one row, computes the RMS of that row, and
> divides every number in the row by it. They all work simultaneously.

##### Line-by-line walkthrough

```python
@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y,
                   hidden_size, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Which row
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < hidden_size

    x = tl.load(x_ptr + pid * stride_x + offs, mask=mask, other=0.0)
    x = x.to(tl.float32)

    var = tl.sum(x * x, axis=0) / hidden_size
    x_norm = x * tl.rsqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = x_norm * w
    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)
```

- **`pid = tl.program_id(0)`**: Here, pid is the **row index**. Each block
  processes one entire row.
- **`offs = tl.arange(0, BLOCK_SIZE)`**: Notice this is NOT `pid * BLOCK_SIZE +
  tl.arange(...)`. That is because we are not splitting a single 1D array; we
  are processing one row per block. `offs` is just the column indices within
  the row.
- **`x = tl.load(x_ptr + pid * stride_x + offs, ...)`**: Load row `pid`.
  `stride_x` is the number of elements between consecutive rows in memory. For
  a contiguous 2D tensor of shape `(rows, cols)`, stride_x equals `cols`.
- **`x = x.to(tl.float32)`**: Cast to float32 for numerical stability in the
  reduction. Summing many squared values in float16 would overflow quickly.
- **`var = tl.sum(x * x, axis=0) / hidden_size`**: This is the key reduction
  step. `x * x` squares every element (element-wise). `tl.sum(..., axis=0)`
  adds them all up into a single scalar. Dividing by `hidden_size` gives the
  mean of squares.
- **`x_norm = x * tl.rsqrt(var + eps)`**: `tl.rsqrt` computes 1/sqrt(x). The
  `eps` (a tiny number like 1e-6) prevents division by zero when the input is
  all zeros. We multiply by rsqrt instead of dividing by sqrt because
  multiplication is faster than division on GPU hardware.
- **`w = tl.load(w_ptr + offs, ...)`**: Load the learned weight vector. This is
  shared across all rows (same weight applies to every token).
- **`y = x_norm * w`**: Apply the learned scale.
- **`tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)`**: Write the
  normalized row back.

**Grid:** `(num_rows,)` -- one block per row.

#### 4.4 LayerNorm Kernel

##### What problem does LayerNorm solve?

LayerNorm (Layer Normalization) is used in the **audio encoder** part of the
model. It is the "full" version of normalization, compared to the "lite" RMSNorm:

```
y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
```

##### How does it differ from RMSNorm?

Two differences:

1. **Mean subtraction (centering):** LayerNorm first subtracts the mean of each
   row, then computes the variance. RMSNorm skips the centering and just uses
   the root mean square. This means LayerNorm has two reductions (one for the
   mean, one for the variance), while RMSNorm has only one.

2. **Bias:** LayerNorm adds a learned bias term after scaling, while RMSNorm
   only scales. This adds one extra parameter vector and one extra element-wise
   addition.

The kernel structure is the same as RMSNorm: one block per row, load the row,
do reductions, normalize, store. You just need to add the mean computation and
bias application.

#### 4.5 Softmax Kernel

##### What problem does softmax solve?

Softmax converts a vector of arbitrary real numbers into a probability
distribution (all values between 0 and 1, summing to 1):

```
softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
```

It is used in the attention mechanism to turn raw attention scores into
attention weights.

##### Why do we need numerical stability?

The naive formula `exp(x_i) / sum(exp(x_j))` has a fatal flaw. If any `x_i` is
large (say 1000), then `exp(1000)` is astronomically huge -- it overflows to
infinity in any floating-point representation. The computation produces `inf /
inf = NaN` (Not a Number), and your entire model output is corrupted.

The fix is the **numerically stable softmax**:

```
y = exp(x - max(x)) / sum(exp(x - max(x)))
```

By subtracting the maximum value first, the largest exponent becomes `exp(0) =
1`, and all others are `exp(negative) < 1`. This guarantees no overflow.
Mathematically, the result is identical (shifting all exponents by a constant
does not change the ratios).

> **Think of it this way:** If you need to compare buildings by height, you
> could measure each one from sea level (huge numbers, hard to work with) or
> you could measure them relative to the tallest building (all numbers are zero
> or negative, easy to work with). The relative heights are the same either way.

##### How does the GPU parallelism work?

Like RMSNorm and LayerNorm, softmax operates on rows independently. One block
per row. Within each row, there are two reductions: `max(x)` and `sum(exp(x -
max(x)))`.

### Phase 3: Tiled Matrix Multiplication

#### 4.6 Linear (Matmul) Kernel

##### What problem does this solve?

Matrix multiplication (`C = A @ B`) is the single most important operation in
deep learning. Every linear layer, every attention score computation, every
projection -- they are all matrix multiplications. In this model, the linear
kernel is called hundreds of times during inference.

##### Why do we need a special kernel for matrix multiplication?

Matrix multiplication is deceptively expensive. Multiplying an `(M x K)` matrix
by a `(K x N)` matrix requires `M * N * K` multiply-add operations. For the
sizes in this model (M and N in the thousands, K in the hundreds), that is
billions of operations.

The naive triple-loop algorithm is catastrophically slow even on a GPU because
of **memory access patterns**. To understand why, we need to understand **tiling**.

##### What is tiling and why does it matter?

GPU memory comes in two flavors:

1. **Global memory** (HBM -- High Bandwidth Memory): Large (tens of gigabytes),
   but relatively slow to access. Every `tl.load` and `tl.store` goes to global
   memory.
2. **Shared memory** (SRAM): Tiny (about 100-230 KB per block depending on GPU),
   but extremely fast -- roughly 10-100x faster than global memory.

> **Think of it this way:** Global memory is like a library across campus. It
> has every book you could ever need, but walking there takes 5 minutes each
> way. Shared memory is like a small fast notebook on your desk. It only holds a
> few pages, but you can read it instantly. Registers (inside the compute unit)
> are like your short-term memory -- the absolute fastest, but the most limited.

The naive matrix multiply algorithm computes each output element by walking
through an entire row of A and column of B. If A is 4096x4096, that means each
output element loads 4096 values from global memory. And you have 4096 x 4096 =
16 million output elements, each loading 4096 values. That is 67 billion global
memory reads.

**Tiling** fixes this by dividing the output matrix into small blocks (tiles)
and loading just the data those tiles need into fast on-chip storage. The
algorithm:

1. Divide the output C into tiles of size `(BLOCK_M x BLOCK_N)`.
2. Each GPU block is responsible for one output tile.
3. To compute its tile, the block iterates over the K dimension in chunks of
   `BLOCK_K`:
   - Load a tile of A: shape `(BLOCK_M x BLOCK_K)` from global memory
   - Load a tile of B: shape `(BLOCK_K x BLOCK_N)` from global memory
   - Multiply them with `tl.dot()` and **accumulate** into the result
4. After all K chunks are processed, write the accumulated result to global memory.

This drastically reduces global memory reads because each tile of A and B is
loaded once and reused for all the output elements in that tile.

> **Think of it this way:** Instead of individually fetching each ingredient
> from the pantry every time a recipe needs it, you bring a small batch of
> ingredients to your prep station and use them for multiple dishes at once
> before going back for the next batch. The fewer trips to the pantry, the
> faster you cook.

##### What is `tl.dot()` and why is it special?

`tl.dot(a, b)` is not just a regular multiply-and-add. On modern NVIDIA GPUs,
it compiles to **tensor core** instructions (HMMA/WMMA). Tensor cores are
specialized hardware units that can multiply small matrices (like 16x16) in a
single clock cycle. This gives roughly 10x the throughput of regular FP32
multiply-add. Using `tl.dot` instead of manual element-wise multiplication is
the difference between "fast" and "really fast."

**Grid:** `(ceil(M/BLOCK_M), ceil(N/BLOCK_N))` -- a **2D grid** of output
tiles. This is the first kernel that uses a 2D grid. Each block's position in
the grid corresponds to which tile of the output matrix it computes.

### Phase 4: Attention Kernels

#### 4.7-4.9 Legacy Attention Kernels (REMOVED)

The original assignment had three separate attention kernels:
- **Attention Scores**: `Q @ K^T * scale` per query position
- **Softmax In-Place**: writes softmax back to input buffer
- **Attention Output**: `attn_weights @ V` weighted sum

These were **removed** from the codebase (Session 7, ~175 lines) -- they were fully
superseded by the fused Flash Attention kernel and never invoked at runtime.

#### 4.10 Fused Flash Attention Kernel (Advanced)

##### What problem does Flash Attention solve?

Standard attention computes:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

In the naive implementation, you:
1. Compute the full `Q @ K^T` matrix (size: `seq_q x seq_kv`)
2. Apply softmax to every row
3. Multiply the result by V

The problem: step 1 creates a **massive intermediate matrix**. If your sequence
length is 4096, that is a 4096 x 4096 matrix = 16 million elements, just for
attention scores. This matrix must be written to global memory and then read back
for the softmax and the final multiplication. For long sequences, this becomes
the bottleneck.

Flash Attention **fuses** all three steps into a single kernel that never
materializes the full attention matrix. Instead, it processes the K and V
matrices in blocks, maintaining a running softmax computation.

##### What is online softmax and why is it better?

Normal softmax needs two passes over the data:
1. First pass: find the maximum value (for numerical stability)
2. Second pass: compute `exp(x - max)` and the sum, then divide

This means you need to store all the intermediate values between passes. For
attention, that means storing the entire `seq_q x seq_kv` score matrix.

**Online softmax** (also called "streaming softmax") computes the softmax
incrementally as new data arrives, using a clever rescaling trick. It maintains
three running values:
- `m_i`: the maximum value seen so far
- `l_i`: the sum of exponentials seen so far (the denominator)
- `acc`: the accumulated weighted sum (the numerator of the final result)

When a new block of K/V arrives, it updates all three values, **rescaling** the
previous accumulations to account for the new maximum.

> **Think of it this way:** Imagine you are computing a weighted average of
> exam scores as students turn in their papers one by one. With normal softmax,
> you would wait for ALL papers, find the best score, then grade relative to
> that. With online softmax, you start grading immediately, and every time a
> new best score arrives, you adjust all your previous grades proportionally.
> You get the same final answer but never need to hold all papers at once.

##### Line-by-line walkthrough of the inner loop

```python
# Inner loop: iterate over K/V blocks
for start_n in range(0, kv_len, BLOCK_N):
    k = tl.load(...)                          # K block [BLOCK_N, BLOCK_D]
    s = tl.dot(q, tl.trans(k))                # Q @ K^T [BLOCK_M, BLOCK_N]
    # Apply causal/attention masks if needed
    m_ij = tl.max(s, axis=1)                  # Block max
    m_new = tl.maximum(m_i, m_ij)             # Running max update
    alpha = tl.exp(m_i - m_new)               # Rescale factor
    p = tl.exp(s - m_new[:, None])            # Attention weights
    l_i = alpha * l_i + tl.sum(p, axis=1)     # Running sum
    acc = alpha[:, None] * acc                 # Rescale accumulator
    v = tl.load(...)                           # V block
    acc += tl.dot(p.to(v.dtype), v)           # Accumulate P @ V
    m_i = m_new
acc = acc / l_i[:, None]                      # Final normalization
```

Let us trace through this carefully:

- **`k = tl.load(...)`**: Load a block of K values. Shape is
  `[BLOCK_N, BLOCK_D]` -- BLOCK_N key vectors, each of dimension BLOCK_D.
  We iterate over the key/value sequence in chunks of BLOCK_N.

- **`s = tl.dot(q, tl.trans(k))`**: Compute attention scores for this block.
  `q` has shape `[BLOCK_M, BLOCK_D]` (BLOCK_M query vectors), `tl.trans(k)`
  has shape `[BLOCK_D, BLOCK_N]`, so the result `s` has shape
  `[BLOCK_M, BLOCK_N]`. Each entry `s[i, j]` is the dot product of query i
  with key j -- how much query i "attends to" key j.

- **`m_ij = tl.max(s, axis=1)`**: Find the maximum score in each row of this
  block. Shape `[BLOCK_M]`. This is needed for numerical stability.

- **`m_new = tl.maximum(m_i, m_ij)`**: Update the running maximum. `m_i` is
  the max from all previous blocks; `m_new` is the max including this block.

- **`alpha = tl.exp(m_i - m_new)`**: This is the magic rescaling factor. If the
  new block has a higher max than before, `m_i - m_new` is negative, and alpha
  is less than 1, which scales down the previous accumulations. If the max
  did not change, alpha is `exp(0) = 1` and nothing is rescaled.

- **`p = tl.exp(s - m_new[:, None])`**: Compute the (unnormalized) attention
  weights for this block, using the updated maximum. The `[:, None]` broadcasts
  the per-row max across columns.

- **`l_i = alpha * l_i + tl.sum(p, axis=1)`**: Update the running sum. The old
  sum is rescaled by alpha (to account for the potentially new maximum), then
  the new weights are added.

- **`acc = alpha[:, None] * acc`**: Rescale the accumulated output. Same logic:
  if the maximum changed, all previous weighted contributions need to be
  adjusted.

- **`acc += tl.dot(p.to(v.dtype), v)`**: Add this block's contribution:
  attention weights times V values. `p.to(v.dtype)` ensures the types match
  for `tl.dot`.

- **`m_i = m_new`**: Update the running max for the next iteration.

After the loop:
- **`acc = acc / l_i[:, None]`**: Final normalization. The accumulated sum is
  divided by the total sum of weights to get the true weighted average.

### Phase 5: Positional Encoding

#### 4.11 RoPE Frequency Kernel

##### What problem does RoPE solve?

Transformers have no inherent notion of position. If you shuffle all the tokens
in a sentence, the attention mechanism would produce the same scores (because
it only looks at dot products between token vectors, not their positions).

RoPE (Rotary Position Embedding) injects position information by **rotating**
each token's vector by an angle proportional to its position. Tokens at
position 0 get rotated by 0 degrees, position 1 by a small angle, position 2
by a larger angle, and so on. Different dimensions of the vector are rotated at
different frequencies, creating a rich position signal.

The kernel precomputes `cos(position * inv_freq)` and `sin(position * inv_freq)`
for all positions and frequencies. The output is duplicated into both halves
because `apply_rotary_pos_emb` splits the input and applies the same
frequencies to each half.

---

## 5. Testing Your Implementation

### Unit tests (test individual kernels):
```bash
cd hw1-asr/glm_asr_triton_template
python layers.py        # Tests RMSNorm, LayerNorm, GELU, SiLU, Linear, Softmax, MLP
python attention.py     # 17-case numerical parity suite for Flash Attention
python rope.py          # Tests RoPE frequency computation
```

### End-to-end benchmark:
```bash
cd hw1-asr
python benchmark_student.py glm_asr_triton_template --warmup 2 --runs 5

# Compare against baseline
python benchmark_student.py glm_asr_triton_example --warmup 1 --runs 3
```

Expected output: `Concord returned to its place amidst the tents.`

---

## 6. Optimization Strategies

### Preface: Why Optimization Matters and How to Think About It

Before diving into specific optimizations, it is essential to understand
**why** GPU programs are slow and what levers you can pull to speed them up.

There are three fundamental physical resources that limit GPU performance:

1. **Memory bandwidth**: How fast you can move data between GPU memory (HBM)
   and the compute units. Measured in GB/s or TB/s. Modern GPUs like the
   RTX 5090 can compute much faster than they can feed data to the compute
   units. This means many GPU programs are **memory-bandwidth bound**: they
   spend more time waiting for data to arrive than actually computing.

2. **Compute throughput**: How many arithmetic operations the GPU can perform
   per second. Measured in TFLOPS (trillions of floating-point operations per
   second). This only matters for compute-heavy workloads like large matrix
   multiplications.

3. **Latency**: The fixed overhead of starting a GPU operation. Each kernel
   launch takes microseconds of setup time, regardless of how much work the
   kernel does. If you launch many tiny kernels, the launch overhead can
   dominate the actual computation time.

> **Think of it this way:** Imagine a factory with three potential bottlenecks.
> The delivery trucks (memory bandwidth) can only bring in so many parts per
> hour. The assembly machines (compute throughput) can only build so many
> products per hour. And starting up a machine (latency) takes 10 minutes no
> matter what you are building. Optimization is figuring out which bottleneck
> is limiting you and addressing it.

Most of the optimizations below fall into one of these categories:
- **Reduce memory traffic**: fp16 (half the bytes), kernel fusion (fewer
  round-trips to memory)
- **Increase compute utilization**: tensor cores, appropriate tile sizes
- **Reduce latency**: fewer kernel launches, KV caching

### 6.1 Backend Selection

```python
# In __init__.py:
layers.Linear.BACKEND = "torch"   # cuBLAS -- fastest on RTX 5090
layers.Linear.BACKEND = "triton"  # strict Triton kernel path
```

**Why cuBLAS is faster:** cuBLAS is NVIDIA's hand-tuned matrix multiplication
library. NVIDIA engineers have spent years optimizing it for every GPU
architecture, using assembly-level tricks that Triton's compiler cannot match.
For large matrix multiplications, cuBLAS is typically 10-30% faster than a
well-written Triton kernel.

**When to use Triton:** The assignment requires you to implement a Triton matmul
kernel. For grading, you may need to use `"triton"` mode to demonstrate your
kernel works. For final performance benchmarks, switch to `"torch"` (cuBLAS).

### 6.2 Runtime Flags

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

**What is TF32?** TF32 (TensorFloat-32) is a number format that uses the range
of float32 (8 exponent bits) but the precision of float16 (10 mantissa bits).
This lets tensor cores process "float32" multiplications at near-float16 speed,
with only a tiny precision loss (undetectable in neural network inference).

**What does `cudnn.benchmark = True` do?** cuDNN (CUDA Deep Neural Network
library) has multiple algorithms for operations like convolution. When this flag
is True, the first time a new input shape is encountered, cuDNN benchmarks all
available algorithms and picks the fastest one. It caches this choice for future
calls with the same shape.

### 6.3 Kernel Fusion

```python
layers.MLP.FUSED = True           # Fused SwiGLU in decoder MLP -- EFFECTIVE
layers.EncoderMLP.FUSED = True    # NOT USED by origin/main model.py
# LinearGELU.FUSED = False        # NOT USED by origin/main model.py
```

**What is kernel fusion?** Instead of running multiple separate kernels
(each reading from and writing to global memory), a fused kernel combines
multiple operations into one.

**Why does fusion help?** Physics. Each kernel launch involves:
1. Reading input from global memory (slow)
2. Computing (fast)
3. Writing output to global memory (slow)
4. Next kernel reads that output from global memory (slow again)

A fused kernel eliminates steps 3 and 4 between operations: the intermediate
result stays in registers or shared memory (fast), and only the final result
is written to global memory.

> **Think of it this way:** Unfused kernels are like a relay race where runners
> must drop the baton in a locker (slow) and the next runner picks it up from
> the locker. Fused kernels let runners pass the baton hand-to-hand (fast)
> without the locker detour.

**Important:** Only `MLP.FUSED` actually affects performance. The origin/main `model.py`
does NOT use `EncoderMLP` or `LinearGELU` -- it uses plain `Linear` + `gelu()` for the
encoder MLP and projector.

### 6.4 fp16 Weights (formerly bfloat16)

```python
Linear.BF16 = True                     # Class default in layers.py (flag name retained)
Linear._HALF_DTYPE = torch.float16     # Actual dtype: fp16 HGEMM on RTX 5090
```

##### What is fp16?

**fp16** (half-precision floating point) uses 16 bits per number instead of the
usual 32 bits (float32). This means:
- Each number takes **half the memory** (2 bytes instead of 4 bytes)
- Moving numbers between memory and compute takes **half the time**
- Tensor core instructions specifically optimized for fp16 (HGEMM) are faster

The tradeoff is precision: fp16 has about 3-4 decimal digits of precision
versus float32's 7-8 digits. For neural network inference, this precision loss
is imperceptible in the final output.

##### Why does half the bytes equal faster?

Because most operations in this model are **memory-bandwidth bound**, not
compute-bound. The GPU can compute faster than it can read data from memory.
Halving the data size means the memory system can deliver data twice as fast,
which directly translates to nearly 2x speedup for bandwidth-bound operations.

> **Think of it this way:** Imagine you are reading a book aloud, and you read
> at a fixed speed. If someone gives you the same book printed in half-sized
> font on half as many pages, you can flip through it twice as fast. The
> information is the same; the medium is just more compact.

Caches fp16 copies of weights. Must be set as class default (not just `__init__.py`)
because `__init__.py` is not always executed during benchmark imports.
Output stays fp16 -- the `.float()` conversion was removed from `Linear._forward_torch()`,
so fp16 cascades through the entire pipeline. Triton kernels handle float32 precision
internally via `.to(tl.float32)` after loading.

### 6.5 Fused Flash Attention (GPU-Tier Aware)

Single Triton kernel with online softmax. Tile sizes set by GPU tier:

**Consumer GPUs** (RTX 4090/5090, ~100KB shared mem):
- `head_dim=64` (encoder): `BLOCK_M=64, BLOCK_N=64`, `num_stages=1, num_warps=4`
- `head_dim=128` (decoder): `BLOCK_M=32, BLOCK_N=32`, `num_stages=1, num_warps=4`
- `seq_q <= 16`: `BLOCK_M=16` (optimized for KV-cached decode)

**Datacenter GPUs** (H200/B200, ~228KB shared mem):
- Larger tiles, `num_stages=2, num_warps=8`

**Why do different GPUs need different tile sizes?** Tile sizes are limited by
shared memory. The Flash Attention kernel needs to hold tiles of Q, K, and V in
shared memory simultaneously. Larger tiles mean fewer iterations (less loop
overhead) and better data reuse, but they require more shared memory.

Consumer GPUs have about 100KB of shared memory per block. Datacenter GPUs have
about 228KB. If your tiles are too large for the available shared memory, the
kernel fails with an `OutOfResources` error.

> **Think of it this way:** Tile size is like the size of your desk. A bigger
> desk lets you spread out more papers and work more efficiently. But the desk
> has to fit in your office. A datacenter GPU has a bigger office.

For **KV-cached decode steps** (seq_q <= 4), SDPA fallback is used instead of the
Triton kernel -- avoids kernel launch overhead for tiny problems (-3ms).

### 6.6 fp16 Kernel Outputs (formerly bf16)

```python
# RMSNorm fp16 output -- avoids fp32 -> fp16 cast in next Linear
rmsnorm_bf16_kernel(...)  # stores y as float16 (via .to(tl.float16))

# LayerNorm fp16 output -- same approach for encoder norm
layernorm_kernel(...)     # stores y as float16 when Linear.BF16=True
```

**Why does this help?** If the next operation after normalization is a linear
layer that expects fp16 input, and our normalization kernel outputs fp32,
PyTorch has to insert an extra dtype conversion step. That conversion reads
every element from memory and writes it back in a different format -- pure
waste. By having the normalization kernel output fp16 directly, we skip that
entirely.

### 6.10 fp16-Throughout Pipeline (Session 10)

The biggest single optimization was eliminating unnecessary dtype conversions
between operations. Triton kernels already compute in float32 internally
(`.to(tl.float32)` after loading), so Python-side float32 casts in wrapper
functions are redundant.

**Changes (cumulative -11.5ms):**
1. **Remove Linear `.float()`** -- output stays fp16, cascades through pipeline (**-7.5ms**)
2. **Remove silu/gelu float32 cast** -- kernels handle internally (**-3.7ms**)
3. **Remove RMSNorm/LayerNorm float32 cast** -- same reasoning (~-0.5ms)
4. **fp16 embedding output** -- decoder pipeline starts in fp16
5. **fp16 fused SwiGLU/EncoderMLP** -- halves intermediate allocations
6. **Remove flash attention float32 conversion** -- pass fp16 to kernel (~-1ms)

##### Why does this work without losing precision?

This is a subtle but important point. You might worry: "If everything is fp16,
do we not lose precision in our computations?"

The answer is no, because of how Triton kernels work internally. Every kernel
follows this pattern:

1. **Load** data from memory (might be fp16)
2. **Cast** to float32: `.to(tl.float32)`
3. **Compute** in float32 (full precision)
4. **Store** result (auto-converts back to the output tensor's dtype)

So the actual arithmetic always happens in float32. The fp16 is only used for
**storage and transport** between kernels. Since the bottleneck is memory
bandwidth (how fast data moves between memory and compute), making the stored
data smaller speeds everything up without sacrificing compute precision.

> **Think of it this way:** You can write your calculations on a full-sized
> whiteboard (float32 compute) while shipping results between rooms on index
> cards (fp16 storage). The work is done at full size; the shipping is compact.

### 6.7 GPU Portability: GPUProfile + Dynamic Tiles

##### What is GPUProfile and why do we need it?

Different GPU models have different amounts of shared memory, different numbers
of compute units (SMs), and different architectural features. A tile size that
is optimal on an RTX 5090 might crash with an out-of-memory error on an older
RTX 3080, or be suboptimally small on a datacenter H200.

**GPUProfile** is a class that detects the GPU architecture at import time and
looks up optimal tile sizes from a table of tested configurations.

##### How does it work?

```python
# layers.py -- 3-tier GPU portability system

# 1. Known configs for tested GPUs (fastest path)
_KNOWN_CONFIGS = {
    "blackwell_consumer": {  # RTX 5090 (sm_120, 99KB optin smem)
        "attn_tiles": {64: (64, 64, 1, 4), 128: (32, 32, 1, 4)},
        "matmul_tiles": (64, 64, 32),
        "rope_nstages": 1, "rope_nwarps": 4,
    },
    "hopper": {              # H100/H200 (sm_90, 228KB optin smem)
        "attn_tiles": {64: (128, 128, 2, 8), 128: (128, 64, 2, 8)},
        "matmul_tiles": (128, 128, 64),
        "rope_nstages": 2, "rope_nwarps": 8,
    },
    # ... 4 more architectures: ada, blackwell_dc, ampere_dc, ampere_consumer
}

# 2. GPUProfile reads sm_version + shared_memory_per_block_optin, classifies arch
GPU = GPUProfile()  # Computed once at import time

# 3. For unknown GPUs, tiles computed dynamically:
# _compute_attention_tiles(head_dim, smem_bytes)
#   Formula: (BLOCK_M + 2*BLOCK_N) * BLOCK_D * 4 + 20KB overhead
# _compute_matmul_tiles(smem_bytes)
#   Formula: TILE_K * (TILE_M + 2*TILE_N) * 4 + 20KB overhead (SwiGLU worst case)
```

The system has three tiers of fallback:

1. **Known configs** (fastest): If your GPU matches one of the tested
   architectures, the tile sizes are looked up from a hardcoded table. These
   have been experimentally verified.
2. **Dynamic computation**: For unknown GPUs, tile sizes are computed from the
   available shared memory using formulas that estimate how much memory each
   tile configuration requires.
3. **Conservative defaults**: If even the dynamic computation fails, very small
   (safe) tile sizes are used.

##### What is `shared_memory_per_block_optin`?

This is a subtle hardware detail. NVIDIA GPUs have a pool of on-chip memory
that can be split between shared memory and L1 cache. The **default**
allocation gives shared memory only 48KB. But the GPU can be configured to
give more memory to shared memory (up to 99KB on consumer Blackwell, 228KB on
Hopper). The "optin" value is the maximum shared memory available when you
opt in to the larger allocation. Triton automatically handles the opt-in, but
you need to query the right attribute to know how much memory is actually
available.

The code uses a fallback chain (`shared_memory_per_block_optin` ->
`max_shared_memory_per_block` -> `shared_memory_per_block`) for compatibility
with older PyTorch versions that may not expose the newer attributes.

All tile selection across the codebase uses `GPU.*`:
- attention.py: `GPU.get_attention_tiles(head_dim, seq_q)` for Flash Attention
- layers.py: `GPU.matmul_tile_m/n/k` for Linear, MLP, EncoderMLP
- rope.py: `GPU.rope_nstages`, `GPU.rope_nwarps` for fused RoPE pair kernel

**Defensive input conversion:** `_generate_v8b` converts all inputs (input_features,
input_features_mask, input_ids, attention_mask) to PyTorch CUDA tensors via the
`_to_torch_tensor()` helper. This handles numpy arrays, CuPy arrays (from CuTile benchmarks),
and generic array-likes. Uses `torch.as_tensor()` instead of `torch.from_numpy()` because
some cluster environments (e.g., cu12 with mismatched numpy versions) fail with
`TypeError: expected np.ndarray (got ndarray)`.

A warmup autotune (`warmup_attention_tiles()`) was tested but found worse configs
in practice (101.6ms vs 98.5ms) and was removed. The 2-tier fallback
(`_KNOWN_CONFIGS` -> dynamic computation) handles all cases.

### 6.8 KV-Cached Generation (generate_v8b)

##### What is a KV cache and why does it matter?

To understand the KV cache, you first need to understand how autoregressive
generation works.

A language model generates text one token at a time. To generate token N+1, it
computes attention over all previous tokens 1 through N. Naively, this means:

- Step 1: process tokens [1] -> generate token 2
- Step 2: process tokens [1, 2] -> generate token 3
- Step 3: process tokens [1, 2, 3] -> generate token 4
- ...
- Step N: process tokens [1, 2, ..., N] -> generate token N+1

At each step, the attention mechanism computes Key and Value vectors for every
token in the sequence, even though tokens 1 through N-1 have not changed since
the last step. The total work is 1 + 2 + 3 + ... + N = **O(n^2)**.

The KV cache eliminates this redundancy. After computing the Key and Value
vectors for tokens 1 through N, it **caches** them. At the next step, only the
new token N+1 needs to have its K and V computed. The cached K and V from
previous tokens are simply reused.

```
Without KV cache:   O(n^2) total work (recompute everything each step)
With KV cache:      O(n) total work (compute only the new token each step)
```

For this model generating ~13 tokens, the difference is modest but measurable:
O(n^2) = ~91 units of work vs O(n) = ~13 units. For longer sequences, the
savings are dramatic.

> **Think of it this way:** Imagine you are taking notes in a meeting. Without
> a KV cache, every time someone new speaks, you re-read ALL your previous
> notes from the beginning before writing the new note. With a KV cache, you
> just add the new note to your existing notebook and glance back at previous
> notes as needed.

##### Implementation

```python
# Monkey-patched onto GlmAsrModel from layers.py
# Uses model.decode(use_cache=True) per instructor guidance
logits, past_kv = self.decode(inputs_embeds=..., use_cache=True)       # Prefill
logits, past_kv = self.decode(inputs_embeds=new_embeds,                # Decode loop
                              past_key_values=past_kv, use_cache=True)
```

- **Prefill**: Process all input tokens at once. This computes K and V for every
  token and caches them. It also produces the first output logits.
- **Decode loop**: For each subsequent token, pass only the new token's
  embedding plus the cached K/V. The attention mechanism only needs to compute
  K/V for the single new token and can reuse the cached values for all
  previous tokens.

Uses `decode(use_cache=True)` which returns `(logits, past_key_values)` --
the concatenation-based KV cache from model.py's `TextDecoder.__call__`.

### 6.9 Optimization Results Summary

| Optimization | Source | Impact | Status |
|-------------|--------|--------|--------|
| Fused Q+K RoPE kernel | **meave** | **-14ms** | ADOPTED |
| bf16 RMSNorm output | **meave** (adapted) | **-3ms** | ADOPTED |
| bf16 LayerNorm output | internal | **-0.7ms** | ADOPTED |
| generate_v8b (KV cache) | internal | **-7.6ms** | ADOPTED |
| SDPA fallback for seq_q<=4 | internal | **-3ms** | ADOPTED |
| GPUProfile + _KNOWN_CONFIGS + dynamic tiles | internal | portability | ADOPTED |
| Dead code cleanup | internal | -320 lines | ADOPTED |
| fp16 pipeline (remove float32 casts) | internal | **-11.5ms** | ADOPTED |
| fp16 cuBLAS HGEMM (was bf16) | internal | ~-0.4ms | ADOPTED |
| Smaller flash attention tiles | **meave** | improved prefill | ADOPTED |
| Swizzled SwiGLU | **yash/optimize** | +18ms regression | Rejected |
| @triton.autotune (lightweight) | **majed** | +0.7ms overhead | Rejected |
| @triton.autotune (heavy kernels) | internal | massive regression | Rejected |
| Softmax bf16 output | internal | 0ms (not in hot path) | Rejected |
| Flash Attention num_stages=2 | **yash/optimize** | OOM on consumer GPUs | Rejected |
| PyTorch SDPA for prefill/encoder | internal | +6ms regression | Rejected |
| SDPA enable_gqa=True for decode | internal | +13ms regression | Rejected |
| Fused gate+up Linear in MLP | internal | Neutral | Rejected |

**Lessons from the rejected optimizations:**

- **Swizzled SwiGLU (+18ms):** More complex memory access patterns are not
  always better. The overhead of the swizzling logic outweighed any memory
  access improvements.
- **@triton.autotune (+0.7ms to massive regression):** Autotuning tries
  multiple configurations at runtime and picks the best one. This sounds great
  but the benchmarking overhead of trying different configs adds latency, and
  for short inference runs, the tuning never pays back its cost.
- **Flash Attention num_stages=2 (OOM on consumer GPUs):** `num_stages` controls
  how many tiles are prefetched simultaneously. More stages means better latency
  hiding but more shared memory usage. On consumer GPUs with limited shared
  memory, this causes the kernel to exceed the available memory and crash.

---

## 7. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `CUDA error: invalid configuration argument` | BLOCK_SIZE too large | Reduce to power of 2, max ~1024 |
| `triton.CompilationError` | Mismatched tensor shapes | Check mask dimensions match data |
| `CUBLAS_STATUS_INVALID_VALUE` | cuBLAS version mismatch | `pip uninstall nvidia-cublas` |
| `expected np.ndarray (got ndarray)` | numpy version mismatch (cu12) | Use `torch.as_tensor()` instead of `torch.from_numpy()` |
| `Out Of Memory` (SLURM) | Insufficient RAM for weight loading | Use `--mem=32G` in srun/sbatch |
| `OutOfResources: shared memory` | Fused kernel tiles too large | Reduce tile sizes or disable fusion |
| Values all zero | Mask not applied correctly | Verify `offs < size` mask |
| NaN/Inf in output | Missing numerical stability | Subtract max before exp in softmax |
| `__init__.py` settings not taking effect | Benchmark imports modules directly | Set defaults as class attributes in layers.py |

### Detailed Debugging Tips

**"My output is all zeros":** This almost always means your mask is wrong.
Common mistakes:
- Using `offs < BLOCK_SIZE` instead of `offs < n_elements` (the mask should
  check against the data size, not the block size)
- Forgetting to apply the mask to `tl.store` (data is computed but never
  written)
- Off-by-one error in stride calculation (writing to the wrong memory location)

**"I get NaN or Inf":** This almost always means a numerical stability issue.
Common causes:
- Computing `exp()` on large values without subtracting the max first
- Dividing by zero (forgot to add epsilon in normalization)
- Using fp16 for intermediate reductions (overflow from summing many values)

**"My kernel compiles but gives wrong results":** Debug by:
1. Testing with tiny inputs (e.g., a 4x4 matrix) where you can verify by hand
2. Comparing against the PyTorch reference implementation element by element
3. Printing intermediate values (Triton supports `tl.device_print` for debugging)

---

## 8. Performance Results

| Implementation | Time | Speed | vs Baseline |
|----------------|------|-------|-------------|
| Our template (fp16 pipeline + KV cache + SDPA) | **98.5ms** | 7.58ms/tok | **62.3% faster** |
| Our template (bf16 pipeline + KV cache + SDPA) | 110.0ms | 8.46ms/tok | 57.9% faster |
| Our template (without KV cache) | 120.7ms | 9.29ms/tok | 53.8% faster |
| Example baseline | 261.3ms | 20.10ms/tok | -- |
| CPU fallback (no GPU) | ~14,000ms | ~1,000ms/tok | -- |

##### How to read this table

- **Time**: Total wall-clock time to transcribe the test audio clip (13 tokens).
- **Speed**: Time per output token (total time / 13 tokens).
- **vs Baseline**: Percentage improvement over the example baseline.

Notice the progression:
- **53.8% faster** from just implementing the kernels well and using cuBLAS
- **57.9% faster** by adding KV caching (avoids redundant computation)
- **62.3% faster** by switching from bf16 to fp16 pipeline (better tensor core utilization on RTX 5090)

Key optimizations ranked by impact:
1. **cuBLAS-backed `F.linear`** + TF32 flags -- cuBLAS outperforms Triton linear kernel
2. **fp16 weights + fp16-throughout pipeline** -- halves memory traffic, eliminates dtype casts (-11.5ms)
3. **Fused Flash Attention** -- Triton kernel with online softmax (GPU-tier aware tiles)
4. **Fused Q+K RoPE pair kernel** -- single kernel for both Q and K rotations (-14ms)
5. **generate_v8b with KV cache** -- O(n) decode instead of O(n^2) (-7.6ms)
6. **Remove Linear `.float()` conversion** -- fp16 output cascades through pipeline (-7.5ms)
7. **Remove silu/gelu float32 casts** -- kernels handle precision internally (-3.7ms)
8. **Fused SwiGLU** -- reduces kernel launch overhead for decoder MLP
9. **SDPA fallback for decode** -- PyTorch SDPA for seq_q<=4 (-3ms)

**Competition standings:** ankush 98.5ms, meave 127.8ms, yash 128ms, majed 187.9ms.

### Teaching Cluster Results (H200 MIG 3g.71gb, 60 SMs, 2026-03-16)

| Implementation | Time | Speed | Accuracy |
|----------------|------|-------|----------|
| **Our template (fp16 pipeline + KV cache + SDPA)** | **204.6ms** | 15.74ms/tok | 100% |
| Example baseline | ~550ms (est.) | -- | 100% |

The H200 MIG slice has only 60 SMs (vs RTX 5090's 170), so times are proportionally slower.
GPUProfile correctly detects Hopper (sm_90) and uses datacenter tile configs (128x128, nstages=2).

**NOTE:** `benchmark_detailed.py` fails with the fp16 pipeline because the benchmark code
expects float32 projector output. The student benchmark (authoritative) works perfectly.

Detailed profiling shows decoder decode steps dominate (82.8% of total time with
50 tokens) because stock `generate()` is O(n^2). The `generate_v8b` KV-cached path
reduces this by caching K/V states and only processing 1 new token per step.
SDPA fallback further saves ~3ms by avoiding Triton kernel launch overhead for
the tiny single-token decode attention calls.

---

## 9. Quick Reference: Triton API

```python
# Thread/block identification
tl.program_id(axis)           # Block index (0, 1, or 2)
                               # "Which block am I?" -- like a station number
tl.arange(start, end)         # Vector of indices [start, start+1, ..., end-1]
                               # "Which elements within my block?"

# Memory operations
tl.load(ptr, mask, other)     # Load from GPU memory with bounds check
                               # mask=False positions get the 'other' value
tl.store(ptr, val, mask)      # Write to GPU memory with bounds check
                               # mask=False positions are skipped

# Reductions
tl.sum(x, axis)               # Sum reduction along an axis
tl.max(x, axis)               # Max reduction along an axis

# Math
tl.dot(a, b)                  # Matrix multiply (uses tensor cores!)
                               # ~10x faster than manual multiply-add
tl.exp(x)                     # Exponential (e^x)
tl.rsqrt(x)                   # 1/sqrt(x) -- faster than separate div+sqrt
tl.cos(x), tl.sin(x)          # Trigonometric functions
tl.extra.cuda.libdevice.tanh(x)  # tanh via NVIDIA's libdevice library

# Type conversion
x.to(tl.float32)              # Cast to float32 (for precision in computation)
x.to(tl.float16)              # Cast to float16 (for compact storage)

# Control
tl.where(cond, a, b)          # Conditional select: a if cond else b
                               # Like numpy.where -- no branching
```

### When to use each function

| Task | Function | Example |
|------|----------|---------|
| Identify my block | `tl.program_id(0)` | Which row/tile am I processing? |
| Generate indices | `tl.arange(0, N)` | Column offsets within a row |
| Bounds checking | `mask = offs < size` | Do not read past the end |
| Safe load/store | `tl.load(..., mask=...)` | Respects the bounds mask |
| Normalize (RMS/Layer) | `tl.sum`, `tl.rsqrt` | Compute statistics then scale |
| Softmax stability | `tl.max`, `tl.exp` | Subtract max before exp |
| Matrix multiply | `tl.dot` | Attention scores, linear layers |
| Activation functions | `tl.exp`, `libdevice.tanh` | SiLU, GELU |
| RoPE | `tl.cos`, `tl.sin` | Rotary position embeddings |
| Branching | `tl.where(cond, a, b)` | Causal mask in attention |
