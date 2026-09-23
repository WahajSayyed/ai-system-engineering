# Chapter 4: Your First Kernels: Naive GPU Attention

**Series:** Flash Attention from Scratch in CUDA · Part 1 (CUDA foundations, driven by attention)
**Language:** CUDA C++ (C++ plus a few extensions)
**Builds on:** Chapter 1 (the `N × N` problem), Chapter 3 (pointers, row-major indexing, header/source files, how compilation works)

---

## Read this first: what could and could not be verified

This is the first chapter where the real thing needs a GPU, and **the sandbox I wrote it in has no GPU and no `nvcc`**. So the labels mean something different from Chapters 1 to 3:

| Label | Meaning |
|---|---|
| **[compiled]** | The code was compiled by a real CUDA front end (clang's CUDA mode, and NVIDIA's own NVRTC compiler), and assembled to GPU machine code by NVIDIA's `ptxas`. It was **not** run on a GPU. |
| **[emulated]** | The unchanged kernel and host code was run **on the CPU**, through a small emulation of CUDA's launch model that I wrote (§4.8). It tests the index arithmetic and boundary logic. It says nothing about GPU speed or GPU-specific behaviour. |
| **[source]** | From NVIDIA's CUDA Programming Guide (v13.4.2 pages, last updated Sep 10, 2026) or another named source (§4.15). |
| **[derived]** | Computed from a formula shown in the text. |
| **[predicted]** | My reasoning about something I could not measure. |
| **[not run]** | Requires a GPU. **You** run it; §4.9 gives the exact commands and a table to fill in. |

I will not claim a single GPU timing in this chapter. The numbers you will need to fill in are on your RTX 3090 and T4.

## What you will be able to do after this chapter

1. Explain what a **kernel**, a **thread**, a **block** and a **grid** are, and compute which element a thread owns from `threadIdx`, `blockIdx` and `blockDim`.
2. Write a kernel that has one thread per output element, with the boundary check that makes it safe for any size.
3. Move data between **host** (CPU) memory and **device** (GPU) memory with `cudaMalloc`, `cudaMemcpy` and `cudaFree`.
4. Launch kernels with `<<<grid, block>>>`, choose a launch geometry, and use the integer ceiling-division idiom.
5. Write and read a `CUDA_CHECK` macro, including the C preprocessor tricks it uses (`#define`, `\`, `do { } while (0)`, `#call`, `__FILE__`, `__LINE__`), and explain why a kernel launch needs a different kind of check.
6. Build the three-kernel version of attention from Chapter 1 and say precisely why it is slow (setting up Chapters 5 to 7).
7. Test GPU code on a machine without a GPU, and understand what that does and does not prove.

---

## 4.1 The GPU execution model

### From a Python loop to a kernel

Suppose you want to compute `S[i][j] = dot(Q[i], K[j])` for every pair. In Python or in Chapter 3's C++ you write nested loops. Here is the loop body pulled out:

```python
def body(i, j):
    S[i][j] = dot(Q[i], K[j])

for i in range(N):
    for j in range(N):
        body(i, j)
```

A **kernel** is that loop body. A **kernel launch** says "run this body once for every index in this range, all at once". Each run is a **thread**. Instead of the loop counters `i` and `j`, each thread reads **built-in variables** that tell it which one it is.

### Threads, blocks, grids

The threads of one launch are organised in two levels **[source]**:

- Threads are grouped into **blocks** ("thread blocks"). All threads of a block run on the same streaming multiprocessor (SM) and share its resources. On current GPUs a block may contain **up to 1,024 threads**.
- Blocks are arranged in a **grid**. A launch specifies the grid size (how many blocks) and the block size (how many threads per block); the total number of threads is the product.

Both grid and block can be 1-, 2- or 3-dimensional. Inside a kernel, four built-in variables (each with `.x`, `.y`, `.z` members; dimensions you did not specify default to 1) describe where the thread is **[source]**:

| Built-in | Meaning |
|---|---|
| `threadIdx` | this thread's index **within its block** (0-based) |
| `blockDim` | the size of a block (as given in the launch) |
| `blockIdx` | this block's index **within the grid** (0-based) |
| `gridDim` | the size of the grid (as given in the launch) |

The **global** position of a thread along one axis is

```
index = blockIdx.x * blockDim.x + threadIdx.x
```

which is the pattern NVIDIA's own example uses for adding two vectors **[source]**. Worked example **[derived]**: with `16 × 16` blocks, the thread with `blockIdx = (2, 1)` and `threadIdx = (5, 3)` has `j = 2·16 + 5 = 37` (x direction) and `i = 1·16 + 3 = 19` (y direction).

The grid and block sizes are given in the launch, the same kernel can be launched with different geometries, and the limits are real: the CUDA guide's own error-handling example launches `k<<<8192, 4096>>>()` and is told `invalid argument`, with a driver log message that the block dimensions `(4096,1,1)` exceed the maximum `(1024,1024,64)` **[source]**.

### Three things that make this different from a loop

1. **Concurrency without an order.** The runs happen in parallel, in no guaranteed order. A thread must never depend on another thread's result (within one launch). Blocks are meant to be independent of each other; synchronisation between blocks is only supported in special circumstances, and the best performance is usually achieved by keeping synchronisation within a block **[source]**.
2. **Launches are asynchronous.** The CPU code that launches a kernel does **not** wait for it: it continues, possibly before the kernel has even started. To know a kernel has finished you must synchronise (for example `cudaDeviceSynchronize()`). This is why Chapter 1 told you to call `torch.cuda.synchronize()` before timing **[source]**.
3. **Kernels return nothing.** A kernel is declared `void`; results are written to memory.

### Warps

The hardware runs threads of a block in groups of **32 consecutive threads** called **warps**, which execute in lockstep. (I am taking this from a secondary source, a Rust binding's documentation; the number is well known and `main_gpu.cu` prints `prop.warpSize` so you can confirm it on your GPUs.) Chapter 7 is about warps. For now, remember only: block sizes should be a multiple of 32; `16 × 16 = 256 = 8 warps`.

NVIDIA's guide says 256 threads per block is "arbitrary" but "quite often a good value to start with" **[source]**. We use it.

---

## 4.2 Mapping attention to threads

Chapter 1 wrote attention as three steps. Give each step its own kernel and choose what one thread does:

| Kernel | Computes | One thread does | Launch geometry | Threads (for `N = 1024, d = 64`) |
|---|---|---|---|---|
| `scores_kernel` | `S = scale · Q Kᵀ` (masked) | one element `S[i][j]` | 2-D grid of 2-D blocks over `N × N` | 1,048,576 |
| `softmax_rows_kernel` | `P = softmax(S)` per row, in place; also `L` | one **whole row** of `S` | 1-D | 1,024 |
| `pv_kernel` | `O = P V` | one element `O[i][c]` | 2-D over `N × d` | 65,536 |

A picture of the first kernel. Each `16 × 16` block computes a `16 × 16` tile of `S`, and each thread within it computes one element:

```
                 j (key)  →   0..15      16..31     32..47
                          ┌───────────┬───────────┬───────────┐
 i (query)   0..15        │ block     │ block     │ block     │   each block = 256 threads
   ↓                      │ (0,0)     │ (1,0)     │ (2,0)     │   = one 16 x 16 tile of S
             16..31       │ block     │ block     │ block     │
                          │ (0,1)     │ (1,1)     │ (2,1)     │   blockIdx = (x, y) = (column tile, row tile)
             32..47       ├───────────┼───────────┼───────────┤
                          │ (0,2)     │ (1,2)     │ (2,2)     │
                          └───────────┴───────────┴───────────┘
```

Notice this is **Chapter 3's tile grid**: a block is a tile, with `Br = Bc = 16`. In Chapter 3 one CPU thread visited tiles in a loop; here every tile is a block and every element is a thread. Also note the convention: **`x` is the column direction and `y` is the row direction**. Getting these swapped is a classic mistake (§4.8 tests what it does).

---

## 4.3 Host memory and device memory

The CPU (the **host**) and the GPU (the **device**) have separate memories on the cards we use. A kernel can only read device memory, so the sequence is always:

1. allocate device memory (`cudaMalloc`),
2. copy inputs from host to device (`cudaMemcpy`),
3. launch kernels,
4. copy results back (`cudaMemcpy`),
5. free (`cudaFree`).

```cpp
float* dQ = nullptr;                                  // will hold a DEVICE address
CUDA_CHECK(cudaMalloc(&dQ, bytes));                   // fill it in
CUDA_CHECK(cudaMemcpy(dQ, hQ, bytes, cudaMemcpyHostToDevice));   // (destination, source, bytes, direction)
...
CUDA_CHECK(cudaMemcpy(hO, dO, bytes, cudaMemcpyDeviceToHost));
CUDA_CHECK(cudaFree(dQ));
```

**C++ decoded:**

- **`cudaMalloc(&dQ, bytes)`**: you pass **the address of the pointer** `dQ` (type `float**`), so that `cudaMalloc` can write the new device address into it. This is the "output parameter" pattern from Chapter 3's pointer section: a function can only fill in a variable you hand it the address of. NVIDIA's examples use exactly this form **[source]**.
- **A device pointer is just a number.** `dQ` is an ordinary `float*`, but it points into GPU memory. **Dereferencing it in CPU code is a bug** (it will crash or read garbage), and passing a CPU pointer to a kernel is the mirror-image bug. A naming convention helps: `hQ` for host, `dQ` for device.
- **`cudaMemcpy(dst, src, bytes, kind)`**: destination first, source second, size **in bytes** (not elements), then the direction: `cudaMemcpyHostToDevice`, `cudaMemcpyDeviceToHost` or `cudaMemcpyDeviceToDevice`. `cudaMemcpy` is **synchronous**: it does not return until the copy is done **[source]**.
- **Bytes, not elements:** `bytes = N * d * sizeof(float)`. Passing `N * d` (a forgotten `sizeof`) copies a quarter of the data and silently corrupts the result; §4.8 tests it.
- **`sizeof(float)`** is 4. `static_cast<size_t>(N) * d * sizeof(float)` does the multiplication in 64-bit (Chapter 3, §3.6): `N * d` alone would be 32-bit `int` arithmetic.
- **Python analogy:** roughly, `x.cuda()` in PyTorch does a device allocation and a host-to-device copy, and `x.cpu()` does the reverse. PyTorch adds a caching allocator on top, so it does not call `cudaMalloc` each time.

There is also **unified memory** (`cudaMallocManaged`), where the driver moves data for you **[source]**. Explicit copies are more verbose but show you exactly where the traffic is, and this series is about traffic, so we use them.

---

## 4.4 The three kernels

The kernels live in `attention_kernels.cuh` (`.cuh` is the conventional extension for a header containing device code **[source]**).

### Kernel 1: scores

```cpp
__global__ void scores_kernel(const float* Q, const float* K, float* S,
                              int N, int d, float scale, bool causal) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // column of S: which key
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // row of S:    which query
    if (i >= N || j >= N) return;                          // the grid is rounded UP: extra threads exit

    if (causal && j > i) {                                 // query i may not look at key j > i
        S[static_cast<size_t>(i) * N + j] = -INFINITY;
        return;
    }
    const float* q = Q + static_cast<size_t>(i) * d;       // row i of Q
    const float* k = K + static_cast<size_t>(j) * d;       // row j of K
    float dot = 0.0f;
    for (int x = 0; x < d; ++x) {
        dot += q[x] * k[x];
    }
    S[static_cast<size_t>(i) * N + j] = dot * scale;
}
```

**C++ / CUDA decoded:**

- **`__global__`** marks a **kernel**: a function that runs on the GPU and can be launched from the CPU. It must return `void` **[source]**.
- **`blockIdx.x * blockDim.x + threadIdx.x`**: the global index formula of §4.1. `blockDim.x` is 16 here, so block 0 covers `j = 0..15`, block 1 covers `16..31`, and so on.
- **The bounds check `if (i >= N || j >= N) return;`.** The grid is sized by rounding *up* (§4.5), so for `N = 37` we launch `3 × 16 = 48` threads per axis and the last ones point past the end. Extra threads must exit without touching memory. Launching a few surplus threads is cheap; NVIDIA's guide makes exactly this point, and also warns that whole blocks in which no thread does work should be avoided **[source]**. §4.8 shows what happens without the check.
- **`return;` in a kernel** ends *this thread's* run, not the launch. Every thread has its own `return`.
- **`static_cast<size_t>(i) * N + j`**: the 64-bit offset arithmetic from Chapter 3. `i` is an `int`; casting *before* the multiply makes the product 64-bit, so it cannot overflow at `N = 65,536` (`65536² = 2³²`).
- **`const float* q = Q + i * d`** is Chapter 3's row-pointer idiom, unchanged. On the GPU, `Q`, `K` and `S` are device pointers into **global memory**, the big off-chip memory (GDDR6 or GDDR6X on your cards) that Chapter 1 called "device memory".
- **`-INFINITY`** comes from `<cmath>`; it works in device code with `nvcc`. (Under NVRTC I had to define it myself, see §4.7.)
- **Every thread runs the same code.** The `if` branches are per thread. Threads in the same warp that take different branches are *divergent*: the hardware runs both paths one after another. Chapter 7 returns to this; here the only divergent threads are those on the `causal` diagonal.

### Kernel 2: softmax, one thread per row

```cpp
__global__ void softmax_rows_kernel(float* S, float* L, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // which row
    if (i >= N) return;

    float* s = S + static_cast<size_t>(i) * N;             // row i of S
    float m = -INFINITY;
    for (int j = 0; j < N; ++j) m = fmaxf(m, s[j]);        // pass 1: max
    float l = 0.0f;
    for (int j = 0; j < N; ++j) {                          // pass 2: exp and sum
        s[j] = expf(s[j] - m);
        l += s[j];
    }
    const float inv = 1.0f / l;
    for (int j = 0; j < N; ++j) s[j] *= inv;               // pass 3: normalise
    L[i] = m + logf(l);
}
```

This is Chapter 2's three-pass safe softmax, executed by one thread per row, entirely serially within the row. It is **correct and slow**, on purpose: it is the baseline Chapter 7 (warp-level reductions) improves.

**Decoded:**

- **`fmaxf`, `expf`, `logf`** are the single-precision math functions. They are available in device code (`nvcc` compiles the same names for the GPU). Two cautions: `fmaxf(NaN, x)` returns `x`, so like `std::max` it hides a NaN (Chapter 3); and compiling with `--use_fast_math` changes what `expf` computes. We do **not** use that option. Chapter 12 tests it before we rely on anything.
- **The kernel writes in place** (`s[j] = ...`), reusing `S` as `P`. The row belongs to this thread alone, so there is no conflict.
- `L[i]` is the logsumexp from Chapter 3, so we can compare `L` as well as `O`.

### Kernel 3: `P · V`

```cpp
__global__ void pv_kernel(const float* P, const float* V, float* O, int N, int d) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;   // output column, 0 .. d-1
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // output row (query)
    if (i >= N || c >= d) return;

    const float* p = P + static_cast<size_t>(i) * N;       // row i of P
    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += p[j] * V[static_cast<size_t>(j) * d + c];
    }
    O[static_cast<size_t>(i) * d + c] = acc;
}
```

One thread per element of `O`, each summing `N` products. Note the **grid is not square** here: `d` columns by `N` rows, so `x` and `y` are *not* interchangeable (§4.8 shows the bug you get if you swap them).

---

## 4.5 The host code

`attention_gpu.cu` allocates memory, copies, launches the three kernels and copies back. The core:

```cpp
// Integer ceiling division: the number of size-b blocks needed to cover a items.
static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

...
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const dim3 block2d(16, 16);                                    // 256 threads per block

    // kernel 1: one thread per element of S (N x N)
    const dim3 grid_s(ceil_div(N, block2d.x), ceil_div(N, block2d.y));
    timer_begin(t0);
    scores_kernel<<<grid_s, block2d>>>(dQ, dK, dS, N, d, scale, causal);
    CUDA_CHECK_LAUNCH();
    kt.scores_ms = timer_end(t0, t1);

    // kernel 2: one thread per row
    const int threads_1d = 128;
    timer_begin(t0);
    softmax_rows_kernel<<<ceil_div(N, threads_1d), threads_1d>>>(dS, dL, N);
    CUDA_CHECK_LAUNCH();
    kt.softmax_ms = timer_end(t0, t1);

    // kernel 3: one thread per element of O (N x d)
    const dim3 grid_o(ceil_div(d, block2d.x), ceil_div(N, block2d.y));
    timer_begin(t0);
    pv_kernel<<<grid_o, block2d>>>(dS, dV, dO, N, d);
    CUDA_CHECK_LAUNCH();
    kt.pv_ms = timer_end(t0, t1);

    CUDA_CHECK(cudaDeviceSynchronize());                           // surface any asynchronous error
```

(The allocation, copies and clean-up around it are as in §4.3; see the file.)

**Decoded:**

- **`kernel<<<grid, block>>>(args)`** is the **launch**: three chevrons on each side, the grid size, then the block size, then the ordinary argument list **[source]**. When both are 1-D you may pass plain integers (`<<<ceil_div(N, 128), 128>>>`).
- **`dim3`** is a CUDA type holding three unsigned integers `x, y, z`; `dim3 block2d(16, 16)` constructs it with `z` defaulting to 1 **[source]**. `block2d.x` reads a member. Using `block2d.x` in `ceil_div` instead of a second `16` keeps the launch geometry and the grid computation from drifting apart.
- **Ceiling division: `(a + b - 1) / b`.** Integer division truncates, so adding `b - 1` first makes it round up: `(37 + 15) / 16 = 3`, `(32 + 15) / 16 = 2`. This is the idiom NVIDIA's guide gives **[source]**. Rounding *down* would silently leave the last rows and columns uncomputed (§4.8).
- **Arguments are passed by value**, including pointers: a kernel receives copies of `dQ`, `N`, `scale` and so on. That is why a device pointer can be handed to a kernel: it is just a number.
- **Timing with events.** A CUDA *event* is a timestamp recorded on the GPU's own timeline. We record one before and one after a launch and ask for the difference. `cudaEventSynchronize(t1)` makes the CPU wait until the GPU has reached the second timestamp. This is what `torch.cuda.synchronize()` plus a timer approximates. Timing with a CPU clock alone would measure launch overhead, because the launch returns immediately.
- **`cudaDeviceSynchronize()`** blocks until all previously issued GPU work has finished **[source]**. Here it is redundant with the event waits, but it makes the intent explicit and is where an asynchronous kernel failure would surface.
- **Three launches, one after another.** Launches into the same stream (the default) run in order, so kernel 2 sees all of kernel 1's writes. (This is standard CUDA stream behaviour; it is not quoted from the pages I fetched.)

---

## 4.6 Error checking: the `CUDA_CHECK` macro

Almost every CUDA runtime function returns a `cudaError_t`; `cudaSuccess` means all is well. In production code, check every one **[source]**. The macro (from `cuda_check.h`):

```cpp
#define CUDA_CHECK(call)                                                             \
    do {                                                                             \
        const cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error at %s:%d\n  call : %s\n  error: %s (%d)\n",  \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err_),             \
                    static_cast<int>(err_));                                         \
            exit(EXIT_FAILURE);                                                      \
        }                                                                            \
    } while (0)

#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
```

NVIDIA's guide shows a similar macro that prints and continues; this one also prints the text of the failing call and stops.

### The preprocessor tricks, one by one

- **`#define NAME(param) replacement`** defines a **macro**: before compilation, the preprocessor replaces every `CUDA_CHECK(anything)` with the replacement text, substituting the argument for `call`. It is textual substitution, not a function call. Python has no equivalent.
- **The trailing `\`** continues the definition onto the next line. A macro must be a single logical line.
- **`#call`** (a `#` before a parameter) turns the argument into a **string literal**. So the error message can print the exact source text of the failing call.
- **`__FILE__` and `__LINE__`** are predefined macros for the current source file name and line number.
- **`do { ... } while (0)`** looks pointless, and is essential. It makes a multi-statement macro behave like **one statement** that needs a trailing semicolon. Here is the expansion of two uses, produced by `g++ -E -P` **[ran]** (line breaks added):

```
do { const cudaError_t err_ = (cudaMalloc(&p, 16 * sizeof(float)));
     if (err_ != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d\n  call : %s\n  error: %s (%d)\n",
         "macro_demo.cpp", 10, "cudaMalloc(&p, 16 * sizeof(float))", cudaGetErrorString(err_), static_cast<int>(err_));
         exit(1); } } while (0);
```

  You can see `#call` became the string `"cudaMalloc(&p, 16 * sizeof(float))"`, and `__LINE__` became `10`.
  **Why the wrapper:** a macro written as a bare `{ ... }` block breaks when used inside an `if ... else`. **[ran]**:

```
macro_demo_bad.cpp: In function 'int main()':
macro_demo_bad.cpp:11:5: error: 'else' without a previous 'if'
   11 |     else
```

  With `do { } while (0)`, the same `if (verbose) CUDA_CHECK(...); else ...` compiles and runs **[ran]**.
- **`err_` with a trailing underscore** keeps the macro's local variable from colliding with a variable of the same name in the caller's code. (Macros are not scoped like functions.)
- **`static_cast<int>(err_)`** because `%d` in `printf` expects an `int` and `cudaError_t` is an enumeration.

### Why launches need a second kind of check

A kernel launch **returns nothing**, so it cannot be wrapped in `CUDA_CHECK(...)`. NVIDIA's guide says to check the error state right after the launch, then synchronise to catch errors from the kernel's *execution* **[source]**:

```cpp
vecAdd<<<blocks, threads>>>(...);
CUDA_CHECK(cudaGetLastError());        // launch-time errors: bad configuration, or an earlier async error
CUDA_CHECK(cudaDeviceSynchronize());   // errors that happened WHILE the kernel ran
```

Facts from the guide **[source]**:

- `cudaSuccess` from the check right after a launch does **not** mean the kernel executed successfully, or even started. It only means the launch parameters were valid and no earlier error is pending.
- An error during execution (for example an out-of-bounds access by a kernel) is an **asynchronous** error: it is reported by the next CUDA call that returns an error code, which may be a completely unrelated call later in the program.
- Such an error state is **sticky**: it is returned by every runtime API call until `cudaGetLastError()` clears it.
- In newer drivers (r570 and later) you can set the environment variable `CUDA_LOG_FILE` (for example to `stderr`) and the driver logs the reason for an error such as `invalid argument`, which is very useful when code does not check errors.

`CUDA_CHECK_LAUNCH()` is our name for the first line. **[emulated]** With the emulation layer, a launch of `k<<<8192, 4096>>>()` (the guide's own example) is refused, and the check reports:

```
launched (no error visible yet)
CUDA error at error_demo.cpp:12
  call : cudaGetLastError()
  error: invalid argument (1)
```

Notice the first line: the launch statement itself printed nothing, and the program went on. Without the check the failure would have been silent, and the results would simply be missing. (The text `invalid argument` matches what the guide shows for the real runtime; the rest of this output comes from my emulation.)

---

## 4.7 Building it, and what `nvcc` does

### The commands (for you to run)

From `ch04_code/`, for the RTX 3090 (compute capability 8.6) and for the T4 (7.5):

```bash
nvcc -O2 -arch=sm_86 -I../ch03_code -o attn_gpu \
     main_gpu.cu attention_gpu.cu ../ch03_code/tiled_attention.cpp
nvcc -O2 -arch=sm_75 -I../ch03_code -o attn_gpu_t4 \
     main_gpu.cu attention_gpu.cu ../ch03_code/tiled_attention.cpp
```

`main_gpu.cu` also links Chapter 3's `tiled_attention.cpp` as the CPU reference (plain host code; `nvcc` passes `.cpp` files to the host compiler). Useful extra flags, all documented in the guide **[source]**: `-lineinfo` (device line numbers, for tools such as `compute-sanitizer`), `-res-usage` (print registers per thread for each kernel), `-Xptxas=-warn-spills` (warn if registers spill to slow "local" memory).

`-arch=sm_86` means: generate PTX **and** machine code (cubin) for compute capability 8.6 **[source]**. `nvcc` compiles GPU code with `-O3` by default **[source]**. We do not pass `--use_fast_math`.

### What `nvcc` actually does (the stages, like Chapter 3's)

**[source]** `nvcc` first **splits** the file into device code and host code. The host code goes to a normal C++ compiler (`g++`, MSVC, ...). The device code is compiled by NVIDIA's GPU compiler to **PTX**, a virtual assembly language, once per *virtual* architecture (`compute_86`). Then **`ptxas`** turns the PTX into machine code (**cubin**) for the *real* architecture (`sm_86`). The results are embedded into the executable in a **fatbin**, which can hold several architectures. The device and host halves are then linked together as usual. `nvcc -v` shows every step; `-keep` saves the intermediate files.

Which architecture number goes with which GPU: the RTX 3090 is compute capability 8.6 and the T4 is 7.5 (Chapter 1's sources). A binary built for `sm_86` will not run on the T4 (it needs `sm_75` machine code, or PTX to JIT-compile). Build separately for each, as above.

### What I ran instead: **[compiled]**

`verify_compile.py` (needs no GPU and no `nvcc`; it exists because this sandbox has neither) does three things:

1. **clang's CUDA front end, host-only pass**, on `attention_gpu.cu` and `main_gpu.cu`, with `-Wall -Wextra`: **0 diagnostics** for both, so the host code, the `<<<>>>` launches and the runtime calls are well-formed.
2. **clang's CUDA front end, device-only pass**, for `sm_75` and `sm_86`, then NVIDIA's real **`ptxas` 12.9**: **0 diagnostics**, and all three kernels assemble for both architectures.
3. **NVRTC 12.9**, NVIDIA's own runtime compiler, on `attention_kernels.cuh` for `compute_75` and `compute_86`, then `ptxas`. Return code 0 and an empty log (no errors, no warnings). NVRTC has no `<cmath>`, so the script strips the `#include` lines and defines `INFINITY` on the command line: the only difference from the file you compile.

Resource usage reported by `ptxas -v` **[compiled]** (registers per thread; **0 bytes spilled** in every case):

```
  source of PTX   arch   kernel                regs
  nvrtc           sm_75  scores_kernel           49
  nvrtc           sm_75  softmax_rows_kernel     43
  nvrtc           sm_75  pv_kernel               50
  nvrtc           sm_86  scores_kernel           38
  nvrtc           sm_86  softmax_rows_kernel     40
  nvrtc           sm_86  pv_kernel               40
  clang           sm_75  scores_kernel           20      (16 / 18 for softmax / pv)
  clang           sm_86  scores_kernel           25      (16 / 22 for softmax / pv)
```

Two compilers, two very different register counts for the same source. The NVRTC numbers (38 to 50) are the ones to trust more: NVRTC shares NVIDIA's compiler back end with `nvcc` (general knowledge, not verified here), whereas the clang numbers come from a different compiler's PTX. But **neither is what `nvcc` will report**; run `nvcc -res-usage` yourself. The lesson is that register use is a property of the compiler as well as the source, and we will care about it in Part 3.

---

## 4.8 Testing GPU code on a machine with no GPU

### The emulation layer

`emulation/cuda_runtime.h` is a small stand-in for `<cuda_runtime.h>`. It defines `__global__` as nothing, `dim3`, the built-in variables `threadIdx`, `blockIdx`, `blockDim`, `gridDim` as global variables, `cudaMalloc` as `malloc`, `cudaMemcpy` as `memcpy`, and a launcher that runs the kernel body as four nested loops (blocks in the grid, threads in the block), setting the built-ins before each call. It also refuses impossible launch configurations (more than 1,024 threads per block, and so on). `emulate_cuda.py` compiles the **unchanged** `attention_kernels.cuh`, `attention_gpu.cu` and `main_gpu.cu` with `g++`, after one mechanical rewrite: `name<<<grid, block>>>(args);` becomes `EMU_LAUNCH(name, grid, block, args);`, because `g++` cannot parse the chevrons. It verified that exactly **3** launch statements were rewritten.

**What this can test:** index formulas, boundary checks, argument order, byte counts, launch-size arithmetic, and the algorithm itself.
**What it cannot test:** anything about real GPU hardware. Emulated kernels run one thread after another, so races and ordering bugs cannot show up (our kernels have none, since threads never communicate); GPU rounding differences (fused multiply-add, the GPU's `expf`) do not appear; the real runtime's checks may differ from my imitation; and there is nothing to say about speed.

### Results **[emulated]**

```
launch statements rewritten: 3   compiler warnings: 0
CPU EMULATION of the CUDA launch model (not a GPU)
49 configurations, 0 failures, worst |dO| = 1.192e-07, worst |dL| = 0.000e+00
```

The 49 configurations are 6 sizes `N` × 4 head dimensions `d` × causal on/off, plus one large-scores case (§4.8 below). The reference is Chapter 3's `attention_naive`. The exact zero in `|dL|` appears because the emulated kernel performs the same floating-point operations in the same order as the reference on the same CPU; **do not expect a real GPU to match that closely**. The test uses a tolerance of `1e-4`, because a GPU and a CPU differ in the last bits: the GPU's `expf`/`logf` are not bit-identical to the CPU library's, and `nvcc` by default may fuse multiply-adds (general knowledge, not verified here).

### The launch geometry, counted **[emulated]**

The emulator logs every launch. Threads launched vs threads that have work (`idle` = the threads that hit the bounds check and return):

```
N=1000 d=64
  scores_kernel        grid=(63,63) block=(16,16)  blocks=3969  threads=1016064  useful=1000000  idle=1.58%
  softmax_rows_kernel  grid=(8,1)   block=(128,1)  blocks=8     threads=1024     useful=1000     idle=2.34%
  pv_kernel            grid=(4,63)  block=(16,16)  blocks=252   threads=64512    useful=64000    idle=0.79%
N=1024 d=64
  scores_kernel        grid=(64,64) block=(16,16)  blocks=4096  threads=1048576  useful=1048576  idle=0.00%
  softmax_rows_kernel  grid=(8,1)   block=(128,1)  blocks=8     threads=1024     useful=1024     idle=0.00%
  pv_kernel            grid=(4,64)  block=(16,16)  blocks=256   threads=65536    useful=65536    idle=0.00%
N=37 d=8
  scores_kernel        grid=(3,3)   block=(16,16)  blocks=9     threads=2304     useful=1369     idle=40.58%
  softmax_rows_kernel  grid=(1,1)   block=(128,1)  blocks=1     threads=128      useful=37       idle=71.09%
  pv_kernel            grid=(1,3)   block=(16,16)  blocks=3     threads=768      useful=296      idle=61.46%
```

At realistic sizes the rounding waste is about 1 to 2%; for tiny sizes most threads are idle, which is harmless there.

### Do the tests have teeth? **[emulated]**

As in Chapter 3, break the code on purpose and see whether the tests notice (`mutants_gpu.py`; the outcomes are pasted here):

| Deliberate bug | What the test suite reported |
|---|---|
| drop the bounds check in `scores_kernel`, built with AddressSanitizer | caught: `heap-buffer-overflow` at the faulty write |
| the same, plain build | **the program aborts with `free(): invalid pointer`**: a crash far from the bug, at `cudaFree` time |
| swap `x` and `y` in `scores_kernel` (`i` from `x`, `j` from `y`) | **not caught: 0 failures** (see below) |
| swap `x` and `y` in `pv_kernel` | caught (e.g. `|dO|` = 1.65) |
| `ceil_div` rounds down (`a / b`) | caught by the launch check: `invalid argument`, at the first tiny size (a zero-sized grid) |
| the same, and the test grid only has `N ≥ 130`, `d ≥ 17` | caught: `|dO|` = 0.94, `|dL|` = 4.9 (rows and columns left uncomputed) |
| 64 × 32 = 2,048 threads per block | caught by the launch check: `invalid argument` |
| forget the causal mask | caught, only on the causal configurations (e.g. `|dO|` = 1.29) |
| `expf(s)` instead of `expf(s − m)`, `L` left as it was | caught (`|dL|` = 0.28 even at `N = 1`) |
| no max at all (`m = 0`), so the maths stays right on ordinary inputs | **caught only by the large-scores case** (`|dO|` = nan, `|dL|` = inf); the other 48 pass |
| copy back `O` with `L`'s byte count | caught (e.g. `|dO|` = 0.97) |

Four lessons:

1. **A missing bounds check is a memory-corruption bug, not just a wrong answer.** With the sanitizer it is reported at the exact line. Without it, in this emulation the process crashed later, at a `free`, far from the cause. On a real GPU the guide says an invalid memory access by a kernel is an asynchronous error reported by a later CUDA call **[source]**, which is how such a bug reaches you as an error message about a line that is completely innocent. (I could not test the GPU behaviour.)
2. **An "equivalent mutant".** Swapping `x` and `y` in `scores_kernel` is **not a correctness bug**: every `(i, j)` pair is still visited exactly once, and each thread writes the correct value into `S[i][j]`. Only *which thread does which element* changed. That changes the **memory access pattern**, and therefore performance, which no test of the answer can see. Chapter 6 (coalescing) explains why. In `pv_kernel` the same swap *is* a correctness bug, because the grid is not square (`d` columns by `N` rows).
3. **Small sizes hide rounding bugs.** With `ceil_div` rounding down, the tiny sizes fail loudly (a zero-sized grid), but a suite with only larger sizes gets silent wrong answers. The two rows above are the same bug seen two ways. Keep small, odd and large sizes in the grid.
4. **The large-scores case earns its place.** Softmax without the max subtraction is *exactly right* on ordinary inputs and breaks only where `exp` overflows. Only a test with large scores (Chapter 1, §1.3) can see it. That is the same lesson as Chapter 3's `−inf` guard: a test suite only covers what its inputs reach.

---

## 4.9 Your turn: run it on the GPU **[not run]**

On each machine, with the toolkit installed:

```bash
# from ch04_code/
nvcc -O2 -arch=sm_86 -lineinfo -res-usage -I../ch03_code -o attn_gpu \
     main_gpu.cu attention_gpu.cu ../ch03_code/tiled_attention.cpp
./attn_gpu
compute-sanitizer ./attn_gpu          # memory checker for GPU code (mentioned in the nvcc docs)
```

(Use `-arch=sm_75` on the T4.) `-res-usage` prints registers per thread for each kernel; compare with §4.7.

What the program prints (shape only; the numbers are yours to fill in):

```
GPU: <name> | compute capability <M.m> | <n> SMs | <x> GiB | warp size <32?> | max threads/block <1024?>
49 configurations, 0 failures, worst |dO| = <~1e-6?>, worst |dL| = <~1e-6?>

     N  scores ms  softmax ms      pv ms   total ms   S matrix
   512        ...
  1024        ...
  2048        ...
  4096        ...
```

- **It skips the timing if any test fails**, so a timing table means the answers were right to `1e-4`.
- The times are the **best of 5** runs after one warm-up run, measured with CUDA events around each launch. They exclude the host-device copies.

### What to compare them with **[derived]**

Chapter 1's arithmetic applied to this three-kernel pipeline in fp32, `d = 64`, no mask. Memory: assume the `N × N` matrix crosses the DRAM boundary four times (kernel 1 writes it; kernel 2 reads and writes it; kernel 3 reads it), ignoring the extra passes kernel 2 makes over each row (a row of `N` floats should stay in cache after its first pass). Compute: the two matrix products. Bandwidths and peaks are Chapter 1's (936 GB/s and 35.6 TFLOPS for the 3090; 300 GB/s and 8.1 TFLOPS for the T4):

```
    N   S (MiB)     FLOP   traffic |  RTX 3090: memory / compute |  Tesla T4: memory / compute
  512         1      67M     4 MiB |         4.5 us /   1.9 us   |        14.0 us /   8.3 us
 1024         4     268M    16 MiB |        17.9 us /   7.5 us   |        55.9 us /  33.1 us
 2048        16    1074M    64 MiB |        71.7 us /  30.2 us   |       223.7 us / 132.6 us
 4096        64    4295M   256 MiB |       286.8 us / 120.6 us   |       894.8 us / 530.2 us
```

Your measured **total** must be above the larger number in its column (a lower bound; nothing can beat the datasheet). **How far above** is the interesting quantity. I expect it to be well above, for reasons the next section counts, but I am not going to guess a factor, because I cannot measure it. Fill in a table of `measured / bound`, and keep it: from Chapter 5 on, every optimisation is judged against it.

### Why it should be slow: three counts **[derived]**

1. **Loads issued vs data that exists.** In `scores_kernel` every thread loads a row of `Q` and a row of `K` (`2d` floats). For `N = 1024, d = 64` that is `N² · 2d = 134,217,728` element loads, **512 MiB** of loads issued, from inputs that hold only `2Nd = 131,072` distinct elements: every element is requested **`N = 1,024` times**. The GPU's caches will absorb much of this repetition, how much I cannot say from here, but the design relies on caches to hide a 1,024× re-read. Chapter 5 fixes that with shared memory: a block loads a tile once and its 256 threads reuse it.
2. **Almost the whole GPU is idle during the softmax.** `softmax_rows_kernel` at `N = 1024` runs 1,024 threads in **8 blocks**. Chapter 1 gave 82 SMs (RTX 3090) and 40 SMs (T4) (the T4 figure derived from its core count, not sourced), so at most 8 SMs can be busy: at most 10% and 20% of the chip **[derived, assuming blocks are spread across SMs]**. And each thread walks its row in three sequential passes. Chapter 7 gives each row to a whole warp.
3. **Memory access pattern.** In `scores_kernel`, consecutive threads in a block (consecutive `threadIdx.x`) have consecutive `j`, and each reads *its own row* of `K`: addresses `d` floats (256 bytes for `d = 64`) apart, rather than side by side. Whether reads are "side by side" matters to the hardware (Chapter 6). I have not measured this effect.

None of these is a bug. The point is that the version that works and the version that is fast are different programs, and each of the coming chapters removes one of these three problems.

---

## 4.10 How this maps onto the rest of the series

| This chapter | Where it goes |
|---|---|
| One thread per output element | The same idea in every kernel, until Chapter 15 gives whole warps a matrix tile at once |
| A block = a `16 × 16` tile of `S` | Chapter 3's tiles, now with real blocks; Chapter 9 makes the block the unit of the fused kernel (a Q tile per block) |
| `S` written to global memory, then read back | The thing FlashAttention removes (Chapter 13) |
| Three launches | One fused kernel (Chapter 14) |
| `ceil_div`, bounds checks | Chapter 10 (ragged tiles) |
| `CUDA_CHECK`, events | Used unchanged in every later chapter |
| Kernel 2's serial rows | Warp-level reductions, Chapter 7 |
| Loads issued 1,024× | Shared-memory tiling, Chapter 5 |
| The `x`/`y` swap that no test sees | Coalescing, Chapter 6 |
| `float` everywhere | `half` and Tensor Cores, Part 3 |

---

## 4.11 Exercises

1. **By hand.** For `N = 100` and `16 × 16` blocks, how many blocks does `scores_kernel` launch? How many threads are idle? Then for `pv_kernel` with `d = 33`. Check with `python emulate_cuda.py stats`-style counting (or edit `launch_stats_main.cpp`).
2. **Which thread?** In a `16 × 16` block, which `(i, j)` does thread `(threadIdx.x, threadIdx.y) = (3, 9)` of block `(blockIdx.x, blockIdx.y) = (4, 2)` compute? Which linear position is it within the block (`x + 16·y`)? Which warp is it in (`linear / 32`)?
3. **Run it.** Build and run on both GPUs (§4.9). Fill in the `measured / bound` table for `N = 1024, 2048, 4096`. Which of the three kernels takes most of the time at each `N`? Does the answer change with `N`? Write your prediction *first*.
4. **Try the swap.** Swap `x` and `y` in `scores_kernel` (it changes no answer). Does the runtime change on your GPUs? If it does, by how much?
5. **Block shape.** Change `block2d` to `(32, 8)`, then `(8, 32)`, then `(32, 32)`. All are valid launches. Which is fastest for each kernel? Then try `(64, 32)` and read the error message.
6. **Break the check.** Remove `CUDA_CHECK_LAUNCH()` after one launch, then use the emulated `64 × 32` block. What do you get? Set `CUDA_LOG_FILE=stderr` (drivers r570 and later) on a real GPU and repeat with the real runtime.
7. **Sanitise.** Run `compute-sanitizer ./attn_gpu`. Then remove the bounds check in `pv_kernel` and run it again. What does it report, and at which line (with `-lineinfo`)?
8. **Overflow.** Set `N = 65,536` (you will need a device with enough memory: how much does `S` take?). Which lines of the code would overflow if `size_t` were `int`?
9. **Registers.** Compare `-res-usage` with the numbers in §4.7. Does `nvcc` agree with NVRTC or with clang?
10. **A different mask.** Add a sliding-window mask to `scores_kernel` (Chapter 3, §3.4). Which part of `softmax_rows_kernel` would need the `−inf` guard from Chapters 2 and 3, and why?

---

## 4.12 Common pitfalls

- **Forgetting the bounds check.** Memory corruption, reported by a later, unrelated call (§4.8).
- **Rounding the grid down** (`N / 16`) instead of up. Wrong answers for sizes that are not a multiple of the block size.
- **`x` versus `y`.** `x` is the column (fastest-varying) direction. Swapping them may not change results yet still changes performance, or, on a non-square grid, breaks results.
- **Host pointer in a kernel, device pointer on the host.** Name them `h…` and `d…`.
- **Elements versus bytes** in `cudaMalloc` and `cudaMemcpy`.
- **`cudaMemcpy` argument order** (destination, source), and the wrong direction constant.
- **Not checking errors**, and not knowing that a launch needs `cudaGetLastError` (§4.6).
- **Timing without synchronising.** Launches return immediately (§4.5).
- **More than 1,024 threads per block**, or a zero-sized grid.
- **32-bit index arithmetic** on large tensors (Chapter 3).
- **A build for the wrong architecture.** `sm_86` binaries need an Ampere GA10x GPU; the T4 needs `sm_75`.
- **Trusting an emulation.** It cannot see races, real rounding, or performance (§4.8).

---

## 4.13 Summary and bridge to Chapter 5

- A kernel is a loop body run by many threads at once. Threads form blocks (up to 1,024 threads), blocks form a grid, and `blockIdx.x * blockDim.x + threadIdx.x` gives each thread its index.
- The CPU and GPU have separate memories: allocate, copy in, launch, copy out, free. Launches are asynchronous; `cudaMemcpy` is synchronous.
- Naive attention is three kernels: scores (one thread per element of `S`), row softmax (one thread per row), `P·V` (one thread per element of `O`).
- Every launch needs a bounds check and a rounded-up grid; every runtime call needs `CUDA_CHECK`; every launch needs `cudaGetLastError`, and a synchronise to see execution errors.
- The code compiles with a real CUDA front end and NVIDIA's assembler for both `sm_75` and `sm_86`, and passes 49 configurations against the Chapter 3 reference on a CPU emulation. **It has not been run on a GPU** (§4.9).
- It should be slow, and I can name three reasons: each input element is requested `N` times, the softmax uses a tenth of the chip or less, and the access pattern is untuned.

**Next: Chapter 5, the memory hierarchy and shared memory.** We keep the algorithm and fix the first problem: a block loads a tile of `Q` and `K` into fast on-chip **shared memory** once, and its threads reuse it. You will meet `__shared__`, `__syncthreads()`, and the first kernel whose threads must cooperate, which the emulation in this chapter cannot handle, so Chapter 5 needs a different testing story (and will say so).

---

## 4.14 Files for this chapter (`ch04_code/`)

| File | Purpose | Status |
|---|---|---|
| `cuda_check.h` | `CUDA_CHECK` and `CUDA_CHECK_LAUNCH` | **[compiled]**, **[emulated]** |
| `attention_kernels.cuh` | the three kernels (device code) | **[compiled]** (clang, NVRTC, `ptxas` for `sm_75` and `sm_86`), **[emulated]** |
| `attention_gpu.h`, `attention_gpu.cu` | host code: allocate, copy, launch, time, free | **[compiled]**, **[emulated]** |
| `main_gpu.cu` | tests against the CPU reference, device query, timing | **[compiled]**, **[emulated]** (timing: **[not run]**) |
| `emulation/cuda_runtime.h`, `emulation/launch_stats_main.cpp` | the CPU launch-model emulation and launch counter | **[ran]** |
| `emulate_cuda.py` | builds and runs everything through the emulation | **[ran]** |
| `mutants_gpu.py` | mutation tests of §4.8 | **[ran]** |
| `verify_compile.py` | clang + NVRTC + `ptxas` compile check (sandbox aid) | **[ran]** |
| `macro_demo.cpp`, `macro_demo_bad.cpp`, `error_demo.cpp` | the macro expansion, the `do/while(0)` failure and the launch-error demo | **[ran]** |

The Chapter 3 files `tiled_attention.h` and `tiled_attention.cpp` supply the CPU reference.

---

## 4.15 Verification status, loose ends and sources

| Item | Status |
|---|---|
| Host and device code compile with clang's CUDA front end (`-Wall -Wextra`, 0 diagnostics), for `sm_75` and `sm_86` | **Executed.** |
| Kernels compile with NVRTC 12.9 (`compute_75`, `compute_86`), no errors or warnings; `ptxas` 12.9.86 assembles them; register counts | **Executed.** |
| Unchanged kernels and host code run on the CPU emulation: 49 configurations, launch geometry, 11 mutation experiments | **Executed.** |
| `CUDA_CHECK` expansion, the `do/while(0)` failure, the launch-error demo | **Executed.** |
| **Anything on a GPU**: correctness on real hardware, timing, `compute-sanitizer`, `nvcc` itself | **Not run.** §4.9 is your part. |
| CUDA facts (thread hierarchy, launch syntax, limits, memory API, asynchrony, error semantics, `nvcc` workflow and flags) | **Read from the CUDA Programming Guide** pages listed below. |

**Loose ends.**

- **No `nvcc` was run.** I used clang's CUDA mode and NVRTC (NVIDIA's compiler) as stand-ins. A file that compiles with those should compile with `nvcc`, but I have not proved it; a warning or a small incompatibility is possible.
- **The emulation is my own code**, and it mirrors the runtime only where I said so. Its error text `invalid argument` for an oversized block matches the guide. I did **not** check which error the real runtime reports for a zero-sized grid.
- **Register counts differ between compilers** (20 to 50). Use `nvcc -res-usage` for the real numbers.
- **The bandwidth-bound table** assumes four crossings of `S` and ignores caches; it is a bound, not a prediction.
- **The warp size (32)** comes from a secondary source in this chapter; the program prints it on your GPU.
- **SM counts** are Chapter 1's (the T4's derived from its core count, not sourced).
- **Default `nvcc` floating-point behaviour** (fused multiply-add) is stated from general knowledge, not checked.

**Sources.**

- NVIDIA. *CUDA Programming Guide*, v13.4.2, last updated Sep 10, 2026. §2.1 "Intro to CUDA C++": https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (kernels, launch syntax, thread and grid index intrinsics, 1,024-thread limit, bounds checking, ceiling division, `cudaMalloc`/`cudaMemcpy`, unified memory, synchronisation, error checking, `CUDA_LOG_FILE`)
- NVIDIA. *CUDA Programming Guide*, §2.5 "Asynchronous Execution": https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html (launches are asynchronous; errors may surface at a later synchronisation)
- NVIDIA. *CUDA Programming Guide*, §2.7 "NVCC: The NVIDIA CUDA Compiler": https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html (device/host split, PTX, `ptxas`, cubin, fatbin, `-arch`, `-gencode`, `-res-usage`, `-lineinfo`, `-Xptxas=-warn-spills`, `compute-sanitizer`, `.cu` and `.cuh`, default `-O3`)
- *The CUDA Execution Model* (cuda-oxide documentation; secondary source, for warps of 32 threads): https://nvlabs.github.io/cuda-oxide/gpu-programming/execution-model.html
- Hardware figures (SM counts, bandwidths, peaks, compute capabilities): Chapter 1, §1.12.
