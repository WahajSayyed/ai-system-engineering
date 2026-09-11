# Chapter 18: PTX, SASS & Compiler Internals

> **Prerequisites:** Chapters 1–17 (CUDA fundamentals through Triton)
> **Language:** C/CUDA
> **Domain:** Systems, GPU Architecture
> **Estimated Time:** 8–10 hours

---

## 18.1 Why Go This Deep?

After 17 chapters you can write high-performance CUDA C++ and Triton kernels, profile them with Nsight, and reason about occupancy and rooflines. So why go deeper?

Because the compiler is the final arbiter of performance. Everything you write — CUDA C++, Triton, even cuBLAS — ultimately becomes PTX and then SASS instructions running on the SM. When a kernel is slower than it should be despite good occupancy and memory access patterns, the answer is often in the assembly: a missed vectorization, excess register spill, a suboptimal instruction sequence, or a barrier the compiler inserted unnecessarily.

Reading PTX and SASS gives you:
- **Verification** — confirm the compiler did what you intended (vectorized loads, fused multiply-adds)
- **Debugging** — find register spills, unexpected memory traffic, redundant computations
- **Optimization** — use inline PTX to force specific instruction sequences the compiler won't generate
- **Understanding** — build the mental model that makes you a 10× better CUDA programmer

---

## 18.2 The NVCC Compilation Pipeline

```
Your .cu file
     │
     ▼
┌─────────────────────────────────────┐
│  NVCC front-end (Clang-based)       │
│  - Separates device/host code       │
│  - Preprocesses, parses, type-checks│
└────────────────┬────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
  Host code (C++)    Device code (CUDA)
  → host compiler        │
    (gcc/clang/MSVC)     ▼
                   ┌─────────────┐
                   │  NVVM IR    │  (LLVM IR subset)
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │    PTX      │  Parallel Thread eXecution
                   │  (text IR)  │  Virtual ISA — architecture-independent
                   └──────┬──────┘
                          │  ptxas (PTX assembler)
                          ▼
                   ┌─────────────┐
                   │    SASS     │  Streaming ASSembler
                   │ (binary)    │  Real GPU machine code
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  .cubin /   │
                   │  fatbinary  │  Embedded in executable
                   └─────────────┘
```

### Key distinction: PTX vs SASS

| | PTX | SASS |
|---|---|---|
| Type | Virtual ISA (text) | Real machine code (binary) |
| Architecture | Architecture-independent | SM-specific (sm_75, sm_86, …) |
| Register count | Unlimited (virtual regs) | Fixed (physical regs) |
| Instruction set | Stable across GPU generations | Changes each architecture |
| When generated | Compile time | JIT at first run, or offline via `ptxas` |
| Use for | Portability, inline asm | Performance analysis, reverse engineering |

---

## 18.3 Generating and Reading PTX

### 18.3.1 Generating PTX from NVCC

```bash
# Compile to PTX only (no binary)
nvcc -ptx kernel.cu -o kernel.ptx

# Compile for specific architecture + generate PTX
nvcc -arch=sm_75 -ptx kernel.cu -o kernel.ptx

# Compile normally BUT also save PTX
nvcc -arch=sm_86 --keep kernel.cu -o kernel
# Look for kernel.ptx in the current directory

# Maximum optimization + PTX
nvcc -O3 -arch=sm_86 -ptx kernel.cu -o kernel.ptx

# From Python (for Triton kernels) — see Section 17.9
```

### 18.3.2 A Simple Kernel to Dissect

```cuda
// kernel.cu
__global__ void add_scaled(float* out, const float* a, const float* b,
                            float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] + b[idx]) * scale;
    }
}
```

Compile and examine:
```bash
nvcc -O3 -arch=sm_86 -ptx kernel.cu -o kernel.ptx
cat kernel.ptx
```

### 18.3.3 Reading PTX: The Essential Vocabulary

A PTX file looks like this:

```ptx
.version 7.5
.target sm_86
.address_size 64

.visible .entry add_scaled(
    .param .u64 add_scaled_param_0,    // out pointer
    .param .u64 add_scaled_param_1,    // a pointer
    .param .u64 add_scaled_param_2,    // b pointer
    .param .f32 add_scaled_param_3,    // scale
    .param .u32 add_scaled_param_4     // n
)
{
    // Register declarations
    .reg .pred  %p<2>;       // Predicate (boolean) registers
    .reg .f32   %f<5>;       // Float32 registers
    .reg .b32   %r<5>;       // 32-bit general purpose
    .reg .b64   %rd<8>;      // 64-bit (pointers)

    // Load kernel parameters from .param space
    ld.param.u64    %rd0, [add_scaled_param_0];  // out ptr
    ld.param.u64    %rd1, [add_scaled_param_1];  // a ptr
    ld.param.u64    %rd2, [add_scaled_param_2];  // b ptr
    ld.param.f32    %f0,  [add_scaled_param_3];  // scale
    ld.param.u32    %r0,  [add_scaled_param_4];  // n

    // Compute global thread index: blockIdx.x * blockDim.x + threadIdx.x
    mov.u32         %r1, %ctaid.x;      // blockIdx.x
    mov.u32         %r2, %ntid.x;       // blockDim.x
    mov.u32         %r3, %tid.x;        // threadIdx.x
    mad.lo.s32      %r4, %r1, %r2, %r3; // r4 = r1*r2 + r3 (idx)

    // Bounds check: if (idx >= n) exit
    setp.ge.s32     %p0, %r4, %r0;      // p0 = (idx >= n)
    @%p0 bra        EXIT;                // branch if true

    // Compute byte offset: idx * 4 (sizeof float)
    mul.wide.s32    %rd3, %r4, 4;
    add.s64         %rd4, %rd1, %rd3;   // &a[idx]
    add.s64         %rd5, %rd2, %rd3;   // &b[idx]
    add.s64         %rd6, %rd0, %rd3;   // &out[idx]

    // Load a[idx] and b[idx] from global memory
    ld.global.f32   %f1, [%rd4];
    ld.global.f32   %f2, [%rd5];

    // Compute (a[idx] + b[idx]) * scale
    add.f32         %f3, %f1, %f2;
    mul.f32         %f4, %f3, %f0;

    // Store result
    st.global.f32   [%rd6], %f4;

EXIT:
    ret;
}
```

### 18.3.4 PTX Instruction Reference

**Memory spaces:**
```ptx
ld.global.f32   %f0, [%rd0];       // Load from global memory
ld.shared.f32   %f0, [%rd0];       // Load from shared memory
ld.param.u32    %r0, [param_name]; // Load kernel parameter
st.global.f32   [%rd0], %f0;       // Store to global memory
```

**Arithmetic:**
```ptx
add.f32     %f2, %f0, %f1;         // f2 = f0 + f1
mul.f32     %f2, %f0, %f1;         // f2 = f0 * f1
fma.rn.f32  %f3, %f0, %f1, %f2;   // f3 = f0*f1 + f2 (fused multiply-add)
mad.lo.s32  %r2, %r0, %r1, %r2;   // r2 = r0*r1 + r2 (integer, low 32 bits)
```

**Type conversions:**
```ptx
cvt.f32.f16  %f0, %h0;            // half → float
cvt.f16.f32  %h0, %f0;            // float → half
cvt.rn.f32.s32 %f0, %r0;          // int → float (round to nearest)
```

**Predicates and branching:**
```ptx
setp.lt.s32  %p0, %r0, %r1;       // p0 = (r0 < r1)
setp.ge.u32  %p1, %r0, %r2;       // p1 = (r0 >= r2) unsigned
@%p0 bra     LABEL;                // if p0: goto LABEL
@!%p0 bra    LABEL;                // if !p0: goto LABEL
```

**Special registers:**
```ptx
mov.u32  %r0, %tid.x;             // threadIdx.x
mov.u32  %r0, %tid.y;             // threadIdx.y
mov.u32  %r0, %ctaid.x;           // blockIdx.x
mov.u32  %r0, %ntid.x;            // blockDim.x
mov.u32  %r0, %nctaid.x;          // gridDim.x
mov.u64  %rd0, %clock64;          // 64-bit clock (for timing)
```

**Synchronization:**
```ptx
bar.sync  0;                       // __syncthreads()
membar.gl;                         // Memory barrier (global)
membar.cta;                        // Memory barrier (CTA/block)
```

---

## 18.4 Vectorized Memory Access in PTX

One of the most important things to verify in PTX is whether loads are vectorized. A coalesced 128-byte load is 4× more efficient than four separate 32-byte loads.

```cuda
// C++ kernel — will this vectorize?
__global__ void load_float4(float4* out, const float4* in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx];
}
```

In PTX, look for `ld.global.v4.f32` — the `v4` suffix means 4-element vector load:

```ptx
ld.global.v4.f32  {%f0, %f1, %f2, %f3}, [%rd0];  // ✓ 128-bit vector load
st.global.v4.f32  [%rd1], {%f0, %f1, %f2, %f3};  // ✓ 128-bit vector store
```

Compare to scalar (bad):
```ptx
ld.global.f32  %f0, [%rd0];       // 32-bit scalar load — inefficient
ld.global.f32  %f1, [%rd0+4];
ld.global.f32  %f2, [%rd0+8];
ld.global.f32  %f3, [%rd0+12];
```

To force vectorization from C++:
```cuda
// Use float4 explicitly
reinterpret_cast<float4*>(out)[idx] = reinterpret_cast<const float4*>(in)[idx];

// Or use __ldg() for read-only cache
float4 val = __ldg(reinterpret_cast<const float4*>(in) + idx);
```

In PTX, `__ldg` appears as:
```ptx
ld.global.nc.v4.f32  {%f0,%f1,%f2,%f3}, [%rd0];  // nc = non-coherent (L1 read-only)
```

---

## 18.5 Reading FMA Patterns

Fused multiply-add (`fma.rn.f32`) is the key arithmetic instruction for FLOP efficiency. Verify the compiler generates FMAs instead of separate mul+add:

```cuda
// Should generate FMA
float result = a * b + c;
```

```ptx
// ✓ FMA — one instruction, full precision
fma.rn.f32  %f3, %f0, %f1, %f2;   // f3 = f0*f1 + f2

// ✗ Separate mul+add — two instructions, rounded twice
mul.f32     %f3, %f0, %f1;
add.f32     %f4, %f3, %f2;
```

If you see separate mul+add, the compiler may be respecting strict IEEE 754. Use `--fmad=true` (NVCC default with `-O3`) or `#pragma STDC FP_CONTRACT ON`.

---

## 18.6 Inspecting SASS with `cuobjdump`

PTX is the virtual ISA. SASS is the real machine code. `cuobjdump` extracts SASS from compiled binaries.

### 18.6.1 Basic Usage

```bash
# Compile with debug info (keeps SASS readable)
nvcc -O3 -arch=sm_86 -lineinfo kernel.cu -o kernel

# Dump SASS for all kernels
cuobjdump --dump-sass kernel

# Dump SASS for a specific kernel
cuobjdump --dump-sass --function add_scaled kernel

# Dump PTX embedded in binary
cuobjdump --dump-ptx kernel

# Dump both
cuobjdump --dump-sass --dump-ptx kernel

# Show all embedded code objects
cuobjdump --list-elf kernel
```

### 18.6.2 SASS vs PTX: What Changes

| PTX | SASS (sm_86 example) |
|---|---|
| Virtual registers (%f0, %r0) | Physical registers (R0, R1, …) |
| Unlimited registers | Limited (255 per thread) |
| `fma.rn.f32` | `FFMA R2, R0, R1, R3` |
| `ld.global.v4.f32` | `LDG.E.128 R0, [R2]` |
| `bar.sync` | `BAR.SYNC 0` |
| `setp` + `bra` | `ISETP` + `BRA` |

### 18.6.3 Sample SASS Output and How to Read It

```sass
        /*0008*/                   MOV R1, c[0x0][0x28] ;    // Load stack ptr
        /*0010*/                   S2R R0, SR_CTAID.X ;       // blockIdx.x
        /*0018*/                   S2R R3, SR_TID.X ;         // threadIdx.x
        /*0020*/                   IMAD R0, R0, c[0x0][0x168], R3 ; // idx = bIdx*bDim + tIdx
        /*0028*/                   ISETP.GE.AND P0, PT, R0, c[0x0][0x16c], PT ; // p0 = idx >= n
        /*0030*/              @P0  EXIT ;                      // if p0: exit
        /*0038*/                   IMAD.WIDE R2, R0, 0x4, c[0x0][0x160] ; // &a[idx]
        /*0048*/                   IMAD.WIDE R4, R0, 0x4, c[0x0][0x158] ; // &b[idx]
        /*0058*/                   LDG.E R2, [R2.64] ;        // a[idx]
        /*0060*/                   LDG.E R4, [R4.64] ;        // b[idx]
        /*0068*/                   IMAD.WIDE R6, R0, 0x4, c[0x0][0x150] ; // &out[idx]
        /*0078*/                   FADD R2, R2, R4 ;          // a[idx] + b[idx]
        /*0080*/                   FMUL R2, R2, c[0x0][0x164] ; // * scale
        /*0088*/                   STG.E [R6.64], R2 ;        // store
        /*0090*/                   EXIT ;
```

Key things to look for:
- `c[0x0][offset]` — constant memory (kernel parameters live here)
- `S2R` — "Special to Register" (reads special registers like threadIdx)
- `LDG.E` — Load from Global memory (`.128` suffix = 128-bit = vectorized)
- `STG.E` — Store to Global memory
- `IMAD.WIDE` — Integer multiply-add, wide (produces 64-bit result for pointer arithmetic)
- `FFMA` — Fused Float Multiply-Add
- `@P0 EXIT` — Predicated exit (bounds check)

### 18.6.4 Register Allocation in SASS

Count the highest register number used to understand register pressure:

```bash
# Check register usage per kernel
cuobjdump --dump-sass kernel | grep "// Reg count:"

# Or from nvcc
nvcc -O3 -arch=sm_86 --ptxas-options=-v kernel.cu 2>&1 | grep "registers"
# Output: ptxas info: Used 32 registers, 384 bytes smem, 400 bytes cmem[0]
```

If register count is high (>64), the compiler may spill to local memory (LMEM), which appears in SASS as:
```sass
STL [R1+0x4], R5 ;    // Spill: store register to local memory
LDL R5, [R1+0x4] ;    // Reload: load from local memory
```

Local memory is backed by L2/DRAM — spilling kills performance. To reduce register pressure:
```cuda
// Option 1: Launch with fewer threads per block (less parallelism, more registers)
kernel<<<grid, 128>>>(...)   // vs 256

// Option 2: Limit registers at compile time
nvcc --maxrregcount=32 kernel.cu

// Option 3: __launch_bounds__ hint
__global__ void __launch_bounds__(256, 2) my_kernel(...) { ... }
// maxThreadsPerBlock=256, minBlocksPerSM=2
```

---

## 18.7 Inline PTX with `asm()`

Sometimes the compiler doesn't generate the instruction you need. Inline PTX lets you write assembly directly inside CUDA C++.

### 18.7.1 Syntax

```cuda
asm volatile (
    "ptx_instruction_template"
    : outputs          // "=constraint"(variable)
    : inputs           // "constraint"(variable)
    : clobbers         // registers you modify
);
```

Constraint codes:
- `r` — 32-bit register
- `l` — 64-bit register  
- `f` — 32-bit float register
- `d` — 64-bit float register
- `=` prefix — output (written)
- `+` prefix — read-write

### 18.7.2 Example 1: Read the Clock

```cuda
__device__ uint64_t read_clock() {
    uint64_t clock;
    asm volatile ("mov.u64 %0, %%clock64;" : "=l"(clock));
    return clock;
}

__global__ void timed_kernel(float* out, const float* in, int n) {
    uint64_t start = read_clock();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = in[idx] * 2.0f;

    uint64_t end = read_clock();

    // Store cycle count for analysis (only first thread)
    if (idx == 0) out[n] = (float)(end - start);
}
```

### 18.7.3 Example 2: Warp Shuffle with Explicit PTX

```cuda
// Standard C++ warp shuffle (compiler generates shfl.sync.bfly.b32)
float val = __shfl_xor_sync(0xffffffff, x, 16);

// Explicit PTX version — same result, more control
__device__ float shfl_xor_ptx(float val, int lane_mask) {
    float result;
    asm volatile (
        "shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(result)
        : "f"(val), "r"(lane_mask)
    );
    return result;
}
```

### 18.7.4 Example 3: Vectorized Global Load (Force LDG.128)

```cuda
// Force a 128-bit non-coherent (texture-cache) load
__device__ void load_float4_ldg(float* dst, const float* src) {
    float4 tmp;
    asm volatile (
        "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(tmp.x), "=f"(tmp.y), "=f"(tmp.z), "=f"(tmp.w)
        : "l"(src)
    );
    dst[0] = tmp.x; dst[1] = tmp.y;
    dst[2] = tmp.z; dst[3] = tmp.w;
}
```

### 18.7.5 Example 4: Tensor Core with WMMA PTX (sm_80+)

For maximum control over tensor core operations, use MMA PTX instructions directly:

```cuda
// 16×8×16 half-precision matrix multiply (Ampere)
// d = a * b + c  where a:16×16 half, b:16×8 half, c/d:16×8 float
__device__ void mma_m16n8k16(
    float* d,           // 4 float output registers (16×8 tile = 128 elements / 32 threads = 4 each)
    const uint32_t* a,  // 8 uint32 = 16 fp16 values per thread
    const uint32_t* b,  // 4 uint32 = 8 fp16 values per thread
    const float* c      // 4 float accumulator values per thread
) {
    asm volatile (
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "           // D (output)
        "{%4,%5,%6,%7}, "           // A
        "{%8,%9}, "                 // B
        "{%10,%11,%12,%13};"        // C (accumulator)
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}
```

This is the raw tensor core instruction that cuBLAS, cuDNN, and (at a higher level) Triton's `tl.dot` all eventually compile to.

---

## 18.8 The NVCC Compilation Pipeline in Detail

### 18.8.1 Separating Host and Device Code

NVCC is a compiler driver, not a compiler itself. It splits `.cu` files:

```bash
# See what NVCC actually invokes
nvcc -v kernel.cu -o kernel 2>&1 | head -50
```

You'll see NVCC calls:
1. `cudafe++` — Separates device and host code, handles `__global__`, `__device__`, `<<<>>>` syntax
2. `cicc` (or `nvopencc`) — Generates PTX from device IR
3. `ptxas` — Assembles PTX to SASS (cubin)
4. `fatbinary` — Packages cubins for multiple architectures
5. Host compiler (`g++`, `clang++`, `cl.exe`) — Compiles host code with the fatbinary embedded

### 18.8.2 Controlling the Pipeline

```bash
# Only preprocess (useful for debugging macros)
nvcc -E kernel.cu -o kernel.i

# Only generate PTX (no SASS)
nvcc -ptx kernel.cu -o kernel.ptx

# Generate PTX AND cubin
nvcc -arch=sm_86 -cubin kernel.cu -o kernel.cubin

# Embed multiple architectures (fat binary)
nvcc -gencode arch=compute_75,code=sm_75 \
     -gencode arch=compute_86,code=sm_86 \
     kernel.cu -o kernel

# Keep all intermediate files
nvcc --keep kernel.cu -o kernel
# Produces: kernel.ptx, kernel.cubin, kernel.cpp1.ii, etc.

# Print compiler phase commands without running them
nvcc --dryrun kernel.cu -o kernel
```

### 18.8.3 Virtual vs Real Architecture

```bash
# Virtual architecture only (PTX embedded, JIT compiled at runtime)
nvcc -arch=compute_86 kernel.cu -o kernel
# Portable but slower first run (JIT compilation)

# Real architecture (SASS embedded, no JIT)
nvcc -arch=sm_86 kernel.cu -o kernel
# Fast but only runs on sm_86

# Both (best practice for production)
nvcc -gencode arch=compute_86,code=[sm_86,compute_86] kernel.cu -o kernel
# Runs natively on sm_86, JIT-compiles for anything else
```

---

## 18.9 Optimization Flags That Affect PTX/SASS

```bash
# Fast math (trades precision for speed)
nvcc -use_fast_math kernel.cu -o kernel
# Enables: --fmad=true, --prec-div=false, --prec-sqrt=false
# PTX effect: rsqrt.approx, sin.approx, cos.approx instead of full-precision

# Specific fast math options
nvcc --fmad=true           # Enable FMA generation (default with -O3)
nvcc --prec-div=false      # Use approximate reciprocal for division
nvcc --prec-sqrt=false     # Use approximate square root

# Loop unrolling
nvcc -Xptxas -O3 kernel.cu   # Pass -O3 to ptxas

# Force loop unroll in source
#pragma unroll 4
for (int i = 0; i < 16; i++) { ... }
// In PTX: generates 4 copies of the loop body per iteration
```

### Effect of `-use_fast_math` on PTX:

```ptx
// Without --prec-sqrt=false
sqrt.rn.f32  %f1, %f0;           // Full precision, slow

// With --prec-sqrt=false
rsqrt.approx.f32  %f1, %f0;     // Approximate reciprocal sqrt
rcp.approx.f32    %f2, %f1;     // Then approximate reciprocal
// Faster but ~2 ULP error
```

---

## 18.10 Practical Workflow: Debugging a Slow Kernel with Assembly

Here's a systematic workflow for using assembly analysis to find performance issues:

### Step 1: Profile to identify the slow kernel
```bash
ncu --set full -o profile.ncu-rep ./my_program
```

### Step 2: Check register usage and shared memory
```bash
nvcc -O3 -arch=sm_86 --ptxas-options=-v kernel.cu 2>&1
# Look for: "registers", "smem", "lmem" (local memory = spilling!)
```

### Step 3: Generate PTX and look for vectorization
```bash
nvcc -O3 -arch=sm_86 -ptx kernel.cu -o kernel.ptx
grep -E "ld\.global|st\.global" kernel.ptx
# Want to see: ld.global.v4.f32 (128-bit)
# Bad to see: ld.global.f32 (32-bit scalar)
```

### Step 4: Inspect SASS for instruction mix
```bash
nvcc -O3 -arch=sm_86 -lineinfo kernel.cu -o kernel
cuobjdump --dump-sass kernel | grep "FFMA\|FMUL\|FADD" | wc -l
# Count arithmetic instructions vs total instructions
```

### Step 5: Look for local memory spills in SASS
```bash
cuobjdump --dump-sass kernel | grep -E "STL|LDL"
# Any STL/LDL = register spill → bad
```

### Step 6: Measure instruction-level bottleneck
```bash
ncu --metrics sm__inst_executed_pipe_fma_type_fp32.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
./my_program
```

---

## 18.11 Case Study: Optimizing via PTX Analysis

Let's walk through a real optimization. Start with a naive dot product kernel:

```cuda
// Version 1: Naive — each thread does one multiply-add
__global__ void dot_v1(float* out, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(out, a[idx] * b[idx]);  // One atomic per thread — terrible
    }
}
```

PTX shows: `atom.global.add.f32` — every thread serializes on the same location.

```cuda
// Version 2: Warp reduction
__global__ void dot_v2(float* out, const float* a, const float* b, int n) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) sum = a[idx] * b[idx];

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}
```

PTX shows: `shfl.sync.down.b32` — warp shuffle, one atomic per warp (32× fewer).

```cuda
// Version 3: Vectorized load + warp reduce
__global__ void dot_v3(float* out, const float* a, const float* b, int n) {
    float sum = 0.0f;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;  // 4 elements per thread

    if (idx + 3 < n) {
        float4 va = *reinterpret_cast<const float4*>(a + idx);
        float4 vb = *reinterpret_cast<const float4*>(b + idx);
        sum = va.x*vb.x + va.y*vb.y + va.z*vb.z + va.w*vb.w;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (threadIdx.x % 32 == 0) atomicAdd(out, sum);
}
```

PTX shows:
```ptx
ld.global.v4.f32  {%f0,%f1,%f2,%f3}, [%rd0];   // 128-bit load
ld.global.v4.f32  {%f4,%f5,%f6,%f7}, [%rd1];
fma.rn.f32  %f8, %f0, %f4, 0f00000000;          // FMA chain
fma.rn.f32  %f8, %f1, %f5, %f8;
fma.rn.f32  %f8, %f2, %f6, %f8;
fma.rn.f32  %f8, %f3, %f7, %f8;
shfl.sync.down.b32 ...                           // Warp reduce
```

Vectorized loads + FMA chain = maximum memory bandwidth and compute efficiency.

---

## 18.12 PTX for Shared Memory: Verifying Bank Conflicts

Bank conflicts appear in assembly as serialized transactions. Compare:

```cuda
// No conflict: stride-1 access
extern __shared__ float smem[];
float val = smem[threadIdx.x];   // Each thread in a different bank
```

```ptx
// PTX — single shared memory load
ld.shared.f32  %f0, [%rd0];
```

```cuda
// Potential conflict: stride-32 access (all threads hit bank 0)
float val = smem[threadIdx.x * 32];
```

```ptx
// PTX still shows single load instruction —
// bank conflicts are detected at runtime by hardware, not visible in PTX/SASS
// Use Nsight Compute to measure:
// metric: l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
```

This is why profiling complements assembly reading — some issues (bank conflicts, cache misses) are runtime behaviors invisible in the static instruction stream.

---

## 18.13 Exercises

**Exercise 1 — PTX Scavenger Hunt**
Write a kernel that computes `y = sin(x) + cos(x)`. Compile with and without `-use_fast_math`. Diff the PTX. Identify which instructions change and measure the performance difference.

**Exercise 2 — Vectorization Verification**
Write a kernel that copies a float array. Check the PTX to verify it uses `ld.global.v4.f32`. If it doesn't, modify the kernel until it does. Benchmark the scalar vs vectorized version.

**Exercise 3 — Register Pressure**
Write a kernel with a large unrolled loop that processes 16 floats per thread. Use `--ptxas-options=-v` to check register usage. Use `__launch_bounds__` to reduce register count and observe the effect on occupancy.

**Exercise 4 — Inline PTX Benchmark**
Use `mov.u64 %0, %%clock64` to measure the cycle count of different operations: a global load, a shared memory load, an FFMA, a warp shuffle. What are the latencies?

**Exercise 5 — SASS Analysis Pipeline**
Take your softmax kernel from Chapter 16. Generate SASS with `cuobjdump`. Find: (a) how many FFMA instructions per iteration, (b) whether there are any STL/LDL spills, (c) what the memory access instructions look like. Based on your findings, suggest one optimization.

**Exercise 6 — Decode a Mystery Kernel**
Compile the following kernel and read the SASS without looking at the source code first. Can you reconstruct what the kernel does from the assembly alone?

```cuda
__global__ void mystery(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = in[i];
    float y = x * x;
    float z = y * x;
    out[i] = 0.5f * x * (3.0f - x * z);
}
// Hint: what does this converge to?
```

---

## 18.14 Chapter Summary

| Concept | Key Point |
|---|---|
| PTX | Virtual ISA; text format; architecture-independent; generated by NVCC |
| SASS | Real machine code; SM-specific; extracted with `cuobjdump` |
| Compilation pipeline | `.cu` → NVVM IR → PTX → SASS → fatbinary |
| `nvcc -ptx` | Dump PTX without compiling to binary |
| `cuobjdump --dump-sass` | Disassemble SASS from any compiled binary |
| `ld.global.v4.f32` | 128-bit vectorized load — what you want for coalesced access |
| `fma.rn.f32` / `FFMA` | Fused multiply-add — verify the compiler generates these |
| `STL` / `LDL` in SASS | Register spill to local memory — performance killer |
| Inline PTX `asm()` | Force specific instructions; access special registers |
| `--ptxas-options=-v` | Print register, shared mem, local mem usage per kernel |

---

## 18.15 What's Next

Chapter 19 moves back to the CUDA runtime API for two powerful features: **CUDA Graphs** (capture and replay entire kernel sequences to slash launch overhead, critical for inference latency) and **Dynamic Parallelism** (kernels that launch kernels). These are the techniques that production inference engines like TensorRT use to squeeze out the last microseconds.

---

*Chapter 18 of 20 — CUDA GPU Programming: Beginner to Expert*
