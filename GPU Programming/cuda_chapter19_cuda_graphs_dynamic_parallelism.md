# Chapter 19: CUDA Graphs & Dynamic Parallelism

> **Prerequisites:** Chapters 1–18 (CUDA fundamentals through PTX/SASS internals)
> **Language:** C/CUDA
> **Domain:** Systems, GPU Architecture
> **Estimated Time:** 8–10 hours

---

## 19.1 The Problem: Kernel Launch Overhead

Every CUDA kernel launch costs time — not GPU time, but CPU time. The CUDA driver must:

1. Validate arguments
2. Set up the kernel descriptor
3. Push a command into the command queue
4. (Optionally) synchronize with the GPU

On modern hardware this is roughly **5–20 µs per kernel launch** from the CPU side. For a single large kernel computing for milliseconds, this is negligible. But inference workloads are different:

```
Typical transformer inference forward pass (simplified):
  Layer 1: embedding lookup        → 3 µs kernel
  Layer 2: QKV projection          → 8 µs kernel
  Layer 3: attention               → 12 µs kernel
  Layer 4: output projection       → 8 µs kernel
  Layer 5: layer norm              → 2 µs kernel
  Layer 6: FFN gate                → 6 µs kernel
  ...× 32 layers = ~192 kernel launches

  Launch overhead alone: 192 × 15 µs = ~2.9 ms
  Actual GPU work: maybe 8 ms
  → 26% of wall time is just launch overhead!
```

**CUDA Graphs** solve this by recording the entire sequence of operations once, then replaying it as a single GPU-side operation with near-zero CPU involvement.

---

## 19.2 CUDA Graphs: Core Concepts

A CUDA Graph is a **DAG (Directed Acyclic Graph)** of operations with explicit dependency edges. Node types include:

- **Kernel nodes** — `cudaGraphNode_t` wrapping a kernel launch
- **Memcpy nodes** — `cudaMemcpy` operations
- **Memset nodes** — `cudaMemset` operations
- **Event record/wait nodes** — synchronization
- **Child graph nodes** — nesting graphs
- **Host function nodes** — CPU callbacks

```
Without graphs (every inference call):
CPU: launch K1 → launch K2 → launch K3 → ... → launch KN
     [15µs]      [15µs]      [15µs]           [15µs]
     CPU is busy the whole time submitting work

With graphs:
CPU: [capture once] → instantiate → replay()  replay()  replay()
                                    [1µs]      [1µs]      [1µs]
     GPU runs the entire graph from a single CPU trigger
```

---

## 19.3 Stream Capture: The Easy Path to Graphs

The simplest way to create a CUDA graph is **stream capture** — wrap existing CUDA calls in begin/end capture markers and let the driver record everything automatically.

### 19.3.1 Basic Stream Capture

```cuda
#include <cuda_runtime.h>
#include <stdio.h>

// Simple kernels to demonstrate graph capture
__global__ void scale_kernel(float* x, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void add_kernel(float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__global__ void relu_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = fmaxf(x[i], 0.0f);
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a,   N * sizeof(float));
    cudaMalloc(&d_b,   N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));

    // Initialize data
    cudaMemset(d_a, 0, N * sizeof(float));
    cudaMemset(d_b, 0, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ── Step 1: Capture ─────────────────────────────────────────────────
    cudaGraph_t graph;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    // Everything launched on `stream` from here is recorded, not executed

    scale_kernel<<<GRID, BLOCK, 0, stream>>>(d_a, 2.0f, N);
    add_kernel  <<<GRID, BLOCK, 0, stream>>>(d_out, d_a, d_b, N);
    relu_kernel <<<GRID, BLOCK, 0, stream>>>(d_out, N);

    cudaStreamEndCapture(stream, &graph);
    // `graph` now holds the DAG — no GPU work has run yet

    // ── Step 2: Instantiate ─────────────────────────────────────────────
    // Compiles the graph into an executable form (validates, optimizes)
    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    // ── Step 3: Replay (as many times as needed) ─────────────────────────
    for (int i = 0; i < 1000; i++) {
        cudaGraphLaunch(graph_exec, stream);   // Single CPU call per iteration!
    }
    cudaStreamSynchronize(stream);

    // ── Cleanup ─────────────────────────────────────────────────────────
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);

    printf("Done.\n");
    return 0;
}
```

### 19.3.2 Benchmarking: Eager vs Graph

```cuda
#include <cuda_runtime.h>
#include <chrono>
#include <stdio.h>

// Micro-kernel: simulates a small operation (like a layer norm or bias add)
__global__ void micro_kernel(float* x, float bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 1.0001f + bias;
}

void benchmark(int num_kernels, int N) {
    const int BLOCK = 256;
    const int GRID  = (N + BLOCK - 1) / BLOCK;

    float* d_x;
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMemset(d_x, 0, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    for (int i = 0; i < 10; i++)
        micro_kernel<<<GRID, BLOCK, 0, stream>>>(d_x, 0.1f, N);
    cudaStreamSynchronize(stream);

    // ── Eager (no graph) ────────────────────────────────────────────────
    const int ITERS = 200;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; iter++) {
        for (int k = 0; k < num_kernels; k++)
            micro_kernel<<<GRID, BLOCK, 0, stream>>>(d_x, 0.1f * k, N);
        cudaStreamSynchronize(stream);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double eager_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;

    // ── CUDA Graph ──────────────────────────────────────────────────────
    cudaGraph_t     graph;
    cudaGraphExec_t exec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int k = 0; k < num_kernels; k++)
        micro_kernel<<<GRID, BLOCK, 0, stream>>>(d_x, 0.1f * k, N);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // Warmup graph
    for (int i = 0; i < 5; i++) cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; iter++) {
        cudaGraphLaunch(exec, stream);
        cudaStreamSynchronize(stream);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double graph_ms = std::chrono::duration<double, std::milli>(t3 - t2).count() / ITERS;

    printf("Kernels: %3d | N: %7d | Eager: %6.3f ms | Graph: %6.3f ms | Speedup: %.2fx\n",
           num_kernels, N, eager_ms, graph_ms, eager_ms / graph_ms);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_x);
}

int main() {
    printf("%-12s %-10s %-15s %-15s %-10s\n",
           "Kernels", "N", "Eager (ms)", "Graph (ms)", "Speedup");
    printf("%s\n", std::string(65, '-').c_str());

    benchmark(10,  1 << 10);   // Many small kernels
    benchmark(50,  1 << 10);
    benchmark(100, 1 << 10);
    benchmark(10,  1 << 20);   // Fewer large kernels
    benchmark(50,  1 << 20);
    return 0;
}
```

Expected output on a modern GPU:
```
Kernels       N          Eager (ms)      Graph (ms)     Speedup
-----------------------------------------------------------------
Kernels:  10 | N:    1024 | Eager:  0.312 ms | Graph:  0.024 ms | Speedup: 13.0x
Kernels:  50 | N:    1024 | Eager:  1.423 ms | Graph:  0.031 ms | Speedup: 45.9x
Kernels: 100 | N:    1024 | Eager:  2.891 ms | Graph:  0.038 ms | Speedup: 76.1x
Kernels:  10 | N: 1048576 | Eager:  1.102 ms | Graph:  1.078 ms | Speedup:  1.02x
Kernels:  50 | N: 1048576 | Eager:  5.201 ms | Graph:  5.162 ms | Speedup:  1.01x
```

The speedup is dramatic for many small kernels (launch overhead dominates) and negligible for large kernels (GPU compute dominates). This exactly characterizes LLM inference: many small kernels per token.

---

## 19.4 Updating Graph Parameters Without Re-Instantiation

A key limitation: graphs capture pointer values and launch configs at capture time. If your input tensor changes address (e.g. new allocation), you must update the graph. Fortunately, CUDA provides `cudaGraphExecKernelNodeSetParams` to update without full re-instantiation:

```cuda
// After graph instantiation, update a kernel node's parameters
cudaGraphNode_t* nodes;
size_t num_nodes;
cudaGraphGetNodes(graph, nullptr, &num_nodes);
nodes = new cudaGraphNode_t[num_nodes];
cudaGraphGetNodes(graph, nodes, &num_nodes);

// Find the kernel node you want to update
for (size_t i = 0; i < num_nodes; i++) {
    cudaGraphNodeType type;
    cudaGraphNodeGetType(nodes[i], &type);
    if (type == cudaGraphNodeTypeKernel) {
        cudaKernelNodeParams params;
        cudaGraphKernelNodeGetParams(nodes[i], &params);

        // Update kernel argument (e.g., new input pointer)
        float* new_ptr = new_input_buffer;
        void* new_args[] = { &new_ptr, &scale, &N };
        params.kernelParams = new_args;

        // Apply update to executable graph (no re-instantiation needed)
        cudaGraphExecKernelNodeSetParams(graph_exec, nodes[i], &params);
        break;
    }
}
delete[] nodes;
```

### PyTorch's `torch.cuda.make_graphed_callables`

In practice, use PyTorch's graph API for ML workloads:

```python
import torch

# Define the workload
def model_forward(x, weight):
    x = torch.nn.functional.linear(x, weight)
    x = torch.nn.functional.relu(x)
    x = torch.nn.functional.layer_norm(x, x.shape[-1:])
    return x

# Static inputs (shapes must match at replay time)
x      = torch.randn(32, 512, device='cuda')
weight = torch.randn(512, 512, device='cuda')

# Warmup (required before capture — initializes cuBLAS handles, etc.)
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        _ = model_forward(x, weight)
torch.cuda.current_stream().wait_stream(s)

# Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model_forward(x, weight)

# Replay: just update the input data in-place
for i in range(1000):
    x.copy_(new_input_batch[i])  # Update input buffer in-place
    g.replay()                    # Single call triggers the entire graph
    result = out.clone()          # Read output
```

---

## 19.5 Multi-Stream Graphs and Fork/Join Patterns

CUDA Graphs can capture multi-stream workloads, encoding the parallelism as graph edges. This is the real power: not just a flat sequence, but a true DAG.

```cuda
cudaStream_t s_main, s_branch_a, s_branch_b;
cudaStreamCreate(&s_main);
cudaStreamCreate(&s_branch_a);
cudaStreamCreate(&s_branch_b);

cudaEvent_t fork_event, join_event_a, join_event_b;
cudaEventCreate(&fork_event);
cudaEventCreate(&join_event_a);
cudaEventCreate(&join_event_b);

// Begin capture on main stream
cudaStreamBeginCapture(s_main, cudaStreamCaptureModeGlobal);

// Pre-fork: runs before the parallel branches
preprocess_kernel<<<grid, block, 0, s_main>>>(d_input, N);

// Fork: record event on main stream, wait on both branches
cudaEventRecord(fork_event, s_main);
cudaStreamWaitEvent(s_branch_a, fork_event, 0);
cudaStreamWaitEvent(s_branch_b, fork_event, 0);

// Parallel branch A
branch_a_kernel<<<grid, block, 0, s_branch_a>>>(d_out_a, d_input, N);
cudaEventRecord(join_event_a, s_branch_a);

// Parallel branch B (runs concurrently with A)
branch_b_kernel<<<grid, block, 0, s_branch_b>>>(d_out_b, d_input, N);
cudaEventRecord(join_event_b, s_branch_b);

// Join: main stream waits for both branches
cudaStreamWaitEvent(s_main, join_event_a, 0);
cudaStreamWaitEvent(s_main, join_event_b, 0);

// Post-join: runs after both branches complete
merge_kernel<<<grid, block, 0, s_main>>>(d_merged, d_out_a, d_out_b, N);

// End capture — the graph encodes the fork/join structure
cudaStreamEndCapture(s_main, &graph);

// The captured graph looks like:
//   preprocess ──→ branch_a ──┐
//                 branch_b ──┴──→ merge
```

---

## 19.6 Dynamic Parallelism

Dynamic Parallelism (DP) allows **device code to launch kernels**. A kernel running on the GPU can spawn child kernels without returning to the CPU.

### 19.6.1 When Does This Help?

- **Adaptive algorithms** — child work depends on parent's result (e.g., adaptive mesh refinement, sparse computation)
- **Recursive algorithms** — quicksort, tree traversal, fractal rendering
- **Variable-width computation** — different threads need different amounts of work

### 19.6.2 Setup

```bash
# Dynamic parallelism requires separate compilation + device linking
nvcc -arch=sm_70 -rdc=true -o kernel.o -c kernel.cu
nvcc -arch=sm_70 -dlink -o device_link.o kernel.o -lcudadevrt
nvcc -o program kernel.o device_link.o -lcudadevrt
```

Or with a single command:
```bash
nvcc -arch=sm_70 -rdc=true kernel.cu -o program -lcudadevrt
```

### 19.6.3 Basic Dynamic Parallelism

```cuda
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Child kernel — launched from device code
__global__ void child_kernel(float* data, int start, int width, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width) {
        data[start + i] *= scale;
    }
}

// Parent kernel — launches child kernels
__global__ void parent_kernel(float* data, int* segment_sizes, int num_segments) {
    int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= num_segments) return;

    // Each thread in the parent launches a child grid sized to its segment
    int start = 0;
    for (int i = 0; i < seg; i++) start += segment_sizes[i];  // prefix sum

    int width   = segment_sizes[seg];
    float scale = 1.0f / width;  // Normalize by segment size

    // Launch child kernel FROM THE GPU
    int child_block = 128;
    int child_grid  = (width + child_block - 1) / child_block;

    child_kernel<<<child_grid, child_block>>>(data, start, width, scale);

    // Optional: wait for child to complete before parent thread continues
    cudaDeviceSynchronize();  // Note: device-side sync, not host-side!
}

int main() {
    const int NUM_SEGMENTS = 8;
    int h_sizes[NUM_SEGMENTS] = {100, 200, 50, 300, 150, 75, 400, 125};

    // Compute total size
    int total = 0;
    for (int i = 0; i < NUM_SEGMENTS; i++) total += h_sizes[i];

    float*  d_data;
    int*    d_sizes;
    cudaMalloc(&d_data,  total * sizeof(float));
    cudaMalloc(&d_sizes, NUM_SEGMENTS * sizeof(int));
    cudaMemset(d_data, 0, total * sizeof(float));
    cudaMemcpy(d_sizes, h_sizes, NUM_SEGMENTS * sizeof(int), cudaMemcpyHostToDevice);

    // Parent launches one thread per segment
    parent_kernel<<<1, NUM_SEGMENTS>>>(d_data, d_sizes, NUM_SEGMENTS);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    cudaFree(d_sizes);
    printf("Done.\n");
    return 0;
}
```

### 19.6.4 Recursive Example: GPU Quicksort

A classic dynamic parallelism use case — recursive partition:

```cuda
__device__ int partition(float* arr, int lo, int hi) {
    float pivot = arr[hi];
    int i = lo - 1;
    for (int j = lo; j < hi; j++) {
        if (arr[j] <= pivot) {
            i++;
            float tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }
    }
    float tmp = arr[i+1]; arr[i+1] = arr[hi]; arr[hi] = tmp;
    return i + 1;
}

__global__ void quicksort_kernel(float* arr, int lo, int hi, int depth) {
    if (lo >= hi || depth > 10) return;  // Base case or depth limit

    // Single thread does the partition (serial within this kernel)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int pivot_idx = partition(arr, lo, hi);

        // Launch child kernels for each half (if large enough)
        if (pivot_idx - 1 - lo > 32) {
            quicksort_kernel<<<1, 1>>>(arr, lo, pivot_idx - 1, depth + 1);
        }
        if (hi - (pivot_idx + 1) > 32) {
            quicksort_kernel<<<1, 1>>>(arr, pivot_idx + 1, hi, depth + 1);
        }
        // Small partitions handled serially (overhead not worth child launch)
    }
}
```

### 19.6.5 Dynamic Parallelism: Performance Considerations

| Consideration | Detail |
|---|---|
| Child launch overhead | ~5–10 µs per child launch (slower than host launch) |
| `cudaDeviceSynchronize()` on device | Stalls the parent thread until ALL device work completes |
| Memory visibility | Parent's writes are visible to children; child writes visible after `cudaDeviceSynchronize()` |
| Depth limit | Maximum nesting depth is 24 on modern GPUs |
| Register overhead | Parent threads must hold state while children run → fewer parent threads can occupy SM |
| Occupancy | Usually lower than flat kernel due to resource overhead |

**Rule of thumb:** Only use dynamic parallelism when the child work is large enough (>10,000 elements) to amortize the launch overhead. For small adaptive work, prefer warp-level masking or cooperative groups instead.

---

## 19.7 Cooperative Groups

Cooperative Groups (CG) is the modern CUDA API for flexible, composable synchronization — a generalization of `__syncthreads()` that works at warp, block, grid, and multi-GPU scope.

### 19.7.1 The Problem with `__syncthreads()`

`__syncthreads()` is all-or-nothing: every thread in the block must reach it. This makes it impossible to synchronize only part of a block, or to write functions that work at multiple granularities.

### 19.7.2 Basic Cooperative Groups

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void cg_demo(float* data, int n) {
    // Thread block group — equivalent to __syncthreads() scope
    cg::thread_block block = cg::this_thread_block();

    // Warp — 32 threads, no explicit sync needed (lockstep)
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Half-warp
    cg::thread_block_tile<16> half_warp = cg::tiled_partition<16>(block);

    int idx = block.thread_rank();  // Thread index within block

    // ── Block-level shared memory reduction ─────────────────────────────
    __shared__ float smem[256];
    smem[idx] = (idx < n) ? data[idx] : 0.0f;

    block.sync();  // Same as __syncthreads()

    // Tree reduction using warp tiles
    float val = smem[idx];
    for (int offset = warp.size() / 2; offset > 0; offset >>= 1) {
        val += warp.shfl_down(val, offset);  // Warp shuffle reduce
    }

    // Lane 0 of each warp writes partial sum to shared mem
    if (warp.thread_rank() == 0) {
        smem[block.thread_rank() / 32] = val;
    }

    block.sync();

    // Final reduction by first warp
    if (block.thread_rank() < (blockDim.x / 32)) {
        val = smem[block.thread_rank()];
        for (int offset = (blockDim.x / 32) / 2; offset > 0; offset >>= 1)
            val += warp.shfl_down(val, offset);
    }

    if (block.thread_rank() == 0) data[0] = val;
}
```

### 19.7.3 Warp-Level Collective Operations via CG

```cuda
__global__ void warp_collectives(float* out, const float* in, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int gidx = block.group_index().x * block.size() + block.thread_rank();
    float val = (gidx < n) ? in[gidx] : 0.0f;

    // Warp-level reduce (all lanes get the result)
    float warp_sum = cg::reduce(warp, val, cg::plus<float>());
    float warp_max = cg::reduce(warp, val, cg::greater<float>());
    float warp_min = cg::reduce(warp, val, cg::less<float>());

    // Inclusive scan (prefix sum within warp)
    float prefix_sum = cg::inclusive_scan(warp, val, cg::plus<float>());

    // Ballot — which lanes satisfy a condition?
    unsigned mask = warp.ballot(val > 0.0f);  // Bitmask of lanes with val > 0

    // Any/All predicates
    bool any_positive = warp.any(val > 0.0f);
    bool all_positive = warp.all(val > 0.0f);

    if (warp.thread_rank() == 0) {
        out[block.group_index().x] = warp_sum;
    }
}
```

### 19.7.4 Grid-Level Cooperative Groups (Full Grid Sync)

Normal CUDA has no way to synchronize across blocks within a single kernel. Grid-level CG enables this — but requires launching with `cudaLaunchCooperativeKernel`:

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void grid_sync_demo(float* data, float* partial_sums, int n) {
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();

    int gidx = grid.thread_rank();

    // Phase 1: Each block computes its partial sum
    __shared__ float smem[256];
    smem[block.thread_rank()] = (gidx < n) ? data[gidx] : 0.0f;
    block.sync();

    // Block-level reduction
    float val = smem[block.thread_rank()];
    for (int s = block.size()/2; s > 0; s >>= 1) {
        if (block.thread_rank() < s) smem[block.thread_rank()] += smem[block.thread_rank() + s];
        block.sync();
    }

    if (block.thread_rank() == 0)
        partial_sums[block.group_index().x] = smem[0];

    // ── GRID-WIDE SYNC — all blocks reach this point before continuing ──
    grid.sync();

    // Phase 2: Block 0 reduces partial sums (now all partial_sums are ready)
    if (block.group_index().x == 0) {
        float total = 0.0f;
        for (int i = block.thread_rank(); i < gridDim.x; i += block.size())
            total += partial_sums[i];
        // ... reduce total within block 0
        if (block.thread_rank() == 0) data[0] = total;
    }
}

// Must launch with cudaLaunchCooperativeKernel
void launch_grid_sync(float* d_data, float* d_partial, int n) {
    int block_size = 256;
    int grid_size;

    // Query max blocks that can run simultaneously (hardware limit)
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &grid_size, grid_sync_demo, block_size, 0);

    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    grid_size *= num_sms;  // Total blocks = blocks/SM × SM count

    void* args[] = { &d_data, &d_partial, &n };
    cudaLaunchCooperativeKernel(
        (void*)grid_sync_demo,
        grid_size, block_size,
        args
    );
}
```

---

## 19.8 Combining Graphs + Cooperative Groups: Production Inference Engine Pattern

Here's how these concepts combine in a real inference scenario:

```cuda
// inference_engine.cu
//
// Pattern used by TensorRT and similar systems:
//   1. Preallocate all buffers statically
//   2. Build a CUDA graph once at load time
//   3. Each inference call: copy input → graph.replay() → read output

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
namespace cg = cooperative_groups;

// Fused bias + activation kernel (typical post-linear op)
__global__ void fused_bias_gelu(float* x, const float* bias, int N, int C) {
    cg::thread_block_tile<32> warp =
        cg::tiled_partition<32>(cg::this_thread_block());

    int n   = blockIdx.y;        // Batch index
    int c   = blockIdx.x * blockDim.x + threadIdx.x;  // Channel index
    if (c >= C || n >= N) return;

    float val = x[n * C + c] + bias[c];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715x³)))
    float tanh_arg = 0.7978845608f * (val + 0.044715f * val * val * val);
    x[n * C + c] = 0.5f * val * (1.0f + tanhf(tanh_arg));
}

__global__ void layer_norm(float* x, const float* gamma, const float* beta,
                            int N, int C, float eps) {
    int n = blockIdx.x;
    if (n >= N) return;

    cg::thread_block block = cg::this_thread_block();
    __shared__ float smem[2];  // [mean, var]

    // Compute mean
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) sum += x[n * C + c];

    // Block reduce
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    sum = cg::reduce(warp, sum, cg::plus<float>());
    if (warp.thread_rank() == 0) atomicAdd(&smem[0], sum);
    block.sync();

    float mean = smem[0] / C;

    // Compute variance
    float var_sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float diff = x[n * C + c] - mean;
        var_sum += diff * diff;
    }
    // ... (similar block reduce for variance)
    block.sync();

    float inv_std = rsqrtf(smem[1] / C + eps);

    // Normalize
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float norm = (x[n * C + c] - mean) * inv_std;
        x[n * C + c] = gamma[c] * norm + beta[c];
    }
}

struct InferenceEngine {
    // Static buffers (allocated once, reused every inference)
    float *d_input, *d_hidden, *d_output;
    float *d_weight1, *d_bias1, *d_weight2, *d_bias2;
    float *d_gamma, *d_beta;

    cudaGraph_t     graph;
    cudaGraphExec_t exec;
    cudaStream_t    stream;

    int batch_size, input_dim, hidden_dim, output_dim;

    void build(int B, int D_in, int D_hidden, int D_out) {
        batch_size = B; input_dim = D_in;
        hidden_dim = D_hidden; output_dim = D_out;

        cudaStreamCreate(&stream);

        // Allocate all buffers
        cudaMalloc(&d_input,   B * D_in     * sizeof(float));
        cudaMalloc(&d_hidden,  B * D_hidden * sizeof(float));
        cudaMalloc(&d_output,  B * D_out    * sizeof(float));
        cudaMalloc(&d_weight1, D_in * D_hidden * sizeof(float));
        cudaMalloc(&d_bias1,   D_hidden * sizeof(float));
        cudaMalloc(&d_weight2, D_hidden * D_out * sizeof(float));
        cudaMalloc(&d_bias2,   D_out * sizeof(float));
        cudaMalloc(&d_gamma,   D_hidden * sizeof(float));
        cudaMalloc(&d_beta,    D_hidden * sizeof(float));

        // Initialize weights (normally loaded from checkpoint)
        cudaMemset(d_weight1, 0, D_in * D_hidden * sizeof(float));
        cudaMemset(d_weight2, 0, D_hidden * D_out * sizeof(float));
        // ... load real weights here

        // Warmup to initialize cuBLAS state before capture
        run_forward_ops();
        cudaStreamSynchronize(stream);

        // Capture
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        run_forward_ops();
        cudaStreamEndCapture(stream, &graph);

        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

        printf("Graph built: %d kernels captured.\n", count_graph_nodes());
    }

    void run_forward_ops() {
        dim3 block(256);
        dim3 grid_hidden((hidden_dim + 255) / 256, batch_size);
        dim3 grid_out   ((output_dim  + 255) / 256, batch_size);

        // Linear 1: hidden = input @ weight1  (use cublasSgemm in practice)
        // ... cublas call on stream ...

        // Fused bias + GELU
        fused_bias_gelu<<<grid_hidden, block, 0, stream>>>(
            d_hidden, d_bias1, batch_size, hidden_dim);

        // Layer norm
        layer_norm<<<batch_size, 256, 0, stream>>>(
            d_hidden, d_gamma, d_beta, batch_size, hidden_dim, 1e-5f);

        // Linear 2: output = hidden @ weight2
        // ... cublas call on stream ...

        // Bias add
        fused_bias_gelu<<<grid_out, block, 0, stream>>>(
            d_output, d_bias2, batch_size, output_dim);
    }

    void infer(const float* h_input, float* h_output) {
        // Copy new input into static buffer (in-place update)
        cudaMemcpyAsync(d_input, h_input,
            batch_size * input_dim * sizeof(float),
            cudaMemcpyHostToDevice, stream);

        // Single call triggers the entire captured forward pass
        cudaGraphLaunch(exec, stream);

        // Copy output back
        cudaMemcpyAsync(h_output, d_output,
            batch_size * output_dim * sizeof(float),
            cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);
    }

    int count_graph_nodes() {
        size_t n; cudaGraphGetNodes(graph, nullptr, &n); return (int)n;
    }

    void destroy() {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        cudaStreamDestroy(stream);
        cudaFree(d_input); cudaFree(d_hidden); cudaFree(d_output);
        // ... free weights ...
    }
};
```

---

## 19.9 Graphs in PyTorch: `torch.cuda.CUDAGraph`

For ML workloads, use PyTorch's graph API:

```python
import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x, need_weights=False)[0])
        x = self.norm2(x + self.ffn(x))
        return x

# ── Setup ──────────────────────────────────────────────────────────────────
device = torch.device('cuda')
model  = TransformerLayer(d_model=512).to(device).eval()

B, T, D = 8, 128, 512  # Batch, sequence length, model dim

# Static inputs — MUST be the same shape every replay
static_input  = torch.randn(B, T, D, device=device)
static_output = torch.empty(B, T, D, device=device)

# ── Warmup (critical!) ────────────────────────────────────────────────────
# Runs the model a few times to initialize all lazy state:
# cuBLAS handle, workspace allocation, algorithm selection
warmup_stream = torch.cuda.Stream()
with torch.cuda.stream(warmup_stream):
    for _ in range(3):
        static_output = model(static_input)
torch.cuda.current_stream().wait_stream(warmup_stream)

# ── Capture ───────────────────────────────────────────────────────────────
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_output = model(static_input)

# ── Benchmark ─────────────────────────────────────────────────────────────
import time

def benchmark(fn, n=200):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000

# Eager
eager_ms = benchmark(lambda: model(static_input))

# Graph
def graph_replay():
    g.replay()

graph_ms = benchmark(graph_replay)

print(f"Eager: {eager_ms:.3f} ms | Graph: {graph_ms:.3f} ms | "
      f"Speedup: {eager_ms/graph_ms:.2f}x")

# ── Inference loop ────────────────────────────────────────────────────────
for batch in dataloader:
    # Copy new data into the static buffer in-place
    static_input.copy_(batch.to(device))

    # Replay the captured graph
    g.replay()

    # Read result (static_output updated in-place)
    result = static_output.clone()
```

### Caveats for PyTorch Graph Capture

```python
# ✗ Cannot capture: dynamic shapes
# Each replay must use the same tensor shapes as capture time

# ✗ Cannot capture: CPU-GPU synchronization inside the graph
# No .item(), .numpy(), print(tensor) during capture

# ✗ Cannot capture: Python control flow that depends on tensor values
# No: if tensor.sum() > 0: ...

# ✓ Can capture: all standard torch ops, custom CUDA kernels, cuBLAS

# For variable sequence lengths: use padding + static max length
MAX_LEN = 512
input_padded = torch.zeros(B, MAX_LEN, D, device=device)
# Capture at MAX_LEN, mask out the padding logically
```

---

## 19.10 Common Pitfalls

### Pitfall 1: Not Warming Up Before Capture

```cuda
// ✗ Wrong — cuBLAS allocates its workspace on first call
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
cublasSgemm(...);  // First call: allocates workspace — captured as alloc node!
                   // Second replay: tries to re-allocate → undefined behavior
cudaStreamEndCapture(stream, &graph);

// ✓ Correct — warm up first
cublasSgemm(...);  // Warmup: workspace allocated, cuBLAS initialized
cudaStreamSynchronize(stream);
// NOW capture — workspace already exists, no alloc node captured
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
cublasSgemm(...);
cudaStreamEndCapture(stream, &graph);
```

### Pitfall 2: Capturing Memory Allocations

Graphs capture pointer values. If you allocate inside the captured region, those allocations happen once at instantiation and the same pointer is reused every replay:

```cuda
// ✗ Dangerous — allocates inside capture
cudaStreamBeginCapture(stream, ...);
float* d_tmp;
cudaMallocAsync(&d_tmp, size, stream);  // This allocation is captured
kernel<<<...>>>(d_tmp, ...);
cudaFreeAsync(d_tmp, stream);
cudaStreamEndCapture(stream, &graph);
// This actually works if using cudaMallocAsync/cudaFreeAsync (stream-ordered allocator)
// but using cudaMalloc inside capture will likely fail
```

### Pitfall 3: Dynamic Parallelism + Graph Capture

Graph capture does not support kernels that themselves launch kernels (dynamic parallelism). Flatten your kernel hierarchy before capturing.

### Pitfall 4: `cudaDeviceSynchronize()` Inside Dynamic Parallelism

```cuda
// ✗ Stalls ALL device work — child kernels plus every other concurrent kernel
__global__ void bad_parent(...) {
    child<<<1,1>>>(...);
    cudaDeviceSynchronize();  // Blocks entire GPU
}

// ✓ Stream-scoped sync using cudaStreamSynchronize with child stream
__global__ void good_parent(...) {
    cudaStream_t child_stream;
    cudaStreamCreateWithFlags(&child_stream, cudaStreamNonBlocking);
    child<<<1, 1, 0, child_stream>>>(...);
    cudaStreamSynchronize(child_stream);  // Only waits for this child
    cudaStreamDestroy(child_stream);
}
```

---

## 19.11 Exercises

**Exercise 1 — Measure Launch Overhead**
Write a micro-benchmark that launches N empty kernels (`__global__ void noop() {}`) with and without graph capture. Plot launch time vs N. At what N does graph capture break even with the instantiation cost?

**Exercise 2 — Multi-Stream Graph**
Build a graph that captures two parallel branches (branch A: 3 kernels, branch B: 3 kernels) that merge into a final kernel. Verify with Nsight Systems that branch A and B execute concurrently in the replay.

**Exercise 3 — Cooperative Groups Reduction**
Implement a global sum reduction using grid-level cooperative groups (a single-kernel approach). Compare it to the two-pass approach (block reductions + second pass). Use `cudaLaunchCooperativeKernel`.

**Exercise 4 — Dynamic Parallelism: Adaptive Computation**
Implement a kernel that computes the L2 norm of each row of a matrix. If a row's norm exceeds a threshold, launch a child kernel to normalize it in-place. Profile to find the threshold at which dynamic parallelism is beneficial vs always normalizing.

**Exercise 5 — Production Inference Loop**
Using the `InferenceEngine` pattern from Section 19.8, build a complete pipeline:
- 2 linear layers with GELU activation
- Input: (32, 512), hidden: 2048, output: (32, 10)
- Benchmark: eager vs graph, measure throughput (inferences/sec)
- Vary batch size from 1 to 128. At what batch size does graph speedup diminish?

---

## 19.12 Chapter Summary

| Concept | Key Point |
|---|---|
| Kernel launch overhead | ~5–20 µs per launch; dominates inference for many-small-kernel workloads |
| CUDA Graph | DAG of GPU operations; captured once, replayed with single CPU call |
| `cudaStreamBeginCapture` | Records all CUDA calls on stream without executing them |
| `cudaGraphInstantiate` | Compiles the graph into executable form |
| `cudaGraphLaunch` | Replays the entire graph with one CPU call |
| Warmup before capture | Critical — initializes cuBLAS, workspace allocations must precede capture |
| `cudaGraphExecKernelNodeSetParams` | Update kernel args without re-instantiation |
| Dynamic Parallelism | Device-side kernel launch; `-rdc=true` required; child launch ~5–10 µs |
| Cooperative Groups | Flexible sync: warp, block, grid scope; replaces `__syncthreads()` |
| Grid-level CG sync | `cg::grid_group::sync()` — all blocks sync; needs `cudaLaunchCooperativeKernel` |
| PyTorch `CUDAGraph` | `torch.cuda.graph(g)` captures; `g.replay()` replays; inputs updated in-place |

---

## 19.13 What's Next

Chapter 20 is the **Capstone** — bring together everything from the curriculum to build a miniature GPU-accelerated ML library from scratch: a custom GEMM kernel, fused attention, multi-stream pipeline, Triton softmax, and a PyTorch extension wrapper that makes it all usable from Python. Every concept from Chapters 1–19 will appear at least once.

---

*Chapter 19 of 20 — CUDA GPU Programming: Beginner to Expert*
