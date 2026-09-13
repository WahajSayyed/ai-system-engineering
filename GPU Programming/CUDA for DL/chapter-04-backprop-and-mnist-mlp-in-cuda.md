# Chapter 4 — Backpropagation & a Neural Net in Pure CUDA

*Part 3: Backpropagation & a Neural Net in Pure CUDA. Confirmed title, from the book's companion repo: `book.cu/2_mnist/README.md` states these are "progressive implementations of a two-layer MLP for MNIST digit classification, from Chapter 04 of the book."*

**A transparency note before we start:** the repo's README lists `v1.py` as "NumPy reference implementation" and `v2.py` as "PyTorch implementation." Having pulled the actual files, that's backwards relative to their content — `v1.py`'s own docstring says *"MNIST Neural Network Training with PyTorch CUDA"* and it's built entirely on `torch.nn`, while `v2.py`'s docstring says *"MNIST Neural Network Training with NumPy (CPU Implementation)"* and hand-derives every gradient. I'm presenting what the files actually contain, in the order that makes pedagogical sense (manual NumPy math first, PyTorch automation second), rather than force-fitting the stale table. Everything else about this chapter's scope — the five-way rebuild, the exact hyperparameters, the file roles — is confirmed directly from the README and the source.

The whole chapter rebuilds **one network** five times: a 784→256→10 MLP (ReLU hidden layer, softmax + cross-entropy output) trained on MNIST. Same architecture, same hyperparameters, five languages/frameworks — so you can watch backpropagation survive the trip from hand-written NumPy math down to raw CUDA kernels and back up to a library call.

Confirmed hyperparameters (from `v3.c`/`v4.cu`/`v5.cu`, identical across all three):

```c
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 8
#define EPOCHS 10
#define LEARNING_RATE 0.01
#define TRAIN_SIZE 10000
#define TEST_SIZE 10000
```

Note that tiny `BATCH_SIZE 8` — it's deliberate, and it's going to matter a lot in §4.6.

---

## 4.1 Autodiff Theory: The Chain Rule as a Graph Traversal

A forward pass is a computational graph: input → linear layer 1 → ReLU → linear layer 2 → softmax → loss. Backpropagation walks that same graph in reverse, applying the chain rule at every node: the gradient of the loss with respect to any node's *input* is the gradient with respect to its *output*, multiplied by that node's own local derivative. Three local derivatives do all the work in this network, and you can see all three, in their cleanest form, in the book's own NumPy code (`v2.py`):

**Linear layer** (`y = xW + b`):

```python
def linear_backward(grad_output, x, weights):
    grad_weights = x.T @ grad_output
    grad_bias = np.sum(grad_output, axis=0, keepdims=True)
    grad_input = grad_output @ weights.T
    return grad_input, grad_weights, grad_bias
```

Three gradients, three matrix products, and it's worth internalizing why each shape is what it is: `x` is `(batch, in)`, `grad_output` is `(batch, out)`, so `x.T @ grad_output` is `(in, out)` — exactly `weights`' shape. `grad_output @ weights.T` is `(batch, out) @ (out, in) = (batch, in)` — exactly `x`'s shape, ready to keep flowing backward into whatever produced `x`.

**ReLU** (`y = max(0, x)`): its derivative is 1 where `x > 0` and 0 elsewhere, so backprop through it is just an elementwise mask:

```python
def relu_derivative(x):
    return (x > 0).astype(float)
# applied as: grad_relu = grad_fc2 * relu_derivative(fc1_output)
```

**Softmax + cross-entropy, combined:** this is the one derivation actually worth doing by hand once, because the two functions' individual Jacobians are messy but their *composition*'s gradient collapses to something almost embarrassingly simple. For cross-entropy loss `L = -log(p_y)` where `p = softmax(logits)` and `y` is the true class, the gradient with respect to the *logits* (not the probabilities) works out to:

```
∂L/∂logit_i = p_i - 1{i == y}
```

i.e. **softmax's output, minus 1 at the true class's position, done.** No separate softmax-Jacobian and cross-entropy-Jacobian multiplication required — they cancel into a subtraction. The book's own code implements exactly this, batch-averaged:

```python
softmax_probs = softmax(y_pred)
y_true_one_hot = np.zeros_like(y_pred)
y_true_one_hot[np.arange(len(batch_y)), batch_y] = 1
grad_output = (softmax_probs - y_true_one_hot) / len(batch_y)
```

Every one of the five implementations in this chapter computes this exact quantity, in this exact form — it's the cleanest single thread you can pull to see the same math survive across languages.

**Deep dive: seeing the simplification hold with actual numbers.** It's worth doing the "long way" once, by hand, on numbers small enough to check in your head, rather than taking the cancellation on faith. Take a tiny 3-class problem, logits `[1.0, 2.0, 0.5]`, true class `y=1` (the middle one):

```
softmax([1.0, 2.0, 0.5]) ≈ [0.2312, 0.6285, 0.1402]   (p_0, p_1, p_2)
```

**The short way** (this section's formula): `∂L/∂logit_i = p_i − 1{i==y}` → `[0.2312, 0.6285−1, 0.1402] = [0.2312, −0.3715, 0.1402]`.

**The long way:** cross-entropy's gradient with respect to the *probabilities* is `∂L/∂p_i = −1{i==y}/p_y` — here, `[0, −1/0.6285, 0] ≈ [0, −1.5910, 0]`, nonzero *only* at the true class. Softmax's own Jacobian is the messy part — `∂p_i/∂logit_j = p_i(δ_ij − p_j)`, a full 3×3 matrix, not just a diagonal. Chaining the two together, `∂L/∂logit_j = Σᵢ (∂L/∂p_i)(∂p_i/∂logit_j)`, and since only the `i=1` term of that sum is nonzero, it collapses to `(−1.591) × p_1 × (δ_{1j} − p_j)` for each `j`. Working it out: at `j=0`, that's `(−1.591)×0.6285×(0−0.2312) ≈ 0.2312`; at `j=1`, `(−1.591)×0.6285×(1−0.6285) ≈ −0.3714`; at `j=2`, `(−1.591)×0.6285×(0−0.1402) ≈ 0.1402`. All three match the short way's `[0.2312, −0.3715, 0.1402]` to rounding. Two structurally different-looking derivations, same numeric answer, because the messy off-diagonal terms of softmax's Jacobian are *exactly* what's needed to cancel cross-entropy's division by `p_y` back down to a plain subtraction. Exercise 2 below asks you to confirm this in code, for arbitrary logits, rather than one hand-checked example.

## 4.2 `v0.py` — Data Preparation

Before any of the five training implementations run, `v0.py` downloads MNIST via `torchvision`, flattens each 28×28 image to a 784-length vector, normalizes pixel values from `[0,255]` to `[0.0,1.0]`, and writes four raw binary files:

```python
X_train = mnist_train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = mnist_train.targets.numpy().astype(np.int32)
X_train.tofile(os.path.join(save_dir, "X_train.bin"))
y_train.tofile(os.path.join(save_dir, "y_train.bin"))
```

This is why the C and CUDA versions (§4.4–§4.6) can read the dataset with a five-line `fread`-based loader instead of needing any image-decoding library — the binary files are just flat `float32`/`int32` arrays, exactly matching what a C `float*`/`int*` buffer expects.

## 4.3 The NumPy Implementation — Backprop Derived by Hand

This is the file the README calls `v1.py` but which actually ships as `v2.py`: every gradient from §4.1, wired into a full training loop, with zero autograd framework involved.

```python
def relu(x):
    return np.maximum(0, x)

def initialize_weights(input_size, output_size):
    # He initialization: suited to ReLU activations
    scale = np.sqrt(6.0 / input_size)
    return (np.random.rand(input_size, output_size) * 2.0 - 1.0) * scale

def linear_forward(x, weights, bias):
    return x @ weights + bias

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    batch_size = y_pred.shape[0]
    probabilities = softmax(y_pred)
    correct_log_probs = np.log(probabilities[np.arange(batch_size), y_true])
    return -np.sum(correct_log_probs) / batch_size

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = initialize_weights(input_size, hidden_size)
        self.bias1 = initialize_bias(hidden_size)
        self.weights2 = initialize_weights(hidden_size, output_size)
        self.bias2 = initialize_bias(output_size)

    def forward(self, x):
        batch_size = x.shape[0]
        fc1_input = x.reshape(batch_size, -1)
        fc1_output = linear_forward(fc1_input, self.weights1, self.bias1)
        relu_output = relu(fc1_output)
        fc2_output = linear_forward(relu_output, self.weights2, self.bias2)
        return fc2_output, (fc1_input, fc1_output, relu_output)

    def backward(self, grad_output, cache):
        x, fc1_output, relu_output = cache
        grad_fc2, grad_weights2, grad_bias2 = linear_backward(grad_output, relu_output, self.weights2)
        grad_relu = grad_fc2 * relu_derivative(fc1_output)
        grad_fc1, grad_weights1, grad_bias1 = linear_backward(grad_relu, x, self.weights1)
        return grad_weights1, grad_bias1, grad_weights2, grad_bias2

    def update_weights(self, gw1, gb1, gw2, gb2, learning_rate):
        self.weights1 -= learning_rate * gw1
        self.bias1 -= learning_rate * gb1
        self.weights2 -= learning_rate * gw2
        self.bias2 -= learning_rate * gb2
```

Notice `forward()` returns a **cache** — `(fc1_input, fc1_output, relu_output)` — alongside its output. This is the essential bookkeeping every autodiff system has to do: `backward()` needs the *forward pass's intermediate values* (the pre-ReLU output to know which units were zeroed, the post-ReLU output to compute `weights2`'s gradient) to compute correct gradients. This explicit cache is what PyTorch's autograd engine builds *automatically* and invisibly for you — seeing it written out by hand here is the whole point of doing this chapter before Part 4's PyTorch integration.

The training loop instruments five phases separately (data loading, forward, loss+gradient computation, backward, weight update) — the same five-way timing breakdown reappears, essentially unchanged, in every remaining implementation this chapter, letting you compare where time actually goes across languages later.

**Deep dive: how much actual arithmetic this network does, end to end.** It's easy to lose sight of the real compute cost behind such a small architecture, so it's worth counting once. One forward pass for a single sample costs `2×784×256` FLOPs for layer 1 plus `2×256×10` for layer 2 (the standard `2×in×out` count for a matmul, one multiply and one add per term):

```
Layer 1: 2 × 784 × 256 = 401,408 FLOPs
Layer 2: 2 × 256 × 10  =   5,120 FLOPs
Forward total ≈ 406,528 FLOPs/sample
```

At `BATCH_SIZE=8`, that's **≈3.25M FLOPs per forward pass**. Backpropagation does roughly the same amount of matrix-multiply work again (§4.1's three gradients are each one more matmul of comparable size to a forward-pass matmul), so a full step — forward + backward, ignoring the comparatively tiny elementwise/reduction ops — costs somewhere in the neighborhood of **3× the forward FLOPs, or ≈9.75M FLOPs per step**. With `TRAIN_SIZE=10000` and `BATCH_SIZE=8`, one epoch is `10000/8 = 1250` steps; over `EPOCHS=10`, that's **12,500 total steps**, or roughly **122 GFLOPs of total training compute** across the entire run. That's a genuinely tiny number by modern standards — a single RTX 3090 at its confirmed 35.58 TFLOPS FP32 peak could, in principle, finish that much raw arithmetic in **under 4 milliseconds** if it ran at peak the whole time. It obviously doesn't (data loading, kernel-launch overhead, and the batch size of 8 leaving the GPU mostly idle all eat into that), which is exactly why §4.6's kernel-launch-count arithmetic below matters so much *more* here than it would for a network doing real, GPU-saturating amounts of work per step.

## 4.4 The PyTorch Comparison — What Autograd Automates

The file the README calls `v2.py` but which actually ships as `v1.py` reimplements the identical architecture using `nn.Module`:

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.reshape(batch_size, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

criterion = nn.CrossEntropyLoss()  # combines LogSoftmax + NLLLoss
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# ...per batch:
optimizer.zero_grad()
outputs = model(data)
loss = criterion(outputs, target)
loss.backward()      # <- every §4.1 derivative, computed automatically
optimizer.step()
```

One line, `loss.backward()`, replaces §4.3's entire hand-written `backward()` method and its explicit cache. It's worth sitting with that contrast for a moment: PyTorch's autograd engine is doing *exactly* the same chain-rule graph traversal you just read in §4.1 and §4.3 — building the same forward cache, applying the same three local derivatives — just automatically, for a graph of arbitrary shape, instead of one hand-coded for this specific two-layer network. Even the weight initialization matches deliberately: the book manually overwrites PyTorch's default init with the identical He-uniform scheme (`scale = sqrt(6/fan_in)`) used in the NumPy version, so the two implementations start from comparable conditions.

## 4.5 `v3.c` — Single-Threaded C: Autograd at the Metal

Same math, no arrays-as-objects, no broadcasting — just raw pointers and nested loops. The first new thing this version needs, that NumPy's `@` operator hid from you, is that **forward and backward need different matrix-multiply shapes**, so the book implements three explicit variants:

```c
// C = A * B        — used in the forward pass
void matmul_a_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++)
                C[i * k + j] += A[i * n + l] * B[l * k + j];
        }
}

// C = A * B^T      — used for grad_input = grad_output @ weights^T
void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < n; l++)
                C[i * k + j] += A[i * n + l] * B[j * n + l];
        }
}

// C = A^T * B      — used for grad_weights = x^T @ grad_output
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++) {
            C[i * k + j] = 0.0f;
            for (int l = 0; l < m; l++)
                C[i * k + j] += A[l * n + i] * B[l * k + j];
        }
}
```

Map these straight back to §4.1's three gradient formulas and the reason there are exactly three becomes obvious: `linear_forward` needs `matmul_a_b`; `grad_input = grad_output @ weights.T` needs `matmul_a_bt`; `grad_weights = x.T @ grad_output` needs `matmul_at_b`. NumPy's `.T` and `@` were quietly picking the right one of these three loop structures for you the entire time.

**Deep dive: not all three variants access memory the same way, and it shows.** Look at the innermost loop of each — the one that actually walks through memory fastest, once per multiply-add:

- `matmul_a_b`: inner loop varies `l` in both `A[i*n+l]` and `B[l*k+j]`. For `A`, incrementing `l` moves one element at a time — contiguous, cache-friendly. For `B`, incrementing `l` jumps by `k` elements each step — strided.
- `matmul_a_bt`: inner loop varies `l` in `A[i*n+l]` and `B[j*n+l]`. **Both** operands are indexed by `l` in their *last* position — both contiguous. This is the one clean case of the three.
- `matmul_at_b`: inner (reduction) loop varies `l` in `A[l*n+i]` and `B[l*k+j]`. For `A`, incrementing `l` jumps by `n` elements — strided. For `B`, contiguous.

Every one of these three access patterns is doing the *exact same count* of multiply-adds — the difference is purely which operand's memory gets walked contiguously versus strided, and it's a direct, mechanical consequence of *which* operand is transposed. This is Chapter 3 §3.3.1's transpose-coalescing lesson, showing up again one level up: the moment `.T` appears in an expression, something's access pattern gets worse, whether that expression is being evaluated inside NumPy, by hand in C, or (once you reach §4.6) inside a CUDA kernel's own indexing arithmetic.

The forward and backward passes, fully instrumented per-operation (this per-phase timing breakdown is what lets you compare all five implementations later):

```c
void forward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int batch_size, TimingStats *stats) {
    matmul_a_b(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    bias_forward(hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    relu_forward(hidden, batch_size * HIDDEN_SIZE);
    matmul_a_b(hidden, nn->weights2, output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    bias_forward(output, nn->bias2, batch_size, OUTPUT_SIZE);
    softmax(output, batch_size, OUTPUT_SIZE);
    // (each line individually wrapped in clock_gettime() calls in the real file)
}

void compute_output_gradients(float *grad_output, float *output, int *labels, int batch_size) {
    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < OUTPUT_SIZE; i++)
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;   // exactly §4.1's p_i - 1{i==y}
    }
    for (int i = 0; i < batch_size * OUTPUT_SIZE; i++)
        grad_output[i] /= batch_size;
}

void backward_timed(NeuralNetwork *nn, float *input, float *hidden, float *output, int *labels, int batch_size, TimingStats *stats) {
    float *grad_output = malloc(batch_size * OUTPUT_SIZE * sizeof(float));
    compute_output_gradients(grad_output, output, labels, batch_size);

    matmul_at_b(hidden, grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE); // grad_weights2
    bias_backward(nn->grad_bias2, grad_output, batch_size, OUTPUT_SIZE);

    float *dX2 = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    matmul_a_bt(grad_output, nn->weights2, dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);          // grad through weights2

    float *d_ReLU_out = malloc(batch_size * HIDDEN_SIZE * sizeof(float));
    for (int i = 0; i < batch_size * HIDDEN_SIZE; i++)
        d_ReLU_out[i] = dX2[i] * (hidden[i] > 0);                                               // ReLU backward

    matmul_at_b(input, d_ReLU_out, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);     // grad_weights1
    // ...grad_bias1 similarly...
}
```

Read straight through, this function *is* §4.1's chain rule, applied node-by-node in reverse order, with every intermediate tensor from the forward pass (`hidden`, `output`) explicitly kept around and reused — the cache from §4.3, made concrete as ordinary C pointers instead of a Python tuple.

## 4.6 `v4.cu` — Custom CUDA Kernels: One Kernel Per C Function

The translation from §4.5 to CUDA is close to mechanical: every `void` function that loops over an array becomes a `__global__` kernel where each thread handles one loop iteration — you've done exactly this translation for eight different operations already, in Chapter 3.

```cuda
__global__ void matmul_a_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += A[row * n + i] * B[i * k + col];
        C[row * k + col] = sum;
    }
}
// matmul_a_bt_kernel and matmul_at_b_kernel follow the same one-thread-per-output-element
// pattern, just indexing A/B with the transposed access pattern from §4.5.

__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;              // one block per batch sample
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) max_val = fmaxf(max_val, x[b * size + i]);
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) { x[b*size+i] = expf(x[b*size+i]-max_val); sum += x[b*size+i]; }
        for (int i = 0; i < size; ++i) x[b*size+i] = fmaxf(x[b*size+i] / sum, 1e-7f);
    }
}

__global__ void compute_output_gradients_kernel(float *grad_output, float *output, int *labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) grad_output[b*OUTPUT_SIZE+i] = output[b*OUTPUT_SIZE+i];
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i) grad_output[b*OUTPUT_SIZE+i] /= batch_size;
    }
}

__global__ void weight_update_kernel(float *weights, float *grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) weights[idx] -= LEARNING_RATE * grad_weights[idx];
}
```

Launch configuration, from the real `forward_timed`:

```cuda
dim3 block_size(32, 32);   // 32*32 = 1024 threads/block — the hardware maximum on every current GPU
dim3 grid_size1((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);
matmul_a_b_kernel<<<grid_size1, block_size>>>(input, nn->weights1, hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
CUDA_CHECK(cudaDeviceSynchronize());
// ...
softmax_kernel<<<batch_size, 1>>>(output, batch_size, OUTPUT_SIZE);
```

Two things worth stopping on, because both are genuine, real inefficiencies in the book's own shipped code, not hypotheticals:

- **`dim3 block_size(32, 32)` deliberately hits the 1024-threads-per-block hardware ceiling** from Chapter 1 (§1.3.2) head-on — worth confirming that's really the max on your own hardware via the `deviceQuery` output from Chapter 1's exercises.
- **`softmax_kernel<<<batch_size, 1>>>` launches only `batch_size` threads total** — with `BATCH_SIZE 8`, that's **8 threads running on a GPU with 10,496 CUDA cores** (your RTX 3090) or 2,560 (your T4). Each of those 8 threads then does the exact same triple-pass-over-`size`-elements work Chapter 3's naive softmax did per-thread — except here there's no *redundant* work across threads (only one thread per row), just an almost total failure to use the GPU's parallelism at all. This is the single clearest real-world illustration in this chapter of Chapter 1 §1.4.2's warning: a kernel that's technically "parallel across the grid" can still leave 99%+ of the hardware idle if you picked the wrong axis to parallelize over. Fixing this — one block *per row*, many cooperating threads *within* the row via shared-memory reduction — is exactly Part 5, Chapter 16's subject.

Also worth noting structurally: every single kernel call above is followed by `CUDA_CHECK(cudaDeviceSynchronize())` before the next one launches. With this many small kernels (ten-plus per forward+backward pass) and a batch size of 8, Chapter 1 §1.3.2's "kernel launches aren't free — a few microseconds each" stops being a footnote and starts being a measurable fraction of total training time. You'll quantify this directly in the hands-on lab below.

**Deep dive: counting the actual launches, for the actual training run.** One training step launches: 6 kernels in the forward pass (`matmul_a_b`×2, `bias_forward`×2, `relu_forward`, `softmax`), 7 in the backward pass (`compute_output_gradients`, `matmul_at_b`×2 for the two weight gradients, `matmul_a_bt` for `dX2`, the ReLU-backward mask multiply, `bias_backward`×2), and 4 for the weight update (one `weight_update_kernel` launch per parameter tensor — `weights1`, `bias1`, `weights2`, `bias2`). That's **17 kernel launches per training step.** Multiply by §4.3's deep dive's own count of 1,250 steps/epoch × 10 epochs = 12,500 steps, and the *entire training run* issues **12,500 × 17 = 212,500 kernel launches.** At even a conservative few microseconds of fixed overhead each, that's **over a second of pure launch overhead** — accumulated purely from the fixed cost of *starting* kernels, before any of them have done a single multiply-add — spent training a network whose entire useful arithmetic (§4.3's deep dive) totals a bit over 100 GFLOPs, work a single GPU could finish in single-digit milliseconds if it ran at peak with zero overhead. This is the concrete, arithmetic reason `BATCH_SIZE=8` is such a deliberately punishing choice for this chapter to make: it maximizes the *ratio* of fixed per-launch cost to actual useful work per launch, making every one of this section's inefficiencies as visible as possible.

**Deep dive: exactly how much of the GPU `softmax_kernel<<<8,1>>>` actually touches.** With `batch_size=8` and one thread per block, this launch uses exactly 8 threads, each on its own block, each block scheduled onto (at most) one SM. Your RTX 3090 has 82 SMs — **at most 8 of them (≈9.8%) receive any work from this kernel at all; at least 90% of the device's SMs sit completely idle** for this launch. Your T4 has 40 SMs — 8 of 40 is 20% active, 80% idle. And even on the SMs that *do* get a block, that block's single thread is a minute fraction of what the SM can actually run concurrently (both Ampere and Turing SMs support over a thousand resident threads each) — so "20% of SMs active" is itself an optimistic upper bound on real utilization, not a measure of how *busy* those active SMs actually are.

## 4.7 `v5.cu` — CUDA + cuBLAS: Two New Techniques

Swapping the hand-written matmul kernels for `cublasSgemm` calls introduces the single trickiest real-world CUDA gotcha you'll hit early: **cuBLAS is column-major**, but every array in this chapter — NumPy, C, and the custom kernels above — has been row-major the entire time.

```cuda
const float alpha = 1.0f, beta = 0.0f;
// Forward, layer 1: computes fc1_output = weights1 @ input_batch (in cuBLAS's column-major view),
// which is exactly the row-major operation input_batch @ weights1 the earlier versions computed —
// row-major C = A@B is column-major C^T = B^T@A^T, so cuBLAS is called with the operand
// order and dimensions swapped relative to what you'd write for a row-major library.
CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       HIDDEN_SIZE, batch_size, INPUT_SIZE,
                       &alpha, nn->d_weights1, HIDDEN_SIZE,
                       nn->d_input_batch, INPUT_SIZE, &beta,
                       nn->d_fc1_output, HIDDEN_SIZE));

// Backward: grad_weights2 = grad_output^T @ hidden  (note CUBLAS_OP_T on the second operand)
CUBLAS_CHECK(cublasSgemm(nn->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                       OUTPUT_SIZE, HIDDEN_SIZE, batch_size,
                       &alpha, nn->d_grad_output, OUTPUT_SIZE,
                       nn->d_fc1_output, HIDDEN_SIZE, &beta,
                       nn->d_grad_weights2, OUTPUT_SIZE));

// Weight update — no custom kernel needed at all; SAXPY (y = alpha*x + y) is a standard BLAS Level-1 op
float neg_lr = -lr;
CUBLAS_CHECK(cublasSaxpy(nn->cublas_handle, INPUT_SIZE * HIDDEN_SIZE,
                       &neg_lr, nn->d_grad_weights1, 1, nn->d_weights1, 1));
```

The `CUBLAS_OP_N`/`CUBLAS_OP_T` flags on each operand, plus the reordered `(M, N, K)` arguments, are exactly cuBLAS's standard trick for calling a column-major library from row-major code — every one of the five `cublasSgemm` calls in this file (forward ×2, backward ×3) applies the identical transformation Chapter 3's `matmul_a_b`/`matmul_a_bt`/`matmul_at_b` did explicitly with loops. **Exercise 4 below walks you through deriving this yourself.**

**Deep dive: the general rule behind the trick, before applying it to one specific call.** The identity underneath every one of these five calls is a single, reusable fact of linear algebra: for any matrices, `(AB)ᵀ = BᵀAᵀ`. A row-major array of shape `(m, n)`, reinterpreted with no data movement at all as a column-major array, is read as its own *transpose*, shape `(n, m)`. So if your data is genuinely row-major and you hand it to a column-major library asking it to compute `C = A@B`, what you actually get — for free, with zero conversion — is `Cᵀ = Bᵀ@Aᵀ`, i.e., cuBLAS computing *your* `C` transposed, from *your* `B` and `A`, also each transposed relative to what cuBLAS thinks it's looking at. The practical recipe this collapses to: **swap the order of the two operands, and swap `M` and `N`**, leaving `K` (the shared reduction dimension) untouched. That's the whole trick — it's general to *every* row-major-code-calling-column-major-library situation, not specific to this network's shapes. Exercise 4 has you apply it concretely to this file's first `cublasSgemm` call and check every dimension by hand.

One more piece of real context worth having: `cublasSgemm` (a matrix-matrix product) and `cublasSaxpy` (`y = alpha*x + y`, a vector-vector operation) aren't two arbitrary function names — they sit at two different rungs of the standard **BLAS** (Basic Linear Algebra Subprograms) hierarchy every vendor's math library follows: **Level 1** ops are vector-vector, `O(n)` work (`saxpy`, dot products); **Level 2** are matrix-vector, `O(n²)` work (`sgemv` — exactly Chapter 5 §5.4's inference-time kernel); **Level 3** are matrix-matrix, `O(n³)` work (`sgemm`). The weight update in this file needing nothing more than a Level-1 call, while the forward/backward passes need Level-3 calls, is a direct reflection of which operations are actually matrix-shaped (the matmuls) versus which are just elementwise updates wearing a linear-algebra name (SGD's `weights -= lr * grad`, which is genuinely just a vector operation regardless of how many dimensions the parameter tensor logically has).

The second new technique is in the bias-gradient reduction — instead of the serial per-feature loop from §4.5/§4.6, `v5.cu` reduces across the batch with an **atomic add**:

```cuda
__global__ void bias_backward_kernel(float *grad_output, float *grad_bias, int batch, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch * size) {
        int bias_idx = idx % size;
        atomicAdd(&grad_bias[bias_idx], grad_output[idx]);
    }
}
```

`atomicAdd` is a hardware-guaranteed read-modify-write: when many threads across the batch try to add into the *same* `grad_bias[bias_idx]` simultaneously (every sample in the batch contributes to the same bias element), the hardware serializes just those conflicting updates instead of losing data to a race condition. It's a genuinely different strategy from §4.6's approach (one thread per output feature, looping serially over the batch inside that single thread) for the identical reduction problem — more parallelism, at the cost of some serialization when updates collide. You'll improve on both approaches directly in Exercise 5.

## 4.8 What's Identical, What Changes

| | NumPy (v2.py) | PyTorch (v1.py) | C (v3.c) | CUDA (v4.cu) | cuBLAS (v5.cu) |
|---|---|---|---|---|---|
| `grad_weights = xᵀ @ grad_out` | `x.T @ grad_output` | automatic (`.backward()`) | `matmul_at_b` | `matmul_at_b_kernel` | `cublasSgemm(OP_N,OP_T,...)` |
| Output gradient | `(softmax_probs - one_hot)/batch` | inside `CrossEntropyLoss` | `compute_output_gradients` | `compute_output_gradients_kernel` | (same, host/device split) |
| Parallelism | none (NumPy vectorizes internally) | GPU via PyTorch's own kernels | none (single CPU thread) | thread-per-output-element | cuBLAS's internal tiling |
| Weight update | manual `-=` | `optimizer.step()` | manual loop | `weight_update_kernel` | `cublasSaxpy` |

The math in row one is *identical* in every column — that's the whole point of this chapter. What changes is only **who does the parallelization and the bookkeeping**: you, by hand, in NumPy and C; you, explicitly, per-thread in raw CUDA; or a library (PyTorch's autograd, cuBLAS's kernels), invisibly. Part 4 picks up exactly here — wiring custom kernels like the ones in `v4.cu` into PyTorch's autograd system directly, so you get hand-written kernels *and* automatic differentiation at the same time.

---

## Hands-On Lab

```bash
cd book.cu/2_mnist
python v0.py                                    # one-time: downloads MNIST, writes data/*.bin

python v2.py                                     # NumPy from scratch
python v1.py                                     # PyTorch (requires CUDA)

gcc -O2 v3.c -o mnist_c -lm && ./mnist_c
nvcc -O3 v4.cu -o mnist_cuda && ./mnist_cuda
nvcc -O3 v5.cu -o mnist_cublas -lcublas && ./mnist_cublas
```

1. **Compare the per-phase timing breakdowns.** All five implementations print the same categories (data loading / forward / loss+gradient / backward / weight update). Build a small table of where time actually goes in each — you should see the C and CUDA versions spend a much larger *relative* share of time in the same phases that dominate in NumPy, but the *absolute* CUDA numbers should be far more sensitive to `BATCH_SIZE` than the CPU ones are.
2. **Verify the launch-overhead estimate empirically.** §4.6's deep dive computed 17 launches/step and ≈212,500 total across training, purely by counting. Now measure it: wrap `mnist_cuda`'s full training run with `nsys profile --stats=true` (this course won't formally introduce Nsight Systems until Chapter 10, but it's usable right now) and check its reported kernel-launch count against that hand-counted figure, and its reported average launch overhead against the "few microseconds" estimate used above. Then re-run with `BATCH_SIZE` bumped to 256 (recompile required) and see how the *fraction* of total time spent on launch overhead changes, even though the *count* of launches per step stays exactly the same.
3. **Watch the softmax anti-pattern get worse.** With `BATCH_SIZE=8`, `softmax_kernel<<<8,1>>>` runs 8 threads. Bump `BATCH_SIZE` to 1024 and re-run — the *absolute* GPU utilization for softmax barely improves relative to the matmul kernels around it, because it's still one thread per row doing all the row's work serially. Time just this kernel (CUDA events, from Chapter 2 §2.6) at both batch sizes.
4. **Run `compute-sanitizer`** on `mnist_cuda` and `mnist_cublas`. Both should report clean.

## Exercises

1. **Parallelize `softmax_kernel` for real.** Rewrite it so each *block* (not each thread) handles one row, with `size` threads cooperating: a shared-memory max-reduction, then a shared-memory sum-of-exp reduction, then a fully parallel normalize step. This is a direct preview of Part 5, Chapter 16's "online softmax" — you're allowed to look ahead.
2. **Verify the softmax+cross-entropy gradient simplification numerically.** In NumPy, compute `∂L/∂logits` two ways for the same random logits and label: (a) the one-line simplification `softmax_probs - one_hot`, and (b) the "long way" — differentiate cross-entropy with respect to the softmax *probabilities* first, then chain that through softmax's own (messy, off-diagonal) Jacobian with respect to the logits. Confirm both give numerically identical results (within float tolerance).
3. **Add momentum to the SGD update.** Extend `weight_update_kernel` (or the NumPy `update_weights`) to classical momentum: maintain a velocity buffer `v`, update `v = mu*v - lr*grad`, then `weights += v`. You'll need one extra device buffer per parameter tensor in the CUDA version.
4. **Derive the cuBLAS row-major trick yourself.** Starting from the identity "row-major `C = A@B`" is equivalent to "column-major `Cᵀ = Bᵀ@Aᵀ`", work through why the first `cublasSgemm` call in §4.7 passes `(HIDDEN_SIZE, batch_size, INPUT_SIZE)` as `(M, N, K)` instead of the naively-expected `(batch_size, HIDDEN_SIZE, INPUT_SIZE)`. Write out the shapes at each step.
5. **Reduce atomic contention with privatization.** Rewrite `bias_backward_kernel`'s `atomicAdd` version so each *block* first accumulates its own partial sum for each bias element in shared memory (no atomics needed within a block), and only one `atomicAdd` per block, per bias element, happens against global memory at the end. This "privatization" pattern is standard in real GPU histogram and reduction kernels — you're building a piece of Part 5's toolkit early.

---

**Next:** Chapter 5 — Integrating CUDA into PyTorch: The Transformer (Part 4). The PyTorch C++/CUDA extension pipeline via `pybind11`, then a character-level GPT trained with custom kernels wired directly into PyTorch's autograd — the natural next step after watching that autograd engine's job done by hand in this chapter.
