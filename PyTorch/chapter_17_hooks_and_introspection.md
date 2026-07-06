# Chapter 17: Hooks & Introspection

## Introduction

So far, inspecting what happens inside a model has meant either reading its `forward()` code directly or printing intermediate values by hand. **Hooks** offer a more powerful alternative: functions that PyTorch calls automatically at specific points during the forward or backward pass, without requiring any change to the model's own code. This is invaluable for debugging — inspecting activations, catching NaN gradients, visualizing what a layer is doing — especially on models you didn't write yourself, or don't want to modify just to add temporary debug prints.

By the end of this chapter you'll be able to:
- Register and remove forward hooks, forward pre-hooks, and backward hooks
- Use hooks to inspect activations and gradients at any point in a model
- Debug vanishing/exploding gradients and NaN issues using hooks
- Understand exactly why `nn.Module.__call__` matters (completing the Chapter 5 discussion)

---

## 17.1 Why Hooks, and Why `__call__` Matters

Recall from Chapter 5, Section 5.3: you always call a module as `model(x)`, never `model.forward(x)`, because `__call__` wraps `forward()` with additional bookkeeping. Hooks are the concrete reason that bookkeeping exists — `nn.Module.__call__` checks whether any hooks are registered and, if so, invokes them at the right point around the actual `forward()` call. Calling `.forward()` directly silently skips all of this, which is precisely why that's the wrong way to invoke a module.

There are three kinds of hooks you'll use in practice:

| Hook type | Registered via | Fires... | Typical use |
|---|---|---|---|
| Forward hook | `register_forward_hook` | After a module's `forward()` completes | Inspecting/modifying activations |
| Forward pre-hook | `register_forward_pre_hook` | Before a module's `forward()` runs | Inspecting/modifying inputs |
| Backward hook | `register_full_backward_hook` | After gradients w.r.t. a module's inputs are computed | Inspecting/modifying gradients |

---

## 17.2 Forward Hooks: Inspecting Activations

A forward hook is a function called with `(module, input, output)` after that module's forward pass runs. Registering one doesn't change the model's behavior at all by default — it's purely observational unless you explicitly choose to modify the output.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
)

activations = {}

def make_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()   # detach -- Chapter 3, don't track this copy in the graph
    return hook

model[0].register_forward_hook(make_hook("linear1"))
model[1].register_forward_hook(make_hook("relu"))

x = torch.rand(4, 10)
output = model(x)

print(activations["linear1"].shape)   # torch.Size([4, 20])
print(activations["relu"].shape)       # torch.Size([4, 20])
print((activations["relu"] >= 0).all())  # True -- confirms ReLU's non-negativity, directly observed
```

`.detach()` inside the hook matters for the same reason established in Chapter 3, Section 3.5 and Chapter 4, Section 4.2: without it, `activations[name]` would hold a live reference into the computational graph, keeping it (and all the memory associated with it) alive for as long as the dictionary exists — a real memory leak risk if you're logging activations across many training steps.

### Removing hooks

Hooks persist until explicitly removed — `register_forward_hook()` returns a handle specifically for this purpose:

```python
handle = model[0].register_forward_hook(make_hook("linear1"))
# ... use the model ...
handle.remove()   # hook no longer fires on subsequent calls
```

Forgetting to remove a debugging hook before, say, saving or sharing a model is a real and easy mistake — the hook function itself isn't saved as part of `state_dict()` (hooks aren't parameters or buffers), but if a script keeps registering new hooks on every debug run without removing old ones, memory usage from the accumulating hook closures and any references they hold can grow. Prefer a `with`-style pattern (Section 17.5) for anything more than one-off, throwaway debugging.

---

## 17.3 Forward Pre-Hooks: Inspecting or Modifying Inputs

A forward pre-hook runs *before* a module's `forward()`, receiving `(module, input)` — useful for validating or modifying inputs before they reach a layer, without editing the layer's own code:

```python
def input_shape_check(module, input):
    x = input[0]   # input is always a tuple, even for single-input modules
    if torch.isnan(x).any():
        raise ValueError(f"NaN detected in input to {module}")

model[0].register_forward_pre_hook(input_shape_check)
```

Note `input` is a **tuple** of positional arguments passed to `forward()`, even when there's only one — `input[0]` is the actual tensor. This is a common point of confusion the first time you write a hook.

### Modifying inputs with a pre-hook

A forward pre-hook can also *return* a new value, which replaces the original input before it reaches `forward()` — useful for things like injecting noise for a robustness test, without permanently modifying the model's code:

```python
def add_noise(module, input):
    x = input[0]
    noisy_x = x + torch.randn_like(x) * 0.01
    return (noisy_x,)   # must return a tuple, matching the input signature

handle = model[0].register_forward_pre_hook(add_noise)
output_with_noise = model(x)
handle.remove()
output_without_noise = model(x)
```

---

## 17.4 Backward Hooks: Inspecting Gradients

Backward hooks let you inspect (or modify) gradients as they flow backward through a specific module, using the same `register_full_backward_hook` API, called with `(module, grad_input, grad_output)`:

```python
gradients = {}

def make_backward_hook(name):
    def hook(module, grad_input, grad_output):
        gradients[name] = grad_output[0].detach()
    return hook

model[0].register_full_backward_hook(make_backward_hook("linear1"))

x = torch.rand(4, 10, requires_grad=True)
output = model(x)
loss = output.sum()
loss.backward()

print(gradients["linear1"].shape)   # torch.Size([4, 20]) -- gradient flowing INTO linear1's output
```

`grad_output` is the gradient of the loss with respect to this module's *output* (the gradient flowing in from later layers); `grad_input` is the gradient with respect to this module's *input* (what it's about to pass further backward) — exactly the same directional relationship as `forward()`'s input/output, just for gradients flowing the opposite way. Both are tuples, mirroring the tuple convention from pre-hooks.

### A practical debugging use: detecting vanishing or exploding gradients

This is one of the most genuinely useful applications of backward hooks — registering hooks across every layer of a deep model to catch exactly where a gradient problem originates, rather than only observing the symptom (a NaN loss, or a model that simply isn't learning) without knowing which layer is responsible:

```python
def check_gradient_health(name):
    def hook(module, grad_input, grad_output):
        grad = grad_output[0]
        grad_norm = grad.norm().item()
        if torch.isnan(grad).any():
            print(f"[{name}] NaN gradient detected!")
        elif grad_norm > 100:
            print(f"[{name}] Large gradient norm: {grad_norm:.2f} -- possible explosion")
        elif grad_norm < 1e-7:
            print(f"[{name}] Tiny gradient norm: {grad_norm:.2e} -- possible vanishing")
    return hook

for name, layer in model.named_modules():   # Chapter 5, Section 5.6
    if isinstance(layer, (nn.Linear, nn.Conv2d)):   # only layers with meaningful gradients
        layer.register_full_backward_hook(check_gradient_health(name))
```

This pattern directly extends the debugging instincts built in Chapter 8 (gradient clipping as a *response* to exploding gradients) and Chapter 11 (vanishing gradients in deep recurrent stacks) — hooks give you the diagnostic visibility to confirm *which specific layer* a gradient problem is occurring at, rather than only observing that training is unstable somewhere in the network.

---

## 17.5 A Clean Pattern: Context-Manager-Style Hook Usage

Since forgetting to remove a hook is an easy mistake (Section 17.2), a reusable helper that guarantees cleanup, even if an exception occurs during the forward/backward pass, is worth having in your toolkit:

```python
from contextlib import contextmanager

@contextmanager
def capture_activations(module, name="activation"):
    storage = {}
    def hook(module, input, output):
        storage[name] = output.detach()
    handle = module.register_forward_hook(hook)
    try:
        yield storage
    finally:
        handle.remove()   # guaranteed cleanup, even if an exception occurs inside the `with` block

# Usage:
with capture_activations(model[1], name="relu_output") as captured:
    output = model(x)

print(captured["relu_output"].shape)
# hook is already removed here -- no manual cleanup needed, and no risk of a leftover hook
```

This pattern — `contextlib.contextmanager` wrapping a `try`/`finally` around hook registration and removal — is worth reaching for any time hooks are used for one-off inspection or debugging, rather than as a permanent, intentional part of a model's behavior.

---

## 17.6 Hooks vs. Just Modifying `forward()`

Worth being clear about when hooks are the right tool versus unnecessary indirection: if you're writing a model from scratch and want to log or inspect an intermediate value, the simplest approach is usually to just do it directly inside `forward()` — e.g., storing a value as an instance attribute, or explicitly returning it as part of a tuple. Hooks earn their place specifically when:

- You're working with a model you didn't write (a `torchvision.models` pretrained network, Chapter 14, or any third-party architecture) and don't want to fork or modify its source just to add instrumentation.
- You want instrumentation that can be added and removed dynamically, without touching model code at all — e.g., toggling detailed gradient logging on for a single debugging run without leaving debug code scattered through a production training script.
- You're building a general-purpose debugging or visualization tool intended to work across many different model architectures, where hardcoding instrumentation into each `forward()` isn't practical.

---

## Summary

- Hooks are functions PyTorch invokes automatically at specific points in the forward or backward pass, without requiring changes to a model's own `forward()` code — this is a core part of why you always call a module as `model(x)` rather than `model.forward(x)` directly.
- Forward hooks (`register_forward_hook`) inspect or modify a module's output after it runs; forward pre-hooks (`register_forward_pre_hook`) do the same for inputs, before the module runs.
- Backward hooks (`register_full_backward_hook`) inspect or modify gradients flowing through a module, and are a genuinely practical tool for localizing exactly where vanishing/exploding gradients or NaNs originate in a deep network.
- Always `.detach()` tensors captured inside a hook to avoid leaking the computational graph, and remove hooks (via their returned handle) when done — a `contextmanager`-based pattern guarantees this even if an exception occurs.
- Reach for hooks specifically when modifying `forward()` directly isn't practical or desirable — for your own simple models, direct instrumentation inside `forward()` is often simpler.

## Exercises

1. Register forward hooks on every `nn.Linear` layer in a multi-layer MLP, and print the mean and standard deviation of each layer's output on a batch of random input — a simple way to sanity-check that activations aren't collapsing to zero or exploding as they pass through the network.
2. Using the `check_gradient_health` pattern from Section 17.4, construct a deep MLP (10+ layers) with a poor weight initialization (e.g., very large initial weights) deliberately designed to cause exploding gradients, and confirm the hooks correctly flag it.
3. Write a forward pre-hook that clips input values to a module into a fixed range (e.g., `[-1, 1]`) before they're processed, and verify it changes the module's output compared to running without the hook.
4. Implement the `capture_activations` context manager from Section 17.5 yourself, then use it to capture the activations of two different layers in two separate `with` blocks, confirming each hook is cleanly removed after its block exits (e.g., by checking that a plain forward pass afterward produces no captured output).

**Next:** Chapter 18 covers learning rate scheduling — warmup, cosine annealing, and `OneCycleLR` — for shaping how the learning rate changes over the course of training, building directly on the optimizer foundations from Chapter 6.
