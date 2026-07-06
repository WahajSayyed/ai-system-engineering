# Chapter 28: Building a Custom Optimizer

## Introduction

Chapter 6 introduced `torch.optim` as a ready-made toolkit — `AdamW`, `SGD`, and friends — used as black boxes ever since. This chapter opens them up: implementing an optimizer from scratch, starting with plain SGD and building up to a full Adam implementation, using PyTorch's `torch.optim.Optimizer` base class exactly as PyTorch's own built-in optimizers do. Understanding this mechanism matters beyond curiosity — it's a direct prerequisite for Chapter 29's policy-gradient and GRPO material, where the update rule itself (not just the loss function) is often the object of genuine algorithmic interest, and for confidently reading or modifying optimizer implementations in research code.

By the end of this chapter you'll be able to:
- Explain the `torch.optim.Optimizer` base class's structure and responsibilities
- Implement SGD with momentum from scratch as a custom `Optimizer` subclass
- Implement Adam from scratch, understanding exactly what each of its internal state buffers does
- Correctly handle per-parameter state and parameter groups in a custom optimizer
- Verify a custom optimizer's correctness against PyTorch's built-in equivalent

---

## 28.1 The `Optimizer` Base Class

Every optimizer in `torch.optim` — and every optimizer you'll write in this chapter — subclasses `torch.optim.Optimizer`, which handles two pieces of bookkeeping common to all optimizers: tracking parameter groups (Chapter 6, Section 6.3) and providing the `zero_grad()` method (Chapter 3) you've called in every training loop since Chapter 8. What a subclass must implement itself is `step()`: the actual update rule, applied to each parameter given its current gradient.

```python
import torch
from torch.optim import Optimizer

class MyOptimizerTemplate(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)   # default hyperparameters, used unless a param group overrides them
        super().__init__(params, defaults)   # Optimizer.__init__ sets up param_groups bookkeeping

    @torch.no_grad()   # Chapter 3, Section 3.5 -- parameter updates must never be tracked by autograd
    def step(self):
        for group in self.param_groups:   # one iteration per parameter group (Chapter 6, Section 6.3)
            for p in group["params"]:
                if p.grad is None:
                    continue   # this parameter had no gradient this step -- nothing to update
                # ... the actual update rule goes here ...
```

Two structural details worth internalizing before writing a real update rule:

- **`defaults = dict(lr=lr)`**, passed to `super().__init__(params, defaults)`, establishes the hyperparameters every parameter group has access to unless it explicitly overrides them — this is precisely the mechanism underlying the parameter-groups pattern from Chapter 6, Section 6.3 and Chapter 14's differential learning rates; a custom optimizer gets that flexibility for free simply by using `self.param_groups` correctly, without needing to reimplement it.
- **`@torch.no_grad()` on `step()`** is essential, not optional — directly connecting to Chapter 8, Section 8.2's discussion of `torch.no_grad()` during validation: parameter updates (`p -= lr * grad`, or whatever the actual rule is) must never themselves be tracked by autograd, or you'd be building a computational graph *of the optimization process itself*, which is both meaningless and a severe, unbounded memory leak across a full training run.

---

## 28.2 Implementing SGD with Momentum

Starting with the simplest genuinely useful case — SGD with momentum — to establish the pattern before adding Adam's additional complexity. Plain SGD's update is `p = p - lr * grad`; momentum extends this by accumulating an exponentially-weighted moving average of past gradients (the "velocity"), smoothing out noisy per-step gradient directions and often accelerating convergence, particularly in directions where the gradient consistently points the same way across many steps.

$$v_t = \mu v_{t-1} + g_t \qquad p_t = p_{t-1} - \text{lr} \cdot v_t$$

```python
class SGDMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]   # per-parameter persistent state, keyed by the parameter itself

                if "velocity" not in state:
                    state["velocity"] = torch.zeros_like(p)   # initialize velocity to zero on first step

                velocity = state["velocity"]
                velocity.mul_(momentum).add_(grad)   # v = momentum * v + grad, IN PLACE

                p.add_(velocity, alpha=-lr)   # p = p - lr * v, IN PLACE
```

**`self.state[p]`** is the mechanism for persisting information *across* calls to `step()`, for a specific parameter — this is precisely why `Adam`/`AdamW` (Chapter 6) needs to maintain momentum and variance estimates across the entire training run, not just within a single `step()` call: `self.state` is a dictionary, keyed by the parameter tensor itself, that an `Optimizer` subclass can freely use to stash whatever per-parameter buffers its update rule needs, persisting naturally for the optimizer's full lifetime (and, as covered in Chapter 8, Section 8.6, exactly what gets saved and restored via `optimizer.state_dict()`/`load_state_dict()`).

**In-place operations (`.mul_()`, `.add_()`) are used deliberately throughout** — directly connecting back to Chapter 2, Section 2.5's in-place operations discussion: inside `step()`, wrapped in `@torch.no_grad()`, there's no autograd-safety concern about in-place mutation (the concern from Chapter 2/3 was specifically about mutating tensors *tracked by* autograd), and using in-place updates here avoids allocating a fresh tensor on every single optimizer step across a potentially very long training run — a real, meaningful memory and performance consideration at scale.

---

## 28.3 Implementing Adam from Scratch

Adam (Chapter 6) extends the momentum idea with two refinements: it maintains a *second* moving average (of squared gradients, giving each parameter its own adaptive learning rate based on how large its recent gradients have typically been), and it applies **bias correction** to compensate for both moving averages starting at zero and being biased toward zero especially in early steps.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$p_t = p_{t-1} - \text{lr} \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

```python
class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:   # first step for this parameter -- initialize all state
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)         # first moment estimate (m)
                    state["exp_avg_sq"] = torch.zeros_like(p)       # second moment estimate (v)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                # Update biased first and second moment estimates, in place
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)              # m = beta1*m + (1-beta1)*g
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)   # v = beta2*v + (1-beta2)*g^2

                # Bias correction -- essential, especially in early steps (small t)
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Parameter update
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(corrected_exp_avg, denom, value=-lr)   # p = p - lr * corrected_m / denom
```

A few implementation details worth explaining directly, since each maps to a specific piece of the math above:

- **`state["step"]`** tracks how many times this specific parameter has been updated — required for bias correction (`beta1 ** t`, `beta2 ** t`), since the correction factor's strength depends on how many updates have accumulated so far. Note this is tracked *per parameter* in this implementation (via `self.state[p]`), though in practice all parameters typically update in lockstep (same step count), since `step()` is called once per optimizer step, uniformly across every parameter.
- **`addcmul_(grad, grad, value=1 - beta2)`** computes `exp_avg_sq += (1 - beta2) * grad * grad` in a single fused, in-place operation — `addcmul_`'s specific signature (`tensor.addcmul_(t1, t2, value=v)` computes `tensor += value * t1 * t2`) is a common enough pattern (multiply two tensors elementwise, scale, and accumulate) that PyTorch provides it as one fused op rather than requiring three separate operations, which also avoids allocating an intermediate tensor for `grad * grad` before adding it.
- **Why bias correction matters concretely**: at `t=1`, without correction, `exp_avg` would equal `(1 - beta1) * grad` — a value artificially small relative to the true gradient, since `exp_avg` started at exactly zero and has only had one update to move away from that biased starting point. Dividing by `bias_correction1 = 1 - beta1^1 = 1 - beta1` exactly cancels this out at `t=1`, and the correction's effect diminishes naturally as `t` grows (since `beta1^t → 0`), which is precisely the intended behavior: strong correction early, negligible correction later, once the moving averages have had enough steps to reflect their true, unbiased values.

---

## 28.4 Verifying Correctness Against `torch.optim.Adam`

Given how easy it is to introduce a subtle sign error or misplaced bias-correction term (directly analogous to the custom `backward()` correctness concern from Chapter 16, Section 16.5), it's worth explicitly verifying a from-scratch optimizer implementation against PyTorch's own, by running both on identical starting weights, gradients, and hyperparameters, and confirming they produce identical results at each step:

```python
def compare_optimizers(steps=10):
    torch.manual_seed(0)

    # Two identical models, one for each optimizer
    model_custom = torch.nn.Linear(4, 2)
    model_builtin = torch.nn.Linear(4, 2)
    model_builtin.load_state_dict(model_custom.state_dict())   # ensure identical starting weights

    optimizer_custom = Adam(model_custom.parameters(), lr=1e-2)
    optimizer_builtin = torch.optim.Adam(model_builtin.parameters(), lr=1e-2)

    x = torch.rand(8, 4)
    y = torch.rand(8, 2)
    loss_fn = torch.nn.MSELoss()

    for step in range(steps):
        # Identical gradients for both, using the same input/target each step
        optimizer_custom.zero_grad()
        loss_custom = loss_fn(model_custom(x), y)
        loss_custom.backward()
        optimizer_custom.step()

        optimizer_builtin.zero_grad()
        loss_builtin = loss_fn(model_builtin(x), y)
        loss_builtin.backward()
        optimizer_builtin.step()

        for p_custom, p_builtin in zip(model_custom.parameters(), model_builtin.parameters()):
            if not torch.allclose(p_custom, p_builtin, atol=1e-6):
                print(f"Mismatch at step {step}!")
                return False

    print(f"All {steps} steps matched PyTorch's built-in Adam.")
    return True

compare_optimizers()
```

This test structure — identical initialization, identical inputs each step, comparing parameter values after every single step rather than only at the end — is deliberately strict: comparing only final parameters after many steps could mask an early divergence that happens to partially self-correct, while step-by-step comparison catches a discrepancy at the earliest possible point, making the specific line responsible far easier to identify.

---

## 28.5 Toward GRPO: Why Custom Optimizers Matter for RL-Style Training

This chapter's relevance extends directly into Chapter 29's policy-gradient methods and Chapter 30's GRPO material, worth previewing briefly here. In standard supervised fine-tuning (SFT), the loss function is where essentially all of the algorithmic novelty lives — the optimizer itself (`AdamW`, as configured in Chapter 6) is typically used unmodified, exactly as built. **RL-style training algorithms — including GRPO, which underlies your own Qwen2.5-Coder RLVR pipeline — sometimes require modifying the update rule itself**, not just the loss: clipping the *update magnitude* per-parameter in ways standard gradient clipping (Chapter 8, Section 8.5, which clips gradients globally by norm) doesn't directly express, or maintaining additional per-parameter state beyond what standard Adam tracks, tailored to a specific RL objective's needs.

Understanding the `Optimizer` base class's structure — `self.state[p]` for persistent per-parameter buffers, `@torch.no_grad()` for safe in-place updates, `self.param_groups` for hyperparameter management — is precisely the prerequisite for reading, modifying, or extending an existing RL training library's optimizer-level customizations (which you're likely to encounter directly when working with GRPO trainer implementations), rather than treating them as an unreadable black box.

---

## Summary

- Every `torch.optim` optimizer subclasses `Optimizer`, implementing `step()` to define the actual parameter update rule; `Optimizer.__init__` handles parameter-group bookkeeping and `zero_grad()` automatically.
- `self.state[p]`, a dictionary keyed by parameter, is how an optimizer persists per-parameter buffers (momentum, variance estimates) across steps — exactly what `optimizer.state_dict()` captures for checkpointing (Chapter 8, Section 8.6).
- `step()` must be wrapped in `@torch.no_grad()`, since parameter updates should never themselves be tracked by autograd.
- Adam's update combines a first-moment (momentum-like) and second-moment (adaptive learning rate) exponential moving average, with bias correction compensating for both starting at zero — strong early in training, negligible later.
- Verify a custom optimizer implementation against PyTorch's built-in equivalent step-by-step, on identical inputs, to catch subtle bugs at the earliest possible divergence point.
- Custom optimizer logic becomes directly relevant for RL-style training algorithms like GRPO, where the update rule itself — not just the loss — is sometimes the object of genuine algorithmic modification.

## Exercises

1. Implement `SGDMomentum` from Section 28.2, verify it against `torch.optim.SGD(..., momentum=0.9)` using the step-by-step comparison methodology from Section 28.4, and confirm exact agreement.
2. Extend the `Adam` implementation from Section 28.3 to support `AdamW`'s decoupled weight decay (Chapter 19, Section 19.3) — the weight decay term should be applied directly to `p`, separately from the gradient-based moment updates — and verify it against `torch.optim.AdamW`.
3. Modify the `Adam` implementation to add gradient clipping *within* the optimizer itself (clipping each parameter's individual gradient to a maximum absolute value before the moment updates), and explain how this differs from the global-norm gradient clipping covered in Chapter 8, Section 8.5.
4. Add support for `AMSGrad`, a well-known Adam variant that maintains the *maximum* of all past `exp_avg_sq` values (rather than the current exponential moving average) when computing the denominator — research the modification needed and verify it against `torch.optim.Adam(..., amsgrad=True)`.

**Next:** Chapter 29 covers the building blocks of reinforcement learning in PyTorch — policy gradients and PPO components — connecting the optimizer mechanics from this chapter to the RL training landscape you've already begun exploring for Agent RL.
