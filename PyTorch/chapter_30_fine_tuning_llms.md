# Chapter 30: Fine-Tuning LLMs with PyTorch

## Introduction

This chapter is where the curriculum's threads converge directly onto the kind of work you've already done with Qwen2.5-Coder: transformers (Chapter 13), custom autograd (Chapter 16), quantization (Chapter 26), and policy gradients (Chapter 29) combine into two of the most practically important LLM fine-tuning techniques — LoRA (parameter-efficient fine-tuning) and GRPO (the RL algorithm behind your own SFT → GRPO → DPO pipeline). Rather than treating these as black boxes provided by Unsloth or `SFTTrainer`, this chapter builds both from scratch, so the underlying mechanics are fully transparent.

By the end of this chapter you'll be able to:
- Explain LoRA's low-rank decomposition and implement it from scratch as a custom `nn.Module`
- Apply LoRA to an existing pretrained model's linear layers, and merge adapters back in
- Implement a GRPO training step from scratch, building directly on Chapter 29's policy gradient foundations
- Explain group-relative advantage computation precisely, and why it removes the need for a separate value network
- Understand DPO's loss function and how it relates to (and differs from) the RL-based approach

---

## 30.1 LoRA: Low-Rank Adaptation

Full fine-tuning updates every parameter in a model — for an LLM with billions of parameters, this means storing a full-sized gradient and optimizer state (Chapter 24, Section 24.1's memory accounting) for every single weight, even though the *actual change* needed to adapt a pretrained model to a new task is often far smaller in effective complexity than the model's full parameter count would suggest. **LoRA (Low-Rank Adaptation)** exploits this: instead of updating a weight matrix $W$ directly, it freezes $W$ entirely and learns a small, low-rank *update* to it, expressed as the product of two much smaller matrices.

$$W' = W + \Delta W = W + BA$$

where $W \in \mathbb{R}^{d \times k}$ is the frozen pretrained weight, $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ are the newly-introduced trainable matrices, and $r$ (the **rank**) is chosen to be much smaller than $d$ or $k$ — commonly 8, 16, or 32, versus weight dimensions often in the thousands. The parameter count of $BA$ (specifically $r \times (d + k)$) is dramatically smaller than $W$'s own $d \times k$ parameters when $r$ is small — this is the entire source of LoRA's efficiency.

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16, bias=True):
        super().__init__()
        # The original, frozen linear layer -- exactly Chapter 5's nn.Linear internals
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.weight.requires_grad = False   # frozen -- Chapter 14, Section 14.3's exact mechanism
        if self.bias is not None:
            self.bias.requires_grad = False

        # LoRA's low-rank trainable matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * (1 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))   # zero-initialized -- see below
        self.scaling = alpha / rank

    def forward(self, x):
        base_output = x @ self.weight.T
        if self.bias is not None:
            base_output = base_output + self.bias

        lora_output = (x @ self.lora_A.T) @ self.lora_B.T   # x -> rank r -> out_features, two small matmuls
        return base_output + self.scaling * lora_output

layer = LoRALinear(768, 768, rank=8)
x = torch.rand(4, 10, 768)   # (batch, seq_len, features) -- standard transformer activation shape
output = layer(x)
print(output.shape)   # torch.Size([4, 10, 768])
```

Several details worth being precise about, each directly connecting to earlier chapters:

- **`self.weight.requires_grad = False`** is exactly Chapter 14, Section 14.3's freezing mechanism, applied here to the *original* pretrained weight rather than an entire pretrained backbone — LoRA is, at its core, a highly targeted, fine-grained application of the freeze/fine-tune distinction from transfer learning.
- **`lora_B` is zero-initialized, `lora_A` is randomly initialized** — this is a deliberate, important detail: since $BA$ starts at exactly zero (because $B$ is zero), the LoRA-adapted model's output is *identical* to the original pretrained model's output at the very start of fine-tuning, before any training has occurred. This gives training a clean, safe starting point — the adapted model doesn't diverge from the pretrained model's behavior until gradient updates actually begin moving $B$ away from zero.
- **`self.scaling = alpha / rank`** is a normalization convention (from the original LoRA paper) that keeps the effective magnitude of the LoRA update roughly consistent across different choices of `rank`, so that `alpha` (rather than `rank` itself) is the primary hyperparameter you'd tune when adjusting how strongly the adaptation is allowed to shift the model's behavior — directly relevant to hyperparameter choices you've likely already encountered when configuring Unsloth's LoRA settings.
- **Only `lora_A` and `lora_B` have `requires_grad=True`** (their default, since they weren't explicitly frozen) — meaning the optimizer (Chapter 6) only needs gradient and momentum/variance state (Chapter 24, Section 24.1) for these small matrices, not the full frozen weight, which is precisely where LoRA's memory savings during training come from, distinct from the (comparatively minor) parameter-count savings in the adapted weight itself.

---

## 30.2 Applying LoRA to a Real Model

In practice, LoRA is applied by walking through a pretrained model's modules and replacing target linear layers (commonly the attention projection layers in a transformer, Chapter 13) with LoRA-wrapped equivalents, directly reusing the module-tree navigation from Chapter 5, Section 5.6:

```python
def apply_lora_to_model(model, target_modules=("q_proj", "v_proj"), rank=8, alpha=16):
    for name, module in model.named_modules():   # Chapter 5, Section 5.6
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            lora_layer = LoRALinear(
                module.in_features, module.out_features, rank=rank, alpha=alpha,
                bias=module.bias is not None,
            )
            # Copy the pretrained weights into the new LoRA-wrapped layer
            lora_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                lora_layer.bias.data.copy_(module.bias.data)

            # Replace the original module with the LoRA-wrapped one, in place within the model tree
            parent_name, child_name = name.rsplit(".", 1)
            parent_module = model.get_submodule(parent_name)
            setattr(parent_module, child_name, lora_layer)

    return model
```

`target_modules=("q_proj", "v_proj")` reflects a common, well-established practice: applying LoRA specifically to the query and value projection matrices within each transformer block's attention mechanism (Chapter 12's Q/K/V projections) rather than to every linear layer in the model — empirically, adapting these specific projections captures much of the benefit of full fine-tuning at a fraction of the trainable parameter count, though the exact set of target modules is itself a tunable choice (Unsloth and other fine-tuning libraries commonly expose this as a configuration option, often extending to additional projections for more aggressive adaptation).

### Setting up the optimizer correctly

Directly reusing Chapter 14, Section 14.3's filtered-parameters pattern:

```python
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),   # only lora_A/lora_B across the whole model
    lr=2e-4,   # LoRA fine-tuning commonly uses a notably higher lr than full fine-tuning
)
```

The much higher learning rate here compared to full fine-tuning's typical `1e-5`–`2e-5` range (Chapter 14, Section 14.4) reflects LoRA's fundamentally different parameter landscape: `lora_A`/`lora_B` start from a clean initialization (Section 30.1) rather than already being in a well-trained region of parameter space, so they benefit from a learning rate closer to what you'd use training a small network from scratch (Chapter 6) rather than the very gentle nudging appropriate for already-pretrained weights.

### Merging LoRA weights back into the base model

Once fine-tuning is complete, LoRA's adapted weight can be **merged** back into a single, standard `nn.Linear` layer — collapsing $W + BA$ into one matrix, eliminating any LoRA-specific inference-time overhead and producing a model indistinguishable in structure from a fully fine-tuned one:

```python
def merge_lora_layer(lora_layer):
    merged_weight = lora_layer.weight.data + lora_layer.scaling * (lora_layer.lora_B.data @ lora_layer.lora_A.data)

    merged_linear = nn.Linear(lora_layer.weight.shape[1], lora_layer.weight.shape[0],
                                bias=lora_layer.bias is not None)
    merged_linear.weight.data.copy_(merged_weight)
    if lora_layer.bias is not None:
        merged_linear.bias.data.copy_(lora_layer.bias.data)

    return merged_linear
```

This is exactly the operation Unsloth performs when you merge a LoRA adapter and convert to GGUF for Ollama deployment, as in your own fine-tuning pipeline — `lora_B.data @ lora_A.data` reconstructs the full-rank $\Delta W$ update from its low-rank factors, added directly into the original frozen weight to produce a single, standard, deployment-ready linear layer with no remaining trace of the LoRA decomposition.

---

## 30.3 GRPO: Group Relative Policy Optimization

GRPO builds directly on Chapter 29's policy gradient and PPO foundations, with one central simplification that gives it its name: **instead of a separately-trained value network estimating expected reward (as traditional PPO uses, Chapter 29, Section 29.3), GRPO computes the baseline directly from a group of sampled completions for the same prompt** — generate several candidate responses to one prompt, score each with a reward function, and use the *group's own mean reward* as the baseline, exactly the batch-mean baseline pattern previewed in Chapter 29, Section 29.3.

```python
def compute_group_relative_advantages(rewards):
    """rewards: (group_size,) — reward for each of several completions to the SAME prompt."""
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8   # epsilon, Chapter 26/28-style numerical safety

    advantages = (rewards - mean_reward) / std_reward   # normalized, group-relative advantage
    return advantages

# For one prompt, 4 sampled completions with programmatic rewards
# (directly analogous to your own concept_reward_func / notes_reward_func)
rewards = torch.tensor([0.8, 0.3, 0.9, 0.5])
advantages = compute_group_relative_advantages(rewards)
print(advantages)   # completions above the group's mean get positive advantage, below get negative
```

Normalizing by `std_reward`, not just subtracting the mean, is a refinement beyond the simple batch-mean baseline from Chapter 29 — it keeps the advantage scale roughly consistent across prompts whose reward functions might naturally produce different absolute reward ranges or variances, which matters directly when a single training batch contains completions from many different prompts, each with their own group of samples and their own reward distribution.

### A complete GRPO training step

```python
def grpo_step(policy, optimizer, prompts, reward_fn, group_size=4, epsilon=0.2, kl_coef=0.05):
    all_log_probs_new, all_log_probs_old, all_advantages = [], [], []

    for prompt in prompts:
        # 1. Rollout phase: sample `group_size` completions for this prompt, under the CURRENT policy
        with torch.no_grad():   # Chapter 3, Section 3.5 / Chapter 29, Section 29.4 -- no gradients during rollout
            completions, log_probs_old = sample_completions(policy, prompt, group_size)

        # 2. Score each completion with the reward function
        #    (directly analogous to your concept_reward_func / notes_reward_func -- programmatic, not learned)
        rewards = torch.tensor([reward_fn(prompt, c) for c in completions])

        # 3. Group-relative advantage -- the core GRPO mechanism, Section 30.3 above
        advantages = compute_group_relative_advantages(rewards)

        # 4. Re-evaluate log-probs under the CURRENT (possibly already-updated) policy
        log_probs_new = evaluate_log_probs(policy, prompt, completions)

        all_log_probs_new.append(log_probs_new)
        all_log_probs_old.append(log_probs_old)
        all_advantages.append(advantages)

    log_probs_new = torch.cat(all_log_probs_new)
    log_probs_old = torch.cat(all_log_probs_old)
    advantages = torch.cat(all_advantages)

    # 5. PPO-style clipped objective -- Chapter 29, Section 29.4, unchanged
    loss = ppo_clipped_loss(log_probs_new, log_probs_old, advantages, epsilon=epsilon)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()
```

This structure should look immediately familiar as a direct extension of Chapter 29, Section 29.4's PPO rollout/update pattern — the two changes that define GRPO specifically are **(1)** completions are sampled in *groups per prompt* rather than as independent single samples, and **(2)** the advantage baseline comes from that group's own mean/std (Section 30.3's `compute_group_relative_advantages`) rather than from a separately-trained value network — everything else, including the clipped PPO objective itself, carries over unchanged from Chapter 29.

### `kl_coef` and staying close to the reference policy

Real GRPO implementations (including what `SFTTrainer`'s GRPO support configures under the hood) typically add a KL-divergence penalty term, discouraging the policy from drifting too far from a fixed reference model (commonly the SFT checkpoint the GRPO run started from) — this connects directly to Chapter 3's autograd mechanics and Chapter 14's frozen-reference-model pattern:

```python
def kl_penalty(log_probs_new, log_probs_reference):
    # A common, simple approximation to KL divergence between the current and reference policy
    return (log_probs_new - log_probs_reference).mean()

# In the full loss:
# loss = ppo_clipped_loss(...) + kl_coef * kl_penalty(log_probs_new, log_probs_reference)
```

`log_probs_reference` comes from a separate, entirely frozen copy of the model (loaded once before GRPO training begins and never updated, `requires_grad=False` throughout, exactly Chapter 14, Section 14.3's freezing pattern) — this frozen reference model is precisely what you'd have preserved as the SFT checkpoint in your own SFT → GRPO pipeline, used here specifically to prevent the RL phase from drifting so far from the SFT-tuned starting point that it degrades capabilities the SFT phase had already established.

---

## 30.4 DPO: A Non-RL Alternative

Given your own SFT → GRPO → DPO pipeline, it's worth placing **DPO (Direct Preference Optimization)** in context alongside GRPO, since both address a related goal — training on preference/quality signal beyond plain SFT — via meaningfully different mechanisms. Where GRPO is a genuine RL algorithm (sampling completions, computing rewards, applying a policy-gradient-style update), DPO reformulates preference learning as a **direct supervised loss** on *pairs* of (preferred, rejected) completions, with no sampling or reward function required at training time at all:

```python
def dpo_loss(log_probs_chosen, log_probs_rejected,
             log_probs_chosen_ref, log_probs_rejected_ref, beta=0.1):
    """log_probs_*: sum of per-token log-probs for the full chosen/rejected sequence,
       under the policy being trained ('_ref' suffix = frozen reference model)."""
    policy_logratios = log_probs_chosen - log_probs_rejected
    reference_logratios = log_probs_chosen_ref - log_probs_rejected_ref

    logits = policy_logratios - reference_logratios
    loss = -F.logsigmoid(beta * logits).mean()   # Chapter 6-style loss, directly differentiable
    return loss
```

The practical distinction worth internalizing: **DPO's loss is directly differentiable and requires no sampling step at all** — it's structurally closer to the supervised losses from Chapter 6 (a fixed pair of chosen/rejected sequences, a closed-form loss, `.backward()`) than to GRPO's rollout-then-update RL loop (Section 30.3), which is precisely why DPO is often simpler to implement and more stable to train than GRPO, at the cost of requiring pre-existing preference pairs (chosen vs. rejected completions) rather than GRPO's ability to generate and score its own completions on the fly using a programmatic reward function during training itself. This distinction maps directly onto why your own pipeline uses both: GRPO's on-the-fly, reward-function-driven RLVR phase for capabilities a programmatic verifier can score directly, and DPO as a more stable, directly-supervised refinement step using curated preference pairs.

---

## Summary

- LoRA freezes a pretrained weight and learns a low-rank update ($BA$, with $B$ zero-initialized so training starts identical to the unmodified pretrained model), applied to targeted layers (commonly attention Q/V projections) and mergeable back into a standard weight after training.
- LoRA's memory savings come primarily from the optimizer only needing state for the small $A$/$B$ matrices, not the full frozen weight — directly building on Chapter 14's freezing mechanism and Chapter 24's memory accounting.
- GRPO extends Chapter 29's PPO foundations with one key simplification: the advantage baseline comes from a *group* of sampled completions' own mean/std reward, removing the need for a separately-trained value network.
- A frozen reference model (the SFT checkpoint) and a KL penalty keep GRPO training from drifting too far from a known-good starting point, using the same freezing pattern from Chapter 14.
- DPO reformulates preference learning as a direct, differentiable supervised loss over (chosen, rejected) pairs — no sampling or reward function needed at training time — trading GRPO's on-the-fly flexibility for DPO's simplicity and stability.

## Exercises

1. Implement `LoRALinear` from Section 30.1, wrap a small pretrained-style `nn.Linear` layer with it, confirm the wrapped layer's output exactly matches the original layer's output before any training (verifying the zero-initialization property), then train it briefly and confirm the output now differs.
2. Implement `merge_lora_layer` from Section 30.2 and verify numerically (via `torch.allclose`) that the merged layer's output matches the unmerged LoRA layer's output on the same input.
3. Implement `compute_group_relative_advantages` from Section 30.3, and manually verify its output on a small hand-computed example (4-5 reward values), confirming the mean advantage is approximately zero and that the highest-reward completion receives the largest positive advantage.
4. Implement `dpo_loss` from Section 30.4, and using a toy example (short token-ID sequences and a trivial policy), confirm the loss decreases when training pushes `log_probs_chosen` up and `log_probs_rejected` down relative to the frozen reference model.

**Next:** Chapter 31 builds a mini training framework — a lightweight, reusable `Trainer` class (in the spirit of Hugging Face's `Trainer`/`SFTTrainer`) consolidating the training loop, checkpointing, logging, and scheduling patterns from across this entire book into one cohesive, extensible tool.
