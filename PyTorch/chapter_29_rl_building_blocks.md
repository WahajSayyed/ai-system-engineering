# Chapter 29: Reinforcement Learning Building Blocks in PyTorch

## Introduction

Every training signal used so far in this book has been a differentiable loss computed directly from labeled data — cross-entropy against a known correct class, MSE against a known correct value. Reinforcement learning (RL) tackles a fundamentally different setting: there's no fixed "correct answer" to imitate, only a **reward** signal indicating how good an action turned out to be, often only known after a sequence of actions has already played out. This chapter builds the core RL building blocks in PyTorch — policy gradients, the REINFORCE algorithm, advantage estimation, and PPO's clipped objective — as the direct mechanical foundation for GRPO, which underlies your own Qwen2.5-Coder RLVR pipeline and the broader Agent RL landscape you've been exploring.

By the end of this chapter you'll be able to:
- Implement a policy network and sample actions from it
- Derive and implement the REINFORCE policy gradient from scratch
- Understand why a baseline reduces variance, and implement one
- Explain and implement PPO's clipped surrogate objective
- Recognize exactly how GRPO (Chapter 30) builds on and simplifies these components

---

## 29.1 The RL Setting, in PyTorch Terms

An RL agent has a **policy** — a function mapping a state (or, for language models, a prompt/context) to a probability distribution over actions (or, for language models, the next token) — and learns to adjust that policy to maximize expected reward. In PyTorch terms, the policy is simply an `nn.Module` (Chapter 5) whose output, instead of being interpreted as classification logits fed directly into `CrossEntropyLoss` (Chapter 6), is used to *sample* an action, and whose training signal comes from the reward that sampled action eventually produced — not from a known-correct label.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, num_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        logits = self.fc2(x)   # raw logits, exactly as in Chapter 6, Section 6.2
        return logits

policy = PolicyNetwork(state_dim=4, num_actions=2)
state = torch.rand(1, 4)

logits = policy(state)
action_dist = torch.distributions.Categorical(logits=logits)   # turns logits into a sampleable distribution
action = action_dist.sample()
log_prob = action_dist.log_prob(action)   # log-probability of the SAMPLED action, needed for the gradient

print(action.item(), log_prob.item())
```

`torch.distributions.Categorical` is worth knowing well — it wraps a set of logits (or probabilities) into a proper probability distribution object supporting `.sample()` (drawing an action according to the distribution) and `.log_prob(action)` (computing the log-probability of a specific action under that distribution) — this `log_prob` is the quantity policy gradient methods are built directly around, as the next section derives.

---

## 29.2 REINFORCE: The Simplest Policy Gradient

The **REINFORCE** algorithm (Williams, 1992) is the foundational policy gradient method, and its derivation is worth walking through carefully, since every more sophisticated method in this chapter (and GRPO in Chapter 30) is a refinement of the same core idea. The goal: adjust policy parameters $\theta$ to maximize expected reward $\mathbb{E}[R]$. The REINFORCE gradient estimator is:

$$\nabla_\theta \mathbb{E}[R] \approx \nabla_\theta \log \pi_\theta(a|s) \cdot R$$

In words: **increase the log-probability of actions that led to high reward, decrease it for actions that led to low reward** — weighted by the reward itself. This is a genuinely different kind of training signal from every loss function in Chapter 6: there's no fixed target being regressed toward; the "target" is implicitly *whatever action was actually sampled*, reinforced or discouraged in proportion to how good it turned out to be.

```python
def reinforce_loss(log_probs, rewards):
    """log_probs: (batch,) log-probabilities of the SAMPLED actions.
       rewards:   (batch,) the reward each sampled action received."""
    return -(log_probs * rewards).mean()   # negative, since optimizers MINIMIZE by default (Chapter 6)
```

The **negative sign** here deserves explicit attention, since it's the single most common sign-error location for anyone implementing this for the first time: `optimizer.step()` (Chapter 6) always *minimizes* whatever loss is returned, but REINFORCE's underlying goal is to *maximize* expected reward — negating the objective converts "maximize this" into the "minimize this" form every optimizer in this book has assumed since Chapter 6. Getting this sign wrong doesn't crash — it silently trains the policy to do the *opposite* of what's intended, actively minimizing reward, a bug that's often only caught by noticing training reward is getting steadily worse rather than better.

### A complete REINFORCE training step

```python
def train_step(policy, optimizer, states, actions, rewards):
    logits = policy(states)
    action_dist = torch.distributions.Categorical(logits=logits)
    log_probs = action_dist.log_prob(actions)   # log-prob of the actions actually taken

    loss = reinforce_loss(log_probs, rewards)

    optimizer.zero_grad()   # Chapter 3 -- exactly the same gradient accumulation concern as every other loss
    loss.backward()
    optimizer.step()
```

Notice this reuses the exact `zero_grad → backward → step` pattern from every training loop since Chapter 8 — RL's novelty is entirely in *how the loss is constructed* from sampled actions and rewards, not in how it's optimized once constructed. This is a genuinely important framing to internalize: everything from Chapters 1-28 (autograd, `nn.Module`, optimizers, mixed precision, DDP) applies to RL training completely unchanged; only the loss-construction logic in this chapter and the next is new.

---

## 29.3 Baselines: Reducing Variance

Raw REINFORCE, as implemented above, works but suffers from high variance: rewards can have a large, mostly action-irrelevant offset (e.g., in a game where *every* action currently receives a large positive reward simply because the agent is doing reasonably well overall), which the raw reward signal doesn't distinguish from genuine, action-specific quality. A **baseline** — typically an estimate of the *expected* reward from a given state, subtracted from the actual reward — addresses this directly, converting the raw reward into an **advantage**: how much *better or worse* this specific action's outcome was than what was typically expected.

$$A = R - b(s)$$

```python
def reinforce_loss_with_baseline(log_probs, rewards, baseline):
    advantages = rewards - baseline
    return -(log_probs * advantages).mean()
```

A simple, common baseline choice: the *mean reward across the current batch* — subtracting it centers the advantages around zero, so actions with above-average reward are reinforced (positive advantage) and actions with below-average reward are discouraged (negative advantage), regardless of the *absolute* reward scale, which is often uninformative or highly variable on its own:

```python
rewards = torch.tensor([10.0, 12.0, 8.0, 11.0])   # e.g., all "pretty good," but with some spread
baseline = rewards.mean()   # a simple batch-mean baseline
advantages = rewards - baseline
print(advantages)   # tensor([-0.25,  1.75, -2.25,  0.75])  -- reward SPREAD, not absolute scale
```

**This mean-subtraction baseline pattern is, directly and specifically, the mechanical core of GRPO** — "Group Relative Policy Optimization" refers precisely to computing advantages relative to a *group* of sampled outputs' mean reward, rather than using a separately-trained value-function baseline (the more traditional RL approach, and the one PPO, Section 29.4, typically uses). Chapter 30 develops this connection fully; the batch-mean baseline here is worth recognizing as its direct conceptual ancestor.

A more sophisticated (and more traditional) baseline trains a separate **value network** to predict expected reward from a state, refined via its own regression loss — this is closer to what "true" PPO implementations use, at the cost of additional model complexity and training overhead that GRPO specifically avoids, which is a large part of GRPO's practical appeal for LLM fine-tuning at scale (discussed further in Chapter 30).

---

## 29.4 PPO: Clipping the Policy Update

**Proximal Policy Optimization (PPO)** addresses a specific instability in raw policy gradient methods: a single large update — especially likely when an advantage happens to be unusually large — can move the policy so far from where it was that the gradient estimate used to make that update is no longer even a good approximation of the *new* policy's true gradient, potentially destabilizing training badly (a related concern in spirit to the exploding-gradient instability gradient clipping addresses in Chapter 8, Section 8.5, though the underlying mechanism here is distinct — it's about the *policy* changing too much per step, not the raw gradient magnitude).

PPO's core idea: compute the *ratio* between the new and old policy's probability of the action actually taken, and clip that ratio to a bounded range before using it to scale the advantage — explicitly preventing the update from moving the policy too far in a single step, regardless of how large the raw advantage signal suggests it should move.

$$r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} \qquad L^{CLIP} = \min\big(r_t \cdot A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \cdot A_t\big)$$

```python
def ppo_clipped_loss(log_probs_new, log_probs_old, advantages, epsilon=0.2):
    ratio = torch.exp(log_probs_new - log_probs_old)   # exp(log(a) - log(b)) = a/b -- the probability ratio

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages

    loss = -torch.min(unclipped, clipped).mean()   # negative -- same minimize-vs-maximize concern as 29.2
    return loss
```

Two mechanics worth being precise about, since they're easy to implement subtly wrong:

- **`torch.exp(log_probs_new - log_probs_old)`** computes the probability ratio via log-space subtraction rather than computing raw probabilities and dividing — a direct, deliberate numerical-stability choice (probabilities can be extremely small for long sequences, especially relevant for language model policies, where a full-sequence probability is a product of many per-token probabilities; working in log-space avoids the underflow that computing and dividing raw probabilities directly would risk).
- **`torch.min(unclipped, clipped)`, not `clipped` alone**: this is the actual core of PPO's design, worth understanding rather than just pattern-matching. When the advantage is positive (a good action, want to reinforce it more), the `min` takes the *more conservative* (smaller) of the two options — capping how much the update can increase that action's probability. When the advantage is negative (a bad action, want to discourage it), the `min` again takes the more conservative option in the appropriate direction — but critically, the `min` here specifically prevents the *clipped* term from creating a perverse incentive to move the ratio far outside the clip range when doing so would actually *help* the (already negative) objective; the unclipped term serves as a safety net ensuring the clipping mechanism only ever makes the objective more conservative, never accidentally more aggressive.

### Why `log_probs_old` is a separate, detached quantity

`log_probs_old` must come from the policy's state *before* any update in the current optimization step — practically, this means it's computed once (typically during a data-collection/rollout phase, before any gradient updates) and treated as a fixed constant thereafter, exactly the `.detach()` pattern from Chapter 3/4:

```python
with torch.no_grad():   # rollout phase -- Chapter 3, Section 3.5, no gradients needed here
    old_logits = policy(states)
    old_action_dist = torch.distributions.Categorical(logits=old_logits)
    log_probs_old = old_action_dist.log_prob(actions)

# ... later, potentially after several gradient updates using this same rollout data ...
new_logits = policy(states)   # policy may have already been updated since the rollout
new_action_dist = torch.distributions.Categorical(logits=new_logits)
log_probs_new = new_action_dist.log_prob(actions)

loss = ppo_clipped_loss(log_probs_new, log_probs_old, advantages)
```

This two-phase structure — collect rollout data with the current (soon-to-be "old") policy, then perform one or more gradient updates against that fixed rollout data — is characteristic of PPO and RL training generally, and is a structurally different loop shape from every training loop since Chapter 8, which processed each batch exactly once. Chapter 30 revisits this rollout/update distinction concretely in the context of GRPO's actual training loop.

---

## 29.5 Monte Carlo Return Estimation

For sequential decision problems (multiple actions taken in sequence before a final reward, as opposed to the single-action, immediate-reward examples above), the reward attributed to any given action in the sequence is typically the **discounted return** — the sum of rewards from that point forward, discounted by how far in the future they occur:

$$G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$$

```python
def compute_discounted_returns(rewards, gamma=0.99):
    """rewards: list of per-step rewards for ONE episode, in chronological order."""
    returns = []
    running_return = 0.0

    for r in reversed(rewards):   # compute from the END of the episode backward
        running_return = r + gamma * running_return
        returns.insert(0, running_return)   # prepend, to restore chronological order

    return torch.tensor(returns)

episode_rewards = [1.0, 0.0, 0.0, 1.0, 5.0]   # e.g., sparse rewards, mostly at the end
returns = compute_discounted_returns(episode_rewards, gamma=0.99)
print(returns)
```

Computing this **backward** through the episode (`reversed(rewards)`) is the standard, efficient approach — it directly reuses `running_return` from the *next* timestep rather than recomputing a full discounted sum from scratch for every single timestep, an $O(T)$ algorithm rather than the naive $O(T^2)$ approach of summing a fresh discounted series for each timestep independently. This Monte Carlo estimate (using actual, complete episode rewards rather than a learned value function's prediction) is the "Monte Carlo" in "Monte Carlo rollouts" — precisely the term from your own Agent RL exploration, referring to this exact technique of estimating returns from complete, actually-sampled trajectories rather than from a bootstrapped or predicted estimate.

---

## Summary

- A policy is an `nn.Module` whose output parameterizes a sampleable distribution (via `torch.distributions.Categorical`); training uses sampled actions and their resulting rewards, not fixed labels.
- REINFORCE's core gradient (`-log_prob * reward`, negated for minimization) increases the probability of high-reward actions and decreases it for low-reward ones — the sign is a frequent, silent bug source.
- A baseline (subtracted from raw reward to form an advantage) reduces variance by focusing the signal on relative, not absolute, reward — a simple batch-mean baseline is the direct conceptual ancestor of GRPO's "group relative" approach, covered fully in Chapter 30.
- PPO's clipped objective bounds how far a single update can shift the policy, using a log-space probability ratio between old and new policy and `torch.min` of clipped/unclipped terms to keep the clipping mechanism strictly conservative.
- RL training loops have a structurally different shape than supervised training — a rollout phase (data collection under a fixed, detached "old" policy) followed by one or more gradient updates against that fixed data — a pattern Chapter 30's GRPO implementation builds on directly.
- Discounted Monte Carlo returns, computed efficiently backward through an episode, are the standard way to attribute a per-action reward signal in sequential decision problems with sparse or delayed rewards.

## Exercises

1. Implement a complete REINFORCE training loop for a simple discrete-action environment (a custom toy environment is fine — e.g., a 1D "reach the target position" task with +1/-1 rewards), and confirm the policy's average reward improves over training.
2. Add a batch-mean baseline (Section 29.3) to your Exercise 1 implementation, and compare training stability (loss/reward variance across batches) with and without the baseline.
3. Implement the PPO clipped loss from Section 29.4 and integrate it into your Exercise 1/2 training loop, using a rollout/multiple-update structure (collect one batch of rollout data, then perform 3-4 gradient updates against it before collecting new rollout data).
4. Implement `compute_discounted_returns` from Section 29.5, and verify it against a naive, direct summation implementation (recomputing the full discounted sum from scratch for every timestep) to confirm both produce identical results, then compare their computational cost as episode length grows.

**Next:** Chapter 30 covers fine-tuning LLMs with PyTorch — LoRA from scratch and GRPO, connecting this chapter's policy gradient and PPO foundations directly to the Unsloth/GRPO pipeline you've already built for Qwen2.5-Coder.
