# PyTorch: A Code-First Guide

From `torch.tensor([1, 2, 3])` to writing your own mini training framework — a 32-chapter, code-first path through PyTorch, covering tensors and autograd, every major architecture family, training at scale, and the advanced systems and RL techniques behind modern LLM fine-tuning.

Each chapter is self-contained, code-heavy, and builds on the ones before it. Exercises are included throughout — work through them in order, or jump to whatever part is relevant to what you're building.

---

## Table of Contents

### Part I — Foundations
1. [Tensors 101](./chapter_01_tensors_101.md)
2. [Tensor Operations & Memory](./chapter_02_tensor_operations_and_memory.md)
3. [Autograd Fundamentals](./chapter_03_autograd_fundamentals.md)
4. [From NumPy to PyTorch](./chapter_04_numpy_to_pytorch.md)

### Part II — Building Blocks of Neural Networks
5. [The `nn.Module` System](./chapter_05_nn_module_system.md)
6. [Loss Functions & Optimizers](./chapter_06_loss_functions_and_optimizers.md)
7. [Datasets & DataLoaders](./chapter_07_datasets_and_dataloaders.md)
8. [The Training Loop, Deconstructed](./chapter_08_training_loop_deconstructed.md)
9. [Building an MLP from Scratch](./chapter_09_mlp_from_scratch.md)

### Part III — Core Architectures
10. [Convolutional Networks](./chapter_10_convolutional_networks.md)
11. [Recurrent Networks](./chapter_11_recurrent_networks.md)
12. [Attention Mechanisms](./chapter_12_attention_mechanisms.md)
13. [Transformers in PyTorch](./chapter_13_transformers_in_pytorch.md)
14. [Transfer Learning](./chapter_14_transfer_learning.md)

### Part IV — Training at Scale
15. [Mixed Precision Training](./chapter_15_mixed_precision_training.md)
16. [Custom Autograd Functions](./chapter_16_custom_autograd_functions.md)
17. [Hooks & Introspection](./chapter_17_hooks_and_introspection.md)
18. [Learning Rate Scheduling](./chapter_18_learning_rate_scheduling.md)
19. [Regularization Techniques](./chapter_19_regularization_techniques.md)

### Part V — Performance & Systems
20. [Profiling PyTorch Code](./chapter_20_profiling_pytorch_code.md)
21. [`torch.compile` & TorchDynamo](./chapter_21_torch_compile_and_torchdynamo.md)
22. [Multi-GPU Training I — DataParallel vs. DistributedDataParallel](./chapter_22_multi_gpu_training_1.md)
23. [Multi-GPU Training II — Process Groups & `torchrun`](./chapter_23_multi_gpu_training_2.md)
24. [Memory Optimization](./chapter_24_memory_optimization.md)

### Part VI — Advanced Topics
25. [Custom CUDA Extensions in PyTorch](./chapter_25_custom_cuda_extensions.md)
26. [Quantization](./chapter_26_quantization.md)
27. [Model Export & Deployment](./chapter_27_model_export_and_deployment.md)
28. [Building a Custom Optimizer](./chapter_28_building_a_custom_optimizer.md)
29. [Reinforcement Learning Building Blocks in PyTorch](./chapter_29_rl_building_blocks.md)
30. [Fine-Tuning LLMs with PyTorch — LoRA & GRPO](./chapter_30_fine_tuning_llms.md)
31. [Writing a Mini Training Framework](./chapter_31_mini_training_framework.md)
32. [Debugging & Testing PyTorch Code](./chapter_32_debugging_and_testing.md)

---

## How This Book Is Structured

- **Part I (Ch. 1–4)** builds the raw materials: tensors, memory semantics, and autograd.
- **Part II (Ch. 5–9)** assembles those materials into a complete, correct training pipeline, ending in a full end-to-end MLP project.
- **Part III (Ch. 10–14)** covers the major architecture families — CNNs, RNNs, attention, transformers, and transfer learning — all plugging into the same training infrastructure from Part II.
- **Part IV (Ch. 15–19)** covers training mechanics: speed, custom differentiation, introspection, scheduling, and generalization.
- **Part V (Ch. 20–24)** shifts to performance and systems: profiling, compilation, multi-GPU/multi-node training, and memory optimization.
- **Part VI (Ch. 25–32)** covers advanced, specialized techniques — custom CUDA kernels, quantization, deployment, custom optimizers, and the RL/LoRA/GRPO machinery behind modern LLM fine-tuning — closing with a capstone training framework and a consolidated debugging reference.

## Prerequisites

Working Python knowledge and basic familiarity with NumPy are assumed. No prior PyTorch or deep learning experience is required — Chapter 1 starts from `import torch`.

## Companion Material

This book is designed to complement a parallel CUDA curriculum (covering GPU programming from first principles through a capstone GPU-accelerated ML library) — Chapter 25 in particular ties the two together directly.
