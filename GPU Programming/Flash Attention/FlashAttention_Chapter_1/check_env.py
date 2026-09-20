"""Print what we need to know about the GPU before writing kernels."""
import torch

print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("name               :", p.name)
    print("compute capability :", f"{p.major}.{p.minor}", f"(sm_{p.major}{p.minor})")
    print("SM count           :", p.multi_processor_count)
    print("total memory       :", round(p.total_memory / 2**30, 2), "GiB")
    print("fp32 matmul mode   :", torch.get_float32_matmul_precision(), "(want 'highest' for oracle tests)")
