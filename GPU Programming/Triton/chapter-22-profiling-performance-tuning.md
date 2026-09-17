# Chapter 22 — Profiling & Performance Tuning

## 22.1 Two Profiling Philosophies

This chapter uses two genuinely different classes of tool, and the difference between them is a deliberate workflow, not an arbitrary choice of preference. **Proton** ships with Triton itself: lightweight, vendor-agnostic, low overhead, and a good first stop for "is this slow, and roughly where." **Nsight Compute** (NVIDIA) and the analogous ROCm Compute Profiler (AMD) are vendor-specific, hardware-performance-counter-based tools — (cite index="70-1">the Triton Proton profiler and vendor-specific profilers play distinct but complementary roles: Proton provides basic kernel performance metrics and context as a starting point, while the vendor-specific tool uncovers detailed hardware performance data</cite> once you already know roughly where to look. The workflow this chapter builds toward is **Proton first, vendor-specific tool second** — not "always reach for the heaviest tool available."

## 22.2 Proton, Concretely

```python
import triton.profiler as proton

session = proton.start("profile_name", context="shadow")
with proton.scope("forward", {"bytes": bytes_accessed, "flops": flops}):
    kernel[grid](...)
proton.finalize(session)
```

`proton.scope` annotates a region with custom metrics — exactly the `bytes`/`flops` calculations you've been hand-computing inline since Chapter 7's GB/s benchmark and Chapter 9's TFLOP/s benchmark, now attached as first-class, aggregated metrics Proton tracks for you rather than printed ad hoc. Two context modes are available: `"python"` (tracking the actual files/functions/lines invoking kernels) and `"shadow"` (tracking your own annotated regions specifically, independent of source location) — `"shadow"` is generally what you want when you've deliberately wrapped a region with `proton.scope`, as above.

A genuinely convenient mechanism worth knowing: rather than hand-computing `flops`/`bytes` at every call site, you can attach a `launch_metadata` callback directly to the kernel itself:

```python
def metadata_fn(grid, metadata, args):
    return {"name": "my_kernel", "flops": 2 * M * N * K}

@triton.jit(launch_metadata=metadata_fn)
def my_kernel(...):
    ...
```

Proton calls this automatically before every launch when started with `hook="triton"`, supplying the metric without you needing to compute it inline at each call site — useful when the same kernel is called from many places.

View results with the built-in viewer: `proton-viewer -m time/s profile_name.hatchet` (a tree-structured summary), or configure Chrome-trace output and open it in `chrome://tracing` or Perfetto for a timeline view.

**Two honest, current limitations worth flagging, consistent with this curriculum's treatment of maturing tools elsewhere**: Proton does not yet support graph-mode profiling on AMD GPUs, and its automatic FLOP-accumulation hooks don't work correctly under CUDA graph capture (kernels are captured and launched separately, so metrics from `launch_metadata` aren't accumulated correctly) — the documented workaround is supplying FLOPs manually via `scope()` in that specific scenario. And a genuinely frontier capability — **intra-kernel profiling**, logging metrics from *inside* a running kernel to a pre-allocated on-device buffer, read out at the end — exists as an active, not-yet-fully-merged extension at the time of this writing. Worth knowing it's coming; not yet something to depend on.

## 22.3 Nsight Compute: The Roofline Model, Made Measurable

This is where Chapter 7 and Chapter 9's structural claims about bandwidth-bound versus compute-bound kernels stop being something you reason through and become something you can directly *measure*.

**SOL% (Speed-of-Light)** is the central metric: `Compute SOL% = achieved compute throughput / peak compute throughput`; `Memory SOL% = achieved memory throughput / peak memory throughput`. (cite index="36-1">A kernel cannot saturate both simultaneously — the higher of the two reveals which resource is actually the bottleneck</cite>. This is the roofline model from Chapter 7, §7.1 and Chapter 9, §9.1, now a number you read directly off a profiling report rather than an architectural argument you constructed by reasoning about FLOPs versus bytes moved. **Profile Chapter 7's fused softmax and you should see high Memory SOL%, comparatively low Compute SOL%; profile Chapter 9's matmul at a large enough size and you should see the reverse** — a direct, empirical confirmation of claims those chapters made purely structurally.

**Occupancy** is reported directly too — `achieved_occupancy` alongside `theoretical_occupancy` — the measured empirical counterpart to the formula you computed *by hand* in Chapter 7, §7.5 and formalized further in Chapter 15, §15.4. Comparing your own hand-computed theoretical occupancy against Nsight Compute's *achieved* figure, and investigating any gap between them, is genuinely useful diagnostic practice — a divergence can reveal scheduling effects, tail-effect losses (Chapter 19), or an assumption your hand calculation didn't account for.

**A concrete, real illustration of why tile sizing (Chapter 9) matters this much**: a documented profiling session compared a "Small Block" matmul kernel using an undersized `BLOCK_SIZE_K` — producing (cite index="42-1">very small, repeated loads rather than efficiently-sized ones</cite> — against an improved version, and (cite index="42-1">measured duration dropping from 418.63 milliseconds to 10.92 milliseconds</cite> — roughly a 38x improvement, from a tile-size fix alone, made visible through real profiling rather than asserted. Another real, numbers-backed case, comparing two different work-decomposition strategies for a quantized matmul kernel: (cite index="37-1">a SplitK decomposition achieved 27.75% occupancy and 313 GB/s memory throughput at 27.90μs latency, versus a naive data-parallel decomposition's 7.55% occupancy, 161 GB/s, and 52.93μs</cite> — a genuine, measured demonstration that *how* work is decomposed across the grid (SplitK being, in spirit, a cousin of the persistent/grouped decomposition strategies from Chapter 19) has real, large consequences, not hypothetical ones.

**Practical commands, and an escalation discipline worth adopting deliberately**:

```bash
ncu --set default -o profile ./my_program        # cheap: one replay pass, roofline basics
ncu --section Occupancy --section LaunchStatistics -o occ ./my_program   # targeted, cheaper still
ncu --set full -o profile_full ./my_program      # expensive: 1200+ counters, comprehensive
ncu --launch-skip 5 --launch-count 3 ./my_program  # profile only launches 6-8, skipping warmup
```

Start with `--set default` or targeted `--section` flags to confirm *where* the problem likely is (memory-bound? compute-bound? low occupancy?); reach for `--set full`'s comprehensive, expensive counter collection only once you know what you're actually looking for. `--launch-skip`/`--launch-count` lets you profile a specific window of kernel launches — skipping the JIT-compile-tax-contaminated first calls (Chapter 2) and any warmup iterations, rather than paying profiling overhead across an entire run.

**A genuinely practical environment gotcha, worth knowing before you hit it**: (cite index="38-1">`ncu` requires elevated permissions (`--cap-add SYS_ADMIN` or `--privileged`) for hardware counter access; on many serverless or managed GPU platforms, neither flag is available, and profiling fails with `ERR_NVGPUCTRPERM` — a structural constraint of that hosting model, not a configuration issue you can work around</cite>. If you're profiling on managed cloud GPU infrastructure and hit this, the fix isn't a flag you're missing — it's a platform limitation to work around by profiling elsewhere (a dedicated instance you administer) instead.

**One more scope distinction worth being precise about**: **Nsight Systems (`nsys`)** answers "where is time going across my whole program" (a system-level timeline — kernel launches, host-device transfers, CPU work, all interleaved); **Nsight Compute (`ncu`)** answers "why is *this specific kernel* slow" (deep, single-kernel hardware-counter analysis — SOL%, occupancy, everything above). They're complementary, not interchangeable — reach for `nsys` first if you don't yet know *which* kernel is the problem, and `ncu` once you do.

## 22.4 Formalizing `triton.testing.do_bench`

You've used this since Chapter 7 without its parameters being named explicitly:

```python
triton.testing.do_bench(fn, warmup=25, rep=100, quantiles=None)
```

**`warmup`** is the mechanism that absorbs Chapter 2's JIT-compile tax — running `fn` a number of times before any measurement begins, so the first-call compilation cost never contaminates your timing. This is precisely why every benchmark you've written since Chapter 7 has been trustworthy despite never explicitly discussing this parameter — it was doing exactly the job Chapter 2, §2.3 described, silently, the whole time.

**`quantiles`** — e.g. `[0.5, 0.2, 0.8]` — returns not just a single point estimate but a spread (median, 20th percentile, 80th percentile), letting you see timing *variance*, not only central tendency. This matters as a diagnostic signal in its own right: a kernel with unexpectedly wide quantile spread across repeated measurements may be exhibiting real, non-deterministic behavior — lock contention (Chapter 12's `dw`/`db` accumulation kernel is a natural candidate to check), scheduling-luck effects from imbalanced persistent-kernel work distribution (Chapter 19), or hardware-level effects like thermal throttling — worth investigating, not just averaging away.

## 22.5 A Tiered Workflow, Put Together

1. **`triton.testing.do_bench` with `quantiles`** — an honest point estimate plus a variance check, cheap and always available.
2. **Proton scopes with real `flops`/`bytes` metrics** — a quick, portable "roughly where is time going, and how does achieved throughput compare to the ideal" pass, still cheap, still vendor-agnostic.
3. **Nsight Compute (or the ROCm equivalent), escalating from `--set default` to targeted sections to `--set full` only as needed** — read SOL% and achieved-vs-theoretical occupancy, and reconcile against your own hand-computed estimates from Chapters 7 and 15.

## 22.6 Hands-On

**Exercise 1 — Instrument two kernels with Proton.** Take Chapter 7's fused softmax and Chapter 9's matmul, wrap each in a `proton.scope` with hand-computed `flops`/`bytes` metrics (reusing those chapters' own GB/s and TFLOP/s formulas), run both, and inspect the results with `proton-viewer`.

**Exercise 2 (NVIDIA hardware) — Confirm the roofline claim with real measurements.** Profile the same two kernels with `ncu --set default`. Read off Compute SOL% and Memory SOL% for each, and confirm softmax shows high Memory SOL% with comparatively low Compute SOL%, while matmul (at a sufficiently large size) shows the reverse — direct, measured validation of Chapters 7 and 9's structural claims.

**Exercise 3 — Reconcile hand-computed occupancy against measured occupancy.** Take your own occupancy calculation from Chapter 7, §7.5 or Chapter 15 for a kernel of your choice, then measure *achieved* occupancy via Nsight Compute's `Occupancy` section. Investigate any gap between the two, and reason through what might explain it.

**Exercise 4 — Reproduce and fix a real tile-size regression.** Deliberately choose a `BLOCK_SIZE_K` too small relative to your matmul kernel's tile/warp width (echoing the "Small Block MatMul" example in §22.3), confirm via `ncu` that it produces small, inefficient, repeated loads and a correspondingly large duration, then fix it and quantify the improvement — aim to reproduce something like the cited example's dramatic before/after gap.

**Exercise 5 — Use `quantiles` to check for hidden variance.** Benchmark Chapter 12's lock-based LayerNorm backward kernel at a large `M`, using `do_bench`'s `quantiles` parameter, across a few different `GROUP_SIZE_M` values (Chapter 12's own tuning exercise). Check whether timing variance itself changes meaningfully with `GROUP_SIZE_M` — connecting a variance measurement directly to a design choice you already made there for a different reason.

**Exercise 6 (if using managed/cloud GPU infrastructure) — Confirm the permissions constraint firsthand.** Attempt to run `ncu` on your environment and see whether you encounter `ERR_NVGPUCTRPERM`. If so, this is a genuinely practical, current thing to have confirmed for yourself rather than read about.

## 22.7 Check Your Understanding

1. Why does this chapter recommend Proton before Nsight Compute, rather than always reaching for the deepest available tool immediately?
2. Explain, in your own words, why a kernel cannot show high Compute SOL% and high Memory SOL% simultaneously. What would it mean if you observed both at once?
3. What does a large gap between hand-computed theoretical occupancy and Nsight Compute's measured achieved occupancy suggest you should investigate?
4. Why is `do_bench`'s `quantiles` parameter useful beyond just "more precise" timing — what class of problem does variance itself help surface that a single median value would hide?

## 22.8 What's Next

You can now measure, not just reason about, where a kernel's time actually goes. Chapter 23 shifts from standalone kernels to integration: wrapping the forward/backward kernel pairs you've built since Chapter 11 in proper `torch.autograd.Function` and `torch.library` custom-op registration, so they compose correctly inside real, trainable PyTorch models rather than existing as isolated benchmarking scripts.
