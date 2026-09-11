# Chapter 14 — Compiler Pipeline Deep Dive

## 14.1 Why Open the Black Box Now

Every kernel in Part III worked, and you could benchmark it from the outside. But "it's slow" or "it's subtly wrong on this GPU but not that one" are questions the outside view can't always answer — and starting now, Part IV gives you the tools to look *inside* the compiler instead of only reasoning about its output. This chapter builds the map; Chapters 15–16 use it to explain memory and layouts concretely; Chapter 22 uses it for profiling; Chapter 25 uses it as the on-ramp to Gluon.

## 14.2 The Pipeline, End to End

```
Python source (@triton.jit function)
        │  AST walked via a standard SSA-construction algorithm
        ▼
TTIR   — Triton IR:      hardware-independent, tile-level
        ▼   (layout encodings assigned; hardware-aware optimizations)
TTGIR  — Triton GPU IR:  hardware-aware, per-target
        ▼   (standard LLVM optimization passes)
LLVM IR
        ▼
   ┌────┴────┐
   ▼         ▼
 PTX      AMDGCN        — backend-specific instruction-set text
   ▼         ▼
cubin     hsaco         — the actual loadable binary
(via ptxas)
```

Each arrow is a real, inspectable transformation, not a conceptual simplification — §14.6 shows you how to print the literal text at every stage for a kernel you've already written.

## 14.3 Stage 1: Triton IR (TTIR)

Your kernel's Python AST is walked once to construct TTIR using standard SSA (static single assignment) construction — the same foundational technique used by essentially every modern compiler. TTIR is (cite index="17-1">an unoptimized, machine-independent intermediate representation</cite> — at this stage, the compiler knows nothing yet about which GPU you're targeting, how many warps will cooperate, or how data will be laid out across threads. It only knows the tile-level algorithm you wrote: loads, stores, arithmetic, `tl.dot`, control flow — expressed using a mix of Triton's own operations and general-purpose MLIR dialects (`arith` for arithmetic, `math` for transcendental functions, `scf`/`cf` for structured and unstructured control flow). This is genuinely just your kernel, restated in a form a compiler can manipulate — nothing hardware-specific has been decided yet.

## 14.4 Stage 2: Triton GPU IR (TTGIR) — Where Hardware Enters

This is the stage where your kernel stops being an abstract tile-level algorithm and starts being a plan for *specific hardware*. Converting TTIR to TTGIR **attaches a layout encoding to every tensor type** — a precise description of how that tensor's data is actually distributed across threads, warps, and the thread block. A real example of such an encoding, for a `32×32` tensor:

```
sizePerThread = [2, 2], threadsPerWarp = [8, 4], warpsPerCTA = [2, 4]
```

Read this as: each individual thread owns a `2×2` sub-tile of elements; threads within a warp are arranged in an `8×4` grid to cover the tensor's width; warps within the whole thread block (CTA) are arranged `2×4`. **This is the literal, concrete answer to a question Chapter 3 deliberately left abstract**: "how does the compiler map a `BLOCK_SIZE`-shaped tile onto actual threads and warps?" — it does so by choosing exactly this kind of layout encoding, visible directly in TTGIR, once it has enough information (your `constexpr` tile sizes, `num_warps`, and the target GPU) to decide.

A cluster of genuinely hardware-dependent middle-end optimizations happen at this stage: memory coalescing analysis, `tl.dot`-specific scheduling improvements, and — concretely tying back to knobs you've already tuned by hand — **software pipelining is implemented here**. The `num_stages` you passed to `@triton.autotune` (Chapter 8) or to `tl.range(...)` (Chapter 6) is realized as a genuine TTGIR-level transformation at this stage, not something that happens later or is merely a hint that gets forgotten.

**How hardware-specific TTGIR actually gets, made concrete**: a recent detailed comparison compiled the *identical* 165-line TTIR for three different AMD CDNA generations and found genuinely different results at the TTGIR stage — (cite index="33-1">gfx950 issues 24 `v_mfma` instructions where gfx942 issues 48, because gfx950's matrix-multiply instruction is wider</cite>, and (cite index="33-1">gfx942 is assigned a layout kind — "linear" combined with "amd_rotating_shared" — that no other target in the entire compiler uses</cite>. Same input, same algorithm, meaningfully different TTGIR — because TTGIR's entire purpose is to encode exactly these hardware-specific decisions.

**This is also precisely where Gluon (Chapter 25) enters the pipeline** — rather than going through TTIR at all, Gluon code is lowered directly into TTGIR, because Gluon's entire premise is exposing the layout/hardware concepts this stage represents *directly* to the programmer, rather than leaving them to compiler inference. Filing that connection away now will make Chapter 25 click faster.

## 14.5 Stage 3: LLVM IR

TTGIR is converted to LLVM IR, and from here on, Triton is standing on the same compiler infrastructure that backs Clang, Rust, Swift, and most other modern compiled languages — genuinely standard LLVM optimization passes run at this stage (instruction combining, loop transformations, register allocation groundwork), not Triton-specific logic. This matters practically: it means LLVM's own extensive debugging and pass-control machinery is available to you, including the ability to **disable a specific pass you have reason to distrust**. The Triton project's own documentation gives a real example: (cite index="19-1">loop strength reduction is known to cause up to a 10% performance change for certain kernels with heavy register pressure</cite>, and can be disabled specifically via `DISABLE_LLVM_OPT="disable-lsr"` — a genuine, occasionally load-bearing escape hatch you'll have context for once Chapter 15 covers register pressure and Chapter 22 covers profiling for it.

## 14.6 Stage 4: Backend Codegen — PTX/cubin or AMDGCN/hsaco

LLVM IR is finally lowered to a backend-specific instruction-set representation: **PTX** for NVIDIA (itself further compiled into a loadable **cubin** by NVIDIA's own `ptxas` assembler, which ships inside the Triton wheel), or **AMDGCN** for AMD (assembled into a loadable **hsaco**). This is the last stage where the *specific* GPU generation, not just the vendor family, can change the output. A concrete, current illustration: compiling the same kernel for an NVIDIA `sm_120a` target (a Blackwell-generation consumer GPU) versus other Hopper/Blackwell targets shows (cite index="33-1">`sm_120a` emitting zero `wgmma` and zero `tcgen05` instructions at all, using `mma.sync` instead — because `wgmma` is Hopper-specific and `tcgen05` is specific to server-class Blackwell, while `mma.sync` is the one tensor-core instruction family that spans both Hopper and both Blackwell variants</cite>. The lesson generalizes: even within "NVIDIA tensor cores," the specific instruction your kernel actually executes is a real, non-obvious function of the exact target you compiled for — not something you can assume from the vendor name alone.

## 14.7 Inspecting Every Stage Yourself: The `.asm` Dictionary

Any compiled kernel exposes its intermediate representations directly:

```python
print(kernel.asm.keys())
# dict_keys(['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin'])
print(kernel.asm['ttgir'])
```

`kernel.asm['ttgir']` prints the literal TTGIR text — the same layout-encoding syntax from §14.4, for your actual kernel, not a textbook example. This is direct, load-bearing evidence, not a debugging curiosity: (cite index="30-1">the standard methodology for isolating a compiler-level bug is to trace it back through LLVM IR, TTGIR, and even TTIR to pinpoint exactly which stage introduced the problem</cite> — a skill this chapter is specifically building.

### A Genuinely Useful, GPU-Free Technique

Here's something worth knowing even if you don't have every GPU generation on hand: **Triton can compile — though not run — a kernel for an arbitrary target from a plain CPU, with no GPU, no CUDA toolkit, and no driver installed at all**:

```python
from triton.backends.compiler import GPUTarget

kernel = triton.compile(
    src, target=GPUTarget("cuda", 90, 32),
    options={"num_warps": 8, "num_stages": 3},
)
print(list(kernel.asm.keys()))   # ['source', 'ttir', 'ttgir', 'llir', 'ptx', 'cubin']

kernel_amd = triton.compile(
    src, target=GPUTarget("hip", "gfx942", 64),
    options={"num_warps": 8, "num_stages": 3},
)
```

Nothing in this pipeline actually touches a device — compilation is a pure, deterministic, host-side transformation, all the way down to a final `cubin`/`hsaco`. You genuinely do need a real GPU to *run* a kernel, but you don't need one to find out what the compiler decided — and, per §14.4–14.6, what the compiler decided is most of the interesting story. This makes cross-target comparison (an H100 kernel vs. a B200 kernel vs. an AMD MI300 kernel) something you can do from a single machine, entirely offline.

### Where This Connects Back to Chapter 2

Recall the on-disk kernel cache from Chapter 2, §2.4, which you inspected without yet knowing what was actually inside it. Now you do — the cache directory layout is exactly the pipeline stages, one file per stage:

```
~/.triton/cache/[CACHE_KEY]/
├── [KERNEL_NAME].ttir
├── [KERNEL_NAME].ttgir
├── [KERNEL_NAME].llir
├── [KERNEL_NAME].ptx
├── [KERNEL_NAME].cubin      # or .amdgcn / .hsaco on AMD
└── [KERNEL_NAME].json       # compilation metadata: target, options, etc.
```

Every cache entry you generated back in Chapter 2 has been sitting there this whole time, holding the full compilation trace for that specific kernel specialization — you just didn't yet have a reason to open the files.

## 14.8 The Debugging Environment Variables

Beyond `.asm`, Triton exposes a substantial set of environment variables for finer-grained inspection — worth knowing what exists even before you need most of them:

- **`MLIR_ENABLE_DUMP=1`** — dumps the IR *before and after every single MLIR pass*, not just the five-or-six named pipeline stages — far more granular than `.asm`. Defaults to `stderr`; redirect with `MLIR_DUMP_PATH=<file>`.
- **`TRITON_KERNEL_DUMP=1`** with **`TRITON_DUMP_DIR=<dir>`** — a tidier alternative: dumps every stage's IR plus the final PTX/AMDGCN to organized files in a directory, rather than a raw stream to stderr.
- **`TRITON_ALWAYS_COMPILE=1`** — forces recompilation even on a cache hit. Reach for this when you've just set one of the dump variables above and need to *guarantee* a fresh compilation actually happens, rather than silently reusing a cached binary from before you were watching.
- **`TRITON_REPRODUCER_PATH=<path>`** — generates a minimal MLIR reproducer file before each compiler stage; if a stage fails, the file left behind is a local, self-contained reproduction of exactly the failing pass — precisely what you'd attach to a genuine compiler bug report.
- **`MLIR_ENABLE_DIAGNOSTICS=<comma-separated>`** — controls diagnostic verbosity (`warnings`, `remarks`, `stacktraces`, `operations`); by default only errors are shown.
- **`TRITON_ENABLE_LLVM_DEBUG=1`** (optionally narrowed with `TRITON_LLVM_DEBUG_ONLY`) — passes `-debug` through to LLVM itself, for genuinely fine-grained LLVM-pass-level debugging when the problem has been isolated that far down the pipeline.
- **`TRITON_KERNEL_OVERRIDE=1`** with **`TRITON_OVERRIDE_DIR=<dir>`** — lets you substitute a *hand-edited* IR/PTX/AMDGCN file back into the compilation pipeline in place of what Triton would have generated. This is a genuinely powerful technique for hypothesis-testing: dump a stage, hand-edit it to test what a different scheduling or layout choice would do, and feed it back in — without touching Triton's actual compiler source at all.

You will not need most of these day-to-day. The practical habit worth building now is knowing they exist, and reaching for `MLIR_ENABLE_DUMP`/`TRITON_KERNEL_DUMP` the moment `.asm` alone isn't granular enough to isolate where a problem is introduced.

## 14.9 A Practical Debugging Habit, Stated Plainly

Before any deep dive into IR: (cite index="30-1">a genuinely common source of apparent "compiler bugs" is actually environment contamination</cite> — a stale cache entry, a mismatched Triton/PyTorch version pairing (Chapter 2, §2.7), or a leftover installation from an earlier version. The recommended first move, before you distrust the compiler itself, is boring but effective: purge `~/.triton/cache`, and if things still look wrong, consider a clean reinstall. Only once environment issues are ruled out does it make sense to start tracing a discrepancy backward through the pipeline stages — LLVM IR, then TTGIR, then TTIR — to find exactly which transformation introduced it.

## 14.10 Hands-On

**Exercise 1 — Read all five stages of a kernel you already understand.** Take Chapter 7's fused-softmax kernel, compile it (a normal launch is sufficient — `.asm` is populated either way), and print `ttir`, `ttgir`, `llir`, and `ptx` in sequence. Don't aim to understand every line; aim to *recognize the shape* of each stage — TTIR should read as a fairly direct restatement of your Python source; TTGIR should be where you can spot layout-encoding syntax attached to tensor types; PTX should be the first stage that looks like genuine, unfamiliar assembly.

**Exercise 2 — Cross-target TTGIR diff, without needing the hardware.** Using the GPU-free `triton.compile(..., target=GPUTarget(...))` technique from §14.7, compile the *same* kernel source for two different targets (two different NVIDIA compute capabilities, or an NVIDIA target and an AMD target if you want the more dramatic contrast) and diff the resulting `ttgir`. Look specifically for differences in layout encodings and instruction counts, in the spirit of the gfx90a/gfx942/gfx950 comparison in §14.4.

**Exercise 3 — A full `MLIR_ENABLE_DUMP` pass-by-pass trace.** Run any kernel from Part III with `MLIR_ENABLE_DUMP=1` and `MLIR_DUMP_PATH` pointed at a file, then skim the result. Don't try to understand every individual pass — just get a feel for *how many* distinct passes actually run between TTIR and TTGIR, and pick two or three consecutive dumps to compare closely, describing in your own words what changed between them.

**Exercise 4 — Confirm the cache-layout connection.** Set `TRITON_KERNEL_DUMP=1` and `TRITON_DUMP_DIR=<some directory>`, run a kernel, and inspect the resulting directory structure against the `~/.triton/cache` layout you first explored in Chapter 2, §2.4. Confirm they're the same shape — the same named stages, per kernel — closing the loop on a detail Chapter 2 asked you to take on faith.

**Exercise 5 (exploratory) — Produce and inspect a reproducer.** Deliberately trigger a compile-time failure — a `tl.static_assert` violation (Chapter 6) is a convenient, safe way to do this — with `TRITON_REPRODUCER_PATH` set, and open the resulting file. You're not filing a real bug; you're just building familiarity with what such a file looks like *before* the day you actually need one.

## 14.11 Check Your Understanding

1. In your own words, what is the essential difference between what TTIR represents and what TTGIR represents? Why can't hardware-specific decisions (like a layout encoding) be made at the TTIR stage?
2. The gfx90a/gfx942/gfx950 example in §14.4 compiled *identical* TTIR into different TTGIR. What does this tell you about where, in the pipeline, "the same algorithm" stops guaranteeing "the same generated code"?
3. Why is it possible to compile a Triton kernel down to a `cubin` on a machine with no GPU installed at all? What does this tell you about what compilation actually *is*, mechanically?
4. You suspect a numerical discrepancy between two kernel runs is caused by an LLVM-level optimization (rather than a bug in your kernel's logic). Walk through, in order, which tools from this chapter you'd reach for to confirm or rule this out.

## 14.12 What's Next

You can now read every stage of what your kernels actually become. Chapter 15 puts this to direct use, revisiting the GPU memory hierarchy you already know from CUDA — but this time tracing concretely, through TTGIR and LLVM IR, exactly how `tl.load`/`tl.store` become coalesced global-memory accesses, how the compiler decides what lives in shared memory, and what register spilling actually looks like once you know how to go find it.
