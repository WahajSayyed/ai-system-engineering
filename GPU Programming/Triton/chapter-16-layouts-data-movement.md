# Chapter 16 — Layouts & Data Movement

## 16.1 Layouts Are a Design Space, Not Just Metadata

Chapter 14 introduced the layout encoding attached to every TTGIR tensor type, and Chapter 15 showed you two concrete consequences: coalescing (which layout makes `tl.load`/`tl.store` efficient) and staging through shared memory (which layout `tl.dot` needs its operands in). This chapter fills in the piece those two left implicit: **there isn't one layout per kernel — different operations want genuinely different layouts for the same logical tensor, and reconciling that is real, sometimes expensive, compiler work.** Understanding this is the single biggest prerequisite for reading why Chapter 17's attention kernel is structured the way it is.

## 16.2 The Layout Taxonomy

Triton's layouts split into two top-level families: (cite index="42-1">Distributed layouts, where tensor elements are spread across different execution units (threads, warps), and Memory layouts, where elements are stored in a specific kind of physical memory</cite>.

**Distributed layouts** — the ones describing how a tensor's elements map onto threads/warps/registers:
- **Blocked** — (cite index="42-1">used for contiguous memory accesses</cite>; this is the `sizePerThread`/`threadsPerWarp`/`warpsPerCTA` encoding you already met in Chapter 14, and directly, (cite index="46-1">each warp owns a contiguous portion of the tensor</cite> under this layout.
- **Sliced** — (cite index="42-1">restructures and distributes a tensor along a dimension</cite>, typically appearing around reduction-adjacent reshaping.
- **MMA** — (cite index="42-1">the layout produced as the *output* of a tensor-core matrix-multiply operation</cite>, further specialized by exactly which hardware instruction produced it: `nvidia_mma` (covering both `mma` and `wgmma` instruction families), `amd_mfma`, or `amd_wmma`.
- **MMA Input / `dot_operand`** — a distinct layout, paired to a specific MMA layout, describing what shape the *inputs* to `tl.dot` need to be in to feed that particular tensor-core instruction. It carries its own metadata — which operand it is (`opIdx`, 0 or 1, for the two operands of a matmul), how many contiguous elements per thread along the reduction dimension (`kWidth`), and a reference back to the MMA layout it's paired with (`parent`).

**Memory layouts** — describing data actually resident in shared memory rather than distributed across registers:
- **Shared**, further split into **Unswizzled** and **Swizzled** variants — the swizzled form is exactly the bank-conflict-avoidance mechanism Chapter 15, §15.3 described; as of Triton 3.7, this is formalized under the name **`SwizzledShared`** as its own distinct, named layout.

**A genuinely current architectural note**: this taxonomy — one bespoke MLIR attribute class per layout kind — is actively being unified. (cite index="46-1">Triton's own project announcements have signaled a transition to a new "linear layout" representation</cite>, and a 2026 paper formalizes exactly this: representing every layout kind — Blocked, Sliced, MMA, MMA Input, and the memory layouts alike — within one unified linear-algebraic framework (using linear algebra over $\mathbb{F}_2$) rather than as separate, independently-implemented encodings. If you read Triton's source or its issue tracker and see references to "linear layouts" alongside the older named encodings, this is what's being described — a genuine, in-progress consolidation of the taxonomy above, not a separate concept you need to learn alongside it.

## 16.3 Layout "Anchors": Where a Layout Actually Comes From

Not every operation has an opinion about layout. The two places a layout is genuinely *decided*, rather than merely inherited, are:

1. **A load or store** — a dedicated **Coalesce** pass picks a Blocked layout favoring efficient, coalesced global-memory access, exactly per Chapter 15, §15.2.
2. **A `tl.dot`** — its output layout is an MMA layout, and its inputs must be in the matching `dot_operand` layout, both dictated entirely by *which specific tensor-core instruction the compiler selected* (itself a consequence of your input dtypes, per Chapter 10). This is worth being precise about: for a `tl.dot`, the layout isn't something the compiler is free to optimize for cost the way it can for an ordinary load — it's a **structural** requirement of the hardware instruction being issued, not a choice with alternatives to weigh.

Everything *between* these two kinds of anchor points — ordinary elementwise arithmetic, `tl.where`, reductions like `tl.sum`/`tl.max` — generally just **propagates** whatever layout its inputs already carry, since nothing about, say, an elementwise add structurally requires one particular layout over another.

## 16.4 `ttg.convert_layout`: When Propagation Isn't Enough

The interesting case — and the one this chapter exists to prepare you for — is when a single tensor needs to flow from one anchor into a *different* kind of anchor: for instance, the output of one `tl.dot` (an MMA layout) needing to become the input to a second `tl.dot` (a `dot_operand` layout, likely a genuinely different shape of distribution), with some elementwise/reduction work in between. When the layout a value already has doesn't match what its next consumer structurally requires, the compiler inserts an explicit **`ttg.convert_layout`** operation — a real, physical rearrangement of which thread/register holds which logical element, not a free reinterpretation.

**Not all conversions cost the same, and this distinction matters.** (cite index="45-1">Some layout conversions can be completed entirely "intra-warp" — via register-level data shuffling within a single warp, with no memory traffic at all. Others require moving data across warp boundaries, which Triton implements implicitly by staging through shared memory</cite> when an intra-warp shuffle genuinely can't accomplish the needed rearrangement. A cross-warp conversion, in other words, becomes a real store-to-shared-memory followed by a reload in the new layout — the same shared-memory capacity and latency cost as any other use of shared memory, spent purely on reshuffling data that was already correct, just laid out the wrong way for its next consumer.

### A Real Illustration: When This Actually Fails

This isn't a hypothetical performance tax — it can become an outright compilation failure. A documented, real case: a kernel took a `#blocked`-layout tensor, performed a `ttg.local_alloc` (store the **entire** tensor to shared memory) followed by a `ttg.local_load` back out in a different, `dot_op`-compatible layout — precisely the store-then-reload conversion strategy just described. (cite index="47-1">This worked correctly for one tile configuration (N=256, K=128), but failed outright for a larger one (N=512, K=128), because storing that much data — 512×128 `bfloat16` elements — into shared memory purely to change its layout exceeded the GPU's available shared memory budget entirely</cite>. The lesson: a layout conversion you never asked for and can't see in your Python source can, in the wrong combination of tile sizes, actually be the thing that makes a kernel fail to compile at all — not merely run slower. This is exactly the kind of failure Chapter 15's shared-memory-footprint awareness (`kernel.metadata.shared`) and this chapter's TTGIR-reading skill together let you diagnose, rather than experience as a mysterious out-of-resources error.

## 16.5 `RemoveLayoutConversions`: The Compiler Tries to Minimize This For You

You are not expected to hand-eliminate every conversion yourself — Triton's compiler has a dedicated optimization pass for exactly this problem. (cite index="45-1">The Coalesce pass establishes the memory-access-side layout anchor, the dot-operation's structural requirements establish the compute-side anchor, and together these define "boundaries" for how layouts are allowed to propagate; a dedicated `RemoveLayoutConversions` pass then works within those boundaries specifically to eliminate unnecessary conversions</cite>. This is the same category of thing as an LLVM optimization pass (Chapter 14, §14.5) — a real, targeted transformation aimed at a known-expensive pattern — except here it's operating specifically on the layout-anchor structure this chapter has just described. Your job, as a kernel author, is not to manually route around every possible conversion; it's to **recognize when this pass hasn't fully succeeded** (a conversion survives into your compiled kernel where you didn't expect one) and understand why, using exactly the TTGIR-reading skills from Chapter 14.

## 16.6 Why This Chapter Exists Before Chapter 17

Flash attention's core structure is: a `tl.dot` computing `Q @ K^T` (producing an MMA-layout output) → a softmax-style reduction and elementwise rescaling (naturally propagating something closer to a Blocked-style layout) → a *second* `tl.dot`, multiplying by `V` (requiring its own `dot_operand` layout, not necessarily the same shape of distribution as the first dot's output). This is precisely the pattern that forces multiple layout anchors to coexist within a single kernel, and makes both `RemoveLayoutConversions`' success and your own awareness of conversion cost genuinely load-bearing for that kernel's real-world performance — not an academic concern. This chapter exists as its own chapter, rather than being folded silently into Chapter 17, specifically so you have this vocabulary *before* you need it there.

## 16.7 Inspecting This Yourself

Exactly the technique from Chapter 14, §14.7, aimed at a new target: pull `kernel.asm['ttgir']` and search the text for `ttg.convert_layout`, `ttg.local_alloc`, and `ttg.local_load`. Their presence — and, just as importantly, *where* in the kernel they appear relative to your `tl.dot` calls — is direct, concrete evidence of where layout-conversion cost is actually being paid in your compiled kernel, the same kind of static, ground-truth check as grepping for `ld.local`/`st.local` to confirm register spilling in Chapter 15.

## 16.8 Hands-On

**Exercise 1 — Find the anchors in a kernel you already know.** Dump the TTGIR for Chapter 9's matmul kernel and locate: the `#blocked` layout attached to the initial `A`/`B` loads, and the `#mma`/`#dot_op` layouts attached around the `tl.dot` call. Identify whether a `ttg.convert_layout` appears bridging them, or whether the compiler managed to avoid one entirely for this particular kernel shape.

**Exercise 2 — Deliberately force an expensive conversion.** Write a small kernel that loads a tile, performs a `tl.dot` with it in one orientation, and then immediately performs a *second* `tl.dot` using the same underlying data in a transposed or otherwise differently-oriented way. Dump the TTGIR and confirm you can see `ttg.convert_layout` (or a `local_alloc`/`local_load` pair) appear. Then rewrite the kernel to load the data pre-oriented for the second `tl.dot` from the start (using Chapter 5's stride tricks, rather than reorienting after the fact), and confirm the conversion disappears from the TTGIR — benchmark both versions and quantify the cost you just eliminated.

**Exercise 3 — Reproduce a shared-memory-exhaustion failure.** Following §16.4's real example, pick tile sizes large enough that a compiler-inserted store-whole-tensor-then-reload-in-a-different-layout strategy would plausibly exceed your GPU's shared memory budget. Confirm you can trigger a genuine compile-time shared-memory allocation failure, and connect the failure back to `kernel.metadata.shared` (Chapter 15) and the layout-conversion mechanism (this chapter) that caused it — rather than treating it as an opaque error.

**Exercise 4 — Cross-target layout diff.** Using Chapter 14 §14.7's GPU-free compilation technique, compile the same `tl.dot`-containing kernel for two targets that use genuinely different tensor-core instruction families (for instance, an older NVIDIA target using plain `mma` versus a newer one using `wgmma`, or an NVIDIA target versus an AMD `mfma` target). Diff the TTGIR specifically around the `tl.dot`-adjacent layouts, and confirm the `dot_operand`/MMA layout details genuinely differ by target — direct, hands-on confirmation of §16.3's claim that these layouts are structurally dictated by the instruction selected, not by your algorithm.

## 16.9 Check Your Understanding

1. Why does a `tl.dot`'s layout requirement get described as "structural" rather than something the compiler optimizes for cost, in contrast to the Blocked layout chosen for an ordinary load?
2. Explain the difference, in both mechanism and cost, between an intra-warp layout conversion and one that requires staging through shared memory.
3. Using the GH issue #6446 example from §16.4, explain in your own words why a layout conversion strategy that works at one tile size can fail outright at a larger one — what resource, specifically, is being exhausted?
4. Why does flash attention's structure (Chapter 17) make layout-conversion awareness more load-bearing than it was for the matmul kernel in Chapter 9?

## 16.10 What's Next

That completes Part IV — you can now read a kernel's compilation all the way from Python source to hardware instructions, and you understand the layout mechanics that govern both memory access and tensor-core staging. Part V puts all of it to work on the most demanding kernel in this curriculum so far: Chapter 17 builds the forward pass of fused (flash) attention, where online softmax, multiple `tl.dot` calls, and exactly the layout-anchor tensions this chapter described all have to be managed together in one kernel.
