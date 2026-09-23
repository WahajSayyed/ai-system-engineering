"""Chapter 3: do our tests have teeth?  Break the tiled C++ code on purpose, one bug at a time,
rebuild, rerun main_test, and see whether the test suite notices.  Run: python mutants.py"""
import os
import subprocess
import tempfile

here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "tiled_attention.cpp")).read()

MUTANTS = [
    ("forget to shrink the old accumulator",
     "                for (size_t x = 0; x < d; ++x) a[x] *= alpha;        // shrink what we had\n", ""),
    ("forget alpha when updating l",
     "l[r] = alpha * l[r] + row_sum;", "l[r] = l[r] + row_sum;"),
    ("forget the 1/sqrt(d) scale",
     "S[r * Bc + c] = masked ? -INFINITY : dot * scale;", "S[r * Bc + c] = masked ? -INFINITY : dot;"),
    ("forget the causal mask inside a tile",
     "const bool masked = causal && (c0 + c) > (r0 + r);", "const bool masked = false;"),
    ("forget the final division by l",
     "const float inv = (l[r] > 0.0f) ? 1.0f / l[r] : 0.0f;", "const float inv = 1.0f;"),
    ("row stride bc instead of Bc when reading S",
     "float* s = &S[r * Bc];", "float* s = &S[r * bc];"),
    ("every output is NaN (does the test notice?)",
     "const float inv = (l[r] > 0.0f) ? 1.0f / l[r] : 0.0f;", "const float inv = NAN;"),
    ("skip the -inf guard (use m_new directly)",
     "const float m_safe = (m_new == -INFINITY) ? 0.0f : m_new;", "const float m_safe = m_new;"),
]

def build_and_run(mutated, extra_flags=()):
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "tiled_attention.cpp")
    open(path, "w").write(mutated)
    exe = os.path.join(tmp, "t")
    cmd = ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", *extra_flags, "-I", here, path,
           os.path.join(here, "main_test.cpp"), "-o", exe]
    b = subprocess.run(cmd, capture_output=True, text=True)
    if b.returncode:
        return "BUILD FAILED: " + b.stderr.splitlines()[0]
    r = subprocess.run([exe], capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if "configurations" in l]
    err = [l.strip() for l in r.stderr.splitlines() if "ERROR: AddressSanitizer" in l or " in attention_tiled" in l]
    return (out[0] if out else "no summary line") + f"   [exit {r.returncode}]" + ("\n      " + "\n      ".join(err[:2]) if err else "")

print("unmodified          :", build_and_run(src))
for name, old, new in MUTANTS:
    assert src.count(old) == 1, name
    print(f"\n{name}\n   ->", build_and_run(src.replace(old, new)))

print("\n--- ignoring ragged last tile: br = Br instead of min(Br, N - r0), built with AddressSanitizer ---")
old, new = "const size_t br = std::min(Br, N - r0);", "const size_t br = Br;"
assert src.count(old) == 1
res = build_and_run(src.replace(old, new), ["-fsanitize=address", "-g"])
print("   ->", res)
