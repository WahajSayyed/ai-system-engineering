// Prints what each kernel launch looks like, using the emulation's launch log.
#include <cstdio>
#include <vector>
#include "attention_gpu.h"
#include <cuda_runtime.h>

int main() {
    const int cases[][2] = {{1000, 64}, {1024, 64}, {37, 8}, {4096, 128}};
    for (const auto& c : cases) {
        const int N = c[0], d = c[1];
        std::vector<float> Q(N * d, 0.1f), K(N * d, 0.2f), V(N * d, 0.3f), O(N * d), L(N);
        emu_log.clear();
        attention_naive_gpu(Q.data(), K.data(), V.data(), O.data(), L.data(), N, d, false);
        const long long useful[3] = {(long long)N * N, (long long)N, (long long)N * d};
        printf("N=%d d=%d\n", N, d);
        int k = 0;
        for (const auto& l : emu_log) {
            const long long blocks = (long long)l.grid.x * l.grid.y * l.grid.z;
            const long long threads = blocks * l.block.x * l.block.y * l.block.z;
            printf("  %-20s grid=(%u,%u) block=(%u,%u)  blocks=%lld  threads=%lld  useful=%lld  idle=%.2f%%\n",
                   l.name.c_str(), l.grid.x, l.grid.y, l.block.x, l.block.y, blocks, threads, useful[k],
                   100.0 * (threads - useful[k]) / threads);
            ++k;
        }
    }
    return 0;
}
