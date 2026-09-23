#pragma once
#include <cstdint>

// Chapter 9: the kernel skeleton -- grid layout, [B,H,N,D] stride
// arithmetic, and the shared-memory budget/opt-in. No tile loading or
// score computation yet (Chapters 10-13).

struct BlockInfo {
    int batch;
    int head;
    int qStart;
    long long bhOffset;
};

// Q, K, V, O are all [B, H, N, D], row-major. debugOut has B*H*numQTiles
// entries, one per block, in (batch, head, qTile) row-major order.
void launchFlashSkeleton(const float* Q, const float* K, const float* V, float* O,
                         int B, int H, int N, int D_runtime,
                         BlockInfo* debugOut);
