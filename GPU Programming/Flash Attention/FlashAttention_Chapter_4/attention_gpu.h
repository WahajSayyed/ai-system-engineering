// Chapter 4: host-side interface to the three-kernel GPU attention.
#pragma once

struct KernelTimes {          // milliseconds, measured with CUDA events
    float scores_ms = 0.0f;
    float softmax_ms = 0.0f;
    float pv_ms = 0.0f;
};

// Q, K, V, O are HOST arrays [N x d]; L is a HOST array [N]. Everything is copied to the GPU and back.
void attention_naive_gpu(const float* hQ, const float* hK, const float* hV,
                         float* hO, float* hL, int N, int d, bool causal,
                         KernelTimes* times = nullptr);
