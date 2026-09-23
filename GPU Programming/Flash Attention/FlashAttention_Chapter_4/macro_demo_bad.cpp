// The same if / else with a macro written as a bare { ... } block instead of do { ... } while (0).
#include "cuda_check.h"
#define BAD_CHECK(call) { const cudaError_t e_ = (call); if (e_ != cudaSuccess) exit(1); }

int main() {
    float* p = nullptr;
    const bool verbose = true;
    BAD_CHECK(cudaMalloc(&p, 16));
    if (verbose)
        BAD_CHECK(cudaFree(p));
    else
        printf("not freeing\n");
    return 0;
}
