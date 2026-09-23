// Chapter 4: what does CUDA_CHECK turn into? (uses the CPU emulation header, so it builds anywhere)
//   g++ -std=c++17 -Iemulation -I. -E -P macro_demo.cpp | tail -12        show the expansion
//   g++ -std=c++17 -Iemulation -I. -o macro_demo macro_demo.cpp && ./macro_demo
#include "cuda_check.h"

int main() {
    float* p = nullptr;
    const bool verbose = true;

    CUDA_CHECK(cudaMalloc(&p, 16 * sizeof(float)));       // an ordinary use

    if (verbose)                                          // used like ONE statement, in an if / else
        CUDA_CHECK(cudaFree(p));
    else
        printf("not freeing\n");

    printf("done\n");
    return 0;
}
