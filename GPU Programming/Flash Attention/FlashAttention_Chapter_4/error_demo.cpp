// Chapter 4: a launch with an impossible configuration, and what CUDA_CHECK_LAUNCH() reports.
// Real CUDA source would read:   k<<<8192, 4096>>>();     (the CUDA guide uses this exact example)
// The emulation cannot parse <<< >>>, so this file spells it EMU_LAUNCH(k, 8192, 4096).
//   g++ -std=c++17 -Iemulation -I. -o error_demo error_demo.cpp && ./error_demo
#include "cuda_check.h"

__global__ void k() {}

int main() {
    EMU_LAUNCH(k, 8192, 4096);             // 4096 threads per block: more than the 1024 allowed
    printf("launched (no error visible yet)\n");
    CUDA_CHECK_LAUNCH();                   // the launch itself returned nothing; this reads the error state
    printf("not reached\n");
    return 0;
}
