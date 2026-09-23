// Chapter 2: small C++ behaviours that surprise Python programmers.
// Build: g++ -std=c++17 -O2 -Wall -Wextra -o pitfalls pitfalls.cpp
#include <cstdio>
#include <vector>

int main() {
    printf("1 / 2   = %d\n", 1 / 2);          // integer division
    printf("1 / 2.0 = %.1f\n", 1 / 2.0);      // one operand is double -> floating-point division

    float f = 0.1f;                            // 'f' suffix = 32-bit float literal
    double d = 0.1;                            // no suffix  = 64-bit double literal
    printf("0.1f stored exactly as %.17g\n", static_cast<double>(f));
    printf("0.1  stored exactly as %.17g\n", d);

    size_t i = 0;                              // size_t is UNSIGNED
    printf("size_t 0 - 1 = %zu\n", i - 1);     // wraps around instead of going negative

    std::vector<float> v(3);                   // vector<float>(3): three zeros, not garbage
    printf("vector<float>(3) = %g %g %g\n", v[0], v[1], v[2]);

    float big = 16777216.0f;                   // 2^24
    printf("float: 16777216 + 1 = %.1f\n", big + 1.0f);   // 24-bit mantissa runs out
    return 0;
}
