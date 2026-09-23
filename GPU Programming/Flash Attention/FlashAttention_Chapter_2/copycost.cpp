// Chapter 2: what does passing a std::vector BY VALUE cost?
// Build: g++ -std=c++17 -O2 -Wall -Wextra -o copycost copycost.cpp
#include <chrono>
#include <cstdio>
#include <vector>

// 'start' changes on every call so the compiler cannot reuse an earlier result.
__attribute__((noinline)) float sum_by_value(std::vector<float> x, size_t start) {          // copies the vector
    float t = 0.0f;
    for (size_t i = start; i < x.size(); i += 4096) t += x[i];
    return t;
}
__attribute__((noinline)) float sum_by_cref(const std::vector<float>& x, size_t start) {    // no copy
    float t = 0.0f;
    for (size_t i = start; i < x.size(); i += 4096) t += x[i];
    return t;
}
int main() {
    std::vector<float> x(1 << 20, 1.0f);   // 1M floats = 4 MiB
    float sink = 0.0f;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t r = 0; r < 200; ++r) sink += sum_by_value(x, r);
    auto t1 = std::chrono::steady_clock::now();
    for (size_t r = 0; r < 200; ++r) sink += sum_by_cref(x, r);
    auto t2 = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    printf("200 calls, 4 MiB vector:  by value = %.1f ms   by const& = %.3f ms   (sink=%g)\n",
           ms(t0, t1), ms(t1, t2), sink);
}
