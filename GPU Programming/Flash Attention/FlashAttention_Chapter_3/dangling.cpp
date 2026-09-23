#include <cstdio>
#include <vector>
int main() {
    std::vector<float> v{1, 2, 3};
    const float* p = v.data();
    printf("before: p[0] = %g\n", p[0]);
    for (int i = 0; i < 100; ++i) v.push_back(0.0f);   // the vector reallocates: old memory is freed
    printf("did the storage move? %s\n", p == v.data() ? "no" : "yes");
    printf("after : p[0] = %g   <- reading freed memory\n", p[0]);
    return 0;
}
