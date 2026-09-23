#include <algorithm>
#include <cmath>
#include <cstdio>
int main() {
    printf("std::max(0.0f, NAN) = %g\n", std::max(0.0f, NAN));
    printf("std::max(NAN, 0.0f) = %g\n", std::max(NAN, 0.0f));
    return 0;
}
