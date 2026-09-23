#include <cstdio>
#include <vector>
int main() {
    std::vector<float> x(4, 1.0f);
    printf("x[4] = %f\n", x[4]);        // one past the end: no error in Python terms
    return 0;
}
