#include <cstdio>
#include <vector>
int main() {
    const size_t rows = 2, cols = 3;
    std::vector<float> M{0, 1, 2, 3, 4, 5};
    printf("M[1][2] with the right stride (cols): %g\n", M[1 * cols + 2]);
    printf("M[1][2] with the wrong stride (rows): %g\n", M[1 * rows + 2]);
    return 0;
}
