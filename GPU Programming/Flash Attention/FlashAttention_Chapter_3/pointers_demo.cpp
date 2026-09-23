// Chapter 3: pointers and row-major indexing, in the smallest possible program.
// Build: g++ -std=c++17 -O2 -Wall -Wextra -o pointers_demo pointers_demo.cpp
#include <cstdio>
#include <vector>

int main() {
    // A 2 x 3 matrix, stored row by row in ONE flat vector (row-major).
    //   [[0 1 2],
    //    [3 4 5]]
    const size_t rows = 2, cols = 3;
    std::vector<float> M{0, 1, 2, 3, 4, 5};

    const float* p = M.data();                          // pointer to the first element
    printf("sizeof(float) = %zu bytes\n", sizeof(float));
    printf("p+1 is %td bytes after p\n",
           reinterpret_cast<const char*>(p + 1) - reinterpret_cast<const char*>(p));
    printf("*p = %g,  p[3] = %g,  *(p + 3) = %g\n", *p, p[3], *(p + 3));

    // Element (row, col) lives at index row * cols + col.
    const size_t r = 1, c = 2;
    printf("M[%zu][%zu] = p[%zu * %zu + %zu] = %g\n", r, c, r, cols, c, p[r * cols + c]);

    // A pointer to the START OF A ROW: add row * cols. Then index it like a 1-D array.
    const float* row1 = p + 1 * cols;
    printf("row1[0..2] = %g %g %g\n", row1[0], row1[1], row1[2]);
    printf("row1 == &M[3]? %s\n", row1 == &M[3] ? "yes" : "no");

    // Walking a whole matrix with one pointer per row (what our attention loops do).
    float total = 0.0f;
    for (size_t i = 0; i < rows; ++i) {
        const float* row = p + i * cols;
        for (size_t j = 0; j < cols; ++j) total += row[j];
    }
    printf("sum = %g\n", total);

    // A pointer that points at nothing.
    const float* nothing = nullptr;
    printf("nothing == nullptr? %s\n", nothing == nullptr ? "yes" : "no");
    return 0;
}
