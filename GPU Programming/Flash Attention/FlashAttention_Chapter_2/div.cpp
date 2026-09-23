#include <cstdio>
int main() {
    printf("7 / 2      = %d\n", 7 / 2);
    printf("-7 / 2     = %d   (Python: -7 // 2 = -4)\n", -7 / 2);
    printf("-7 %% 3     = %d   (Python: -7 %% 3 = 2)\n", -7 % 3);
    printf("7 / 2.0f   = %.1f\n", 7 / 2.0f);
    printf("(float)7/2 = %.1f\n", static_cast<float>(7) / 2);
    return 0;
}
