#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef USE_DOUBLE
typedef float real_t;
#else
typedef double real_t;
#endif

#define N 10000000
#define K 10000
#define TWO_PI (acos(-1.0) * 2)

int main() {
    real_t *arr = (real_t *)malloc(N * sizeof(real_t));
    if (!arr) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    for (size_t i = 0; i < N; i++) {
		real_t value = (TWO_PI * i / N);
        arr[i] = sin(TWO_PI * i / N);
		if (i % K == 0) {
			printf("Sin(%.10f) = %.10f\n", (float)value, arr[i]);
		}
    }

    real_t sum = 0.0;
    for (size_t i = 0; i < N; i++) {
        sum += arr[i];
    }

    printf("Sum: %.20lf\n", (double)sum);
    
    free(arr);
    return 0;
}