#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef THREADS_CNT
#define THREADS_CNT 1
#endif

#ifndef MATRIX_N
#define MATRIX_N 20000
#endif

#ifndef MATRIX_M
#define MATRIX_M 20000
#endif

void* xmalloc(size_t size) {
    void* ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n) {
    for (int i = 0; i < m; i++) {
        c[i] = 0.0;
        for (int j = 0; j < n; j++)
            c[i] += a[i * n + j] * b[j];
    }
}

void run_serial() {
    double *a, *b, *c;
    a = (double*)xmalloc(sizeof(*a) * MATRIX_M * MATRIX_N);
    b = (double*)xmalloc(sizeof(*b) * MATRIX_N);
    c = (double*)xmalloc(sizeof(*c) * MATRIX_M);

    for (int i = 0; i < MATRIX_M; i++) {
        for (int j = 0; j < MATRIX_N; j++)
            a[i * MATRIX_N + j] = i + j;
    }

    for (int j = 0; j < MATRIX_N; j++)
        b[j] = j;

    double t = omp_get_wtime();
    matrix_vector_product(a, b, c, MATRIX_M, MATRIX_N);
    t = omp_get_wtime() - t;

    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n) {
    #pragma omp parallel num_threads(THREADS_CNT)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++)
                c[i] += a[i * n + j] * b[j];
        }
    }
}

void run_parallel() {
    double *a, *b, *c;
    a = (double*)xmalloc(sizeof(*a) * MATRIX_M * MATRIX_N);
    b = (double*)xmalloc(sizeof(*b) * MATRIX_N);
    c = (double*)xmalloc(sizeof(*c) * MATRIX_M);

    #pragma omp parallel num_threads(THREADS_CNT)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = MATRIX_M / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (MATRIX_M - 1) : (lb + items_per_thread - 1);
        for (int i = lb; i <= ub; i++) {
            for (int j = 0; j < MATRIX_N; j++)
                a[i * MATRIX_N + j] = i + j;
            c[i] = 0.0;
        }
    }

    for (int j = 0; j < MATRIX_N; j++)
        b[j] = j;

    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, MATRIX_M, MATRIX_N);
    t = omp_get_wtime() - t;

    printf("Elapsed time (parallel): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

int main(int argc, char **argv) {
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", MATRIX_M, MATRIX_N);
    printf("Memory used: %" PRIu64 " MiB\n", ((MATRIX_M * MATRIX_N + MATRIX_M + MATRIX_N) * sizeof(double)) >> 20);
    printf("Threads count: %d\n", THREADS_CNT);

    run_serial();
    run_parallel();

    return 0;
}
