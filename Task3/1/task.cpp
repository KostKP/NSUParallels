#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <thread>
#include <chrono>
#include <cinttypes>

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

    auto start = std::chrono::high_resolution_clock::now();
    matrix_vector_product(a, b, c, MATRIX_M, MATRIX_N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double t = diff.count();

    printf("Elapsed time (serial): %.6f sec.\n", t);
    free(a);
    free(b);
    free(c);
}

void run_parallel() {
    double *a, *b, *c;
    a = (double*)xmalloc(sizeof(*a) * MATRIX_M * MATRIX_N);
    b = (double*)xmalloc(sizeof(*b) * MATRIX_N);
    c = (double*)xmalloc(sizeof(*c) * MATRIX_M);

    // Параллельная инициализация матрицы a
    {
        std::vector<std::jthread> threads;
        const int chunk = (MATRIX_M + THREADS_CNT - 1) / THREADS_CNT;
        for (int tid = 0; tid < THREADS_CNT; ++tid) {
            const int start_i = tid * chunk;
            const int end_i = std::min(start_i + chunk, MATRIX_M);
            threads.emplace_back([start_i, end_i, a] {
                for (int i = start_i; i < end_i; ++i) {
                    for (int j = 0; j < MATRIX_N; ++j) {
                        a[i * MATRIX_N + j] = i + j;
                    }
                }
            });
        }
    }

    // Параллельная инициализация вектора b
    {
        std::vector<std::jthread> threads;
        const int chunk = (MATRIX_N + THREADS_CNT - 1) / THREADS_CNT;
        for (int tid = 0; tid < THREADS_CNT; ++tid) {
            const int start_j = tid * chunk;
            const int end_j = std::min(start_j + chunk, MATRIX_N);
            threads.emplace_back([start_j, end_j, b] {
                for (int j = start_j; j < end_j; ++j) {
                    b[j] = j;
                }
            });
        }
    }

    // Параллельное вычисление произведения матрицы на вектор
    auto start = std::chrono::high_resolution_clock::now();
    {
        std::vector<std::jthread> threads;
        const int chunk = (MATRIX_M + THREADS_CNT - 1) / THREADS_CNT;
        for (int tid = 0; tid < THREADS_CNT; ++tid) {
            const int start_i = tid * chunk;
            const int end_i = std::min(start_i + chunk, MATRIX_M);
            threads.emplace_back([start_i, end_i, a, b, c] {
                for (int i = start_i; i < end_i; ++i) {
                    double sum = 0.0;
                    for (int j = 0; j < MATRIX_N; ++j) {
                        sum += a[i * MATRIX_N + j] * b[j];
                    }
                    c[i] = sum;
                }
            });
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double t = diff.count();

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
