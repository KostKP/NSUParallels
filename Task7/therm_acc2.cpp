#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cstdio>

#include <cublas_v2.h>

using namespace std;
using namespace chrono;

#define OUT_FILE "result.dat"

#ifndef NX
#define NX 30
#endif
#ifndef NY
#define NY 20
#endif

#define TAU -0.01
#define EPS 0.01

#define MAX_ITER 1000

#define SIZE ((size_t)(NX) * (size_t)(NY))

double get_a(int row, int col) {
    if (row == col) return -4;
    if (row + 1 == col) return 1;
    if (row - 1 == col) return 1;
    if (row + NX == col) return 1;
    if (row - NX == col) return 1;
    return 0;
}

double get_b(int idx) {
    if (idx == NY / 2 * NX + NX / 3) return 10;
    if (idx == NY * 2 / 3 * NX + NX * 2 / 3) return -25;
    return 0;
}

void init_b(double *b) {
    for (int i = 0; i < SIZE; i++)
        b[i] = get_b(i);
}

double norm(double *x) {
    double result = 0;

    for (int i = 0; i < SIZE; i++)
        result += x[i] * x[i];

    return sqrt(result);
}

double norm_cublas(double *x_d, cublasHandle_t handle) {
    double result = 0.0;
    cublasDnrm2(handle, SIZE, x_d, 1, &result);
    return result;
}

void mul_mv_sub(double *res, double *x, double *y) {
    #pragma acc parallel loop present(x[0:SIZE], y[0:SIZE], res[0:SIZE])
    for (int i = 0; i < SIZE; i++) {
        res[i] = -y[i];
        for (int j = 0; j < SIZE; j++) {
            double a_ij = get_a(i, j);
            if (a_ij != 0) {
                res[i] += a_ij * x[j];
            }
        }
    }
}

void next(double *x, double *delta) {
    #pragma acc parallel loop present(x[0:SIZE], delta[0:SIZE])
    for (int i = 0; i < SIZE; i++)
        x[i] -= TAU * delta[i];
}


void solve_simple_iter(double *x, double *b, int &iterations, double &final_norm) {
    double *Axmb = (double*)malloc(SIZE * sizeof(double));
    double norm_b;
    int iter = 0;

    cublasHandle_t handle;
    cublasCreate(&handle);

    #pragma acc data copyin(b[0:SIZE]) copy(x[0:SIZE]) create(Axmb[0:SIZE])
    {
        #pragma acc host_data use_device(b)
        {
            norm_b = norm_cublas(b, handle);
        }

        do {
            mul_mv_sub(Axmb, x, b);

            #pragma acc host_data use_device(Axmb)
            {
                final_norm = norm_cublas(Axmb, handle);
            }

            next(x, Axmb);

            iter++;
            printf("%lf >= %lf\r", final_norm / norm_b, EPS);
            fflush(stdout);
        } while (final_norm / norm_b >= EPS && iter < MAX_ITER);
    }

    cublasDestroy(handle);
    printf("\33[2K\r");
    fflush(stdout);

    free(Axmb);

    iterations = iter;
}

int main() {
	double *b, *x;
	FILE *f;

	b = (double*)malloc(SIZE*sizeof(double));
	x = (double*)malloc(SIZE*sizeof(double));

	init_b(b);
	memset(x, 0, sizeof(double)*SIZE);

    int iterations;
    double final_norm;

    auto start = high_resolution_clock::now();

	solve_simple_iter(x, b, iterations, final_norm);

    auto end = high_resolution_clock::now();

    duration<double> diff = end - start;

    printf("Matrix %dx%d processing time: %.4f sec. (Iterations: %d/%d, Final norm: %.8lf < %.8lf).\n", NX, NY, diff.count(), iterations, MAX_ITER, final_norm/norm(b), EPS);

    if ((NX == 10 && NY == 10) || (NX == 13 && NY == 13)) {
        cout << "Result matrix:\n";
        for (int i = 0; i < NY; ++i) {
            for (int j = 0; j < NX; ++j) {
                cout << fixed << setw(10) << setprecision(4) << x[i * NX + j] << " ";
            }
            cout << endl;
        }
    }

	f = fopen(OUT_FILE, "wb");

	fwrite(x, sizeof(double), SIZE, f);
	fclose(f);

    printf("Result matrix saved to file '%s'.\r\n", OUT_FILE);

	free(b);
	free(x);
}
