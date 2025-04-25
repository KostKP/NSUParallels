#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <cstdio>

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

#define SIZE ((size_t)(NX) * (size_t)(NY))

double get_a(int row, int col) {
	if (row==col) return -4;
	if (row+1==col) return 1;
	if (row-1==col) return 1;
	if (row+NX==col) return 1;
	if (row-NX==col) return 1;
	return 0;
}

double get_b(int idx) {
	if (idx==NY/2*NX+NX/3) return 10;
	if (idx==NY*2/3*NX+NX*2/3) return -25;
	return 0;
}

void init_matrix(double *M) {
    #pragma acc parallel loop present(M) collapse(2)
	for (int i=0; i < SIZE; i++)
		for (int j=0; j < SIZE; j++)
			M[i * SIZE + j] = get_a(i, j);
}

void init_b(double *b) {
    #pragma acc parallel loop present(b)
	for (int i=0; i < SIZE; i++)
		b[i] = get_b(i);
}

double norm(double *x) {
	double result=0;

    #pragma acc parallel loop present(x) reduction(+:result)
	for (int i=0; i < SIZE; i++)
		result += x[i]*x[i];

	return sqrt(result);
}

void mul_mv_sub(double *res, double *A, double *x, double *y) {
    #pragma acc parallel loop present(A, x, y, res)
    for (int i = 0; i < SIZE; i++) {
        res[i] = -y[i];
        double sum = 0.0;
        #pragma acc loop reduction(+:sum)
        for (int j = 0; j < SIZE; j++)
            sum += A[i * SIZE + j] * x[j];
        res[i] += sum;
    }
}

void next(double *x, double *delta) {
    #pragma acc parallel loop present(x, delta)
	for (int i=0; i < SIZE; i++)
		x[i] -= TAU * delta[i];
}

void solve_simple_iter(double *A, double *x, double *b, int &iterations, double &final_norm) {
    double *Axmb, norm_b;
    norm_b = norm(b);
    iterations = 0;
    Axmb = (double*)malloc(SIZE * sizeof(double));

    #pragma acc data copyin(A[0:(size_t)SIZE*(size_t)SIZE], b[0:SIZE]) copy(x[0:SIZE]) create(Axmb[0:SIZE])
    {
        do {
            mul_mv_sub(Axmb, A, x, b);
            final_norm=norm(Axmb);
            next(x, Axmb);
            iterations++;
            printf("%lf >= %lf\r", final_norm/norm_b, EPS);
            fflush(stdout);
        } while (final_norm/norm_b >= EPS);
    }

	printf("\33[2K\r");
	fflush(stdout);

	free(Axmb);
}

int main() {
	double *A, *b, *x;
	FILE *f;

    A = (double*)malloc((size_t)SIZE * (size_t)SIZE * sizeof(double));
    b = (double*)malloc(SIZE * sizeof(double));
    x = (double*)malloc(SIZE * sizeof(double));

    #pragma acc enter data create(A[0:(size_t)SIZE*(size_t)SIZE], b[0:SIZE], x[0:SIZE])
	init_matrix(A);
	init_b(b);
    #pragma acc parallel loop present(x)
    for (int i = 0; i < SIZE; i++) x[i] = 0;

    int iterations;
    double final_norm;

    auto start = high_resolution_clock::now();

	solve_simple_iter(A, x, b, iterations, final_norm);

    auto end = high_resolution_clock::now();

    duration<double> diff = end - start;

    printf("Matrix %dx%d processing time: %.4f sec. and %d iterations (FN: %.8lf < %.8lf).\n", NX, NY, diff.count(), iterations, final_norm/norm(b), EPS);

    #pragma acc update self(x[0:SIZE])
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

    #pragma acc exit data delete(A, b, x)
	free(A);
	free(b);
	free(x);
}
