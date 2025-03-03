#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

#define STRINGIFY_HELPER(...) #__VA_ARGS__
#define STRINGIFY(x) STRINGIFY_HELPER(x)

#ifndef MATRIX_N
#define MATRIX_N 10000
#endif

#ifndef MAX_ITERS
#define MAX_ITERS 10000000
#endif

using namespace std;

const double tau = 0.000001;
const double epsilon = 1e-5;

double norm(const vector<double>& v) {
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    return sqrt(sum);
}

void solve_parallel_for(vector<vector<double>>& A, vector<double>& b, int num_threads) {
    omp_set_num_threads(num_threads);
    vector<double> x(MATRIX_N, 0.0);
    vector<double> Ax(MATRIX_N, 0.0);
    int iteration = 0;
    double start_time = omp_get_wtime();

    while (iteration < MAX_ITERS) {
        #pragma omp parallel for schedule(SCHEDULE)
        for (int i = 0; i < MATRIX_N; ++i) {
            Ax[i] = 0.0;
            for (int j = 0; j < MATRIX_N; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        vector<double> residual(MATRIX_N);
        #pragma omp parallel for schedule(SCHEDULE)
        for (int i = 0; i < MATRIX_N; ++i) {
            residual[i] = Ax[i] - b[i];
        }

        if (norm(residual) / norm(b) < epsilon) {
            break;
        }

        #pragma omp parallel for schedule(SCHEDULE)
        for (int i = 0; i < MATRIX_N; ++i) {
            x[i] -= tau * residual[i];
        }

        iteration++;
    }

    double end_time = omp_get_wtime();

    if (iteration == MAX_ITERS)  {
        cout << "Maximum number of iterations has been reached!\n";
        return;
    }

    cout << "Solved in " << iteration + 1 << " iterations and " 
         << end_time - start_time << " seconds using " 
         << num_threads << " threads with schedule(" << STRINGIFY(SCHEDULE) << ")\n";
}

int main() {
    vector<vector<double>> A(MATRIX_N, vector<double>(MATRIX_N, 1.0));
    vector<double> b(MATRIX_N, MATRIX_N + 1);
    
    for (int i = 0; i < MATRIX_N; ++i) {
        A[i][i] = 2.0;
    }

    int threads_cnt = omp_get_max_threads();
    
    solve_parallel_for(A, b, threads_cnt);

    return 0;
}
