#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

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
        #pragma omp parallel for
        for (int i = 0; i < MATRIX_N; ++i) {
            Ax[i] = 0.0;
            for (int j = 0; j < MATRIX_N; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
        }

        vector<double> residual(MATRIX_N);
        #pragma omp parallel for
        for (int i = 0; i < MATRIX_N; ++i) {
            residual[i] = Ax[i] - b[i];
        }

        if (norm(residual) / norm(b) < epsilon) {
            break;
        }

        #pragma omp parallel for
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

    cout << "Solved in " << iteration + 1 << " iterations and " << end_time - start_time << " seconds using " << num_threads << " threads\n";
}

void solve_parallel_section(vector<vector<double>>& A, vector<double>& b, int num_threads) {
    omp_set_num_threads(num_threads);
    vector<double> x(MATRIX_N, 0.0);
    vector<double> Ax(MATRIX_N);
    vector<double> residual(MATRIX_N);
    int iteration = 0;
    bool converged = false;
    double residual_norm_sq = 0.0;
    double norm_b = norm(b);

    double start_time = omp_get_wtime();

    #pragma omp parallel num_threads(num_threads) shared(x, Ax, residual, iteration, converged, residual_norm_sq)
    {
        int thread_id = omp_get_thread_num();
        int num_threads_omp = omp_get_num_threads();
        int chunk_size = MATRIX_N / num_threads_omp;
        int start = thread_id * chunk_size;
        int end = (thread_id == num_threads_omp - 1) ? MATRIX_N : start + chunk_size;

        while (iteration < MAX_ITERS && !converged) {
            for (int i = start; i < end; ++i) {
                Ax[i] = 0.0;
                for (int j = 0; j < MATRIX_N; ++j) {
                    Ax[i] += A[i][j] * x[j];
                }
            }

            #pragma omp barrier
            for (int i = start; i < end; ++i) {
                residual[i] = Ax[i] - b[i];
            }

            #pragma omp barrier
            double local_sum = 0.0;
            for (int i = start; i < end; ++i) {
                local_sum += residual[i] * residual[i];
            }

            #pragma omp critical
            {
                residual_norm_sq += local_sum;
            }

            #pragma omp barrier
            #pragma omp single
            {
                double current_residual = sqrt(residual_norm_sq);
                if (current_residual / norm_b < epsilon) {
                    converged = true;
                }
                residual_norm_sq = 0.0;
            }

            #pragma omp barrier

            if (converged) {
                break;
            }

            for (int i = start; i < end; ++i) {
                x[i] -= tau * residual[i];
            }

            #pragma omp barrier
            #pragma omp single
            {
                iteration++;
            }

            #pragma omp barrier
        }
    }

    double end_time = omp_get_wtime();

    if (iteration >= MAX_ITERS && !converged) {
        cout << "Maximum number of iterations has been reached!\n";
        return;
    }

    cout << "Solved in " << iteration + 1 << " iterations and " << end_time - start_time << " seconds using " << num_threads << " threads\n";
}

int main() {
    vector<vector<double>> A(MATRIX_N, vector<double>(MATRIX_N, 1.0));
    vector<double> b(MATRIX_N, MATRIX_N + 1);
    
    for (int i = 0; i < MATRIX_N; ++i) {
        A[i][i] = 2.0;
    }

    int max_threads = omp_get_max_threads();
    cout << "Running Variant 1 with different number of threads:\n";
    for (int num_threads = 1; num_threads <= max_threads; ++num_threads) {
        solve_parallel_for(A, b, num_threads);
    }

    cout << "Running Variant 2 with different number of threads:\n";
    for (int num_threads = 1; num_threads <= max_threads; ++num_threads) {
        solve_parallel_section(A, b, num_threads);
    }

    return 0;
}
