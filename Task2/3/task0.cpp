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

// Функция для вычисления нормы вектора
double norm(const vector<double>& v) {
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    return sqrt(sum);
}

int main() {
    vector<vector<double>> A(MATRIX_N, vector<double>(MATRIX_N, 1.0));
    vector<double> b(MATRIX_N, MATRIX_N + 1);
    vector<double> x(MATRIX_N, 0.0);
    
    // Заполняем матрицу A, главная диагональ = 2.0
    for (int i = 0; i < MATRIX_N; ++i) {
        A[i][i] = 2.0;
    }

    double start_time = omp_get_wtime();
    
    vector<double> Ax(MATRIX_N, 0.0);
    int iteration = 0;
    while (iteration < MAX_ITERS) {
        for (int i = 0; i < MATRIX_N; ++i) {
            Ax[i] = 0.0;
            for (int j = 0; j < MATRIX_N; ++j) {
                Ax[i] += A[i][j] * x[j];
            }
        }
        
        vector<double> residual(MATRIX_N);
        for (int i = 0; i < MATRIX_N; ++i) {
            residual[i] = Ax[i] - b[i];
        }
        if (norm(residual) / norm(b) < epsilon) {
            break;
        }
        
        for (int i = 0; i < MATRIX_N; ++i) {
            x[i] -= tau * residual[i];
        }
        
        iteration++;
    }
    
    double end_time = omp_get_wtime();

    if (iteration == MAX_ITERS)  {
        cout << "Maximum number of iterations has been reached!\n";
        return 1;
    }

    cout << "SOLVED IN " << iteration + 1 << " ITERATIONS AND " << end_time - start_time << " SECONDS\n";
    
    return 0;
}
