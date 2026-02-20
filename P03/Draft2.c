#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


double f(double x) {
    return acos(cos(x) / (1.0 + 2.0 * cos(x)));
}

int main() {

    long n = 1000000;

    const double PI = acos(-1.0);
    double a = 0.0;
    double b = PI / 2.0;
    double h = (b - a) / n;

    const double exact_solution = (5.0 * PI * PI) / 24.0;

    int thread_counts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int num_tests = 9;

    omp_set_dynamic(0);

     printf("Threads Time (sec) Result Error\n");


    for (int t = 0; t < num_tests; t++) {

        int Nthrds = thread_counts[t];
        omp_set_num_threads(Nthrds);

        double start_time = omp_get_wtime();
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (long i = 1; i < n; i++) {

            double x = a + i * h;

            if (i % 2 == 0)
                sum += 2.0 * f(x);
            else
                sum += 4.0 * f(x);
        }

        double integral = (h / 3.0) * (f(a) + f(b) + sum);
        double end_time = omp_get_wtime();

        double error = fabs(integral - exact_solution);
        double duration = end_time - start_time;

        printf("%7d %10.6f %.15f %.3e\n",
               Nthrds, duration, integral, error);
    }

    return 0;
}
