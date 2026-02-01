#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define AVAL 3.0
#define BVAL 5.0
#define TOL 1.1102230246251568E-16
#define UNROLL_FACTOR 4
#define DEFAULT_ORDER 1000

int main(int argc, char *argv[])
{
    int Ndim, Pdim, Mdim;
    int i, j, k;
    double *A, *B, *C;
    double tmp, err, errsq, cval;
    double start_time, run_time;
    double dN, mflops;
    int num_threads;
    int order;
    FILE *fp;
    char *env_threads;
    char *env_order;

    env_order = getenv("MATRIX_ORDER");
    if (env_order == NULL) {
        env_order = getenv("ORDER");
    }
    if (env_order == NULL) {
        order = DEFAULT_ORDER; 
        fprintf(stderr, "Warning: MATRIX_ORDER or ORDER not set, using %d\n", DEFAULT_ORDER);
    } else {
        order = atoi(env_order);
        if (order < 1) {
            order = DEFAULT_ORDER;
            fprintf(stderr, "Warning: Invalid matrix order, using %d\n", DEFAULT_ORDER);
        }
    }

    Ndim = order;
    Pdim = order;
    Mdim = order;

    env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads == NULL) {
        env_threads = getenv("NUM_THREADS");
    }
    if (env_threads == NULL) {
        num_threads = 1; 
        fprintf(stderr, "Warning: OMP_NUM_THREADS or NUM_THREADS not set, using 1 thread\n");
    } else {
        num_threads = atoi(env_threads);
        if (num_threads < 1) {
            num_threads = 1;
            fprintf(stderr, "Warning: Invalid thread count, using 1 thread\n");
        }
    }
    
    
    int max_power = 0;
    int temp = num_threads;
    while (temp > 1) {
        temp >>= 1;
        max_power++;
    }

    
    A = (double *)malloc(Ndim * Pdim * sizeof(double));
    B = (double *)malloc(Pdim * Mdim * sizeof(double));
    C = (double *)malloc(Ndim * Mdim * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for(i = 0; i < Ndim; i++)
        for(j = 0; j < Pdim; j++)
            A[i + j * Ndim] = AVAL;

    for(i = 0; i < Pdim; i++)
        for(j = 0; j < Mdim; j++)
            B[i + j * Pdim] = BVAL;

    printf("Parallel computation:\n");
    printf("  Matrix order: %d x %d\n", order, order);
    printf("  Testing powers of 2 from 2^0 to 2^%d (1 to %d threads)\n", max_power, num_threads);
    
    fp = fopen("data.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Error opening data.txt for writing\n");
    }
    
    int thread_power;
    int current_threads;
    for(thread_power = 0; thread_power <= max_power; thread_power++) {
        current_threads = 1 << thread_power;
        
        for(i = 0; i < Ndim; i++)
            for(j = 0; j < Mdim; j++)
                C[i + j * Ndim] = 0.0;
        
        omp_set_num_threads(current_threads);
        
        start_time = omp_get_wtime();
        
 
        #pragma omp parallel for private(tmp, i, j, k)
        for(i = 0; i < Ndim; i++) {
            for(j = 0; j < Mdim; j++) {
                tmp = 0.0;
                
                for(k = 0; k < Pdim - UNROLL_FACTOR + 1; k += UNROLL_FACTOR) {
                    tmp += A[i + k * Ndim] * B[k + j * Pdim];
                    tmp += A[i + (k+1) * Ndim] * B[(k+1) + j * Pdim];
                    tmp += A[i + (k+2) * Ndim] * B[(k+2) + j * Pdim];
                    tmp += A[i + (k+3) * Ndim] * B[(k+3) + j * Pdim];
                }
                
                for(; k < Pdim; k++) {
                    tmp += A[i + k * Ndim] * B[k + j * Pdim];
                }
                C[i + j * Ndim] = tmp;
            }
        }
        
        run_time = omp_get_wtime() - start_time;
        dN = (double)order;
        mflops = 2.0 * dN * dN * dN / (run_time * 1.0e6);
        
        printf("  Threads: %d (2^%d), Time: %f seconds, Performance: %f mflops\n", 
               current_threads, thread_power, run_time, mflops);
        
        
        cval = Pdim * AVAL * BVAL;
        errsq = 0.0;
        for(i = 0; i < Ndim; i++) {
            for(j = 0; j < Mdim; j++) {
                err = C[i + j * Ndim] - cval;
                errsq += err * err;
            }
        }
        
        if (errsq > TOL) {
            printf("    Error in dgemm: %e\n", errsq);
        }
        
       
        if (fp != NULL) {
            fprintf(fp, "%d %f\n", current_threads, run_time);
        }
    }
    
    
    if (fp != NULL) {
        fclose(fp);
    }

    free(A);
    free(B);
    free(C);

    return 0;
}

