#define _USE_MATH_DEFINES
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                            \
        }                                                                       \
    } while (0)

__device__ static double f_device(double x)
{
    double c = cos(x);
    double arg = c / (1.0 + 2.0 * c);
    if (arg > 1.0) {
        arg = 1.0;
    }
    if (arg < -1.0) {
        arg = -1.0;
    }
    return acos(arg);
}

__global__ static void simpson_panel_reduce_kernel(double a, double h,
                                                   unsigned long long panel_start,
                                                   unsigned long long panel_count,
                                                   double *block_sums)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned long long global_panel =
        panel_start + (unsigned long long)blockIdx.x * blockDim.x + tid;

    double val = 0.0;

    if (global_panel < panel_start + panel_count) {
        double x0 = a + (double)(2ULL * global_panel) * h;
        double x1 = a + (double)(2ULL * global_panel + 1ULL) * h;
        double x2 = a + (double)(2ULL * global_panel + 2ULL) * h;
        val = f_device(x0) + 4.0 * f_device(x1) + f_device(x2);
    }

    sdata[tid] = val;
    __syncthreads();

    /* Tree reduction in shared memory; blockDim must be a power of two. */
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}

static int is_power_of_two(int x)
{
    return (x > 0) && ((x & (x - 1)) == 0);
}

static void parse_args(int argc, char **argv, unsigned long long *n_intervals,
                       int *threads_per_block, int *show_help)
{
    int i;
    *n_intervals = 10000000ULL;
    *threads_per_block = 256;
    *show_help = 0;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && (i + 1) < argc) {
            *n_intervals = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "-t") == 0 && (i + 1) < argc) {
            *threads_per_block = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            *show_help = 1;
        }
    }
}

int main(int argc, char **argv)
{
    const double a = 0.0;
    const double b = 0.5 * M_PI;
    const double exact = 5.0 * (M_PI * M_PI) / 24.0;

    int rank = 0;
    int size = 1;
    int device_count = 0;
    int device_id = 0;

    unsigned long long n_intervals = 0ULL;
    int threads_per_block = 0;
    int show_help = 0;

    unsigned long long total_panels = 0ULL;
    unsigned long long base_panels = 0ULL;
    unsigned long long rem_panels = 0ULL;
    unsigned long long local_panel_start = 0ULL;
    unsigned long long local_panel_count = 0ULL;

    double h = 0.0;
    int blocks = 0;
    double *d_block_sums = NULL;
    double *h_block_sums = NULL;
    double local_simpson_sum = 0.0;
    double local_integral = 0.0;

    double *all_partial = NULL;
    double approx = 0.0;
    double err_abs = 0.0;

    float kernel_ms = 0.0f;
    double local_kernel_s = 0.0;
    double max_kernel_s = 0.0;
    cudaEvent_t ev_start;
    cudaEvent_t ev_stop;

    double wall_start = 0.0;
    double wall_end = 0.0;
    double wall_elapsed = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    parse_args(argc, argv, &n_intervals, &threads_per_block, &show_help);
    if (show_help) {
        if (rank == 0) {
            printf("Usage: ./p11 -n NINTERVALS -t THREADS_PER_BLOCK\n");
            printf("  -n : total Simpson subintervals (even)\n");
            printf("  -t : CUDA threads per block (power of 2)\n");
            printf("Example: mpirun -mca btl \"^openib\" -np 4 ./p11 -n 10000000 -t 256\n");
        }
        MPI_Finalize();
        return EXIT_SUCCESS;
    }

    if (n_intervals < 2ULL) {
        if (rank == 0) {
            fprintf(stderr, "Error: N must be >= 2\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if ((n_intervals & 1ULL) != 0ULL) {
        if (rank == 0) {
            fprintf(stderr, "Error: N must be even for Simpson 1/3\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (!is_power_of_two(threads_per_block)) {
        if (rank == 0) {
            fprintf(stderr, "Error: -t must be a power of 2\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    if (threads_per_block > 1024) {
        if (rank == 0) {
            fprintf(stderr, "Error: -t must be <= 1024\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: no CUDA devices found\n");
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    device_id = rank % device_count;
    CUDA_CHECK(cudaSetDevice(device_id));

    total_panels = n_intervals / 2ULL;
    base_panels = total_panels / (unsigned long long)size;
    rem_panels = total_panels % (unsigned long long)size;

    local_panel_count = base_panels + ((unsigned long long)rank < rem_panels ? 1ULL : 0ULL);
    local_panel_start =
        (unsigned long long)rank * base_panels +
        ((unsigned long long)rank < rem_panels ? (unsigned long long)rank : rem_panels);

    h = (b - a) / (double)n_intervals;

    blocks = (int)((local_panel_count + (unsigned long long)threads_per_block - 1ULL) /
                   (unsigned long long)threads_per_block);

    if (blocks > 0) {
        h_block_sums = (double *)malloc((size_t)blocks * sizeof(double));
        if (h_block_sums == NULL) {
            fprintf(stderr, "Rank %d: host allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        CUDA_CHECK(cudaMalloc((void **)&d_block_sums, (size_t)blocks * sizeof(double)));
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    } else {
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    wall_start = MPI_Wtime();

    CUDA_CHECK(cudaEventRecord(ev_start));
    if (blocks > 0) {
        simpson_panel_reduce_kernel<<<blocks, threads_per_block,
                                      (size_t)threads_per_block * sizeof(double)>>>(
            a, h, local_panel_start, local_panel_count, d_block_sums);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, (size_t)blocks * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, ev_start, ev_stop));
    local_kernel_s = 1.0e-3 * (double)kernel_ms;

    local_simpson_sum = 0.0;
    for (int i = 0; i < blocks; i++) {
        local_simpson_sum += h_block_sums[i];
    }
    local_integral = (h / 3.0) * local_simpson_sum;

    if (rank == 0) {
        all_partial = (double *)malloc((size_t)size * sizeof(double));
        if (all_partial == NULL) {
            fprintf(stderr, "Rank 0: host allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Gather(&local_integral, 1, MPI_DOUBLE, all_partial, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_kernel_s, &max_kernel_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    wall_end = MPI_Wtime();
    wall_elapsed = wall_end - wall_start;

    if (rank == 0) {
        approx = 0.0;
        for (int r = 0; r < size; r++) {
            approx += all_partial[r];
        }
        err_abs = fabs(approx - exact);


        printf("M,threads,N,wall_s,kernel_s_max,approx,error,pass\n");
        printf("%d,%d,%llu,%.9f,%.9f,%.15e,%.6e,%s\n", size, threads_per_block,
               n_intervals, wall_elapsed, max_kernel_s, approx, err_abs,
               (err_abs <= 1e-8) ? "yes" : "no");
        printf("exact=%.15e\n", exact);
        printf("note=run multiple M and -t values to build a 3D runtime surface\n");
    }

    if (all_partial != NULL) {
        free(all_partial);
    }
    if (h_block_sums != NULL) {
        free(h_block_sums);
    }
    if (d_block_sums != NULL) {
        CUDA_CHECK(cudaFree(d_block_sums));
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    MPI_Finalize();
    return 0;
}
