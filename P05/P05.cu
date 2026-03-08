#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16


#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

__global__ void matrixMultiply(float* A, float* B, float* C,
                               int m, int p, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < p; k++) {
            sum += A[row * p + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrixTiledMultiply(float* A, float* B, float* C,
                                    int m, int p, int n)
{
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    int numTiles = (p + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {

        
        if (row < m && (t * TILE_WIDTH + threadIdx.x) < p)
            As[threadIdx.y][threadIdx.x] =
                A[row * p + t * TILE_WIDTH + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        
        if (col < n && (t * TILE_WIDTH + threadIdx.y) < p)
            Bs[threadIdx.y][threadIdx.x] =
                B[(t * TILE_WIDTH + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n)
        C[row * n + col] = sum;
}

void fillRandom(float* mat, int size) {
    for (int i = 0; i < size; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

int main() {

    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096};
    int numSizes = 7;

    printf("Size Basic(ms) Tiled(ms)\n");

    for (int s = 0; s < numSizes; s++) {

        int m = sizes[s];
        int p = sizes[s];
        int n = sizes[s];

        
        float* hA = (float*)malloc(sizeof(float) * m * p);
        float* hB = (float*)malloc(sizeof(float) * p * n);
        float* hC = (float*)malloc(sizeof(float) * m * n);

        fillRandom(hA, m * p);
        fillRandom(hB, p * n);

        
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(float) * m * p));
        CUDA_CHECK(cudaMalloc((void**)&dB, sizeof(float) * p * n));
        CUDA_CHECK(cudaMalloc((void**)&dC, sizeof(float) * m * n));

        
        CUDA_CHECK(cudaMemcpy(dA, hA, sizeof(float)*m*p, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, sizeof(float)*p*n, cudaMemcpyHostToDevice));


        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH,
                     (m + TILE_WIDTH - 1) / TILE_WIDTH);


        matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

   
        cudaEventRecord(start);
        matrixMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float basicTime = 0.0f;
        cudaEventElapsedTime(&basicTime, start, stop);


        cudaEventRecord(start);
        matrixTiledMultiply<<<dimGrid, dimBlock>>>(dA, dB, dC, m, p, n);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);

        float tiledTime = 0.0f;
        cudaEventElapsedTime(&tiledTime, start, stop);

        printf("%d %f %f\n", m, basicTime, tiledTime);

        free(hA);
        free(hB);
        free(hC);

        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return 0;
}