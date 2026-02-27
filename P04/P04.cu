#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 512

__global__ void reduction(float *input, float *output, int len)
{
    __shared__ float partialSum[2 * BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * BLOCK_SIZE;

    if (start + t < len)
        partialSum[t] = input[start + t];
    else
        partialSum[t] = 0;

    if (start + BLOCK_SIZE + t < len)
        partialSum[BLOCK_SIZE + t] = input[start + BLOCK_SIZE + t];
    else
        partialSum[BLOCK_SIZE + t] = 0;


    for (unsigned int stride = BLOCK_SIZE; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (t < stride)
            partialSum[t] += partialSum[t + stride];
    }

    if (t == 0)
        output[blockIdx.x] = partialSum[0];
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: %s N\n", argv[0]);
        return 0;
    }

    int N = atoi(argv[1]);
    size_t bytes = N * sizeof(float);


    float *h_input = (float *)malloc(bytes);


    for (int i = 0; i < N; i++)
        h_input[i] = (float)(i + 1);


    clock_t t;
    double cpu_sum = 0.0;

    t = clock();
    for (int i = 0; i < N; i++)
        cpu_sum += h_input[i];
    t = clock() - t;

    double cpu_time = ((double)t / CLOCKS_PER_SEC) * 1000.0;
    printf("CPU Sum: %.0f\n", cpu_sum);
    printf("CPU Time: %f ms\n", cpu_time);


    float *d_input;
    cudaMalloc((void **)&d_input, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    int currentSize = N;
    float *d_in = d_input;
    float *d_out;

    cudaEvent_t start, stop;
    float gpu_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0); 
  
    while (currentSize > 1)
    {
        int numBlocks = (currentSize + (2 * BLOCK_SIZE - 1)) / (2 * BLOCK_SIZE);
        size_t outBytes = numBlocks * sizeof(float);

        cudaMalloc((void **)&d_out, outBytes);

        reduction<<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, currentSize);
        cudaDeviceSynchronize();

        if (d_in != d_input)
            cudaFree(d_in);

        d_in = d_out;
        currentSize = numBlocks;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_time, start, stop);

    float gpu_sum;
    cudaMemcpy(&gpu_sum, d_in, sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU Sum: %.0f\n", gpu_sum);
    printf("GPU Time: %f ms\n", gpu_time);

    printf("Speedup (CPU/GPU): %f\n", cpu_time / gpu_time);

    cudaFree(d_in);
    cudaFree(d_input);
    free(h_input);

    return 0;
}