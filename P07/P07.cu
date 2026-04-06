#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda.h>

#define TILE_WIDTH 16
#define MASK_WIDTH 5
#define MASK_RADIUS 2

// CUDA Kernel for 2D convolution
__global__ void convolution(
    int *input,
    int *output,
    int *mask,
    int channels,
    int width,
    int height)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;

    if (tx >= width || ty >= height)
        return;

    for (int c = 0; c < channels; c++)
    {
        int sum = 0;

        for (int i = -MASK_RADIUS; i <= MASK_RADIUS; i++)
        {
            for (int j = -MASK_RADIUS; j <= MASK_RADIUS; j++)
            {
                int x = tx + j;
                int y = ty + i;

                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    int imageIdx = (y * width + x) * channels + c;
                    int maskIdx = (i + MASK_RADIUS) * MASK_WIDTH + (j + MASK_RADIUS);

                    sum += input[imageIdx] * mask[maskIdx];
                }
            }
        }

        int outIdx = (ty * width + tx) * channels + c;
        output[outIdx] = sum;
    }
}

int main()
{
    int imageWidth = 512;
    int imageHeight = 384;
    int imageChannels = 3;

    int inputLength = imageWidth * imageHeight * imageChannels;

    // Allocate host memory
    int *hostInputImage = (int*)malloc(inputLength * sizeof(int));
    int *hostOutputImage = (int*)malloc(inputLength * sizeof(int));

    // Read input file
    FILE *f = fopen("peppers.dat", "r");
    if (!f)
    {
        printf("Error opening input file\n");
        return 1;
    }

    for (int i = 0; i < inputLength; i++)
    {
        fscanf(f, "%d", &hostInputImage[i]);
    }
    fclose(f);

// Sobel 5x5 horizontal mask
    int hostMask[5][5] = {
        {2,  2,  4,  2, 2},
        {1,  1,  2,  1, 1},
        {0,  0,  0,  0, 0},
        {-1, -1, -2, -1, -1},
        {-2, -2, -4, -2, -2}
    };

    // Flatten mask
    int flatMask[25];
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 5; j++)
            flatMask[i * 5 + j] = hostMask[i][j];

    // Allocate device memory
    int *deviceInputImage, *deviceOutputImage, *deviceMask;

    cudaMalloc((void**)&deviceInputImage, inputLength * sizeof(int));
    cudaMalloc((void**)&deviceOutputImage, inputLength * sizeof(int));
    cudaMalloc((void**)&deviceMask, 25 * sizeof(int));

    // Copy to device
    cudaMemcpy(deviceInputImage, hostInputImage,
               inputLength * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(deviceMask, flatMask,
               25 * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid/block
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (imageWidth + TILE_WIDTH - 1) / TILE_WIDTH,
        (imageHeight + TILE_WIDTH - 1) / TILE_WIDTH
    );

    // Launch kernel
    convolution<<<dimGrid, dimBlock>>>(
        deviceInputImage,
        deviceOutputImage,
        deviceMask,
        imageChannels,
        imageWidth,
        imageHeight
    );

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(hostOutputImage, deviceOutputImage,
               inputLength * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output file
    FILE *out = fopen("peppers.out", "w");
    for (int i = 0; i < inputLength; i++)
    {
        fprintf(out, "%d\n", hostOutputImage[i]);
    }
    fclose(out);

    // Cleanup
    cudaFree(deviceInputImage);
    cudaFree(deviceOutputImage);
    cudaFree(deviceMask);

    free(hostInputImage);
    free(hostOutputImage);

    printf("Convolution completed successfully.\n");

    return 0;
}
