#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <chrono>

#include "WavFile.h"

#define BUFF_SIZE 16384

using namespace std;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// ---------------- FILTER FUNCTION ----------------
void apply_filter(fftw_complex *data, int N, int Fs) {
    int target_bin = (10000.0 * N) / Fs;
    int mirror_bin = N - target_bin;

    for (int i = target_bin - 10; i <= target_bin + 10; i++) {
        data[i][0] = 0.0;
        data[i][1] = 0.0;
    }

    for (int i = mirror_bin - 10; i <= mirror_bin + 10; i++) {
        data[i][0] = 0.0;
        data[i][1] = 0.0;
    }
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        fprintf(stderr, "usage: %s <input.wav>\n", argv[0]);
        exit(1);
    }

    const char *wavfile = argv[1];

    char *wavfileout = (char *)malloc(strlen(argv[0]) + 20);
    sprintf(wavfileout, "%s_out.wav", argv[0]);

    // ---------------- FFTW SETUP ----------------
    fftw_complex *in, *out, *out2;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
    out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);

    fftw_plan plan  = fftw_plan_dft_1d(BUFF_SIZE, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan2 = fftw_plan_dft_1d(BUFF_SIZE, out, out2, FFTW_BACKWARD, FFTW_ESTIMATE);

    // ---------------- CUDA SETUP ----------------
    cufftHandle cuplan;
    cufftDoubleComplex *d_data;
    cufftDoubleComplex *h_data;

    cudaMalloc((void**)&d_data, BUFF_SIZE * sizeof(cufftDoubleComplex));
    h_data = (cufftDoubleComplex*)calloc(BUFF_SIZE, sizeof(cufftDoubleComplex));

    cufftPlan1d(&cuplan, BUFF_SIZE, CUFFT_Z2Z, 1);

    // ---------------- AUDIO ----------------
    short sampleBuffer[BUFF_SIZE];
    short outputBuffer[BUFF_SIZE];

    WavInFile inFile(wavfile);
    WavOutFile outFile(wavfileout,
                       inFile.getSampleRate(),
                       inFile.getNumBits(),
                       inFile.getNumChannels());

    int Fs = inFile.getSampleRate();

    double total_cpu_time = 0.0;
    double total_gpu_time = 0.0;

    // =========================================================
    // PROCESS AUDIO
    // =========================================================
    while (!inFile.eof()) {

        size_t samplesRead = inFile.read(sampleBuffer, BUFF_SIZE);

        for (int i = 0; i < BUFF_SIZE; i++) {
            in[i][0] = sampleBuffer[i];
            in[i][1] = 0.0;
        }

        // =====================================================
        // CPU PIPELINE (FFTW)
        // =====================================================
        auto cpu_start = chrono::high_resolution_clock::now();

        fftw_execute(plan);              // forward FFT
        apply_filter(out, BUFF_SIZE, Fs);
        fftw_execute(plan2);             // inverse FFT

        for (int i = 0; i < BUFF_SIZE; i++) {
            out2[i][0] /= BUFF_SIZE;
        }

        auto cpu_end = chrono::high_resolution_clock::now();
        total_cpu_time += chrono::duration<double, milli>(cpu_end - cpu_start).count();

        // =====================================================
        // GPU PIPELINE (CUFFT)
        // =====================================================
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        CUDA_CHECK(cudaMemcpy(d_data, in,
                              BUFF_SIZE * sizeof(cufftDoubleComplex),
                              cudaMemcpyHostToDevice));

        cudaEventRecord(start);

        cufftExecZ2Z(cuplan, d_data, d_data, CUFFT_FORWARD);

        CUDA_CHECK(cudaMemcpy(h_data, d_data,
                              BUFF_SIZE * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToHost));

        // filter on CPU (consistent measurement)
        for (int i = 0; i < BUFF_SIZE; i++) {
            h_data[i].x = 0;
            h_data[i].y = 0;
        }

        CUDA_CHECK(cudaMemcpy(d_data, h_data,
                              BUFF_SIZE * sizeof(cufftDoubleComplex),
                              cudaMemcpyHostToDevice));

        cufftExecZ2Z(cuplan, d_data, d_data, CUFFT_INVERSE);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float gpu_time;
        cudaEventElapsedTime(&gpu_time, start, stop);
        total_gpu_time += gpu_time;

        CUDA_CHECK(cudaMemcpy(h_data, d_data,
                              BUFF_SIZE * sizeof(cufftDoubleComplex),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < BUFF_SIZE; i++) {
            h_data[i].x /= BUFF_SIZE;
            outputBuffer[i] = (short)h_data[i].x;
        }

        outFile.write(outputBuffer, BUFF_SIZE);
    }

    // =========================================================
    // OUTPUT RESULTS
    // =========================================================
    printf("\n===== WAV INFO =====\n");
    printf("SampleRate: %d Hz\n", Fs);
    printf("Bits per sample: %d\n", inFile.getNumBits());
    printf("Channels: %d\n", inFile.getNumChannels());

    printf("\n===== PERFORMANCE COMPARISON =====\n");
    printf("CPU FFTW Time (Forward + Inverse): %f ms\n", total_cpu_time);
    printf("GPU cuFFT Time (Forward + Inverse): %f ms\n", total_gpu_time);

    if (total_gpu_time < total_cpu_time)
        printf("GPU faster by %f ms\n", total_cpu_time - total_gpu_time);
    else
        printf("CPU faster by %f ms\n", total_gpu_time - total_cpu_time);

    // =========================================================
    // CLEANUP
    // =========================================================
    fftw_destroy_plan(plan);
    fftw_destroy_plan(plan2);
    fftw_free(in);
    fftw_free(out);
    fftw_free(out2);

    cufftDestroy(cuplan);
    cudaFree(d_data);

    return 0;
}