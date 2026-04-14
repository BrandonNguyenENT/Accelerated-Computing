#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <cufft.h>
#include <cuda_runtime.h>
#include <chrono> 

#include <soundtouch/SoundTouch.h>

#include "WavFile.h"

using namespace std;
using namespace soundtouch;

#define BUFF_SIZE           16384
#define MAX_FREQ            48 //KHz

#define CUDA_CHECK(ans)
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
		      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code),
	    file, line);
    if (abort)
      exit(code);
  }
}

int main(int argc, char *argv[]) {

  const char *wavfile;
  if( argc != 2 )
    {
      fprintf(stderr,"usage: %s <input.wav>\n", argv[0]);
      exit(1);
    }
  else
    {
      wavfile = argv[1];
    }

  char *wavfileout = (char *)malloc(strlen(argv[0]) + strlen("_out.wav") + 1);
  char *logfile = (char *)malloc(strlen(argv[0]) + strlen("_out.log") + 1);
  wavfileout = strcat(strcpy(wavfileout, argv[0]),"_out.wav");
  logfile = strcat(strcpy(logfile, argv[0]),"_out.log");

  FILE *log;
  fftw_complex *in, *out, *out2;
  in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);
  out2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * BUFF_SIZE);

  fftw_plan plan = fftw_plan_dft_1d(BUFF_SIZE, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_plan plan2 = fftw_plan_dft_1d(BUFF_SIZE, out, out2, FFTW_BACKWARD, FFTW_ESTIMATE);

  cufftHandle cuplan;
  cufftDoubleComplex *d_DoubleComplexData;
  cufftDoubleComplex *h_DoubleComplexData;
  int nx = BUFF_SIZE;

  cudaMalloc((void**)&d_DoubleComplexData, nx * sizeof(cufftDoubleComplex));
  h_DoubleComplexData = (cufftDoubleComplex *)calloc(nx, sizeof(cufftDoubleComplex));

  cufftPlan1d(&cuplan, nx, CUFFT_Z2Z, 1);

  SAMPLETYPE sampleBuffer[BUFF_SIZE];
  short buffer[BUFF_SIZE];
  static float power[MAX_FREQ];


  double cpu_total_time = 0.0;
  double gpu_total_time = 0.0;

  WavInFile inFile(wavfile);
  printf("SampleRate: %d Hz\n",inFile.getSampleRate());
  printf("Number of bits per sample: %d\n",inFile.getNumBits());
  printf("Sample data size in bytes: %d\n",inFile.getDataSizeInBytes());
  printf("Total number of samples in file: %d\n",inFile.getNumSamples());
  printf("Number of bytes per audio sample: %d\n",inFile.getBytesPerSample());
  printf("Number of audio channels in the file (1=mono, 2=stereo): %d\n",inFile.getNumChannels());
  printf("Audio file length in milliseconds: %d\n",inFile.getLengthMS());

  WavOutFile outFile(wavfileout,inFile.getSampleRate(),inFile.getNumBits(),inFile.getNumChannels());

  log = fopen(logfile,"w");

  while (inFile.eof() == 0) {

    size_t samplesRead = inFile.read(sampleBuffer, BUFF_SIZE);

    for (int i = 0; i < BUFF_SIZE; i++) {
      in[i][0] = (double) sampleBuffer[i];
      in[i][1] = 0.0;
      out[i][0] = 0.0;
      out[i][1] = 0.0;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();

    fftw_execute(plan);

    int Fs = inFile.getSampleRate();
    int N = BUFF_SIZE;
    int target_bin = (10000.0 * N) / Fs;

    for(int i = target_bin - 10; i <= target_bin + 10; i++) {
      out[i][0] = 0.0;
      out[i][1] = 0.0;
    }

    int mirror_bin = N - target_bin;

    for(int i = mirror_bin - 10; i <= mirror_bin + 10; i++) {
      out[i][0] = 0.0;
      out[i][1] = 0.0;
    }

    fftw_execute(plan2);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    cpu_total_time += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();


    CUDA_CHECK(cudaMemcpy(d_DoubleComplexData, out,
              nx * sizeof(cufftDoubleComplex),
              cudaMemcpyHostToDevice));

    auto gpu_start = std::chrono::high_resolution_clock::now();

    cufftExecZ2Z(cuplan, d_DoubleComplexData, d_DoubleComplexData, CUFFT_FORWARD);
    cufftExecZ2Z(cuplan, d_DoubleComplexData, d_DoubleComplexData, CUFFT_INVERSE);

    CUDA_CHECK(cudaDeviceSynchronize());

    auto gpu_end = std::chrono::high_resolution_clock::now();
    gpu_total_time += std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();


    CUDA_CHECK(cudaMemcpy(h_DoubleComplexData, d_DoubleComplexData,
              nx * sizeof(cufftDoubleComplex),
              cudaMemcpyDeviceToHost));

   
    for (int i = 0; i < nx; i++) {
      h_DoubleComplexData[i].x /= nx;
      h_DoubleComplexData[i].y /= nx;
    }

    for (size_t i = 0; i < samplesRead; i++) {
      int re = out[i][0];
      int im = out[i][1];
      float magnitude = sqrt(re * re + im * im);
      float freq = (i + 1) * Fs / samplesRead;
      int index = freq / 1000;

      if (index <= MAX_FREQ) {
        power[index] += magnitude;
      }
    }


    for (int i = 0; i < BUFF_SIZE; i++) {
        buffer[i] = h_DoubleComplexData[i].x;
    }

    outFile.write(buffer, BUFF_SIZE);
  }

  for (int i = 0; i < MAX_FREQ; i++)
    printf("%2d kHz, power: %9.2f\n",i,power[i]);
 
  printf("\n===== TIMING RESULTS =====\n");
  printf("Total CPU FFTW Time: %f ms\n", cpu_total_time);
  printf("Total GPU cuFFT Time: %f ms\n", gpu_total_time);

  fftw_destroy_plan(plan);
  fftw_destroy_plan(plan2);
  fftw_free(in);
  fftw_free(out);
  fftw_free(out2);

  cufftDestroy(cuplan);
  cudaFree(d_DoubleComplexData);

  fclose(log);
  return 0;
}
