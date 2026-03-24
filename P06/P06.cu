/*
 * How to compile:
 *   nvcc -c -I/usr/local/cuda/include lu.cpp
 *   g++ -o gpulu lu.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

int main(int argc, char*argv[])
{
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;

  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  cudaError_t cudaStat3 = cudaSuccess;
  cudaError_t cudaStat4 = cudaSuccess;

  /* Example system with solution [-1,1,-2,3]'
   *
   *  4  1  2 -3 | -16
   * -3  3 -1  4 |  20
   * -1  2  5  1 |  -4
   *  5  4  3 -1 | -10
   */
  const int m = 4;
  const int lda = m;
  const int ldb = m;
  double A[lda*m] = {4,-3,-1,5,  1,3,2,4,  2,-1,5,3,  -3,4,1,-1}; // define coef. matrix in ***column*** order
  double B[m] = { -16, 20, -4, -10 }; // define RHS

  double X[m]; /* X = A\B */
  double LU[lda*m]; /* L and U */
  int Ipiv[m];      /* host copy of pivoting sequence */
  int info = 0;     /* host copy of error info */

  double *d_A = NULL; /* device copy of A */
  double *d_B = NULL; /* device copy of B */
  int *d_Ipiv = NULL; /* pivoting sequence */
  int *d_info = NULL; /* error info */
  int  lwork = 0;     /* size of workspace */
  double *d_work = NULL; /* device workspace for getrf */
  const int pivot_on = 0;

  /* step 1: create cusolver handle, bind a stream */
  status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  assert(cudaSuccess == cudaStat1);

  status = cusolverDnSetStream(cusolverH, stream);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  /* step 2: copy A to device */
  cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
  cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * m);
  cudaStat2 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
  cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);
  assert(cudaSuccess == cudaStat4);

  cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*m, cudaMemcpyHostToDevice);
  cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)*m, cudaMemcpyHostToDevice);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  /* step 3: query working space of getrf */
  status = cusolverDnDgetrf_bufferSize(
                                       cusolverH,
                                       m,
                                       m,
                                       d_A,
                                       lda,
                                       &lwork);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
  assert(cudaSuccess == cudaStat1);

  /* step 4: LU factorization */
  if (pivot_on){
    status = cusolverDnDgetrf(
                              cusolverH,
                              m,
                              m,
                              d_A,
                              lda,
                              d_work,
                              d_Ipiv,
                              d_info);
  }else{
    status = cusolverDnDgetrf(
                              cusolverH,
                              m,
                              m,
                              d_A,
                              lda,
                              d_work,
                              NULL,
                              d_info);
  }
  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == status);
  assert(cudaSuccess == cudaStat1);

  if (pivot_on){
    cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
  }
  cudaStat2 = cudaMemcpy(LU   , d_A   , sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
  cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);
  assert(cudaSuccess == cudaStat2);
  assert(cudaSuccess == cudaStat3);

  if ( 0 > info ){
    printf("%d-th parameter is wrong \n", -info);
    exit(1);
  }
  if (pivot_on){
    printf("pivoting sequence, matlab base-1\n");
    for(int j = 0 ; j < m ; j++){
      printf("Ipiv(%d) = %d\n", j+1, Ipiv[j]);
    }
  }

  /*
   * step 5: solve A*X = B
   */
  if (pivot_on){
    status = cusolverDnDgetrs(
                              cusolverH,
                              CUBLAS_OP_N,
                              m,
                              1, /* nrhs */
                              d_A,
                              lda,
                              d_Ipiv,
                              d_B,
                              ldb,
                              d_info);
  }else{
    status = cusolverDnDgetrs(
                              cusolverH,
                              CUBLAS_OP_N,
                              m,
                              1, /* nrhs */
                              d_A,
                              lda,
                              NULL,
                              d_B,
                              ldb,
                              d_info);
  }
  cudaStat1 = cudaDeviceSynchronize();
  assert(CUSOLVER_STATUS_SUCCESS == status);
  assert(cudaSuccess == cudaStat1);

  cudaStat1 = cudaMemcpy(X , d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);
  assert(cudaSuccess == cudaStat1);

  printf("solution:\nx = [");
  for(int j = 0 ; j < m ; j++)
    printf(" %f ", X[j]);
  printf("]'\n");

  /* free resources */
  if (d_A    ) cudaFree(d_A);
  if (d_B    ) cudaFree(d_B);
  if (d_Ipiv ) cudaFree(d_Ipiv);
  if (d_info ) cudaFree(d_info);
  if (d_work ) cudaFree(d_work);

  if (cusolverH   ) cusolverDnDestroy(cusolverH);
  if (stream      ) cudaStreamDestroy(stream);

  cudaDeviceReset(); // destroy context

  return 0;
}