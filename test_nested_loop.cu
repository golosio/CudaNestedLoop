/*
Copyright (C) 2016 Bruno Golosio
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda_error.h"
#include "nested_loop.h"

__device__ int *TestArray;

__global__ void InitTestArray(int *test_array)
{
  TestArray = test_array;
}

__device__ void NestedLoopFunction(int ix, int iy, int val)
{
  atomicAdd(&TestArray[ix], iy*val);
}

int SetNy(int Nx, int Ny_max, int *Ny, int k);

int main(int argc, char*argv[])
{
  int Nx_max;
  int Ny_max;
  float k;
    
  if (argc!=4) {
    printf("Usage: %s Nx Ny_max k\n", argv[0]);
    return 0;
  }
    
  sscanf(argv[1], "%d", &Nx_max);
  sscanf(argv[2], "%d", &Ny_max);
  sscanf(argv[3], "%f", &k);

  int Nx = Nx_max;
  
  int *h_Ny;
  int *d_Ny;

  int *h_test_array;
  int *d_test_array;
  int *ref_array;

  h_test_array = new int[Nx_max];
  ref_array = new int[Nx_max];
  CudaSafeCall(cudaMalloc(&d_test_array, Nx_max*sizeof(int)));
  InitTestArray<<<1, 1>>>(d_test_array);
  
  h_Ny = new int[Nx_max];
  CudaSafeCall(cudaMalloc(&d_Ny, Nx_max*sizeof(int)));
  
  NestedLoop::Init();
  
  printf("Testing Frame1DNestedLoop...\n");
  SetNy(Nx, Ny_max, h_Ny, k);
  cudaMemcpy(d_Ny, h_Ny, Nx*sizeof(int), cudaMemcpyHostToDevice);
  
  for(int ix=0; ix<Nx_max; ix++) {
    ref_array[ix] = h_Ny[ix]*(h_Ny[ix] - 1);
  }
  cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
  
  NestedLoop::Frame1DNestedLoop(Nx, d_Ny);

  cudaMemcpy(h_test_array, d_test_array, Nx_max*sizeof(int),
	     cudaMemcpyDeviceToHost);

  for(int ix=0; ix<Nx_max; ix++) {
    if (h_test_array[ix] != ref_array[ix]) {
      printf("Frame1DNestedLoop error at ix = %d\n", ix);
      exit(-1);
    }
  }
  printf("OK\n\n");
  
  printf("Testing Frame2DNestedLoop...\n");
  cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
  
  NestedLoop::Frame2DNestedLoop(Nx, d_Ny);
  
  cudaMemcpy(h_test_array, d_test_array, Nx_max*sizeof(int),
	     cudaMemcpyDeviceToHost);
  
  for(int ix=0; ix<Nx_max; ix++) {
    if (h_test_array[ix] != ref_array[ix]) {
      printf("Frame2DNestedLoop error at ix = %d\n", ix);
      exit(-1);
    }
  }
  printf("OK\n\n");

  printf("Testing Smart2DNestedLoop...\n");
  cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
  
  NestedLoop::Smart2DNestedLoop(Nx, d_Ny);
  
  cudaMemcpy(h_test_array, d_test_array, Nx_max*sizeof(int),
	     cudaMemcpyDeviceToHost);

  for(int ix=0; ix<Nx_max; ix++) {
    if (h_test_array[ix] != ref_array[ix]) {
      printf("Smart2DNestedLoop error at ix = %d\n", ix);
      exit(-1);
    }
  }
  printf("OK\n\n");

  printf("Testing Smart1DNestedLoop...\n");
  cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
  
  NestedLoop::Smart1DNestedLoop(Nx, d_Ny);
  
  cudaMemcpy(h_test_array, d_test_array, Nx_max*sizeof(int),
	     cudaMemcpyDeviceToHost);

  for(int ix=0; ix<Nx_max; ix++) {
    if (h_test_array[ix] != ref_array[ix]) {
      printf("Smart1DNestedLoop error at ix = %d\n", ix);
      exit(-1);
    }
  }
  printf("OK\n\n");

  printf("Testing SimpleNestedLoop...\n");
  cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
  
  NestedLoop::SimpleNestedLoop(Nx, d_Ny);

  cudaMemcpy(h_test_array, d_test_array, Nx_max*sizeof(int),
	     cudaMemcpyDeviceToHost);

  for(int ix=0; ix<Nx_max; ix++) {
    if (h_test_array[ix] != ref_array[ix]) {
      printf("SimpleNestedLoop error at ix = %d\n", ix);
      exit(-1);
    }
  }
  printf("OK\n\n");

  printf("Evaluating execution time...\n");
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  long n_iter = (1000000000l/Nx/Ny_max);
  if (n_iter<1) n_iter=1;
  if (n_iter>1000) n_iter=1000;

  float frame1D_nested_loop_time = 0;
  float frame2D_nested_loop_time = 0;
  float smart1D_nested_loop_time = 0;
  float smart2D_nested_loop_time = 0;
  float simple_nested_loop_time = 0;
  
  for (long i_iter=0; i_iter<n_iter; i_iter++) {
    SetNy(Nx, Ny_max, h_Ny, k);
    cudaMemcpy(d_Ny, h_Ny, Nx*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
    cudaEventRecord(start);
    NestedLoop::Frame1DNestedLoop(Nx, d_Ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    frame1D_nested_loop_time += milliseconds;

    cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
    cudaEventRecord(start);
    NestedLoop::Frame2DNestedLoop(Nx, d_Ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    frame2D_nested_loop_time += milliseconds;

    cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
    cudaEventRecord(start);
    NestedLoop::Smart1DNestedLoop(Nx, d_Ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    smart1D_nested_loop_time += milliseconds;

    cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
    cudaEventRecord(start);
    NestedLoop::Smart2DNestedLoop(Nx, d_Ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    smart2D_nested_loop_time += milliseconds;

    cudaMemset(d_test_array, 0, Nx_max*sizeof(int));  
    cudaEventRecord(start);
    NestedLoop::SimpleNestedLoop(Nx, d_Ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    simple_nested_loop_time += milliseconds;
  }
  frame1D_nested_loop_time = frame1D_nested_loop_time / n_iter;
  frame2D_nested_loop_time = frame2D_nested_loop_time / n_iter;
  smart1D_nested_loop_time = smart1D_nested_loop_time / n_iter;
  smart2D_nested_loop_time = smart2D_nested_loop_time / n_iter;  
  simple_nested_loop_time = simple_nested_loop_time / n_iter;
  
  printf ("Frame1DNestedLoop average time: %f ms\n", frame1D_nested_loop_time);
  printf ("Frame2DNestedLoop average time: %f ms\n", frame2D_nested_loop_time);
  printf ("Smart1DNestedLoop average time: %f ms\n", smart1D_nested_loop_time);
  printf ("Smart2DNestedLoop average time: %f ms\n", smart2D_nested_loop_time);
  printf ("SimpleNestedLoop average time: %f ms\n", simple_nested_loop_time);
  
  return 0;
}

float rnd()
{
  float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  return r;
}


float rnd_distribution(float k)
{
  if (k<1.e-6) return rnd();
  float eps = 0.01;
  float C = k/(1.-exp(-k));

  float x, y, f;
  do {
    x = rnd();
    y = rnd();
    f = eps + (1.-eps)*C*exp(-k*x);
  } while (y>f);
  
  return x;
}

int SetNy(int Nx, int Ny_max, int *Ny, int k)
{
  for (int ix=0; ix<Nx; ix++) {
    int ny = (int)floor(rnd_distribution(k)*Ny_max);
    if (ny == 0) ny = 1;
    Ny[ix] = ny;
  }

  return 0;

}
