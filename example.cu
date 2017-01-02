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

int main()
{
  NestedLoop::Init();
  int Nx = 10;
  int h_Ny[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int *h_test_array = new int[Nx];
  int *d_Ny;
  int *d_test_array;
  CudaSafeCall(cudaMalloc(&d_Ny, Nx*sizeof(int)));
  CudaSafeCall(cudaMalloc(&d_test_array, Nx*sizeof(int)));
  InitTestArray<<<1, 1>>>(d_test_array);
  cudaMemcpy(d_Ny, h_Ny, Nx*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_test_array, 0, Nx*sizeof(int));  
  NestedLoop::Run(Nx, d_Ny);
  cudaMemcpy(h_test_array, d_test_array, Nx*sizeof(int),
	     cudaMemcpyDeviceToHost);
  for(int ix=0; ix<Nx; ix++) {
    printf("ix: %d\ttest array: %d\texpected value: %d\n", ix,
    h_test_array[ix], h_Ny[ix]*(h_Ny[ix] - 1));
  }

  CudaSafeCall(cudaFree(d_Ny));
  CudaSafeCall(cudaFree(d_test_array));
  delete[] h_test_array;

  return 0;
}