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

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
//#include <cub/cub.cuh>

#include "cuda_error.h"
#include "nested_loop.h"

//////////////////////////////////////////////////////////////////////
// declare here the function called by the nested loop 
__device__ void NestedLoopFunction(int ix, int iy, int val);
//////////////////////////////////////////////////////////////////////

namespace NestedLoop
{
  #include "Ny_th.h"
  void *d_sort_storage_;
  size_t sort_storage_bytes_;
  void *d_reduce_storage_;
  size_t reduce_storage_bytes_;

  int Nx_max_;
  int *d_max_Ny_;
  int *d_sorted_Ny_;

  int *d_idx_;
  int *d_sorted_idx_;

  int block_dim_x_;
  int block_dim_y_;
  int frame_area_;
  int area_th_;
  float x_lim_;
}

//////////////////////////////////////////////////////////////////////
__global__ void SimpleNestedLoopKernel(int Nx, int *Ny)
{
  int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
  int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ix<Nx && iy<Ny[ix]) {
    NestedLoopFunction(ix, iy, 2);
  }
}

//////////////////////////////////////////////////////////////////////
__global__ void Frame1DNestedLoopKernel(int ix0, int dim_x, int dim_y,
					int *sorted_idx, int *sorted_Ny)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<dim_x*dim_y) {
    int ix = ix0 + array_idx % dim_x;
    int iy = array_idx / dim_x;
    if (iy<sorted_Ny[ix]) {
      // call here the function that should be called by the nested loop
      NestedLoopFunction(sorted_idx[ix], iy, 2);
    }
  }
}

//////////////////////////////////////////////////////////////////////
__global__ void Frame2DNestedLoopKernel(int ix0, int dim_x, int dim_y,
					int *sorted_idx, int *sorted_Ny)
{
  int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
  int iy = (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ix<dim_x && iy<sorted_Ny[ix+ix0]) {
    // call here the function that should be called by the nested loop
    NestedLoopFunction(sorted_idx[ix+ix0], iy, 2);
  }
}

//////////////////////////////////////////////////////////////////////
__global__ void Smart1DNestedLoopKernel(int ix0, int iy0, int dim_x, int dim_y,
                                 int *sorted_idx, int *sorted_Ny)
{
  int array_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (array_idx<dim_x*dim_y) {
    int ix = ix0 + array_idx % dim_x;
    int iy = iy0 + array_idx / dim_x;
    if (iy<sorted_Ny[ix]) {
      // call here the function that should be called by the nested loop
      NestedLoopFunction(sorted_idx[ix], iy, 2);
    }
  }
}

//////////////////////////////////////////////////////////////////////
__global__ void Smart2DNestedLoopKernel(int ix0, int iy0, int dim_x,
					int dim_y, int *sorted_idx,
					int *sorted_Ny)
{
  int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
  int iy = iy0 + (blockIdx.y * blockDim.y) + threadIdx.y;
  if (ix<dim_x && iy<sorted_Ny[ix+ix0]) {
    // call here the function that should be called by the nested loop
    NestedLoopFunction(sorted_idx[ix+ix0], iy, 2);
  }
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Init()
{
  return Init(65536*1024);
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Init(int Nx_max)
{
  if (Nx_max <= 0) return 0;

  block_dim_x_ = 32;
  block_dim_y_ = 32;
  area_th_ = 100000000;
  frame_area_ = 65536*64;
  x_lim_ = 0.75;
  Nx_max_ = Nx_max;

  CudaSafeCall(cudaMalloc(&d_max_Ny_, sizeof(int)));  
  CudaSafeCall(cudaMalloc(&d_sorted_Ny_, Nx_max*sizeof(int)));
  CudaSafeCall(cudaMalloc(&d_idx_, Nx_max*sizeof(int)));
  CudaSafeCall(cudaMalloc(&d_sorted_idx_, Nx_max*sizeof(int)));

  int *h_idx = new int[Nx_max];
  for(int i=0; i<Nx_max; i++) {
    h_idx[i] = i;
  }  
  CudaSafeCall(cudaMemcpy(d_idx_, h_idx, Nx_max*sizeof(int),
			  cudaMemcpyHostToDevice));
  delete[] h_idx;
    
  // Determine temporary storage requirements for RadixSort
  d_sort_storage_ = NULL;
  sort_storage_bytes_ = 0;
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_sorted_Ny_, d_sorted_Ny_, d_idx_,
				  d_sorted_idx_, Nx_max);
  // Determine temporary device storage requirements for Reduce
  d_reduce_storage_ = NULL;
  reduce_storage_bytes_ = 0;
  int *d_Ny = NULL;
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx_max);

  // Allocate temporary storage
  CudaSafeCall(cudaMalloc(&d_sort_storage_, sort_storage_bytes_));
  CudaSafeCall(cudaMalloc(&d_reduce_storage_, reduce_storage_bytes_));
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Run(int Nx, int *d_Ny)
{
  return Smart2DNestedLoop(Nx, d_Ny);
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::SimpleNestedLoop(int Nx, int *d_Ny)
{
  // Find max value of Ny
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx);
  int max_Ny;
  CudaSafeCall(cudaMemcpy(&max_Ny, d_max_Ny_, sizeof(int),
			  cudaMemcpyDeviceToHost));
  return SimpleNestedLoop(Nx, d_Ny, max_Ny);
}
//////////////////////////////////////////////////////////////////////
int NestedLoop::SimpleNestedLoop(int Nx, int *d_Ny, int max_Ny)
{
  if (max_Ny < 1) max_Ny = 1;
  dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
  dim3 numBlocks((Nx - 1)/threadsPerBlock.x + 1,
		 (max_Ny - 1)/threadsPerBlock.y + 1);
  SimpleNestedLoopKernel <<<numBlocks,threadsPerBlock>>>(Nx, d_Ny);
  cudaDeviceSynchronize();
  CudaCheckError();
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Frame1DNestedLoop(int Nx, int *d_Ny)
{
  if (Nx <= 0) return 0;
  int dim_x, dim_y;

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);
  
  int ix0 = Nx;
  while(ix0>0) {
    CudaSafeCall(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    if (dim_y < 1) dim_y = 1;
    dim_x = (frame_area_ - 1) / dim_y + 1;
    ix0 -= dim_x;
    if (ix0<0) {
      dim_x += ix0;
      ix0 = 0;
    } 
    Frame1DNestedLoopKernel<<<(dim_x*dim_y+1023)/1024, 1024>>>
      (ix0, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);
  }
  cudaDeviceSynchronize();
  CudaCheckError();
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Frame2DNestedLoop(int Nx, int *d_Ny)
{
  if (Nx <= 0) return 0;
  
  int dim_x, dim_y;

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);
  
  int ix0 = Nx;
  while(ix0>0) {
    CudaSafeCall(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    if (dim_y < 1) dim_y = 1;
    dim_x = (frame_area_ - 1) / dim_y + 1;
    ix0 -= dim_x;
    if (ix0<0) {
      dim_x += ix0;
      ix0 = 0;
    }
    
    dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
    
    dim3 numBlocks((dim_x - 1)/threadsPerBlock.x + 1,
		   (dim_y - 1)/threadsPerBlock.y + 1);
    Frame2DNestedLoopKernel <<<numBlocks,threadsPerBlock>>>
      (ix0, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);

  }
  cudaDeviceSynchronize();
  CudaCheckError();
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Smart1DNestedLoop(int Nx, int *d_Ny)
{
  // Find max value of Ny
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx);
  int max_Ny;
  CudaSafeCall(cudaMemcpy(&max_Ny, d_max_Ny_, sizeof(int),
			  cudaMemcpyDeviceToHost));
  if (Nx <= 0) return 0;
  float f_Nx = 2.0*log((float)Nx)-5;
  int i_Nx = (int)floor(f_Nx);
  int Ny_th;
  if (i_Nx<0) {
    Ny_th = Ny_th_arr_[0];
  }
  else if (i_Nx>=Ny_arr_size_-1) {
    Ny_th = Ny_th_arr_[Ny_arr_size_-1];
  }
  else {
    float t = f_Nx - (float)i_Nx;
    Ny_th = Ny_th_arr_[i_Nx]*(1.0 - t) + Ny_th_arr_[i_Nx+1]*t;
  }
  if (max_Ny<Ny_th) {
    return SimpleNestedLoop(Nx, d_Ny, max_Ny);
  }

  if(max_Ny < 1) max_Ny = 1;
  
  int dim_x, dim_y;

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);
  // CudaCheckError(); // uncomment only for debugging
  
  int ix1 = (int)round(x_lim_*Nx);
  if (ix1==Nx) ix1 = Nx - 1;
  int Ny1;
  CudaSafeCall(cudaMemcpy(&Ny1, &d_sorted_Ny_[ix1], sizeof(int),
			  cudaMemcpyDeviceToHost));
  if(Ny1 < 1) Ny1 = 1;

  dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
  int nbx = (Nx - 1)/threadsPerBlock.x + 1;
  int nby = (Ny1 - 1)/threadsPerBlock.y + 1;
  Ny1 = nby*threadsPerBlock.y;
  
  dim3 numBlocks(nbx, nby);
  SimpleNestedLoopKernel <<<numBlocks,threadsPerBlock>>>(Nx, d_Ny);
  //CudaCheckError(); // uncomment only for debugging
  
  int ix0 = Nx;
  while(ix0>ix1) {
    CudaSafeCall(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    dim_y -= Ny1;
    if (dim_y<=0) break;
    dim_x = (frame_area_ - 1) / dim_y + 1;
    ix0 -= dim_x;
    if (ix0<ix1) {
      dim_x += ix0 - ix1;
      ix0 = ix1;
    } 
    Smart1DNestedLoopKernel<<<(dim_x*dim_y+1023)/1024, 1024>>>
      (ix0, Ny1, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);
    //CudaCheckError(); // uncomment only for debugging
  }
  cudaDeviceSynchronize();
  CudaCheckError();
  
  return 0;
}

//////////////////////////////////////////////////////////////////////
int NestedLoop::Smart2DNestedLoop(int Nx, int *d_Ny)
{
  // Find max value of Ny
  cub::DeviceReduce::Max(d_reduce_storage_, reduce_storage_bytes_, d_Ny,
			 d_max_Ny_, Nx);
  int max_Ny;
  CudaSafeCall(cudaMemcpy(&max_Ny, d_max_Ny_, sizeof(int),
			  cudaMemcpyDeviceToHost));
  if (Nx <= 0) return 0;
  float f_Nx = 2.0*log((float)Nx)-5;
  int i_Nx = (int)floor(f_Nx);
  int Ny_th;
  if (i_Nx<0) {
    Ny_th = Ny_th_arr_[0];
  }
  else if (i_Nx>=Ny_arr_size_-1) {
    Ny_th = Ny_th_arr_[Ny_arr_size_-1];
  }
  else {
    float t = f_Nx - (float)i_Nx;
    Ny_th = Ny_th_arr_[i_Nx]*(1.0 - t) + Ny_th_arr_[i_Nx+1]*t;
  }
  if (max_Ny<Ny_th) {
    return SimpleNestedLoop(Nx, d_Ny, max_Ny);
  }

  if(max_Ny < 1) max_Ny = 1;

  int dim_x, dim_y;

  // Run sorting operation
  cub::DeviceRadixSort::SortPairs(d_sort_storage_, sort_storage_bytes_,
				  d_Ny, d_sorted_Ny_, d_idx_, d_sorted_idx_,
				  Nx);
  // CudaCheckError(); // uncomment only for debugging
  
  int ix1 = (int)round(x_lim_*Nx);
  if (ix1==Nx) ix1 = Nx - 1;
  int Ny1;
  CudaSafeCall(cudaMemcpy(&Ny1, &d_sorted_Ny_[ix1], sizeof(int),
			  cudaMemcpyDeviceToHost));
  if(Ny1 < 1) Ny1 = 1;

  dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
  int nbx = (Nx - 1)/threadsPerBlock.x + 1;
  int nby = (Ny1 - 1)/threadsPerBlock.y + 1;
  Ny1 = nby*threadsPerBlock.y;
  
  dim3 numBlocks(nbx, nby);
  SimpleNestedLoopKernel <<<numBlocks,threadsPerBlock>>>(Nx, d_Ny);
  //CudaCheckError(); // uncomment only for debugging
  
  int ix0 = Nx;
  while(ix0>ix1) {
    CudaSafeCall(cudaMemcpy(&dim_y, &d_sorted_Ny_[ix0-1], sizeof(int),
			    cudaMemcpyDeviceToHost));
    dim_y -= Ny1;
    if (dim_y<=0) break;
    dim_x = (frame_area_ - 1) / dim_y + 1;
    ix0 -= dim_x;
    if (ix0<ix1) {
      dim_x += ix0 - ix1;
      ix0 = ix1;
    }

    dim3 threadsPerBlock(block_dim_x_, block_dim_y_);  // block size
    dim3 numBlocks((dim_x - 1)/threadsPerBlock.x + 1,
		   (dim_y - 1)/threadsPerBlock.y + 1);
    Smart2DNestedLoopKernel <<<numBlocks,threadsPerBlock>>>
      (ix0, Ny1, dim_x, dim_y, d_sorted_idx_, d_sorted_Ny_);
    //CudaCheckError(); // uncomment only for debugging      
  }
  cudaDeviceSynchronize();
  CudaCheckError();
  
  return 0;
}

