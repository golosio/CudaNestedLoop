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
__device__ int ArrayNx;
__device__ int ArrayNy;

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
  const int N_Nx = NestedLoop::Ny_arr_size_;
  const int N_Ny = NestedLoop::Ny_arr_size_;
  float algo_arr[N_Nx*N_Ny];

  int k_arr[] ={0, 10, 50, 100};
  int Nk = 4;
  
  int Nx_max = 65536*1024;
  int Ny_max;
  int Nx;
  float k;
    
  int *h_Ny;
  int *d_Ny;

  int *d_test_array;

  CudaSafeCall(cudaMalloc(&d_test_array, Nx_max*sizeof(int)));
  InitTestArray<<<1, 1>>>(d_test_array);
  
  h_Ny = new int[Nx_max];
  CudaSafeCall(cudaMalloc(&d_Ny, Nx_max*sizeof(int)));
  
  NestedLoop::Init();
  NestedLoop::area_th_ = 0;
  NestedLoop::frame_area_ = 50000000;
  NestedLoop::x_lim_ = 0.75;
  for (int i_Ny=0; i_Ny<NestedLoop::Ny_arr_size_; i_Ny++) {
    NestedLoop::Ny_th_arr_[i_Ny] = 0;
  }
  
  for (int i=0; i<N_Nx*N_Ny; i++) {
    algo_arr[i] = 1;
  }

  float worst_err = 0;
  float rmse = 0;
  int Nmse = 0;
  for (int i_Nx=0; i_Nx<N_Nx; i_Nx++) {
    Nx = (int)round(exp((5.0 + i_Nx)/2.0));
    for (int i_Ny=0; i_Ny<N_Ny; i_Ny++) {
      Ny_max = (int)round(exp((5.0 + i_Ny)/2.0));
      if ((long)Nx*Ny_max>1000000000l) continue;
    
      //printf("Evaluating execution time...\n");
      cudaEvent_t start, stop;
      cudaEventCreate(&start);
      cudaEventCreate(&stop);
  
      long n_iter = (500000000l/Nx/Ny_max); //100;
      if (n_iter<25) n_iter=25;
      if (n_iter>500) n_iter=500;
      printf ("n_iter: %ld\t", n_iter);
      float frame1D_nested_loop_time = 0;
      float frame2D_nested_loop_time = 0;
      float smart1D_nested_loop_time = 0;
      float smart2D_nested_loop_time = 0;
      float simple_nested_loop_time = 0;

      float t_test[5];
      float t_worst[5];
  
      for (int ik=0; ik<Nk; ik++) {
	k = k_arr[ik];
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

	/*
	  printf ("Frame1DNestedLoop average time: %f ms\n",
	  frame1D_nested_loop_time);
	  printf ("Frame2DNestedLoop average time: %f ms\n",
	  frame2D_nested_loop_time);
	  printf ("Smart1DNestedLoop average time: %f ms\n",
	  smart1D_nested_loop_time);
	  printf ("Smart2DNestedLoop average time: %f ms\n",
	  smart2D_nested_loop_time);
	  printf ("SimpleNestedLoop average time: %f ms\n",
	  simple_nested_loop_time);
	*/
	t_test[0] = simple_nested_loop_time;
	t_test[1] = smart2D_nested_loop_time;
	t_test[2] = smart1D_nested_loop_time;
	t_test[3] = frame2D_nested_loop_time;
	t_test[4] = frame1D_nested_loop_time;
	float t_min = 0;
	for (int i=0; i<5; i++) {
	  if (i==0 || t_test[i]<t_min) {
	    t_min = t_test[i];
	  }
	}
	for (int i=0; i<5; i++) {
	  float t_rel = (t_test[i]-t_min)/t_min;
	  if (ik==0 || t_rel>t_worst[i]) {
	    t_worst[i] = t_rel;
	  }
	}
      }
      //for (int i=0; i<5; i++) {
      //  printf("%f\n", t_worst[i]);
      //}
      float best_algo;
      float err_best_algo;
      if (t_worst[0]<t_worst[1]) {
	best_algo = 0;
	err_best_algo = t_worst[0];
      }
      else {
	best_algo = 1;
	err_best_algo = t_worst[1];
      }
      if (fabs(t_worst[0]-t_worst[1])<0.05) best_algo = 0.5;
      if (err_best_algo>worst_err) {
	worst_err = err_best_algo;
      }
      rmse += err_best_algo*err_best_algo;
      Nmse++;
      printf("%d\t%d\t%g\t%f\n", Nx, Ny_max, best_algo, err_best_algo);
      algo_arr[i_Ny*N_Nx + i_Nx] = best_algo;
    }
  }
  printf("worst error: %f\n", worst_err);
  rmse = sqrt(rmse/Nmse);
  printf("RMSE: %f\n", rmse);

  FILE *fp=fopen("Ny_th.h", "w");
  fprintf(fp, "  const int Ny_arr_size_ = 24;\n");
  fprintf(fp, "  int Ny_th_arr_[] = {\n");
     
  for (int i_Nx=0; i_Nx<N_Nx; i_Nx++) {
    Nx = (int)round(exp((5.0 + i_Nx)/2.0));
    for (int i_Ny=2; i_Ny<N_Ny-2; i_Ny++) {
      float algo = 0;
      for (int j=-2; j<=2; j++) {
	int j_Ny = i_Ny + j;
	algo += algo_arr[j_Ny*N_Nx + i_Nx];
      }
      algo /= 5;
      if (algo>=0.5) {
	if (algo==0.5) {
	  Ny_max = (int)round(exp((5.0 + i_Ny)/2.0));
	}
	else {
	  Ny_max = (int)round((exp((5.0 + i_Ny)/2.0)
			       + exp((5.0 + i_Ny - 1)/2.0))/2.0);
	}
	printf("%d\t%d\n", Nx, Ny_max);
	fprintf(fp, "    %d", Ny_max);
	if (i_Nx<N_Nx-1) {
	  fprintf(fp, ",");
	}
	fprintf(fp, "\n");
	break;
      }
    }
  }
  fprintf(fp, "};\n");
  fclose(fp);
  
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
