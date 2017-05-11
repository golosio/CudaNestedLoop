CudaNestedLoop
==============

This code uses CUDA to parallelize nested loops of the type:

```
for(int ix=0; ix<Nx; ix++) {
  for(int iy=0; iy<Ny[ix]; iy++) {
    NestedLoopFunction(ix, iy, ...);
	       ...
```

where Ny[] is an array in CUDA global memory.
In order to use it:

1) Install cub from https://nvlabs.github.io/cub/

2) Add to your code the files nested_loop.cu, nested_loop.h, Ny_th.h,
   cuda_error.h

3) Put the body of the nested loop in a __device__ function
   (NestedLoopFunction in this example) and modify as you need the call to
   this function in nested_loop.cu

4) At the beginning of your program call
   NestedLoop::Init();

5) Run the nested loop with the command:
   NestedLoop::Run(Nx, d_Ny);
   where d_Ny is the array Ny[ix] stored in CUDA global memory 

See the example in the file example.cu

