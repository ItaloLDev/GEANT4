
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include <stdio.h>

cudaError_t findPIInitializer(int *circle_points, int *square_points, int size, int const INTERVALO);

__global__ void findPIKernel(int *d_circle_points,  int *d_square_points, int const INTERVALO, unsigned int seed)
{
    curandState_t state;
    curand_init(seed, 0, 0, &state);

    /* curand works like rand - except that it takes a state as a parameter */
    double rand_x, rand_y, origin_dist;

    rand_x = double(curand(&state) % INTERVALO) / INTERVALO;
    rand_y = double(curand(&state) % INTERVALO) / INTERVALO;

    // Distance between (x, y) from the origin 
    origin_dist = rand_x * rand_x + rand_y * rand_y;

    // Checking if (x, y) lies inside the define 
    // circle with R=1 
    if (origin_dist <= 1)
        circle_points+=1;

    // Total number of points generated 
    square_points+=1;
}

int main()
{
   

    int circle_points, square_points; // copias locais do numero de pontos dentro do circulo e fora(no quadrado como um todo)
    int *d_circle_points, *d_square_points; // device copies of a, b, c
    int size = sizeof(int);
    int const INTERVALO = 10;

    // Add vectors in parallel.
    cudaError_t cudaStatus = findPIInitializer(circle_points, square_points, size, INTERVALO);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t findPIInitializer(int *circle_points, int *square_points, int size, int const INTERVALO)
{
    int *d_circle_points, *d_square_points;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate space for device copies of a, b, c.
    cudaStatus = cudaMalloc((void **)&d_circle_points, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void **)&d_square_points, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(d_circle_points, circle_points, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(d_square_points, square_points, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    srand(time(NULL));
    int threadsPerBlock = 256;
    int blocksPerGrid = (INTERVALO + threadsPerBlock - 1) / threadsPerBlock;
    findPIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_circle_points, d_square_points, INTERVALO, time(NULL));

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "findPIKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(circle_points d_circle_points, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(square_points d_square_points, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
