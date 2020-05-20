//
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <curand.h>
//#include <curand_kernel.h>
//#include <time.h>
//
//#include <stdio.h>
//
//#define INTERVALO 100
//
//cudaError_t findPIInitializer(int* circle_points, int* square_points, int size);
//
//__global__ void findPIKernel(int *d_circle_points,  int *d_square_points, unsigned int seed)
//{
//    curandState_t state;
//    curand_init(seed, 0, 0, &state);
//
//    /* curand works like rand - except that it takes a state as a parameter */
//    double rand_x, rand_y, origin_dist;
//
//    rand_x = double(curand(&state) % INTERVALO) / INTERVALO;
//    rand_y = double(curand(&state) % INTERVALO) / INTERVALO;
//
//    // Distance between (x, y) from the origin 
//    origin_dist = rand_x * rand_x + rand_y * rand_y;
//
//    // Checking if (x, y) lies inside the define 
//    // circle with R=1 
//    if (origin_dist <= 1)
//        (*d_circle_points)++;
//
//    // Total number of points generated 
//    (*d_square_points)++;
//}
//
//int main()
//{
//   
//
//    int circle_points = 7;int  square_points = 0; // copias locais do numero de pontos dentro do circulo e fora(no quadrado como um todo)
//    int *h_circle_points, *h_square_points;
//    h_circle_points = &circle_points;// device copies of numero de pontos dentro do circulo e fora(no quadrado como um todo)
//    h_square_points = &square_points;// device copies of numero de pontos dentro do circulo e fora(no quadrado como um todo)
//    int size = sizeof(int);
//
//    // Add vectors in parallel.
//    cudaError_t cudaStatus = findPIInitializer(h_circle_points, h_square_points, size);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "addWithCuda failed!");
//        return 1;
//    }
//
//    fprintf(stderr, "Numero de pontos no circulo: %d\n", circle_points);
//    fprintf(stderr, "Numero de pontos no quadrado: %d\n", square_points);
//
//    // cudaDeviceReset must be called before exiting in order for profiling and
//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
//    /*cudaStatus = cudaDeviceReset();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceReset failed!");
//        return 1;
//    }*/
//
//    return 0;
//}
//
//// Helper function for using CUDA to add vectors in parallel.
//cudaError_t findPIInitializer(int* h_circle_points, int* h_square_points, int size)
//{
//    int* d_circle_points = 0; int* d_square_points = 0;;
//    cudaError_t cudaStatus;
//
//    // Choose which GPU to run on, change this on a multi-GPU system.
//    cudaStatus = cudaSetDevice(0);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
//        goto Error;
//    }
//
//    // Allocate space for device copies of a, b, c.
//    cudaStatus = cudaMalloc((void **)&d_circle_points, size);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMalloc((void **)&d_square_points, size);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMalloc failed!");
//        goto Error;
//    }
//
//    // Launch a kernel on the GPU with one thread for each element.
//    srand(time(NULL));
//    int threadsPerBlock = 256;
//    int blocksPerGrid = (INTERVALO + threadsPerBlock - 1) / threadsPerBlock;
//    findPIKernel<<<blocksPerGrid, threadsPerBlock>>>(d_circle_points, d_square_points, time(NULL));
//
//    // Check for any errors launching the kernel
//    cudaStatus = cudaGetLastError();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "findPIKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
//        goto Error;
//    }
//    
//    // cudaDeviceSynchronize waits for the kernel to finish, and returns
//    // any errors encountered during the launch.
//    cudaStatus = cudaDeviceSynchronize();
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
//        goto Error;
//    }
//
//    // Copy output vector from GPU buffer to host memory.
//    cudaStatus = cudaMemcpy(h_circle_points, d_circle_points, size, cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//    cudaStatus = cudaMemcpy(h_square_points, d_square_points, size, cudaMemcpyDeviceToHost);
//    if (cudaStatus != cudaSuccess) {
//        fprintf(stderr, "cudaMemcpy failed!");
//        goto Error;
//    }
//
//Error:
//    cudaFree(d_circle_points);
//    cudaFree(d_square_points);
//    
//    return cudaStatus;
//}
