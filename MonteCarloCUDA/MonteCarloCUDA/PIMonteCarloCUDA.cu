// Approximation of Pi using a simple, and not optimized, CUDA program
// Copyleft Alessandro Re
//
// GCC 6.x not supported by CUDA 8, I used compat version
//
// nvcc -std=c++11 -ccbin=gcc5 pigreco.cu -c
// g++5 pigreco.o -lcudart -L/usr/local/cuda/lib64 -o pigreco
//
// This code is basically equivalent to the following Python code:
//
// def pigreco(NUM):
//     from random import random as rand
//     def sqrad():
//         x, y = rand(), rand()
//         return x*x + y*y
//     return 4 * sum(1 - int(test()) for _ in range(NUM)) / NUM
//
// Python version takes, on this machine, 3.5 seconds to compute 10M tests
// CUDA version takes, on this machine, 1.6 seconds to compute 20.48G tests
//
#include <iostream>
#include <limits>
#include <cuda.h>
#include <curand_kernel.h>
#include <time.h>

using std::cout;
using std::endl;

typedef unsigned long long Count;
typedef std::numeric_limits<double> DblLim;

const Count WARP_SIZE = 40; // Warp size
const Count NBLOCKS = 896; // total cuda cores GPU
const Count ITERATIONS = 28000; // Numero de pontos(por thread)

// This kernel is 
__global__ void picount(Count* totals) {
	// memoria compartilhada
	__shared__ Count counter[WARP_SIZE];

	//  ID thread
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// RNG
	curandState_t rng;
	curand_init(clock64(), tid, 0, &rng);

	//counter inicio
	counter[threadIdx.x] = 0;

	
	for (int i = 0; i < ITERATIONS; i++) {
		float x = curand_uniform(&rng); // Random x position in [0,1]
		float y = curand_uniform(&rng); // Random y position in [0,1]
		float dist = (x * x) + (y * y);
		if (dist <= float(1))
			counter[threadIdx.x] += 1 - int(dist); 
	}

	// primeira thread do bloco soma os resultados
	if (threadIdx.x == 0) {
		// Reset count for this block
		totals[blockIdx.x] = 0;
		for (int i = 0; i < WARP_SIZE; i++) {
			totals[blockIdx.x] += counter[i];
		}
	}
}

int main(int argc, char** argv) {
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		cout << "CUDA device missing! Do you need to use optirun?\n";
		return 1;
	}
	cout << "Starting simulation with " << NBLOCKS << " blocks, " << WARP_SIZE << " threads, and " << ITERATIONS << " iterations\n";

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	Count* hOut, * dOut;
	hOut = new Count[NBLOCKS]; // Host memory
	cudaMalloc(&dOut, sizeof(Count) * NBLOCKS); // Device memory

	// kernel
	picount << <NBLOCKS, WARP_SIZE >> > (dOut);

	cudaMemcpy(hOut, dOut, sizeof(Count) * NBLOCKS, cudaMemcpyDeviceToHost);
	cudaFree(dOut);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float t_gpu1;
	cudaEventElapsedTime(&t_gpu1, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	Count total = 0;
	for (int i = 0; i < NBLOCKS; i++) {
		total += hOut[i];
	}
	Count tests = NBLOCKS * ITERATIONS * WARP_SIZE;
	cout << "Approximated PI using " << tests << " random tests\n";

	cout.precision(DblLim::max_digits10);
	cout << "PI ~= " << 4.0 * (double)total / (double)tests << endl;
	cout << "TEMPO: " << t_gpu1/1000 << " sec" << endl;

	return 0;
}