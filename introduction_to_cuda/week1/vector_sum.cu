#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_DATA 512

__global__ void vecAdd(int* _a, int* _b, int* _c){
	int tID = threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	int memSize = sizeof(int)*NUM_DATA;

	printf("%d elements, memsize = %d bytes\n", NUM_DATA, memSize);
	
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	
	for (int i = 0; i < NUM_DATA; i++){
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	cudaMalloc(&d_a, memSize);
	cudaMalloc(&d_b, memSize);
	cudaMalloc(&d_c, memSize);
	
	//Data Transfer overhead 1
	clock_t st = clock();
	cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
	clock_t et = clock();
	printf("Data Transfer Overhead 1 - Host to Device : %ldms\n", et - st);

	//computation
	vecAdd<<<1, NUM_DATA>>>(d_a, d_b, d_c);
	
	//Data Transfer overhead 2
	st = clock();
	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
	et = clock();
	printf("Data Transfer Overhead 2 - Device to Host : %ldms\n", et - st);
	
	//check results
	bool result = true;
	for (int i = 0; i < NUM_DATA; i++){
		if(a[i] + b[i] != c[i]){
			printf("[%d] The results is not matched! (%d, %d)\n", i, a[i] + b[i], c[i]);
			result = false;
		}
	}

	if(result)
		printf("GPU works well!\n");

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}

