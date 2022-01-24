#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_DATA 1025 //large number (more than 1024) makes incorrect results in Kernel computation

__global__ void vecAdd(int* _a, int* _b, int* _c){
	int tID = threadIdx.x;
	_c[tID] = _a[tID] + _b[tID];
}

int main(void){
	int *a, *b, *c, *host_c;
	int *d_a, *d_b, *d_c;

	int memSize = sizeof(int)*NUM_DATA;

	printf("%d elements, memsize = %d bytes\n", NUM_DATA, memSize);
	
	a = new int[NUM_DATA]; memset(a, 0, memSize);
	b = new int[NUM_DATA]; memset(b, 0, memSize);
	c = new int[NUM_DATA]; memset(c, 0, memSize);
	host_c = new int[NUM_DATA]; memset(host_c, 0, memSize);

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
	printf("Data Transfer Overhead 1 - Host to Device : %f seconds\n", (double)(et - st)/CLOCKS_PER_SEC);

	//computation on Device
	st = clock();
	vecAdd<<<1, NUM_DATA>>>(d_a, d_b, d_c);
	et = clock();	
	printf("Vector Summation Kernel computation : %f seconds\n",(double)(et - st)/CLOCKS_PER_SEC);
	
	//computation on Host
	st = clock();
	for(int i = 0; i < NUM_DATA; i++)
		host_c[i] = a[i] + b[i];
	et = clock();
	printf("Vector Summation Host computation : %f seconds\n", (double)(et - st)/CLOCKS_PER_SEC);



	//Data Transfer overhead 2
	st = clock();
	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
	et = clock();
	printf("Data Transfer Overhead 2 - Device to Host : %f seconds\n", (double)(et - st)/CLOCKS_PER_SEC);
	
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
	delete[] host_c;

	return 0;
}

