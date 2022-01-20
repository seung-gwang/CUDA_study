#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void helloCUDA(void){
	printf("GPU: Hello CUDA!\n");
}

int main(void){
	printf("CPU: Hey GPU, say 'Hello CUDA'\n");
	helloCUDA<<<10, 10>>>();

	return 0;
}

