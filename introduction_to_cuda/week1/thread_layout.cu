#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ELEM_NUM 1024
#define BLOCKX 20
__global__ void checkIndex(void){
	printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n"
			, threadIdx.x, threadIdx.y, threadIdx.z
			, blockIdx.x, blockIdx.y, blockIdx.z
			, blockDim.x, blockDim.y, blockDim.z
			, gridDim.x, gridDim.y, gridDim.z);

}

int main(int argc, char **argv){
	int nElem = ELEM_NUM;
	
	//Blcok의 모양 지정
	dim3 block(BLOCKX);

	//Grid의 모양 지정  
	dim3 grid((nElem + block.x - 1) / block.x);

	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);
	
	printf("----------------------------Result of checkIndex()----------------------------\n");
	checkIndex <<<grid, block>>> ();
	cudaDeviceReset();
	return(0);
}

