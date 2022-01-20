#include "book.h"

int main(void){
	cudaDeviceProp prop;

	int count;
	HANDLE_ERROR( cudaGetDeviceCount( &count));

	for (int i = 0; i < count; i++){
		HANDLE_ERROR( cudaGetDeviceProperties( &prop, i));
		printf("Total global mem: %ld\n", prop.totalGlobalMem);
		//other queries can be made here

	}
}
