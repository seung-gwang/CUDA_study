#include <iostream>
#include "book.h"

//by adding the keyword __global__, indicate to the compiler that
//we intend to run the function on the GPU.
__global__ void add(int a, int b, int *c){
	*c = a + b;
}

int main(void){
	int c;
	int *dev_c;
	HANDLE_ERROR( cudaMalloc( (void**) &dev_c, sizeof(int)));
	/*
	 cudaMalloc(pointer to the pointer you want to hold the address of the newly allocated memory,
	 		size of  allocation you want to make)
	==>you can pass pointers allocated with cudaMalloc() to...
	1)functions that execute on the device
	2)read or write memory from code that executes on the device
	3)functions that execute on the host but cannot use it to read or write memory from the code that executes on the host.

	DO NOT dereference the pointer returned by cudaMalloc() from the code that executes on the host.
	 */
	add<<<1,1>>>(2, 7, dev_c);

	//Host pointers can access memory from host code,
	//device pointers can access memory from device code..
	HANDLE_ERROR( cudaMemcpy( &c,
				dev_c
				sizeof(int),
				cudaMemcpyDeviceToHost) );

	printf("2 + 7 = %d\n", c);
	cudaFree(dev_c);

	return 0;
	
}

