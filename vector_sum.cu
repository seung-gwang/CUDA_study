#include "book.h"
#define N 60000


__global__ void add(int *a, int *b, int *c){
	int tid = blockIdx.x;
	if(tid < N)//should always be less than N
		//if not checked and subsequently fetched memory that was not ours ==> problem.
		c[tid] = a[tid] + b[tid];
}

int main(void){
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	//GPU memory allocation
	HANDLE_ERROR( cudaMalloc( (void**) &dev_a, N * sizeof(int) ));
	HANDLE_ERROR( cudaMalloc( (void**) &dev_b, N * sizeof(int) ));
	HANDLE_ERROR( cudaMalloc( (void**) &dev_c, N * sizeof(int) ));

	//fill the arrays 'a' and 'b' on the CPU

	for (int i = 0; i < N; i++) {
		a[i] = -i;
		b[i] = i * i;
	}

	//memcpy 'a', 'b' to GPU
	HANDLE_ERROR( cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR( cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));
	add<<<N,1>>>(dev_a,  dev_b, dev_c);

	//copy the array 'c' back from the GPU to the CPU	
	HANDLE_ERROR( cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	//Display
	for (int i = 0; i < N; i++){
	       printf("%d + %d = %d\n", a[i], b[i], c[i] );
	}

	//free mem allocation
	cudaFree(dev_a);	
	cudaFree(dev_b);	
	cudaFree(dev_c);	

	return 0;
}

