#include<stdio.h>

#define N 1024*1024
#define threads_per_block 1024

 
__global__ void V_add(int *a, int *b, int *c)
{

	int idx=threadIdx.x+blockIdx.x*blockDim.x;
	
	c[idx]=a[idx]+b[idx]; 


}




int main(void)
{


	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;
	
	int SIZE= N*sizeof(int);
	
	h_a=(int*)malloc(SIZE);
	h_b=(int*)malloc(SIZE);
	h_c=(int*)malloc(SIZE);

	cudaMalloc(&d_a, SIZE);
	cudaMalloc(&d_b, SIZE);
	cudaMalloc(&d_c, SIZE);

	for(int i=0; i<N; i++)
	{
		h_a[i]=i;
		h_b[i]=i;
		h_c[i]=0;
	}

	cudaMemcpy(d_a,h_a, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,h_b, SIZE, cudaMemcpyHostToDevice);
	
	V_add<<<N/threads_per_block ,threads_per_block >>>(d_a,d_b,d_c);
	
	cudaMemcpy(h_c,d_c, SIZE, cudaMemcpyDeviceToHost);
	
	for(int j=0;j<N;j++)
		printf("%d\n",h_c[j]);

	free(h_a);
	free(h_b);
	free(h_c);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

return 0;
}
