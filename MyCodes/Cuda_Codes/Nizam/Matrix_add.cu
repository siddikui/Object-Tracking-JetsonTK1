#include<stdio.h>

#define H 1024
#define W 1024

__global__ void Matrix_add(int *a, int *b, int *c)

{

	int x=blockIdx.x*blockDim.x+threadIdx.x;
	int y=blockIdx.y*blockDim.y+threadIdx.y;
	
	int sum=0;

	for(int k=0; k<W; k++)
		{
		int aa=a[y*W+k];
		int bb=b[k*W+x];

		sum+=aa*bb;
		}
	c[y*W+x]=sum;

}

int main(void)
{
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;
	
	int SIZE= H*W*sizeof(int);

	h_a=(int*)malloc(SIZE);
	h_b=(int*)malloc(SIZE);
	h_c=(int*)malloc(SIZE);

	for(int i=0; i<W;i++)
	{
		for(int j=0; j<H;j++)
		{
		
		h_c[i*W+j]=0;
		h_b[i*W+j]=1;
		h_a[i*W+j]=1;
		}		
	}
	cudaMalloc(&d_a, SIZE);
	cudaMalloc(&d_b, SIZE);
	cudaMalloc(&d_c, SIZE);

	cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);


	dim3 threads(32, 32);
	dim3 blocks(32, 32);

	Matrix_add<<<blocks,threads>>>(d_a, d_b, d_c);

	cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost);

	for(int i=0; i<W; i++)
	{
		for(int j=0;j<W;j++)
		{
			printf("%d ",h_c[i*W+j]);
				
		}
		printf("\n");
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

return 0;

}

