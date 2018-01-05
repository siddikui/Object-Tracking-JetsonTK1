#include<stdio.h>

#define Width	32
#define Height	32

__global__ void Kernel(int *a, int *b, int *c, int *d)
{
	int x= blockIdx.x * blockDim.x + threadIdx.x;
	int y= blockIdx.y * blockDim.y + threadIdx.y;

	int id= y*Width+x;

	d[id]= (a[id] + b[id]) - (b[id] + c[id]);

}

int main(void)
{
	int *h_a, *h_b, *h_c, *h_d;
	int *d_a, *d_b, *d_c, *d_d;

	int Size= Width*Height*sizeof(int);

	int i,j;

	h_a= (int *)malloc(Size);
	h_b= (int *)malloc(Size);
	h_c= (int *)malloc(Size);
	h_d= (int *)malloc(Size);

	cudaMalloc(&d_a,Size);
	cudaMalloc(&d_b,Size);
	cudaMalloc(&d_c,Size);
	cudaMalloc(&d_d,Size);

	for(i=0;i<Width;i++)
		for(j=0;j<Height;j++)
			{
				h_a[i*Width+j]=32;
				h_b[i*Width+j]=32;
				h_c[i*Width+j]=16;
				h_d[i*Width+j]=0;
			}

	cudaMemcpy(d_a, h_a, Size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, Size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, Size, cudaMemcpyHostToDevice);

	dim3 Blocks(32,32);
	dim3 Grid(Width/32,Height/32);

	Kernel<<<Grid,Blocks>>>(d_a,d_b,d_c,d_d);

	cudaMemcpy(h_d, d_d, Size, cudaMemcpyDeviceToHost);

	for(i=0;i<Width;i++)
		{
		for(j=0;j<Height;j++)
			{
				printf("%d ",h_d[i*Width+j]);
			}
		printf("\n");
		}

	free(h_a);
	free(h_b);
	free(h_c);
	free(h_d);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	cudaFree(d_d);


return 0;
}
