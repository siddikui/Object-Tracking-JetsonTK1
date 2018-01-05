#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

__global__ void bgr_to_gray_kernel( unsigned char* input, unsigned char* output, int width, int height, int widthstep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;	

	float Mask[3*3]={1,2,1,0,0,0,-1,-2,-1};
	//float Mask[3*3]={-1,0,1,-2,0,2,-1,0,1};
	//float Mask[3*3]={1,2,1,0,0,0,-1,-2,-1};
	
	if((xIndex>0)  && (yIndex>0) && (xIndex<width) && (yIndex<height))
	{
		int i,j;
		float gray;		

		const int tid = yIndex * widthstep + xIndex;
		
		for(i=-1;i<=1;i++)
		for(j=-1;j<=1;j++)
		gray += input[(yIndex+i)*widthstep + (xIndex+j)]*Mask[(i+1)*3+(j+1)];

		output[tid] = static_cast<unsigned char>(gray);		
	}
}

void convert_to_gray(const cv::Mat& input, cv::Mat& output)
{
	
	const int Bytes = input.step * input.rows;	

	unsigned char *d_input, *d_output;
	
	cudaMalloc<unsigned char>(&d_input,Bytes);
	cudaMalloc<unsigned char>(&d_output,Bytes);
	
	cudaMemcpy(d_input,input.ptr(),Bytes,cudaMemcpyHostToDevice);
	
	const dim3 block(32,32);
	
	const dim3 grid((input.cols + block.x - 1)/block.x, (input.rows + block.y - 1)/block.y);
	
	bgr_to_gray_kernel<<<grid,block>>>(d_input,d_output,input.cols,input.rows,input.step);
	
	cudaDeviceSynchronize();
	
	cudaMemcpy(output.ptr(),d_output,Bytes,cudaMemcpyDeviceToHost);
	
	cudaFree(d_input);
	cudaFree(d_output);
}

int main()
{
	std::string imagePath = "image.jpg";

	
	cv::Mat input = cv::imread(imagePath,CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat output = cv::imread(imagePath,CV_LOAD_IMAGE_GRAYSCALE);

	if(input.empty())
	{
		std::cout<<"Image Not Found!"<<std::endl;
		std::cin.get();
		return -1;
	}
	
	convert_to_gray(input,output);

	cv::imshow("Input",input);
	cv::imshow("Output",output);
	
	cv::waitKey();

	return 0;
}
