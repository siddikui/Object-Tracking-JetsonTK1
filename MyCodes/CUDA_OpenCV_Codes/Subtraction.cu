#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

__global__ void Kernel( unsigned char* FrameA, unsigned char* FrameB,unsigned char* Frame,unsigned char* FrameF,int width,int height,int colorWidthStep, int grayWidthStep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	if((xIndex>0)  && (yIndex>0) && (xIndex<width) && (yIndex<height))
	{
		const int color_tid = yIndex * colorWidthStep + (3 * xIndex);
		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		Frame[color_tid]=FrameB[color_tid]-FrameA[color_tid];	
		Frame[color_tid+1]=FrameB[color_tid+1]-FrameA[color_tid+1];	
		Frame[color_tid+2]=FrameB[color_tid+2]-FrameA[color_tid+2];		
		__syncthreads();

		const unsigned char blue	= Frame[color_tid];
		const unsigned char green	= Frame[color_tid + 1];
		const unsigned char red		= Frame[color_tid + 2];

		const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

		FrameF[gray_tid] = static_cast<unsigned char>(gray);		
		
	}
}


void ImageGrayConverter(const cv::Mat& FrameA, const cv::Mat& FrameB, cv::Mat& FrameFinal)
{
	
	const int colorBytes = FrameA.step * FrameA.rows;
	const int grayBytes = FrameFinal.step * FrameFinal.rows;

	unsigned char *D_FrameA, *D_FrameB, *D_Frame, *D_FrameFinal;
	
	cudaMalloc<unsigned char>(&D_FrameA,colorBytes);
	cudaMalloc<unsigned char>(&D_FrameB,colorBytes);
	cudaMalloc<unsigned char>(&D_Frame,colorBytes);
		
	cudaMalloc<unsigned char>(&D_FrameFinal,grayBytes);
	
	cudaMemcpy(D_FrameA, FrameA.ptr(),colorBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(D_FrameB, FrameB.ptr(),colorBytes,cudaMemcpyHostToDevice);
		
	const dim3 block(16,16);
	
	const dim3 grid((FrameA.cols + block.x - 1)/block.x, (FrameA.rows + block.y - 1)/block.y);
	
	Kernel<<<grid,block>>>(D_FrameA,D_FrameB,D_Frame,D_FrameFinal,FrameA.cols,FrameA.rows,FrameA.step,FrameFinal.step);
	
	cudaDeviceSynchronize();	
	
	cudaMemcpy(FrameFinal.ptr(),D_FrameFinal,grayBytes,cudaMemcpyDeviceToHost);
	
	cudaFree(D_FrameA);
	cudaFree(D_FrameB);
	cudaFree(D_Frame);
			
	cudaFree(D_FrameFinal);
}

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat InputA;	
	Mat InputB;

	cv::VideoCapture cap;
	cap.open(string(argv[1])); 						

	while(1)
	{	
		cap>>InputA;
		cap>>InputB;				
     
		cv::Mat FrameFinal(InputA.rows,InputA.cols,CV_8U);

		ImageGrayConverter(InputA,InputB,FrameFinal);
			
		cv::imshow("Input",InputA);		
		cv::imshow("Output",FrameFinal);
		
		if(cv::waitKey(33)>=0) break;
	
	}	
	return 0;
}
