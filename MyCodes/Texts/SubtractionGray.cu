#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

__global__ void Kernel( unsigned char* inputA, unsigned char* inputB, 
 
	unsigned char* outputF, int width, int height,	 int grayWidthStep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	//unsigned char outputA[640*480];
	
	
	if((xIndex>0)  && (yIndex>0) && (xIndex<width) && (yIndex<height))
	{
		
		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		outputF[gray_tid]=inputB[gray_tid]-inputA[gray_tid];
		
		__syncthreads();

		
		
	}
}


void ImageGrayConverter(const cv::Mat& FrameA, const cv::Mat& FrameB, cv::Mat& FrameFinal)
{
	
	//const int colorBytes = FrameA.step * FrameA.rows;
	const int grayBytes = FrameFinal.step * FrameFinal.rows;

	unsigned char *D_FrameA, *D_FrameB, *D_FrameFinal;
	
	cudaMalloc<unsigned char>(&D_FrameA,grayBytes);
	cudaMalloc<unsigned char>(&D_FrameB,grayBytes);
		
	cudaMalloc<unsigned char>(&D_FrameFinal,grayBytes);
	
	cudaMemcpy(D_FrameA, FrameA.ptr(),grayBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(D_FrameB, FrameB.ptr(),grayBytes,cudaMemcpyHostToDevice);
		
	const dim3 block(32,32);
	
	const dim3 grid((FrameA.cols + block.x - 1)/block.x, (FrameA.rows + block.y - 1)/block.y);
	
	Kernel<<<grid,block>>>(D_FrameA, D_FrameB, D_FrameFinal, FrameA.cols, FrameA.rows, 

	FrameFinal.step);
	
	cudaDeviceSynchronize();	
	
	cudaMemcpy(FrameFinal.ptr(),D_FrameFinal,grayBytes,cudaMemcpyDeviceToHost);
	
	cudaFree(D_FrameA);
	cudaFree(D_FrameB);
			
	cudaFree(D_FrameFinal);
}

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat InputA;	
	Mat InputB;
	Mat Output;
	Mat Gray;

	cv::VideoCapture cap;
	cap.open(string(argv[1]));	

	//Output = Mat::zeros(Input.rows,Input.cols,CV_8UC1);
	//Mat Gray = Mat::zeros(Input.rows,Input.cols,CV_8UC1);

	//cvtColor(Input,Gray,CV_BGR2GRAY);
	
	        						

	while(1)
	{	
		cap>>InputA;
		cap>>InputB;	
			
                cvtColor(InputA,InputA,CV_BGR2GRAY);
		cvtColor(InputB,InputB,CV_BGR2GRAY);

		//Output=InputB-InputA;
		//cvtColor(Output,Gray,CV_BGR2GRAY);

		cv::Mat FrameFinal(InputA.rows,InputA.cols,CV_8U);

		ImageGrayConverter(InputA,InputB,FrameFinal);		
		cv::threshold(FrameFinal,FrameFinal, 225, 255, cv::THRESH_BINARY);		
		cv::imshow("Input",InputA);		
		cv::imshow("Output",FrameFinal);
		//cv::imshow("Output_CPU",Gray);		
		if(cv::waitKey(33)>=0) break;
	
	}	
	return 0;
}
