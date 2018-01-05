#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>

using std::cout;
using std::endl;

__global__ void Kernel(unsigned char* inputA,unsigned char* inputB,unsigned char* Sub,unsigned char* Binary,unsigned char* Ero,
 
unsigned char* Frame,int width,int height,int grayWidthStep)
{
	
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;	
	
	if((xIndex>0)  && (yIndex>0) && (xIndex<width) && (yIndex<height))
	{
	
		const int gray_tid  = yIndex * grayWidthStep + xIndex;

		Sub[gray_tid]=inputB[gray_tid]-inputA[gray_tid];
		
		__syncthreads();

		if(Sub[gray_tid]>200)
			Binary[gray_tid]=255;
		else
			Binary[gray_tid]=0;

		__syncthreads();	

		const int tidA = (yIndex) * grayWidthStep + (xIndex-1);
		const int tidB = (yIndex) * grayWidthStep + (xIndex+1);
		const int tidC = (yIndex-1) * grayWidthStep + (xIndex);
		const int tidD = (yIndex-1) * grayWidthStep + (xIndex-1);
		const int tidE = (yIndex-1) * grayWidthStep + (xIndex+1);		
	
		if((Binary[tidA]==255)&&(Binary[tidB]==255)&&(Binary[tidC]==255)&&(Binary[tidD]==255)&&(Binary[tidE]==255))
		Ero[gray_tid]=255;
		else
		Ero[gray_tid]=0;
		
		__syncthreads();

		if((Ero[tidA]==255)||(Ero[tidB]==255)||(Ero[tidC]==255)||(Ero[tidD]==255)||(Ero[tidE]==255))
		Frame[gray_tid]=255;
		else
		Frame[gray_tid]=0;		
		
	}
}


void ImageGrayConverter(const cv::Mat& FrameA, const cv::Mat& FrameB, cv::Mat& FrameFinal)
{
	
	//const int colorBytes = FrameA.step * FrameA.rows;
	const int grayBytes = FrameFinal.step * FrameFinal.rows;

	unsigned char *D_FrameA,*D_FrameB,*D_Sub,*D_Binary,*D_Ero,*D_FrameFinal;
	
	cudaMalloc<unsigned char>(&D_FrameA,grayBytes);
	cudaMalloc<unsigned char>(&D_FrameB,grayBytes);
	cudaMalloc<unsigned char>(&D_Sub,grayBytes);
	cudaMalloc<unsigned char>(&D_Binary,grayBytes);
	cudaMalloc<unsigned char>(&D_Ero,grayBytes);		
	cudaMalloc<unsigned char>(&D_FrameFinal,grayBytes);
	
	cudaMemcpy(D_FrameA, FrameA.ptr(),grayBytes,cudaMemcpyHostToDevice);
	cudaMemcpy(D_FrameB, FrameB.ptr(),grayBytes,cudaMemcpyHostToDevice);
		
	const dim3 block(32,32);
	
	const dim3 grid((FrameA.cols + block.x - 1)/block.x, (FrameA.rows + block.y - 1)/block.y);
	
	Kernel<<<grid,block>>>(D_FrameA,D_FrameB,D_Sub,D_Binary,D_Ero,D_FrameFinal,FrameA.cols,FrameA.rows,FrameFinal.step);
	
	cudaDeviceSynchronize();	
	
	cudaMemcpy(FrameFinal.ptr(),D_FrameFinal,grayBytes,cudaMemcpyDeviceToHost);
	
	cudaFree(D_FrameA);
	cudaFree(D_FrameB);
	cudaFree(D_Sub);
	cudaFree(D_Binary);
	cudaFree(D_Ero);		
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
			
                cvtColor(InputA,InputA,CV_BGR2GRAY);
		cvtColor(InputB,InputB,CV_BGR2GRAY);	

		cv::Mat FrameFinal(InputA.rows,InputA.cols,CV_8U);

		ImageGrayConverter(InputA,InputB,FrameFinal);		
				
		cv::imshow("Input",InputA);		
		cv::imshow("Output",FrameFinal);
				
		if(cv::waitKey(33)>=0) break;	
	}	
	return 0;
}
