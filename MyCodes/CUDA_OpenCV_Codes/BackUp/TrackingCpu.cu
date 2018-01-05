#include<iostream>
#include<cstdio>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<cuda_runtime.h>
#include<time.h>


using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	       
	VideoCapture cap(0); 
        if(!cap.isOpened())	
	return -1;

	clock_t Time_Start, Time_End, Time_Difference; // Clock used to measure time for CPU 
	double Time;

	Mat input1;	
	Mat input2;

	Mat output;  
	/*
	cv::VideoCapture cap;
	cap.open(string(argv[1]));*/

	int erosion_size=0;
	int d_size=2;
	
	for(;;)
    	{      

	Time_Start=clock();
	
	cap>>input1;
	cap>>input2;	
	
	output=input1-input2;		
		
	cvtColor(output,output,CV_BGR2GRAY);

	threshold(output,output,100,255,THRESH_BINARY);

	Mat element = getStructuringElement(MORPH_RECT,Size(2*erosion_size+1,2*erosion_size+1),
                                       			Point(erosion_size,erosion_size));

 	erode(output,output,element);

	Mat element1 = getStructuringElement(MORPH_RECT,Size(2*d_size+1,2*d_size+1),
                                       			Point(d_size,d_size));

	dilate(output,output,element1);
	dilate(output,output,element1);
	dilate(output,output,element1);
	dilate(output,output,element1);

	Time_End=clock();
	Time_Difference=Time_End-Time_Start;
	Time=Time_Difference/(double)CLOCKS_PER_SEC ;
	printf ("CPU Frame Rate = %f FPS\n",1/Time);

    	RNG rng(12345);
 	
  	vector<vector<Point> >contours;
  	vector<Vec4i>hierarchy;
  
  	findContours(output,contours,hierarchy,CV_RETR_TREE,CV_CHAIN_APPROX_SIMPLE,Point(0, 0));

	vector<vector<Point> > contours_poly(contours.size());
 	vector<Rect>boundRect(contours.size());
	vector<Point2f>center(contours.size());
 	vector<float>radius(contours.size());

  	for(int i=0;i<contours.size();i++)
		{ 
			approxPolyDP(Mat(contours[i]),contours_poly[i],3,true);
			boundRect[i]=boundingRect(Mat(contours_poly[i]));
			minEnclosingCircle((Mat)contours_poly[i],center[i],radius[i]);
		}


	for(int i=0;i<contours.size();i++)
		{
			Scalar color = Scalar(0,0,255);
			rectangle(input1,boundRect[i].tl(),boundRect[i].br(),color,2,8,0);
		}


	namedWindow( "Motion Tracking",CV_WINDOW_AUTOSIZE);
	imshow( "Motion Tracking", input1 );
	
	
	
      
        if(waitKey(30)>=0) break;
    }
   
    return 0;
}
