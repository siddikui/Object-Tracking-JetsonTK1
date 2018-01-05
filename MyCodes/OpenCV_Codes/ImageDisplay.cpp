#include<opencv2/highgui/highgui.hpp>

int main(int argc, char** argv)
{
	cv::Mat img = cv::imread(argv[1],-1);

	if(img.empty()) return -1;

	cv::namedWindow("Image Display",cv::WINDOW_AUTOSIZE);

	cv::imshow("Image Display",img);

	cv::waitKey(0);
	
	cv::destroyWindow("Image Display");	
}
