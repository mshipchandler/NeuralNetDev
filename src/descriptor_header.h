/*
	Ma'ad Shipchandler
	Descriptor extraction functionality
	21-09-2015
*/

// OpenCV 3 Library List ---------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> 
#include <opencv2/xfeatures2d.hpp> // xfeatures2d, SURF, drawKeypoints()
using namespace cv;
// -------------------------------------------------------------------------

std::vector<std::vector<double>> getDescriptors(Mat image)
{
	int minHessian = 400;
	Ptr<xfeatures2d::SURF> surf = xfeatures2d::SURF::create(minHessian);

	std::vector<KeyPoint> keypoints;
	Mat descriptors;

	surf->detectAndCompute(image, Mat(), keypoints, descriptors);

	std::vector<std::vector<double>> feature_vector;
	std::vector<double> row_feature_vector;

	for(int row = 0; row < descriptors.rows; row++)
	{
		for(int column = 0; column < descriptors.cols; column++)
		{
			row_feature_vector.push_back(descriptors.at<float> (row, column));
		}
		feature_vector.push_back(row_feature_vector);
	}

	return feature_vector;
}