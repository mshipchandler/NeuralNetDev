/*
	Ma'ad Shipchandler
	Neural Net Input Module 2.0 - image_preprocessor.h
	21-09-2015
*/

/*
	SUGGESTED INPUT IMAGE SIZE: 256 x 256
*/

#include <iostream>
#include <vector>
#include <fstream>

// OpenCV 3 Library List -------------------------------------------------------------------
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> // CV_LOAD_IMAGE_COLOR, imread(), imshow(), waitKey()
#include <opencv2/calib3d.hpp> // Blob detection functionality
#include <opencv2/imgproc.hpp> // line, corner, edge detection functionality
using namespace cv;
// -----------------------------------------------------------------------------------------

/*
	This struct holds pixel characteristics for each pixel in an image.
*/
struct PixelChar
{
	Point2f coordinates;
	float intensity;
	bool blobFlag, lineFlag, cornerFlag;

	PixelChar() { coordinates.x = -1, coordinates.y = -1; }
	PixelChar(Point2f point, float _intensity) : coordinates(point), intensity(_intensity)
	{
		blobFlag = false, lineFlag = false, cornerFlag = false;
	}

	void display() // Debug purposes
	{
		std::cout << "Coordinates: " << coordinates << std::endl;
		std::cout << " Intensity: " << intensity << std::endl;
		std::cout << " Blob?: " << blobFlag << std::endl;
		std::cout << " Line?: " << lineFlag << std::endl;
		std::cout << " Corner?: " << cornerFlag << std::endl;
	}
};

/* 
	Function will extract pixel co-ordinates and their itensity values,
	and start adding them into the image_features vector.
*/
void extractIntensity(const Mat& image, std::vector<PixelChar>& image_features)
{
	Mat image_gray;
	cvtColor(image, image_gray, CV_BGR2GRAY);

	for(int row = 0; row < image_gray.rows; row++)
	{
		for(int column = 0; column < image_gray.cols; column++)
		{
			Point2f pixelPoint = Point2f(column, row);
			Vec3b channels = image_gray.at<uchar> (row, column);

			image_features.push_back(PixelChar(pixelPoint, channels[0]));
		}
	}
}

/*
	Function detects blobs in an image and adds the co-ordinates of the 
	pixels in the blob the the vector blobCoords.
*/
void blobDetection(Mat image, std::vector<Point2f>& blobCoords)
{
	SimpleBlobDetector::Params params;
	params.blobColor = 255; // Light blobs
	Ptr<SimpleBlobDetector> blobDetector_white = SimpleBlobDetector::create(params);
	params.blobColor = 0; // Dark blobs
	Ptr<SimpleBlobDetector> blobDetector_black = SimpleBlobDetector::create(params);

	// Detect blobs.
	std::vector<KeyPoint> keypoints_black, keypoints_white, keypoints;

	blobDetector_black->detect(image, keypoints_black);
	blobDetector_white->detect(image, keypoints_white);
	for(size_t i = 0; i < keypoints_black.size(); i++)
		keypoints.push_back(keypoints_black[i]);
	for(size_t i = 0; i < keypoints_white.size(); i++)
		keypoints.push_back(keypoints_white[i]);

	Size image_size = image.size();
	Mat mask = Mat::zeros(image_size.height, image_size.width, CV_8U);

	for(size_t i = 0; i < keypoints.size(); i++)
	{
		Point2f kp_center = keypoints[i].pt;
		float kp_radius = keypoints[i].size / 2;
		circle(mask, kp_center, kp_radius, Scalar(255), -1);
	}

	// Adding pixels located within the blob radius
	for(int row = 0; row < image.rows; row++)
	{
		for(int column = 0; column < image.cols; column++)
		{
			Vec3b channels = mask.at<uchar> (row, column);
			if(channels[0] > 0)
				blobCoords.push_back(Point2f(column, row));
		}
	}

	/*imshow("Mask", mask);
	imshow("Original", image);
	waitKey(0);*/
}

/*
	Function detects presence of lines in an image and adds the co-ordinates
	of the pixel that fall on the line to the vector pointsOnLine.
*/
void lineDetection(Mat image, std::vector<Point2f>& pointsOnLine)
{
	Mat output;

	/*
		Canny(Mat input, Mat output, double threshold1, 
			double threshold2, int apertureSize = 3, bool L2gradient = false);

		input:Single channel 8-bit input image
		output: output edge map (same size and type as image)
		threshold1: first threshold for hysteresis procedure
		threshold2: second threshold for hysteresis procedure
		apertureSize: aperture size for Sobel() operator
		L2gradient: a flag, indicating whether a more accurate 
					L2 norm =\sqrt{(dI/dx)^2 + (dI/dy)^2} should be used to 
					calculate the image gradient magnitude (L2gradient=true), 
					or whether the default L1 norm =|dI/dx|+|dI/dy| is enough (L2gradient=false)
	*/
	Canny(image, output, 300, 400, 3, true);

	/*
		HoughLines(Mat image, vector of lines, double rho, double theta, int thresh,
			double srn = 0, double stn = 0); <-- Returns Polar coordinates
		HoughLinesP(Mat image, vector of lines, doubel rho, double theta, int thresh
			double minLineLength = 0, double maxLineGap = 0); <-- Returns cartesian coordinates
	*/

	#if 0
	// This calculation is done to convert polar coordinates to the cartesian plane.
	std::vector<Vec2f> lines;
	HoughLines(output, lines, 1, CV_PI/180, 100);

	for(size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0];
		float theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		Point pt1(cvRound(x0 + 1000*(-b)),
				  cvRound(y0 + 1000*(a)));
		Point pt2(cvRound(x0 - 1000*(-b)),
				  cvRound(y0 - 1000*(a)));
		line(output, pt1, pt2, Scalar(255, 255, 255), 3, CV_AA);
	}
	#else
	std::vector<Vec4i> lines;
	HoughLinesP(output, lines, 1, CV_PI/180, 80, 50, 50);
	for(size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		//line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);

		// Getting all the points on the line.
		LineIterator lineIter(image, Point(l[0], l[1]), Point(l[2], l[3]), 8);
		for(int i = 0; i < lineIter.count; i++, ++lineIter)
		{
			Point2f pOnL= lineIter.pos();
			//image.at<Vec3b>(pOnL) = 255; // To see if LineIterator works properly
			pointsOnLine.push_back(pOnL);
		}
		
	}
	#endif
}

/*
	Function detects corners in an image and adds the co-ordinates of the corners
	to the vector corners.
*/
void cornerDetection(Mat image, std::vector<Point2f>& corners)
{
	Mat image_gray;
	cvtColor(image, image_gray, CV_BGR2GRAY);
	int thresh = 200, max_thresh = 255;

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(image.size(), CV_32FC1);

	// Detector parameters.
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// Detecting corners.
	cornerHarris(image_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	// Normalizing.
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	// Drawing a circle around corners.
	for(int i = 0; i < dst_norm.rows; i++)
	{
		for(int j = 0; j < dst_norm.cols; j++)
		{
			if((int)dst_norm.at<float>(i,j) > thresh && (int)dst_norm.at<float>(i,j) < max_thresh)
			{
				//circle(dst_norm_scaled, Point(i, j), 5, Scalar(255, 255, 255), 1, CV_AA, 0);
				corners.push_back(Point2f(i, j));
			}
		}
	}
}

/*
	Function updates the image_features vector by checking off flags
	for blobs, points on a line and corners.
*/
void updateImageFeatures(std::vector<PixelChar>& image_features,
						 const std::vector<Point2f>& blobCoords, 
						 const std::vector<Point2f>& pointsOnLine,
						 const std::vector<Point2f>& corners)
{
	std::cout << "Updating the vector" << std::endl;

	#pragma omp parallel for
	for(size_t i = 0; i < image_features.size(); i++)
	{
		#pragma omp parallel for
		for(size_t j = 0; j < blobCoords.size(); j++)
		{
			if(image_features[i].coordinates == blobCoords[j])
				image_features[i].blobFlag = true;
		}

		#pragma omp parallel for
		for(size_t j = 0; j < pointsOnLine.size(); j++)
		{
			if(image_features[i].coordinates == pointsOnLine[j])
				image_features[i].lineFlag = true;
		}

		#pragma omp parallel for
		for(size_t j = 0; j < corners.size(); j++)
		{
			if(image_features[i].coordinates == corners[j])
				image_features[i].cornerFlag = true;
		}
	}
}

/*
	Writing data to file.
*/
void writeToFile(const std::vector<PixelChar>& image_features)
{
	std::ofstream fout("../resources/training_data/chessBoardtrainingData.csv");
	fout << "x, y, intensity, blob, line, corner" << std::endl;

	std::cout << "Writing to file." << std::endl;
	for(size_t i = 0; i < image_features.size(); i++)
	{
		fout << image_features[i].coordinates.x << ", " << 
				image_features[i].coordinates.y << ", " << 
				image_features[i].intensity << ", " << 
				image_features[i].blobFlag << ", " << 
				image_features[i].lineFlag << ", " << 
				image_features[i].cornerFlag << std::endl;
	}	
}

/*
	Controller function to process the image by calling necessary sub 
	functions.
*/
void processImage(const Mat& image, std::vector<PixelChar>& image_features)
{
	extractIntensity(image, image_features);

	std::vector<Point2f> blobCoords;
	blobDetection(image, blobCoords);

	std::vector<Point2f> pointsOnLine;
	lineDetection(image, pointsOnLine);

	std::vector<Point2f> corners;
	cornerDetection(image, corners);

	updateImageFeatures(image_features, blobCoords, pointsOnLine, corners);

	writeToFile(image_features);
}

/*
int main(int argc, char* argv[])
{
	std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

	if(argc != 2)
	{
		std::cerr << "Error: Please enter ONE image." << std::endl;
		std::cerr << " Usage: " << argv[0] << " image.extension" << std::endl;
		return 1;
	}

	Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	if(image.empty())
	{
		std::cerr << "Error: Could not load image." << std::endl;
		return 2;
	}

	imshow("Image", image);
	waitKey(0);

	// -----------------------------------------
	std::vector<PixelChar> image_features;
	processImage(image, image_features);
	// -----------------------------------------

	return 0;
}*/