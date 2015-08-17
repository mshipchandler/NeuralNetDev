/* 
	Ma'ad Shipchandler
	Set up and Run the network implementation file - with descriptor extractor
	for multiple images
	30-7-2015
*/

#include <iostream>
#include <vector> // std::vector
#include <random>
#include <unistd.h> // usleep()
#include "Node.h"
#include "data.h"
#include "descriptor_header.h" // Also includes the OpenCV headers.
#include "activation_functions.h"

#define INPUTNUM 64 // Not including the Bias for the hidden layer.
#define HIDDENNUM 128 // Not including the Bias for output the layer.
#define OUTPUTNUM 1

// Function to calculate RMS (Takes into consideration cases with multiple outputs)
void calcRMS(std::vector<Node>& outputLayer, double ideal_output)
{
	double error = 0, RMS;
	for(size_t i = 0; i < outputLayer.size(); i++)
	{
		double delta = ideal_output - outputLayer[i].getNodeVal();
		error += delta * delta;
	}
	error /= outputLayer.size(); 
	RMS = sqrt(error);

	std::cout << "RMS: " << RMS << std::endl;
}


// Function will return a random weight between -1.0 and 1.0
double randomWeight()
{
	std::random_device rd;
	std::default_random_engine e1(rd());
	std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

	return uniform_dist(e1);
}

void compare(double real_output, double ideal_output)
{
	if((ideal_output == 1 && real_output > 0.9)|| (ideal_output == 0 && real_output < 0.1))
		std::cout << "TARGETS MATCH." << " --> ";
	else
		std::cout << "TARGETS DO NOT MATCH YET." << " --> ";

	std::cout << "[IDEAL OUTPUT: " << ideal_output 
			  << ", REAL OUTPUT: " << real_output << "]" << std::endl;

	//std::cin.ignore(); // For debugging
	//usleep(20000);
}

/*
	Function to check if output is a part of the keyboard or not. Stores keypoint
	if it is a part of the keyboard.
*/

void chessboard_check(double output, const KeyPoint& kp, std::vector<KeyPoint>& keypoints_of_chessboard)
{
	if(output > 0.9)
	{
		std::cout << "Part of the Chessboard." << std::endl;
		keypoints_of_chessboard.push_back(kp);
	}
	else
	{
		std::cout << "Not a part of the Chessboard." << std::endl;	
	}
}


// For display purposes. Highlights or flashes keypoints of detected chessboard
void detect_chessboard(Mat image, std::vector<KeyPoint>& keypoints)
{
	Mat keyPointImage;

	for(int i = 0; i < 500; i++)
	{
		drawKeypoints(image, keypoints, keyPointImage);

		imshow("Chessboard detected", keyPointImage);
		waitKey(10);
	}
	std::cin.ignore();
}

/*
	Structure used for TRAINING purposes. Holds an image and a flag to show if
	it is a keyboard or not.
*/
struct Images
{
	Mat image;
	double ideal_output;

	Images(Mat _image, double _ideal_output) : image(_image), ideal_output(_ideal_output) { }
};

int main(int argc, char* argv[])
{
	std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

	// Loading images -----------------------------------------------
	Mat temp;
	std::vector<Images> image_stream;
	std::string baseLoc = "./resources/images/";

	// Adding images to the image stream.
	for(int i = 0; i < 6; i++)
	{
		std::string updatedLoc = baseLoc + "positive/" + std::to_string(i + 1) + ".jpg";
		temp = imread(updatedLoc, CV_LOAD_IMAGE_COLOR);
		if(!temp.data)
		{
			std::cerr << "Error: Could not load image: " << updatedLoc << std::endl;
			return -2;
		}
		image_stream.push_back(Images(temp, 1));

		updatedLoc = baseLoc + "negative/" + std::to_string(i + 1) + ".jpg";
		temp = imread(updatedLoc, CV_LOAD_IMAGE_COLOR);
		if(!temp.data)
		{
			std::cerr << "Error: Could not load image: " << updatedLoc << std::endl;
			return -2;
		}
		image_stream.push_back(Images(temp, 0));
	}
	// --------------------------------------------------------------

	std::vector<Node> inputLayer;
	std::vector<Node> hiddenLayer;
	std::vector<Node> outputLayer;
	int ID = 0;
	// Bias node activation is always 1, generally.
	double biasNodeForHidden_Val = 1.0, biasNodeForOutput_Val = 1.0;

	// Setup Layers -------------------------------------------------

	/*  Taking some values as initial nodeVal for all layers.
		inputLayer nodeVal will be replaced by training/testing data.
		hiddenLayer and outputLayer nodeVal will be calculated using
		weighted sums as per neural net heuristic. */

	/*  Any activation function can be used. Different activation function for
		for different Nodes can be used as well. */

	// DO NOT FORGET TO DEALLOCATE ----------------------------------
	//iActivationFunction* act_func = new ActivationStep();
	//iActivationFunction* act_func = new ActivationSigmoid();
	//iActivationFunction* act_func = new ActivationSigmoidBipolar();
	iActivationFunction* act_func = new ActivationTanh();
	//iActivationFunction* act_func = new ActivationLinear();
	// --------------------------------------------------------------

	for(int i = 0; i < INPUTNUM; i++)
		inputLayer.push_back(Node(INPUT, ID++, 0.0, false, act_func)); 
	inputLayer.push_back(Node(INPUT, ID++, biasNodeForHidden_Val, true, act_func)); // Bias Node

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer.push_back(Node(HIDDEN, ID++, 0.0, false, act_func));
	hiddenLayer.push_back(Node(INPUT, ID++, biasNodeForOutput_Val, true, act_func)); // Bias Node

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer.push_back(Node(OUTPUT, ID++, 0.0, false, act_func));
	
	// --------------------------------------------------------------

	// Setting up connections and randomizing weights ---------------

	for(int i = 0; i < HIDDENNUM; i++)
	{
		for(int j = 0; j < INPUTNUM + 1; j++) // + 1 for Bias Node
		{
			inputLayer[j].setWeight(randomWeight(), &hiddenLayer[i]);
			hiddenLayer[i].setWeightPort(&inputLayer[j]);
		}
	}

	for(int i = 0; i < OUTPUTNUM; i++)
	{
		for(int j = 0; j < HIDDENNUM + 1; j++) // + 1 for Bias Node
		{
			hiddenLayer[j].setWeight(randomWeight(), &outputLayer[i]);
			outputLayer[i].setWeightPort(&hiddenLayer[j]);
		}
	}

	// --------------------------------------------------------------

	// Running the Neural Net----------------------------------------

	/*  Here, for the inputLayer, nodeVal is taken from the data set
		and for the hiddenLater and the outputLayer, nodeVal is 
		calculated using weighted sums as per neural net 
		heuristic. */

	int trainingCount = 0, ideal_output, feature_vector_size = 0;
	std::string image_name;
	std::vector<std::vector<double>> feature_vector_train;
	int image_counter = 0;

	while(1)
	{
		// Add a new image to process once an older one has been processed.
		if(trainingCount >= feature_vector_size)
		{
			/*trainingCount = 0;
			std::cout << "Enter the image to train('None' to " <<
				"move onto testing): ";
			std::cin >> image_name;
			if(image_name.compare("None") == 0)
				break;
			std::cout << "Positive training? (1 or 0): ";
			std::cin >> ideal_output;

			Mat image = imread(image_name, CV_LOAD_IMAGE_COLOR);
			if(!image.data)
			{
				std::cerr << "Error: Could not load image: " << 
				image_name << std::endl;
				return -2;
			}

			std::vector<KeyPoint> keypoints;
			feature_vector_train = getDescriptors(image, keypoints);
			feature_vector_size = (int)feature_vector_train.size();*/

			if(image_counter == (int)image_stream.size())
				break; // break out of while loop

			trainingCount = 0;
			std::cout << "Training phase: " << std::endl;
			Mat image = image_stream[image_counter].image.clone();
			ideal_output = image_stream[image_counter].ideal_output;
			image_counter++;

			imshow("Processing Image", image);
			waitKey(1000);

			std::vector<KeyPoint> keypoints;
			feature_vector_train = getDescriptors(image, keypoints);
			feature_vector_size = (int)feature_vector_train.size();
		}


		// 'Running' the Net --------------------------------------------
		for(int i = 0; i < INPUTNUM; i++)
		{
			inputLayer[i].setNodeVal(feature_vector_train[trainingCount][i]);
		}
		
		for(int i = 0; i < HIDDENNUM; i++) // Exclude the bias since it has a constant nodeVal.
		{
			hiddenLayer[i].calculateNodeVal();
		}

		for(int i = 0; i < OUTPUTNUM; i++) // Exclude the bias since it has a constant nodeVal.
		{
			outputLayer[i].calculateNodeVal();	
		}
		// --------------------------------------------------------------

		// Comparing to see if the outputs match.
		compare(outputLayer.back().getNodeVal(), ideal_output);
		calcRMS(outputLayer, ideal_output);

		// Now, to calculate error and update weights -------------------
		for(int i = 0; i < OUTPUTNUM; i++) // Bias Node not used
		{
			outputLayer[i].calculateErrorGradients(ideal_output);
			//outputLayer[i].updateWeights();
		}

		for(int i = 0; i < HIDDENNUM; i++) // Bias Node not used
		{
			hiddenLayer[i].calculateErrorGradients(UNDEF);
			//hiddenLayer[i].updateWeights();
		}

		for(int i = 0; i < OUTPUTNUM; i++) // Bias Node not used
			outputLayer[i].updateWeights();

		for(int i = 0; i < HIDDENNUM; i++) // Bias Node not used
			hiddenLayer[i].updateWeights();

		// --------------------------------------------------------------

		trainingCount++;
		std::cout << "Training Cycle: " << trainingCount << std::endl << std::endl;
	}

	// TESTING THE IMAGE --------------------------------------------

	while(1)
	{
		std::string test_image_name;
		std::cout << "Enter the image to test ('Exit' to quit): ";
		std::cin >> test_image_name;

		if(test_image_name.compare("Exit") == 0)
			break;

		Mat test_image = imread(test_image_name, CV_LOAD_IMAGE_COLOR);
		if(!test_image.data)
		{
			std::cerr << "Error: Could not load image: " << test_image_name << std::endl;
			continue; // rerun the loop
		}

		std::vector<KeyPoint> keypoints;
		std::vector<KeyPoint> keypoints_of_chessboard;
		std::vector<std::vector<double>> feature_vector_test = 
											getDescriptors(test_image, keypoints);	

		int totalTrainingCount_test_image = (int)feature_vector_test.size();
		int trainingCount_test_image = 0;

		while(trainingCount_test_image < totalTrainingCount_test_image)
		{

			// 'Running' the Net
			for(int i = 0; i < INPUTNUM; i++)
			{
				inputLayer[i].setNodeVal(feature_vector_test[trainingCount_test_image][i]);
			}

			// 1 - Chessboard, 0 - No Chessboard.
			ideal_output = 1; // THIS WILL CHANGE. 1 for now, since I am only feeding chessboards to the net.
			
			for(int i = 0; i < HIDDENNUM; i++) // Exclude the bias since it has a constant nodeVal.
			{
				hiddenLayer[i].calculateNodeVal();
			}

			for(int i = 0; i < OUTPUTNUM; i++) // Exclude the bias since it has a constant nodeVal.
			{
				outputLayer[i].calculateNodeVal();	
			}

			chessboard_check(outputLayer.back().getNodeVal(), keypoints[trainingCount_test_image], keypoints_of_chessboard);

			trainingCount_test_image++;
		}

		detect_chessboard(test_image, keypoints_of_chessboard);
	}

	// --------------------------------------------------------------

	// Display Node information -------------------------------------

	/*for(int i = 0; i < INPUTNUM; i++)
		inputLayer[i].display();

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer[i].display();

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer[i].display();*/

	// --------------------------------------------------------------

	delete act_func; // Deallocate dynamically allocated memory
	return 0;
}