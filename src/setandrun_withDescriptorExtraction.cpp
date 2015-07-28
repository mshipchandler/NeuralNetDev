/* 
	Ma'ad Shipchandler
	Set up and Run the network implementation file - with image preprocessor
	24-7-2015
*/

#include <iostream>
#include <vector> // std::vector
#include <random>
#include <unistd.h> // usleep()
#include "Node.h"
#include "data.h"
#include "descriptor_header.h"
#include "activation_functions.h"

#define INPUTNUM 64 // Not including the Bias for the hidden layer.
#define HIDDENNUM 128 // Not including the Bias for output the layer.
#define OUTPUTNUM 1

// Function to calculate RMS (Takes into consideration cases with multiple outputs)
void calcRMS(std::vector<Node>& outputLayer, double ideal_output)
{
	double error = 0, RMS;
	for(int i = 0; i < (int)outputLayer.size(); i++)
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
	usleep(20000);
}

void chessboard_check(double output)
{
	if(output > 0.9)
	{
		std::cout << "Part of the Chessboard!" << std::endl;
	}
	else
	{
		std::cout << "Not a part of the Chessboard!" << std::endl;	
	}

	usleep(20000);
}

int main(int argc, char* argv[])
{
	std::cout << "OpenCV Version: " << CV_VERSION << std::endl;

	if(argc != 4)
	{
		std::cerr << "Error: Please enter TWO images." << std::endl;
		std::cerr << " Usage: " << argv[0] << 
			" train_image.extension test_image.extenstion" << std::endl;
		return 1;
	}

	Mat train_image1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat train_image2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	if(train_image1.empty() || train_image2.empty())
	{
		std::cerr << "Error: Could not load test image: " << 
			argv[1] << std::endl;
		return 2;
	}

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


	// Image Processing Work-----------------------------------------

	std::vector<std::vector<double>> feature_vector_train = 
										getDescriptors(train_image1);
	int image1_feature_vector_limit = (int)feature_vector_train.size();
	std::vector<std::vector<double>> temp = getDescriptors(train_image2);

	feature_vector_train.insert(feature_vector_train.end(), temp.begin(), temp.end());

	// --------------------------------------------------------------


	// Running the Neural Net----------------------------------------

	/*  Here, for the inputLayer, nodeVal is taken from the data set
		and for the hiddenLater and the outputLayer, nodeVal is 
		calculated using weighted sums as per neural net 
		heuristic. */
	int totalTrainingCount = (int)feature_vector_train.size(); // or image.rows * image.cols;
	//std::cout << totalTrainingCount = Total pixels = " << totalTrainingCount << std::endl;

	int trainingCount = 0, ideal_output;

	while(trainingCount < totalTrainingCount)
	{
		// 'Running' the Net --------------------------------------------
		for(int i = 0; i < INPUTNUM; i++)
		{
			inputLayer[i].setNodeVal(feature_vector_train[trainingCount][i]);
		}

		// 1 - Chessboard, 0 - No Chessboard.
		ideal_output = 1; // THIS WILL CHANGE. 1 for now, since I am only feeding chessboards to the net.
		if(trainingCount > image1_feature_vector_limit)
			ideal_output = 0;
		
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

	Mat test_image = imread(argv[3], CV_LOAD_IMAGE_COLOR);

	std::vector<std::vector<double>> feature_vector_test = 
										getDescriptors(test_image);	

	int totalTrainingCount_test_image = (int)feature_vector_test.size();
	int trainingCount_test_image = 0;

	while(trainingCount_test_image < totalTrainingCount_test_image)
	{

		// 'Running' the Net --------------------------------------------
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

		chessboard_check(outputLayer.back().getNodeVal());

		trainingCount_test_image++;
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