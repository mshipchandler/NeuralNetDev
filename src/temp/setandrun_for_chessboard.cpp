/* 
	Ma'ad Shipchandler
	Set up and Run the network implementation file
	2-6-2015
*/

#include <iostream>
#include <vector> // std::vector
#include <random>
#include <unistd.h>
#include "../Node.h"
#include "data_chess.h"

#define INPUTNUM 4 // Excluding Bias for hidden layer.
#define HIDDENNUM 10 // Excluding Bias for output layer.
#define OUTPUTNUM 1

double randomWeight()
{
	std::random_device rd;
	std::default_random_engine e1(rd());
	std::uniform_real_distribution<double> uniform_dist(-1.0, 1.0);

	return uniform_dist(e1);
}

void compare(double real_output, double ideal_output)
{
	if((ideal_output == 1 && real_output > 0.9) || (ideal_output == 0 && real_output < 0.1))
		std::cout << "TARGETS MATCH." << " --> ";
	else
		std::cout << "TARGETS DO NOT MATCH YET." << " --> ";

	std::cout << "[IDEAL OUTPUT: " << ideal_output 
			  << ", REAL OUTPUT: " << real_output << "]" << std::endl;

	//std::cin.ignore(); // For debugging
	usleep(20000);
}

int main(int argc, char* argv[])
{
	std::vector<Node> inputLayer;
	std::vector<Node> hiddenLayer;
	std::vector<Node> outputLayer;
	int ID = 0;
	double biasNodeForHiddenVal = 1.0, biasNodeForOutputVal = 1.0;

	// Setup Layers -------------------------------------------------

	/*  Taking some values as initial nodeVal for all layers.
		inputLayer nodeVal will be replaced by training/testing data.
		hiddenLayer and outputLayer nodeVal will be calculated using
		weighted sums as per neural net heuristic. */

	for(int i = 0; i < INPUTNUM; i++)
		inputLayer.push_back(Node(INPUT, ID++, 0.0, false)); 
	inputLayer.push_back(Node(INPUT, ID++, biasNodeForHiddenVal, true)); // Bias Node

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer.push_back(Node(HIDDEN, ID++, 0.0, false));
	hiddenLayer.push_back(Node(INPUT, ID++, biasNodeForOutputVal, true)); // Bias Node

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer.push_back(Node(OUTPUT, ID++, 0.0, false));
	
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

	int trainingCount = 0, row = 0, ideal_output;
	while(trainingCount < 65536)
	{
		// 'Running' the Net --------------------------------------------
		for(int i = 0; i < INPUTNUM; i++)
		{
			inputLayer[i].setNodeVal(chessboard[row][i]);
			/*if(chessboard[row-1][i] == -9.99)
				std::cin.ignore();*/
		}
		ideal_output = 1;
		row++;

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

	// --------------------------------------------------------------

	// Display Node information -------------------------------------

	/*for(int i = 0; i < INPUTNUM; i++)
		inputLayer[i].display();

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer[i].display();

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer[i].display();*/

	// --------------------------------------------------------------

	return 0;
}