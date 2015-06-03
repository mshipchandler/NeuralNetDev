/* 
	Ma'ad Shipchandler
	Set up and Run the network implementation file
	2-6-2015
*/

#include <iostream>
#include <vector> // std::vector
#include <random>
#include "Node.h"
#include "data.h"

#define INPUTNUM 2
#define HIDDENNUM 3
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
	if(real_output == ideal_output)
		std::cout << "TARGETS MATCH!" << std::endl;
	else
		std::cout << "TARGETS DO NOT MATCH YET!" << std::endl;

	std::cout << "IDEAL OUTPUT: " << ideal_output 
			  << " REAL OUTPUT: " << real_output << std::endl;
}

int main(int argc, char* argv[])
{
	std::vector<Node> inputLayer;
	std::vector<Node> hiddenLayer;
	std::vector<Node> outputLayer;
	int ID = 0;

	// Setup Layers -------------------------------------------------

	/*  Taking some values as initial nodeVal for all layers.
		inputLayer nodeVal will be replaced by training/testing data.
		hiddenLayer and outputLayer nodeVal will be calculated using
		weighted sums as per neural net heuristic. */

	for(int i = 0; i < INPUTNUM; i++)
		inputLayer.push_back(Node(INPUT, ID++, 4.2)); 

	for(int i = 0; i < HIDDENNUM; i++)
		hiddenLayer.push_back(Node(HIDDEN, ID++, 0.0));

	for(int i = 0; i < OUTPUTNUM; i++)
		outputLayer.push_back(Node(OUTPUT, ID++, 0.0));
	
	// --------------------------------------------------------------

	// Setting up connections and randomizing weights ---------------

	for(int i = 0; i < HIDDENNUM; i++)
	{
		for(int j = 0; j < INPUTNUM; j++)
		{
			inputLayer[j].setWeight(randomWeight(), &hiddenLayer[i]);
			hiddenLayer[i].setWeightPort(&inputLayer[j]);
		}
	}

	for(int i = 0; i < OUTPUTNUM; i++)
	{
		for(int j = 0; j < HIDDENNUM; j++)
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
	while(trainingCount < 400)
	{
		// 'Running' the Net.
		for(int i = 0; i < INPUTNUM; i++)
		{
			inputLayer[i].setNodeVal(XOR_data[row][i]);
		}
		ideal_output = XOR_data[row][2];
		row++;
		if(row == 4) { row = 0; }

		for(int i = 0; i < HIDDENNUM; i++)
		{
			hiddenLayer[i].calculateNodeVal();
		}

		for(int i = 0; i < OUTPUTNUM; i++)
		{
			outputLayer[i].calculateNodeVal();	
		}

		// Comparing to see if the outputs match.
		compare(outputLayer.back().getNodeVal(), ideal_output);

		// Now, to calculate error and update weights.
		for(int i = 0; i < OUTPUTNUM; i++)
		{
			outputLayer[i].calculateErrorGradients(ideal_output);
			outputLayer[i].updateWeights();
		}

		for(int i = 0; i < HIDDENNUM; i++)
		{
			hiddenLayer[i].calculateErrorGradients(UNDEF);
			hiddenLayer[i].updateWeights();
		}

		//trainingCount++;
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